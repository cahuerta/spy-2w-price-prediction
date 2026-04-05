# =========================================================
# analysis_router.py — ORDER ANALYSIS ENDPOINT
# =========================================================
# GET  /dashboard/order-analysis        → lee reporte cacheado
# POST /dashboard/order-analysis/run    → corre análisis y guarda
# =========================================================

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Any, Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    ALPACA_OK = True
except ImportError:
    ALPACA_OK = False

from broker import get_engine

logger = logging.getLogger("order_analysis")
router = APIRouter(prefix="/dashboard", tags=["analysis"])

DATA_PATH   = Path(os.getenv("DATA_PATH", "/data"))
EVAL_ROOT   = DATA_PATH / "evaluations"
REPORT_FILE = DATA_PATH / "order_analysis_report.json"

_running = False  # lock simple para evitar doble ejecución


# =========================================================
# HELPERS
# =========================================================

def _load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


# =========================================================
# PASO 1 — ÓRDENES ALPACA
# =========================================================

def _fetch_orders() -> List[Dict]:
    if not ALPACA_OK:
        return []
    try:
        engine = get_engine()
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=500,
        )
        orders = engine.client.get_orders(filter=request)
    except Exception as e:
        logger.error(f"Alpaca orders error: {e}")
        return []

    result = []
    for o in orders:
        if o.filled_at is None:
            continue

        client_id = getattr(o, "client_order_id", "") or ""
        is_system = "access_key" in client_id.lower() or client_id.startswith("auto")

        side_str = str(o.side).upper()
        is_buy   = "BUY" in side_str

        result.append({
            "symbol":    o.symbol,
            "is_buy":    is_buy,
            "source":    "system" if is_system else "manual",
            "filled_at": str(o.filled_at),
            "price":     float(o.filled_avg_price) if o.filled_avg_price else None,
        })

    return result


# =========================================================
# PASO 2 — EVALUACIONES EN DISCO
# =========================================================

def _load_evaluations() -> Dict[str, List[Dict]]:
    by_ticker: Dict[str, List[Dict]] = defaultdict(list)
    if not EVAL_ROOT.exists():
        return by_ticker
    for ticker_dir in sorted(EVAL_ROOT.iterdir()):
        if not ticker_dir.is_dir():
            continue
        for f in sorted(ticker_dir.glob("*.json")):
            data = _load_json(f)
            if not data or data.get("real_return_pct") is None:
                continue
            by_ticker[ticker_dir.name].append(data)
    return by_ticker


# =========================================================
# PASO 3 — MÉTRICAS POR GRUPO
# =========================================================

def _group_metrics(tickers: List[str], evals: Dict[str, List[Dict]], label: str) -> Dict:
    hits = 0
    total = 0
    errors, returns = [], []
    rec_stats   = defaultdict(lambda: {"hits": 0, "total": 0, "returns": []})
    horizon_stats = defaultdict(lambda: {"hits": 0, "total": 0})
    found, missing = [], []

    for ticker in tickers:
        evs = evals.get(ticker, [])
        if not evs:
            missing.append(ticker)
            continue
        found.append(ticker)

        for ev in evs:
            real_ret = float(ev["real_return_pct"])
            pred_ret = float(ev.get("predicted_return_pct", 0))
            rec      = (ev.get("recommendation") or "HOLD").strip().upper()
            hit      = bool(np.sign(real_ret) == np.sign(pred_ret)) if pred_ret != 0 else False

            total += 1
            hits  += int(hit)
            returns.append(real_ret)
            if ev.get("error_return_pct") is not None:
                errors.append(float(ev["error_return_pct"]))

            rec_stats[rec]["total"] += 1
            rec_stats[rec]["returns"].append(real_ret)
            if hit:
                rec_stats[rec]["hits"] += 1

            for hkey, hdata in (ev.get("models_diagnostics") or {}).items():
                if not isinstance(hdata, dict):
                    continue
                horizon_stats[hkey]["total"] += 1
                if hdata.get("hit_sign"):
                    horizon_stats[hkey]["hits"] += 1

    hit_rate = round(hits / total * 100, 2)       if total   else None
    avg_err  = round(float(np.mean(errors)), 2)   if errors  else None
    avg_ret  = round(float(np.mean(returns)), 2)  if returns else None

    by_h = {}
    best_h = None
    for hkey in sorted(horizon_stats.keys()):
        s  = horizon_stats[hkey]
        hr = round(s["hits"] / s["total"] * 100, 1) if s["total"] else None
        by_h[hkey] = {"hit_rate_pct": hr, "total": s["total"]}
        if hr and (best_h is None or hr > by_h[best_h]["hit_rate_pct"]):
            best_h = hkey

    by_rec = {}
    for rec, s in rec_stats.items():
        by_rec[rec] = {
            "total":        s["total"],
            "hit_rate_pct": round(s["hits"] / s["total"] * 100, 1) if s["total"] else None,
            "avg_return":   round(float(np.mean(s["returns"])), 2)  if s["returns"] else None,
        }

    return {
        "label":             label,
        "tickers_input":     len(tickers),
        "tickers_found":     len(found),
        "tickers_missing":   len(missing),
        "total_evaluations": total,
        "hit_rate_dir_pct":  hit_rate,
        "avg_error_pct":     avg_err,
        "avg_real_return":   avg_ret,
        "best_horizon":      best_h,
        "by_horizon":        by_h,
        "by_recommendation": by_rec,
        "tickers_list":      sorted(found),
    }


# =========================================================
# ANÁLISIS COMPLETO
# =========================================================

def _run_analysis() -> Dict:
    global _running
    _running = True
    try:
        orders = _fetch_orders()
        evals  = _load_evaluations()

        system_tickers = list(set(o["symbol"] for o in orders if o["is_buy"] and o["source"] == "system"))
        manual_tickers = list(set(o["symbol"] for o in orders if o["is_buy"] and o["source"] == "manual"))
        all_tickers    = list(evals.keys())

        report = {
            "generated_at":       datetime.now(timezone.utc).isoformat(),
            "orders_total":       len(orders),
            "system_tickers_n":   len(system_tickers),
            "manual_tickers_n":   len(manual_tickers),
            "universe":           _group_metrics(all_tickers,    evals, "Universo"),
            "system_buys":        _group_metrics(system_tickers, evals, "Sistema"),
            "manual_buys":        _group_metrics(manual_tickers, evals, "Manual"),
        }

        # Veredicto automático
        sys_hr = report["system_buys"]["hit_rate_dir_pct"]
        man_hr = report["manual_buys"]["hit_rate_dir_pct"]
        uni_hr = report["universe"]["hit_rate_dir_pct"]

        if sys_hr is not None and man_hr is not None:
            if man_hr > sys_hr:
                verdict = f"Manual supera al sistema en {round(man_hr - sys_hr, 1)}pp"
                winner  = "manual"
            elif sys_hr > man_hr:
                verdict = f"Sistema supera al manual en {round(sys_hr - man_hr, 1)}pp"
                winner  = "system"
            else:
                verdict = "Empate"
                winner  = "tie"
        else:
            verdict = "Insuficientes datos para comparar"
            winner  = "unknown"

        report["verdict"] = {"text": verdict, "winner": winner, "universe_baseline": uni_hr}

        _save_json(REPORT_FILE, report)
        logger.info(f"✅ Análisis completado: {verdict}")
        return report

    finally:
        _running = False


# =========================================================
# ENDPOINTS
# =========================================================

@router.get("/order-analysis")
async def get_order_analysis():
    """Lee el reporte cacheado. Si no existe, retorna status pending."""
    if REPORT_FILE.exists():
        data = _load_json(REPORT_FILE)
        if data:
            return {"status": "ready", **data}
    return {"status": "pending", "message": "Análisis no ejecutado aún. Usa POST /run"}


@router.post("/order-analysis/run")
async def run_order_analysis(background_tasks: BackgroundTasks):
    """Lanza el análisis en background. Retorna inmediatamente."""
    global _running
    if _running:
        return {"status": "running", "message": "Análisis ya en curso"}

    background_tasks.add_task(_run_analysis)
    return {"status": "started", "message": "Análisis iniciado. Consulta GET /order-analysis en ~10s"}

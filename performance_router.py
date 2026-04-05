# =========================================================
# performance_router.py — V5
# =========================================================
# V4 → V5:
#   [M1] model-quality agrega métricas de tendencia reciente:
#        - hit_rate_7d_pct   : últimos 7 días
#        - hit_rate_14d_pct  : últimos 14 días
#        - hit_rate_30d_pct  : últimos 30 días
#        - trend             : "mejorando" | "empeorando" | "estable"
#        Permite ver si el fix del theta está funcionando
#        sin resetear el historial completo.
# =========================================================

import os
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter
from broker import get_engine

DATA_PATH             = Path(os.getenv("DATA_PATH", "/data"))
META_FILE             = DATA_PATH / "account_meta.json"
EQUITY_SNAPSHOTS_FILE = DATA_PATH / "equity_snapshots.json"

router = APIRouter(prefix="/dashboard", tags=["performance"])


# =========================================================
# HELPERS
# =========================================================

def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def list_evaluation_files() -> List[Path]:
    root = DATA_PATH / "evaluations"
    if not root.exists():
        return []
    files = []
    for ticker_dir in root.iterdir():
        if ticker_dir.is_dir():
            files.extend(ticker_dir.glob("*.json"))
    return sorted(files)


# =========================================================
# SNAPSHOT DIARIO
# =========================================================

def record_equity_snapshot(equity: float) -> None:
    today     = datetime.now(timezone.utc).date().isoformat()
    snapshots = load_json(EQUITY_SNAPSHOTS_FILE) or {}
    if today not in snapshots:
        snapshots[today] = round(equity, 2)
        save_json(EQUITY_SNAPSHOTS_FILE, snapshots)


# =========================================================
# BLOQUE BROKER — /dashboard/performance
# =========================================================

@router.get("/performance")
async def performance():
    try:
        engine  = get_engine()
        account = await engine.get_account()
        equity  = float(account.equity)
    except Exception:
        equity = None

    meta = load_json(META_FILE)

    if not meta and equity is not None:
        meta = {
            "initial_equity":  equity,
            "start_date":      datetime.now(timezone.utc).isoformat(),
            "high_water_mark": equity,
        }
        save_json(META_FILE, meta)

    total_return_pct = None
    drawdown_pct     = None
    high_water_mark  = None

    if meta and equity is not None:
        initial_equity  = float(meta["initial_equity"])
        high_water_mark = float(meta.get("high_water_mark", equity))

        if equity > high_water_mark:
            high_water_mark          = equity
            meta["high_water_mark"]  = equity
            save_json(META_FILE, meta)

        total_return_pct = round((equity - initial_equity) / initial_equity * 100, 2)
        drawdown_pct     = round((equity - high_water_mark) / high_water_mark * 100, 2)

        record_equity_snapshot(equity)

    return {
        "equity":           equity,
        "initial_equity":   float(meta["initial_equity"]) if meta else None,
        "total_return_pct": total_return_pct,
        "drawdown_pct":     drawdown_pct,
        "high_water_mark":  high_water_mark,
        "since":            meta["start_date"] if meta else None,
    }


# =========================================================
# BLOQUE BROKER — /dashboard/equity-curve
# =========================================================

@router.get("/equity-curve")
async def equity_curve():
    curve  = []
    source = "alpaca"

    try:
        engine  = get_engine()
        history = engine.client.get_portfolio_history(
            period="1M",
            timeframe="1D",
            intraday_reporting="market_hours",
        )
        if history and history.equity:
            for ts, eq in zip(history.timestamp, history.equity):
                if eq is None or eq == 0:
                    continue
                date_str = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
                curve.append({"date": date_str, "equity": round(float(eq), 2)})
            source = "alpaca"
    except Exception:
        source = "snapshots"

    if not curve:
        snapshots = load_json(EQUITY_SNAPSHOTS_FILE) or {}
        for date in sorted(snapshots.keys()):
            curve.append({"date": date, "equity": snapshots[date]})
        source = "snapshots"

    if curve:
        base = curve[0]["equity"]
        for point in curve:
            point["return_pct"] = round(
                (point["equity"] - base) / base * 100, 2
            ) if base > 0 else 0.0

    meta           = load_json(META_FILE)
    initial_equity = float(meta["initial_equity"]) if meta else None
    current_equity = curve[-1]["equity"] if curve else None

    return {
        "curve":          curve,
        "initial_equity": initial_equity,
        "current_equity": current_equity,
        "n_days":         len(curve),
        "source":         source,
        "updated_at":     datetime.now(timezone.utc).isoformat(),
    }


# =========================================================
# HELPER — hit rate para un subset de evaluaciones
# =========================================================

def _hit_rate_for_files(files: List[Path]) -> Optional[float]:
    """Calcula hit rate direccional para una lista de archivos de evaluación."""
    hits  = 0
    total = 0
    for f in files:
        data = load_json(f)
        if not data or data.get("real_return_pct") is None:
            continue
        real_ret = float(data["real_return_pct"])
        pred_ret = float(data.get("predicted_return_pct", 0))
        if pred_ret == 0:
            continue
        total += 1
        if np.sign(real_ret) == np.sign(pred_ret):
            hits += 1
    return round(hits / total * 100, 2) if total >= 5 else None


def _eval_date(f: Path) -> Optional[str]:
    """Extrae evaluation_date del archivo o usa el stem como fallback."""
    data = load_json(f)
    if data:
        d = data.get("evaluation_date") or data.get("date_base")
        if d:
            return str(d)[:10]
    # fallback: nombre del archivo es YYYY-MM-DD.json
    try:
        datetime.strptime(f.stem, "%Y-%m-%d")
        return f.stem
    except ValueError:
        return None


# =========================================================
# BLOQUE MODELO — /dashboard/model-quality
# [M1] Agrega hit rate por ventana temporal reciente
# =========================================================

@router.get("/model-quality")
async def model_quality():

    files = list_evaluation_files()

    empty = {
        "hit_rate_direction_pct": None,
        "hit_rate_7d_pct":        None,
        "hit_rate_14d_pct":       None,
        "hit_rate_30d_pct":       None,
        "trend":                  None,
        "avg_error_pct":          None,
        "evaluated":              0,
        "pending":                0,
        "total":                  0,
        "by_recommendation":      {},
        "by_horizon":             {},
    }

    if not files:
        return empty

    # ── Fechas de corte ──────────────────────────────────────────
    now     = datetime.now(timezone.utc).date()
    cut_7d  = (now - timedelta(days=7)).isoformat()
    cut_14d = (now - timedelta(days=14)).isoformat()
    cut_30d = (now - timedelta(days=30)).isoformat()

    # Separar archivos por ventana temporal
    files_7d  = []
    files_14d = []
    files_30d = []

    total     = 0
    evaluated = 0
    hits_dir  = 0
    errors    = []

    rec_stats     = defaultdict(lambda: {"total": 0, "hits": 0, "errors": []})
    horizon_stats = defaultdict(lambda: {"hits": 0, "total": 0, "errors": []})

    for f in files:
        data = load_json(f)
        if not data:
            continue

        total += 1

        if data.get("real_return_pct") is None:
            continue

        evaluated += 1

        # Fecha de evaluación para ventanas temporales
        eval_date = (
            data.get("evaluation_date")
            or data.get("date_base")
            or f.stem
        )
        eval_date = str(eval_date)[:10] if eval_date else ""

        if eval_date >= cut_7d:
            files_7d.append(f)
        if eval_date >= cut_14d:
            files_14d.append(f)
        if eval_date >= cut_30d:
            files_30d.append(f)

        rec      = (data.get("recommendation") or "HOLD").strip().upper()
        real_ret = float(data["real_return_pct"])
        pred_ret = float(data.get("predicted_return_pct", 0))

        hit = bool(np.sign(real_ret) == np.sign(pred_ret)) if pred_ret != 0 else False
        if hit:
            hits_dir += 1

        if data.get("error_return_pct") is not None:
            err = float(data["error_return_pct"])
            errors.append(err)
            rec_stats[rec]["errors"].append(err)

        rec_stats[rec]["total"] += 1
        if hit:
            rec_stats[rec]["hits"] += 1

        models_diag = data.get("models_diagnostics") or {}
        for hkey, hdata in models_diag.items():
            if not isinstance(hdata, dict):
                continue
            horizon_stats[hkey]["total"] += 1
            if hdata.get("hit_sign"):
                horizon_stats[hkey]["hits"] += 1
            if hdata.get("error_pct") is not None:
                horizon_stats[hkey]["errors"].append(float(hdata["error_pct"]))

    pending      = total - evaluated
    hit_rate_all = round(hits_dir / evaluated * 100, 2) if evaluated > 0 else None
    avg_error    = round(float(np.mean(errors)), 2) if errors else None

    # ── [M1] Hit rates por ventana ───────────────────────────────
    hr_7d  = _hit_rate_for_files(files_7d)
    hr_14d = _hit_rate_for_files(files_14d)
    hr_30d = _hit_rate_for_files(files_30d)

    # Tendencia: compara 7d vs 30d (o vs histórico)
    trend = None
    ref   = hr_30d or hit_rate_all
    if hr_7d is not None and ref is not None:
        diff = hr_7d - ref
        if diff >= 3:
            trend = "mejorando"
        elif diff <= -3:
            trend = "empeorando"
        else:
            trend = "estable"

    # ── Formatear by_recommendation ─────────────────────────────
    by_rec = {}
    for rec, s in rec_stats.items():
        by_rec[rec] = {
            "total":         s["total"],
            "hit_rate_pct":  round(s["hits"] / s["total"] * 100, 1) if s["total"] else None,
            "avg_error_pct": round(float(np.mean(s["errors"])), 2) if s["errors"] else None,
        }

    # ── Formatear by_horizon ─────────────────────────────────────
    by_horizon = {}
    for hkey in sorted(horizon_stats.keys()):
        s = horizon_stats[hkey]
        by_horizon[hkey] = {
            "total":         s["total"],
            "hit_rate_pct":  round(s["hits"] / s["total"] * 100, 1) if s["total"] else None,
            "avg_error_pct": round(float(np.mean(s["errors"])), 2) if s["errors"] else None,
        }

    return {
        # Global histórico
        "hit_rate_direction_pct": hit_rate_all,
        "avg_error_pct":          avg_error,
        "evaluated":              evaluated,
        "pending":                pending,
        "total":                  total,

        # [M1] Ventanas recientes
        "hit_rate_7d_pct":        hr_7d,
        "hit_rate_14d_pct":       hr_14d,
        "hit_rate_30d_pct":       hr_30d,
        "trend":                  trend,
        "recent_window_sizes":    {
            "7d":  len(files_7d),
            "14d": len(files_14d),
            "30d": len(files_30d),
        },

        "by_recommendation": by_rec,
        "by_horizon":        by_horizon,
        "updated_at":        datetime.now(timezone.utc).isoformat(),
    }
    

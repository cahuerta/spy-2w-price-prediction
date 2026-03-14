# =========================================================
# performance_router.py — INSTITUTIONAL PERFORMANCE ENGINE
# =========================================================
#
# FIX v2:
#   [F1] Sharpe y Max Drawdown → solo retornos de COMPRA
#   [F2] Win Rate → todas las decisiones (COMPRA + VENDE + MANTÉN)
#   [F3] buy_sharpe_ratio y buy_max_drawdown_pct explícitos
#   [F4] evaluated_buy_predictions como contador de COMPRAs evaluadas
#
# FIX v3 (equity-curve):
#   [EC1] Agrupación diaria — evita zigzag intraday por múltiples
#         evaluaciones el mismo día (retorno diario = promedio del día)
#   [EC2] ticker vacío → se lee desde path del archivo como fallback
#   [EC3] n_trades ahora refleja días únicos, no evaluaciones individuales
# =========================================================

import os
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter

from broker import get_engine

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
META_FILE = DATA_PATH / "account_meta.json"

router = APIRouter(prefix="/dashboard", tags=["performance"])

# =========================================================
# HELPERS
# =========================================================

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


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
# MODEL METRICS
# =========================================================

def compute_model_metrics():

    files = list_evaluation_files()

    if not files:
        return {
            "win_rate_pct": None,
            "avg_prediction_error_pct": None,
            "total_predictions": 0,
            "evaluated_predictions": 0,
            "evaluated_buy_predictions": 0,
            "pending_predictions": 0,
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
            "models_performance": {},
        }

    total = 0
    evaluated = 0
    correct = 0

    errors = []

    returns_all = []
    returns_buy = []
    evaluated_buy = 0

    model_errors: Dict[str, List[float]] = {}
    model_hits: Dict[str, List[int]] = {}

    for f in files:

        data = load_json(f)
        if not data:
            continue

        total += 1

        if data.get("real_return_pct") is None:
            continue

        evaluated += 1

        if data.get("decision_correct") is True:
            correct += 1

        if data.get("error_return_pct") is not None:
            errors.append(float(data["error_return_pct"]))

        real_ret = float(data["real_return_pct"])
        returns_all.append(real_ret)

        rec = (data.get("recommendation") or "").strip().upper()
        if rec == "COMPRA":
            returns_buy.append(real_ret)
            evaluated_buy += 1

        models = data.get("models_diagnostics", {})
        for model, mdata in models.items():
            err = mdata.get("error_pct")
            hit = mdata.get("hit_sign")
            if err is not None:
                model_errors.setdefault(model, []).append(float(err))
            if hit is not None:
                model_hits.setdefault(model, []).append(1 if hit else 0)

    pending = total - evaluated

    win_rate = round((correct / evaluated) * 100, 2) if evaluated > 0 else None
    avg_error = round(np.mean(errors), 4) if errors else None

    sharpe = None
    max_dd = None

    if returns_buy:
        buy_arr = np.array(returns_buy) / 100.0

        mean_ret = np.mean(buy_arr)
        std_ret = np.std(buy_arr)

        if std_ret > 0:
            sharpe = round((mean_ret / std_ret) * np.sqrt(252), 3)

        equity_curve = np.cumprod(1 + buy_arr)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_dd = round(float(np.min(drawdowns)) * 100, 2)

    models_perf = {}
    for model in model_errors:
        errs = model_errors.get(model, [])
        hits = model_hits.get(model, [])
        if not errs:
            continue
        models_perf[model] = {
            "avg_error_pct": round(float(np.mean(errs)), 4),
            "win_rate_pct": round((sum(hits) / len(hits)) * 100, 2) if hits else None,
            "samples": len(errs),
        }

    return {
        "win_rate_pct": win_rate,
        "avg_prediction_error_pct": avg_error,
        "total_predictions": total,
        "evaluated_predictions": evaluated,
        "evaluated_buy_predictions": evaluated_buy,
        "pending_predictions": pending,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "models_performance": models_perf,
    }


# =========================================================
# PERFORMANCE ENDPOINT
# =========================================================

@router.get("/performance")
async def performance():

    try:
        engine = get_engine()
        account = await engine.get_account()
        equity = float(account.equity)
    except Exception:
        equity = None

    meta = load_json(META_FILE)

    if not meta and equity:
        meta = {
            "initial_equity": equity,
            "start_date": datetime.utcnow().isoformat(),
            "high_water_mark": equity,
        }
        META_FILE.write_text(json.dumps(meta, indent=2))

    if meta and equity:

        initial_equity = float(meta["initial_equity"])
        high_water_mark = float(meta.get("high_water_mark", equity))

        if equity > high_water_mark:
            high_water_mark = equity
            meta["high_water_mark"] = equity
            META_FILE.write_text(json.dumps(meta, indent=2))

        total_return_pct = round(
            (equity - initial_equity) / initial_equity * 100, 2,
        )

        drawdown_pct = round(
            (equity - high_water_mark) / high_water_mark * 100, 2,
        )

    else:

        total_return_pct = None
        drawdown_pct = None
        high_water_mark = None

    model_metrics = compute_model_metrics()

    return {
        "equity": equity,
        "total_return_pct": total_return_pct,
        "drawdown_pct": drawdown_pct,
        "high_water_mark": high_water_mark,
        "since": meta["start_date"] if meta else None,
        "win_rate_pct": model_metrics["win_rate_pct"],
        "avg_prediction_error_pct": model_metrics["avg_prediction_error_pct"],
        "total_predictions": model_metrics["total_predictions"],
        "evaluated_predictions": model_metrics["evaluated_predictions"],
        "evaluated_buy_predictions": model_metrics["evaluated_buy_predictions"],
        "pending_predictions": model_metrics["pending_predictions"],
        "sharpe_ratio": model_metrics["sharpe_ratio"],
        "max_drawdown_pct": model_metrics["max_drawdown_pct"],
        "models_performance": model_metrics["models_performance"],
    }


# =========================================================
# EQUITY CURVE ENDPOINT
# =========================================================

@router.get("/equity-curve")
async def equity_curve():
    """
    Retorna la curva de equity acumulada desde evaluaciones históricas.

    [EC1] Agrupa por día — evita zigzag intraday por múltiples
          evaluaciones el mismo día. Retorno diario = promedio del día.
    [EC2] Fallback de ticker desde el path del archivo de evaluación.
    [EC3] n_trades = días únicos con al menos una COMPRA evaluada.
    """

    files = list_evaluation_files()

    if not files:
        return {"curve": [], "initial_equity": None, "current_equity": None}

    # ── Recolectar retornos de COMPRA agrupados por fecha ──────────
    # [EC1] defaultdict para acumular múltiples retornos por día
    daily_returns: dict = defaultdict(list)
    daily_tickers: dict = defaultdict(list)

    for f in files:
        data = load_json(f)
        if not data:
            continue

        if data.get("real_return_pct") is None:
            continue

        rec = (data.get("recommendation") or "").strip().upper()
        if rec != "COMPRA":
            continue

        date_str = (
            data.get("evaluation_date")
            or data.get("close_date")
            or data.get("date_base")
        )
        if not date_str:
            continue

        date_key = date_str[:10]

        daily_returns[date_key].append(float(data["real_return_pct"]))

        # [EC2] ticker vacío → fallback desde path del archivo
        ticker = data.get("ticker") or ""
        if not ticker:
            # path: /data/evaluations/AAPL/2026-03-11.json → "AAPL"
            try:
                ticker = f.parent.name
            except Exception:
                ticker = ""

        daily_tickers[date_key].append(ticker)

    if not daily_returns:
        return {"curve": [], "initial_equity": None, "current_equity": None}

    # ── Capital inicial desde meta ─────────────────────────────────
    meta = load_json(META_FILE)
    initial_equity = float(meta["initial_equity"]) if meta else 100_000.0

    # ── Construir curva diaria acumulada ───────────────────────────
    # [EC1] Un punto por día, retorno = promedio de todas las COMPRAs del día
    curve = []
    equity = initial_equity

    for date in sorted(daily_returns.keys()):
        rets = daily_returns[date]
        avg_ret = float(np.mean(rets))           # promedio del día
        equity = equity * (1 + avg_ret / 100.0)

        tickers_today = list(set(daily_tickers[date]))

        curve.append({
            "date": date,
            "equity": round(equity, 2),
            "return_pct": round(avg_ret, 4),     # retorno promedio del día
            "n_trades": len(rets),               # cuántas COMPRAs ese día
            "tickers": tickers_today,
        })

    current_equity = curve[-1]["equity"] if curve else initial_equity

    return {
        "curve": curve,
        "initial_equity": round(initial_equity, 2),
        "current_equity": round(current_equity, 2),
        "n_days": len(curve),        # [EC3] días únicos con COMPRAs
        "n_trades": sum(            # total de evaluaciones individuales
            p["n_trades"] for p in curve
        ),
        "updated_at": datetime.utcnow().isoformat(),
        }
    

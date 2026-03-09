# evaluator.py — WALKFORWARD REAL (LOOKBACK CORRECTO)
# Evalúa HOY la predicción hecha hace N días
# Compatible 100% con contratos existentes

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import concurrent.futures
import numpy as np
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError

# ======================================================
# CONFIG
# ======================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")
MAX_WORKERS = min(int(os.getenv("EVAL_MAX_WORKERS", "4")), 16)
YF_TIMEOUT = 10
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
PRICE_CACHE = {}

logger = logging.getLogger(__name__)

# ======================================================
# STRUCT
# ======================================================
@dataclass
class EvaluationResult:
    meta: Dict[str, Any]
    prediction_date: str
    evaluation_date: str
    price_now: float
    price_pred: float
    price_real: float
    predicted_return_pct: float
    real_return_pct: float
    error_price_pct: float
    error_return_pct: float
    hit_sign: bool
    hit_threshold: bool
    recommendation: str
    decision_correct: bool
    evaluated_at: str


# ======================================================
# UTILS
# ======================================================
def load_json(path: str | Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_json(path: str | Path, data: Dict[str, Any]) -> bool:
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Save failed {path}: {e}")
        return False


def parse_date(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


# ======================================================
# PRECIO REAL HOY (último close disponible)
# ======================================================

def get_price_today(ticker: str, today):

    key = f"{ticker}_{today}"

    if key in PRICE_CACHE:
        return PRICE_CACHE[key]

    try:
        client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=today - timedelta(days=7),
            end=today,
        )

        bars = client.get_stock_bars(request).df
        if bars is None or bars.empty:
            return None

        bars = bars.reset_index()
        bars = bars[bars["timestamp"].dt.date <= today]

        if len(bars) == 0:
            return None

        price = float(bars["close"].iloc[-1])

        PRICE_CACHE[key] = price

        return price

    except Exception:
        return None


# ======================================================
# EVALUACIÓN INDIVIDUAL (LOOKBACK CORRECTO)
# ======================================================

def evaluate_prediction(prediction_path: Path) -> Optional[EvaluationResult]:

    pred = load_json(prediction_path)

    if "meta" not in pred or "prediction" not in pred:
        return None

    meta = pred["meta"]
    p = pred["prediction"]

    ticker = meta["ticker"]
    horizon = int(meta.get("horizon_days", 1))
    theta = float(meta.get("theta", 0.75))

    base_date = parse_date(p["date_base"])
    if base_date is None:
        return None

    today = datetime.utcnow().date()

    # Target date real
    target_date = base_date + timedelta(days=horizon)

    # Si aún no corresponde evaluar → no hacer nada
    if target_date > today:
        return None

    real_price = get_price_today(ticker, today)

    price_now = float(p["price_now"])
    price_pred = float(p["price_pred"])
    predicted_return = float(p["ret_ens_pct"])
    rec = p["recommendation"]

    # 🔥 Si no hay precio real, igual grabamos archivo
    if real_price is None:
        return EvaluationResult(
            meta=meta,
            prediction_date=str(base_date),
            evaluation_date=str(today),
            price_now=price_now,
            price_pred=price_pred,
            price_real=None,
            predicted_return_pct=predicted_return,
            real_return_pct=None,
            error_price_pct=None,
            error_return_pct=None,
            hit_sign=None,
            hit_threshold=None,
            recommendation=rec,
            decision_correct=None,
            evaluated_at=datetime.utcnow().isoformat(),
        )

    real_return = (real_price / price_now - 1) * 100.0

    hit_sign = np.sign(real_return) == np.sign(predicted_return)
    hit_threshold = abs(real_return) >= theta

    if rec == "COMPRA":
        decision_correct = real_return >= 0
    elif rec == "VENDE":
        decision_correct = real_return <= 0
    else:
        decision_correct = abs(real_return) < theta

    return EvaluationResult(
        meta=meta,
        prediction_date=str(base_date),
        evaluation_date=str(today),
        price_now=price_now,
        price_pred=price_pred,
        price_real=real_price,
        predicted_return_pct=predicted_return,
        real_return_pct=real_return,
        error_price_pct=abs(price_pred / real_price - 1) * 100.0,
        error_return_pct=abs(predicted_return - real_return),
        hit_sign=bool(hit_sign),
        hit_threshold=bool(hit_threshold),
        recommendation=rec,
        decision_correct=bool(decision_correct),
        evaluated_at=datetime.utcnow().isoformat(),
    )


def evaluate_models(pred: Dict[str, Any], price_real: float):

    p = pred.get("prediction", {})
    price_now = float(p.get("price_now"))

    base_date = parse_date(p["date_base"])
    today = datetime.utcnow().date()

    h = (today - base_date).days

    if h < 1 or h > 10:
        return {}

    curve = pred.get("price_curve", {})
    path = curve.get("price_path", [])

    models = {}

    if h <= 9:

        if len(path) < h:
            return {}

        price_pred = float(path[h-1])

    else:

        price_pred = float(p.get("price_pred"))

    pred_ret = (price_pred / price_now - 1) * 100
    real_ret = (price_real / price_now - 1) * 100

    models[f"H{h}"] = {
        "pred_price": round(price_pred,4),
        "pred_return": round(pred_ret,4),
        "real_return": round(real_ret,4),
        "error_pct": round(abs(pred_ret-real_ret),4),
        "hit_sign": bool(np.sign(pred_ret)==np.sign(real_ret))
    }

    return models

def summarize_models(models: Dict[str, Any]):

    errors = {k:v["error_pct"] for k,v in models.items() if v["error_pct"] is not None}

    if not errors:
        return {}

    best = min(errors, key=errors.get)
    worst = max(errors, key=errors.get)

    mean_error = float(np.mean(list(errors.values())))

    return {
        "best_model": best,
        "worst_model": worst,
        "mean_error": round(mean_error,4)
    }

# ======================================================
# EVALUACIÓN MASIVA
# ======================================================
def evaluate_all(
    ticker: Optional[str] = None,
    max_workers: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:

    pred_root = Path(DATA_PATH) / "predictions"
    eval_root = Path(DATA_PATH) / "evaluations"

    results = {"evaluated": [], "skipped": [], "errors": [], "summary": {}}

    if not pred_root.exists():
        return results

    ticker_dirs = (
        [pred_root / ticker] if ticker else [d for d in pred_root.iterdir() if d.is_dir()]
    )

    pending = []

    for td in ticker_dirs:
        (eval_root / td.name).mkdir(parents=True, exist_ok=True)
        for f in td.glob("*.json"):
            if not (eval_root / td.name / f.name).exists():
                pending.append(f)

    if dry_run:
        return {"pending": len(pending)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or MAX_WORKERS) as ex:
        future_map = {ex.submit(evaluate_prediction, f): f for f in pending}

        for fut in concurrent.futures.as_completed(future_map):
            f = future_map[fut]
            try:
                ev = fut.result()
                if ev:
                    ev_dict = asdict(ev)
                    # cargar predicción original
                    pred = load_json(f)

                    if ev_dict.get("price_real") is not None:
                        models_diag = evaluate_models(pred, ev_dict["price_real"])
                        models_summary = summarize_models(models_diag)

                        ev_dict["models_diagnostics"] = models_diag
                        ev_dict["models_summary"] = models_summary

                    save_json(
                        eval_root / f.parent.name / f.name,
                        ev_dict,
                    )
                    results["evaluated"].append(str(f))
                else:
                    results["skipped"].append(str(f))
            except Exception as e:
                logger.error(f"Error evaluating {f}: {e}")
                results["errors"].append(str(f))

    results["summary"] = {
        "evaluated": len(results["evaluated"]),
        "skipped": len(results["skipped"]),
        "errors": len(results["errors"]),
    }

    return results


# ======================================================
# COMPAT
# ======================================================
def evaluate_all_compat(ticker: Optional[str] = None) -> Dict[str, List[str]]:
    r = evaluate_all(ticker)
    return {"evaluated": r["evaluated"], "skipped": r["skipped"]}

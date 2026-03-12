import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import concurrent.futures
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ======================================================
# CONFIG
# ======================================================

DATA_PATH   = os.getenv("DATA_PATH", "/data")
MAX_WORKERS = min(int(os.getenv("EVAL_MAX_WORKERS", "4")), 16)
ALPACA_KEY  = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
EVAL_HORIZON  = int(os.getenv("EVAL_HORIZON", "10"))   # ← ahora configurable
PRICE_CACHE: Dict[str, float] = {}

logging.basicConfig(level=logging.INFO)
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
    price_real: Optional[float]
    predicted_return_pct: float
    real_return_pct: Optional[float]
    error_price_pct: Optional[float]
    error_return_pct: Optional[float]
    hit_sign: Optional[bool]
    hit_threshold: Optional[bool]
    recommendation: str
    decision_correct: Optional[bool]
    evaluated_at: str
    models_diagnostics: Dict[str, Any] = None
    models_summary: Dict[str, Any] = None

# ======================================================
# UTILS
# ======================================================

def load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"load_json failed {path}: {e}")   # ← ya no silencioso
        return {}

def save_json(path: Path, data: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"save_json failed {path}: {e}")
        return False

def parse_date(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

# ======================================================
# PRECIO REAL
# ======================================================

# Cliente Alpaca reutilizado durante toda la ejecución
_alpaca_client: Optional[StockHistoricalDataClient] = None

def _get_alpaca_client() -> StockHistoricalDataClient:
    global _alpaca_client
    if _alpaca_client is None:
        _alpaca_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    return _alpaca_client

def get_price_today(ticker: str, today) -> Optional[float]:
    key = f"{ticker}_{today}"
    if key in PRICE_CACHE:
        return PRICE_CACHE[key]
    try:
        client = _get_alpaca_client()
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
        if bars.empty:
            return None
        price = float(bars["close"].iloc[-1])
        PRICE_CACHE[key] = price
        return price
    except Exception as e:
        logger.warning(f"get_price_today({ticker}): {e}")
        return None

# ======================================================
# EVALUACIÓN MODELOS H1-H10
# ======================================================

def evaluate_models(
    ticker: str,
    price_real: float,
    today,               # ← recibe today desde afuera
) -> Dict[str, Any]:
    """
    Cómo funciona la búsqueda de H1..H10
    ─────────────────────────────────────
    Para cada horizonte h (1 a 10):

      1. Calcula la fecha pasada:  target_date = today - h días
         Ejemplo con today=2025-07-10:
           H1 → 2025-07-09,  H2 → 2025-07-08, ..., H10 → 2025-06-30

      2. Abre el archivo de predicción de ese día:
         /data/predictions/<ticker>/<target_date>.json

      3. Dentro de ese JSON busca "price_curve.price_path", que es una lista
         de precios proyectados día a día desde la fecha de la predicción:
           price_path[0] = precio proyectado para +1 día
           price_path[1] = precio proyectado para +2 días
           ...
           price_path[h-1] = precio proyectado para exactamente +h días = HOY

      4. Compara ese precio proyectado (price_path[h-1]) con price_real (hoy)
         para calcular error, hit_sign, etc.

    En resumen: cada Hx evalúa "¿qué predijo hace x días para hoy?"
    """
    models: Dict[str, Any] = {}

    for h in range(1, 11):
        target_date = today - timedelta(days=h)
        file_path = Path(DATA_PATH) / "predictions" / ticker / f"{target_date.strftime('%Y-%m-%d')}.json"

        if not file_path.exists():
            continue

        old_pred  = load_json(file_path)
        old_p     = old_pred.get("prediction", {})
        curve     = old_pred.get("price_curve", {}).get("price_path", [])

        if len(curve) < h:
            continue

        price_now = float(old_p.get("price_now", 0))
        if price_now <= 0:
            continue

        price_pred = float(curve[h - 1])
        pred_ret   = (price_pred / price_now - 1) * 100
        real_ret   = (price_real / price_now - 1) * 100

        models[f"H{h}"] = {
            "pred_price":  round(price_pred, 4),
            "pred_return": round(pred_ret,   4),
            "real_return": round(real_ret,   4),
            "error_pct":   round(abs(pred_ret - real_ret), 4),
            "hit_sign":    bool(np.sign(pred_ret) == np.sign(real_ret)),
        }

    return models

def summarize_models(models: Dict[str, Any]) -> Dict[str, Any]:
    errors = {k: v["error_pct"] for k, v in models.items() if v.get("error_pct") is not None}
    if not errors:
        return {"best_model": None, "worst_model": None, "mean_error": 0}
    return {
        "best_model":  min(errors, key=errors.get),
        "worst_model": max(errors, key=errors.get),
        "mean_error":  round(float(np.mean(list(errors.values()))), 4),
    }

# ======================================================
# EVALUACIÓN INDIVIDUAL
# ======================================================

def evaluate_prediction(prediction_path: Path, today=None) -> Optional[EvaluationResult]:
    """today se pasa desde evaluate_all para consistencia en toda la ejecución."""
    if today is None:
        today = datetime.utcnow().date()

    pred = load_json(prediction_path)
    if "meta" not in pred or "prediction" not in pred:
        return None

    meta   = pred["meta"]
    p      = pred["prediction"]
    ticker = meta["ticker"]
    theta  = float(meta.get("theta", 0.75))

    real_price       = get_price_today(ticker, today)
    price_now        = float(p.get("price_now", 0))
    price_pred       = float(p.get("price_pred", 0))
    predicted_return = float(p.get("ret_ens_pct", 0))
    rec              = p.get("recommendation", "HOLD")

    real_return = hit_sign = hit_threshold = decision_correct = None
    error_price_pct = error_return_pct = None

    if real_price and price_now > 0:
        real_return     = (real_price / price_now - 1) * 100.0
        hit_sign        = bool(np.sign(real_return) == np.sign(predicted_return))
        hit_threshold   = bool(abs(real_return) >= theta)
        error_price_pct = abs(price_pred / real_price - 1) * 100.0 if real_price > 0 else 0
        error_return_pct = abs(predicted_return - real_return)

        if rec == "COMPRA":
            decision_correct = real_return >= 0
        elif rec == "VENDE":
            decision_correct = real_return <= 0
        else:
            decision_correct = abs(real_return) < theta

    models_diag = evaluate_models(ticker, real_price, today) if real_price else {}
    models_sum  = summarize_models(models_diag)

    return EvaluationResult(
        meta=meta,
        prediction_date=str(parse_date(prediction_path.stem)),
        evaluation_date=str(today),
        price_now=price_now,
        price_pred=price_pred,
        price_real=real_price,
        predicted_return_pct=predicted_return,
        real_return_pct=real_return,
        error_price_pct=error_price_pct,
        error_return_pct=error_return_pct,
        hit_sign=hit_sign,
        hit_threshold=hit_threshold,
        recommendation=rec,
        decision_correct=decision_correct,
        evaluated_at=datetime.utcnow().isoformat(),
        models_diagnostics=models_diag,
        models_summary=models_sum,
    )

# ======================================================
# EVALUACIÓN MASIVA
# ======================================================

def evaluate_all(
    ticker: Optional[str] = None,
    max_workers: Optional[int] = None,
    dry_run: bool = False,
    horizon: int = EVAL_HORIZON,       # ← configurable, no hardcodeado
) -> Dict[str, Any]:

    pred_root = Path(DATA_PATH) / "predictions"
    eval_root = Path(DATA_PATH) / "evaluations"
    results   = {"evaluated": [], "skipped": [], "errors": [], "summary": {}}

    if not pred_root.exists():
        return results

    # today se calcula UNA sola vez aquí y se pasa hacia abajo
    today          = datetime.utcnow().date()
    target_date    = today - timedelta(days=horizon)
    target_date_str = target_date.strftime("%Y-%m-%d")

    ticker_dirs = (
        [pred_root / ticker]
        if ticker
        else [d for d in pred_root.iterdir() if d.is_dir()]
    )

    pending = [
        td / f"{target_date_str}.json"
        for td in ticker_dirs
        if (td / f"{target_date_str}.json").exists()
        and not (eval_root / td.name / f"{target_date_str}.json").exists()
    ]

    if dry_run:
        return {"pending": len(pending)}

    workers = max_workers or MAX_WORKERS
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(evaluate_prediction, f, today): f
            for f in pending
        }
        for fut in concurrent.futures.as_completed(future_map):
            f = future_map[fut]
            try:
                ev = fut.result()
                if ev:
                    save_json(eval_root / f.parent.name / f.name, asdict(ev))
                    results["evaluated"].append(str(f))
                else:
                    results["skipped"].append(str(f))
            except Exception as e:
                logger.error(f"Error evaluating {f}: {e}")
                results["errors"].append(str(f))

    results["summary"] = {
        "evaluated": len(results["evaluated"]),
        "skipped":   len(results["skipped"]),
        "errors":    len(results["errors"]),
    }
    return results

def evaluate_all_compat(ticker: Optional[str] = None) -> Dict[str, List[str]]:
    r = evaluate_all(ticker)
    return {"evaluated": r["evaluated"], "skipped": r["skipped"]}

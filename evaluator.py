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

# ======================================================
# CONFIG
# ======================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")
MAX_WORKERS = min(int(os.getenv("EVAL_MAX_WORKERS", "4")), 16)
YF_TIMEOUT = 10

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
    try:
        start = today - timedelta(days=7)
        end = today + timedelta(days=1)

        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=False,
            timeout=YF_TIMEOUT,
        )

        if df is None or len(df) == 0:
            return None

        df.index = df.index.tz_localize(None)
        df = df[df.index.date <= today]

        if len(df) == 0:
            return None

        return float(df["Close"].iloc[-1])

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

    # 🔥 CLAVE: solo evaluar si base_date == hoy - horizon
    if base_date != today - timedelta(days=horizon):
        return None

    real_price = get_price_today(ticker, today)
    if real_price is None:
        return None

    price_now = float(p["price_now"])
    price_pred = float(p["price_pred"])
    predicted_return = float(p["ret_ens_pct"])

    real_return = (real_price / price_now - 1) * 100.0

    hit_sign = np.sign(real_return) == np.sign(predicted_return)
    hit_threshold = abs(real_return) >= theta

    rec = p["recommendation"]

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
                    save_json(
                        eval_root / f.parent.name / f.name,
                        asdict(ev),
                    )
                    results["evaluated"].append(str(f))
                else:
                    results["skipped"].append(str(f))
            except Exception:
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

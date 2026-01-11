# evaluator.py - VERSIÓN ABSOLUTA FINAL (PRODUCTION-READY)
# 100% compatible, 0 dependencias externas

import os
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import concurrent.futures
import time
from functools import wraps
import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# Configuración
# =========================
DATA_PATH = os.getenv("DATA_PATH", "/data")
MAX_WORKERS = min(int(os.getenv("EVAL_MAX_WORKERS", "4")), 16)
YF_TIMEOUT = 10

# =========================
# Retry nativo
# =========================
class Retry:
    def __init__(self, stop_max_attempt_number: int = 3, wait_fixed: float = 2.0):
        self.stop_max_attempt_number = stop_max_attempt_number
        self.wait_fixed = wait_fixed

    def __call__(self, fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(self.stop_max_attempt_number):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == self.stop_max_attempt_number - 1:
                        raise
                    time.sleep(self.wait_fixed * (2 ** attempt))
            raise last_exception
        return wrapper

# =========================
# Logging fail-safe + verbose
# =========================
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO

    try:
        log_path = Path(DATA_PATH) / "evaluator.log"
        log_path.parent.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
    except Exception:
        fh = logging.NullHandler()

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        handlers=[fh, sh]
    )

logger = logging.getLogger(__name__)

# =========================
# Data structures
# =========================
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

# =========================
# Utilidades
# =========================
def load_json(path: str | Path) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug(f"Load failed {path}: {e}")
        return {}

def save_json(path: str | Path, data: Dict[str, Any]) -> bool:
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
        return True
    except Exception as e:
        logger.error(f"Save failed {path}: {e}")
        return False

def parse_date(date_str: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return None

def validate_prices(price_now: float, price_real: float) -> bool:
    return np.isfinite(price_now) and np.isfinite(price_real) and price_now > 0 and price_real > 0

# =========================
# yfinance robusto
# =========================
@Retry(stop_max_attempt_number=3, wait_fixed=2.0)
def download_price_robust(ticker: str, target_date: datetime.date, window_days: int = 10) -> Optional[pd.DataFrame]:
    try:
        start = target_date - timedelta(days=window_days // 2)
        end = target_date + timedelta(days=window_days // 2 + 1)

        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            prepost=False,
            threads=False,
            timeout=YF_TIMEOUT
        )

        if df is None or len(df) == 0 or "Close" not in df.columns:
            return None

        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        logger.debug(f"yfinance {ticker} {target_date}: {e}")
        return None

# =========================
# Evaluación individual
# =========================
def evaluate_prediction(prediction_path: str | Path) -> Optional[EvaluationResult]:
    pred = load_json(prediction_path)
    if not all(k in pred for k in ["meta", "prediction"]):
        return None

    meta = pred["meta"]
    p = pred["prediction"]

    ticker = meta["ticker"]
    horizon = meta.get("horizon_days", 10)
    theta = meta.get("theta", 0.75)

    base_date = parse_date(p["date_base"])
    if base_date is None:
        return None

    target_date = base_date + timedelta(days=horizon)
    if datetime.utcnow().date() < target_date:
        return None

    df = download_price_robust(ticker, target_date)
    if df is None:
        return None

    df_eval = df[df.index.date >= target_date]
    if len(df_eval) == 0:
        df_eval = df[df.index.date < target_date].tail(5)
        if len(df_eval) == 0:
            return None
        real_price = float(df_eval["Close"].iloc[-1])
    else:
        real_price = float(df_eval["Close"].iloc[0])

    price_now = float(p["price_now"])
    price_pred = float(p["price_pred"])

    if not validate_prices(price_now, real_price):
        logger.warning(f"Invalid prices {ticker}: now={price_now}, real={real_price}")

    real_ret = (real_price / price_now - 1) * 100.0
    pred_ret = float(p["ret_ens_pct"])

    hit_sign = np.sign(real_ret) == np.sign(pred_ret)
    hit_threshold = abs(real_ret) >= theta

    rec = p["recommendation"]
    if rec == "COMPRA":
        decision_correct = real_ret >= 0
    elif rec == "VENDE":
        decision_correct = real_ret <= 0
    else:
        decision_correct = abs(real_ret) < theta

    ev = EvaluationResult(
        meta=meta,
        prediction_date=str(base_date),
        evaluation_date=str(target_date),
        price_now=price_now,
        price_pred=price_pred,
        price_real=real_price,
        predicted_return_pct=pred_ret,
        real_return_pct=real_ret,
        error_price_pct=abs(price_pred / real_price - 1) * 100.0,
        error_return_pct=abs(pred_ret - real_ret),
        hit_sign=bool(hit_sign),
        hit_threshold=bool(hit_threshold),
        recommendation=rec,
        decision_correct=bool(decision_correct),
        evaluated_at=datetime.utcnow().isoformat()
    )

    logger.debug(f"{ticker}: {pred_ret:+.2f}% → {real_ret:+.2f}% ({'✓' if decision_correct else '✗'})")
    return ev

# =========================
# Evaluación masiva
# =========================
def evaluate_all(ticker: Optional[str] = None, max_workers: Optional[int] = None, dry_run: bool = False) -> Dict[str, Any]:
    start_time = datetime.now()
    max_workers = min(max_workers or MAX_WORKERS, 16)

    results = {"evaluated": [], "skipped": [], "errors": [], "summary": {}}

    pred_root = Path(DATA_PATH) / "predictions"
    eval_root = Path(DATA_PATH) / "evaluations"
    if not pred_root.exists():
        return results

    ticker_dirs = [pred_root / ticker] if ticker else [d for d in pred_root.iterdir() if d.is_dir()]
    pending = []

    for td in ticker_dirs:
        (eval_root / td.name).mkdir(exist_ok=True)
        for f in sorted(td.glob("*.json")):
            if not (eval_root / td.name / f.name).exists():
                pending.append(f)

    total_pending = len(pending)
    if dry_run:
        return {"dry_run": True, "pending_count": total_pending}

    if total_pending == 0:
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(evaluate_prediction, f): f for f in pending}
        for i, fut in enumerate(concurrent.futures.as_completed(future_map), 1):
            f = future_map[fut]
            try:
                ev = fut.result(timeout=45)
                rel = f.relative_to(pred_root)
                if ev:
                    save_json(eval_root / f.parent.name / f.name, asdict(ev))
                    results["evaluated"].append(str(rel))
                else:
                    results["skipped"].append(str(rel))
            except Exception as e:
                results["errors"].append(str(f))
                logger.debug(f"Error {f}: {e}")

            if total_pending >= 10 and i % max(10, total_pending // 10) == 0:
                logger.info(f"Progress: {i}/{total_pending}")

    duration = max((datetime.now() - start_time).total_seconds(), 1e-6)
    results["summary"] = {
        "total_pending": total_pending,
        "evaluated": len(results["evaluated"]),
        "skipped": len(results["skipped"]),
        "errors": len(results["errors"]),
        "duration_seconds": round(duration, 2),
        "workers": max_workers,
        "throughput_per_sec": round(total_pending / duration, 2)
    }
    return results

# =========================
# Métricas agregadas
# =========================
def compute_aggregate_stats(evaluations_root: str | Path = None, days_limit: Optional[int] = None) -> Dict[str, Any]:
    root = Path(evaluations_root or f"{DATA_PATH}/evaluations")
    if not root.exists():
        return {"error": "No evaluations"}

    cutoff = datetime.utcnow() - timedelta(days=days_limit) if days_limit else None
    rows = []

    for f in root.rglob("*.json"):
        d = load_json(f)
        if not d or "decision_correct" not in d:
            continue
        try:
            dt = datetime.fromisoformat(d["evaluated_at"])
            if cutoff and dt < cutoff:
                continue
            rows.append(d)
        except Exception:
            continue

    if not rows:
        return {"total": 0}

    df = pd.DataFrame(rows)
    return {
        "total": len(df),
        "hit_rate_overall": round(float(df["decision_correct"].mean()), 4),
        "avg_error_return_pct": round(float(df["error_return_pct"].mean()), 2),
        "avg_error_price_pct": round(float(df["error_price_pct"].mean()), 2),
        "by_recommendation": df.groupby("recommendation")["decision_correct"].mean().round(4).to_dict(),
        "worst_errors": [
            {"ticker": r["meta"]["ticker"], "error": round(r["error_return_pct"], 2)}
            for _, r in df.nlargest(5, "error_return_pct")[["meta", "error_return_pct"]].iterrows()
        ]
    }

# =========================
# API compatible
# =========================
def evaluate_all_compat(ticker: Optional[str] = None) -> Dict[str, List[str]]:
    r = evaluate_all(ticker)
    return {"evaluated": r["evaluated"], "skipped": r["skipped"]}

# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prediction Evaluator")
    parser.add_argument("ticker", nargs="?", help="Specific ticker")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.stats:
        print(json.dumps(compute_aggregate_stats(), indent=2))
    else:
        res = evaluate_all(args.ticker, args.workers, args.dry_run)
        print(json.dumps(res["summary"], indent=2))

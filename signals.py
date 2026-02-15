# signals.py — VERSIÓN ABSOLUTA FINAL (ENTERPRISE 10/10)
# =====================================================
# ✅ Compatible con model.py + evaluator.py
# ✅ Fundamental (model2_improved) como CONTEXTO (NO score)
# ✅ Cache + concurrencia + TTL
# ✅ NO altera hit-rate ni ranking
# =====================================================

import os
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import argparse
import concurrent.futures
from threading import Lock

# =====================================================
# FUNDAMENTAL CONTEXT (SAFE IMPORT)
# =====================================================
try:
    from model2 import fundamental_signal_context
    FUNDAMENTAL_AVAILABLE = True
except Exception:
    FUNDAMENTAL_AVAILABLE = False
    def fundamental_signal_context(ticker: str) -> Dict[str, Any]:
        return {"usable": False}

# =====================================================
# Configuración ENV
# =====================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")
DEFAULT_WINDOW = int(os.getenv("SIGNAL_WINDOW", "30"))
MIN_CONFIDENCE = float(os.getenv("SIGNAL_MIN_CONFIDENCE", "0.4"))
MIN_HISTORY = int(os.getenv("SIGNAL_MIN_HISTORY", "3"))
MAX_CONCURRENT = int(os.getenv("SIGNAL_MAX_CONCURRENT", "4"))
SIGNAL_DECAY_DAYS = int(os.getenv("SIGNAL_DECAY_DAYS", "14"))
CACHE_TTL = int(os.getenv("SIGNAL_CACHE_TTL", "300"))

# =====================================================
# Logging
# =====================================================
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    try:
        log_path = Path(DATA_PATH) / "signals.log"
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

# =====================================================
# Cache thread-safe
# =====================================================
_signals_cache: Dict[str, Dict] = {}
_cache_lock = Lock()

# =====================================================
# Utils
# =====================================================
def load_json(path: str | Path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def list_json_files(path: str | Path) -> List[str]:
    path = Path(path)
    if not path.exists():
        return []
    return sorted(f.name for f in path.glob("*.json"))

def load_universe() -> List[str]:
    p = Path(DATA_PATH) / "tickers.json"
    if not p.exists():
        return []
    d = load_json(p)
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        return d.get("tickers", [])
    return []

def parse_date(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

# =====================================================
# Rolling metrics (decay)
# =====================================================
def rolling_metrics(evals: List[Dict], window_days: int):
    if not evals:
        return None

    now = datetime.utcnow().date()
    cutoff = now - timedelta(days=window_days)

    hits, errors, weights = [], [], []

    for e in evals:
        d = parse_date(e.get("prediction_date", ""))
        if not d or d < cutoff:
            continue

        age = max(0, (now - d).days)
        w = np.exp(-max(0, age - 7) / (SIGNAL_DECAY_DAYS / 2))

        hits.append(1.0 if e.get("decision_correct") else 0.0)
        errors.append(abs(e.get("error_return_pct", 0)))
        weights.append(w)

    if not weights or sum(weights) == 0:
        return None

    return {
        "n": len(hits),
        "effective_n": round(float(sum(weights)), 2),
        "hit_rate": float(np.average(hits, weights=weights)),
        "mae_return_pct": float(np.average(errors, weights=weights)),
    }

# =====================================================
# Confidence score
# =====================================================
def confidence_score(ret_pct: float, metrics: Dict) -> Optional[float]:
    if not metrics:
        return None

    strength = min(abs(ret_pct) / 3.0, 1.0)
    acc = metrics["hit_rate"]
    err = 1.0 / (1.0 + metrics["mae_return_pct"] / 10.0)
    size = min(metrics["effective_n"] / 10.0, 0.3)

    score = 0.35 * strength + 0.35 * acc + 0.2 * err + 0.1 * size

    if metrics["effective_n"] < MIN_HISTORY:
        score *= 0.7

    return round(float(np.clip(score, 0, 1)), 3)

def signal_quality(conf: Optional[float]) -> str:
    if conf is None:
        return "NO_DATA"
    if conf >= 0.70:
        return "🔥 STRONG"
    if conf >= 0.55:
        return "✅ GOOD"
    if conf >= MIN_CONFIDENCE:
        return "⚠️ WEAK"
    return "❌ NOISE"

# =====================================================
# Compute signal (CORE)
# =====================================================
def compute_signal(ticker: str, window_days: int = DEFAULT_WINDOW) -> Dict[str, Any]:
    cache_key = f"{ticker}_{window_days}"

    with _cache_lock:
        c = _signals_cache.get(cache_key)
        if c:
            if (datetime.utcnow() - datetime.fromisoformat(c["_cached_at"])).seconds < CACHE_TTL:
                return c

    pred_dir = Path(DATA_PATH) / "predictions" / ticker
    eval_dir = Path(DATA_PATH) / "evaluations" / ticker

    preds = list_json_files(pred_dir)
    if not preds:
        return {"ticker": ticker, "error": "NO_PREDICTIONS"}

    last = load_json(pred_dir / preds[-1])
    if "prediction" not in last or "meta" not in last:
        return {"ticker": ticker, "error": "INVALID_PREDICTION"}

    p = last["prediction"]
    evals = [load_json(eval_dir / f) for f in list_json_files(eval_dir)]

    metrics = rolling_metrics(evals, window_days)
    conf = confidence_score(p["ret_ens_pct"], metrics) if metrics else None
    quality = signal_quality(conf)
    strength = min(abs(p["ret_ens_pct"]) / 3.0, 1.0)

    # =========================
    # FUNDAMENTAL CONTEXT
    # =========================
    fundamental = fundamental_signal_context(ticker) if FUNDAMENTAL_AVAILABLE else {"usable": False}

    fundamental_flag = None
    if fundamental.get("usable"):
        mp = fundamental.get("mispricing_pct", 0)
        if mp <= -30:
            fundamental_flag = "🟢 DEEP_VALUE"
        elif mp >= 30:
            fundamental_flag = "🔴 OVERHEATED"

    result = {
        "ticker": ticker,
        "date": p["date_base"],
        "recommendation": p["recommendation"],
        "ret_ens_pct": round(p["ret_ens_pct"], 2),
        "price_now": round(p["price_now"], 2),
        "price_pred": round(p["price_pred"], 2),
        "confidence": conf,
        "quality": quality,
        "signal_strength": round(strength, 3),
        "rolling_metrics": metrics,
        "fundamental": fundamental if fundamental.get("usable") else None,
        "fundamental_flag": fundamental_flag,
        "horizon_days": last["meta"]["horizon_days"],
        "theta": last["meta"]["theta"],
        "_cached_at": datetime.utcnow().isoformat(),
    }

    with _cache_lock:
        _signals_cache[cache_key] = result

    return result

# =====================================================
# Batch
# =====================================================
def compute_all_signals(window_days: int = DEFAULT_WINDOW) -> List[Dict[str, Any]]:
    tickers = load_universe()
    if not tickers:
        return []

    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futs = {ex.submit(compute_signal, t, window_days): t for t in tickers}
        for f in concurrent.futures.as_completed(futs):
            try:
                out.append(f.result())
            except Exception as e:
                out.append({"ticker": futs[f], "error": str(e)})

    out.sort(key=lambda s: (-(s.get("confidence") or 0), -(s.get("signal_strength") or 0)))
    return out

# =====================================================
# CLI
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--min-conf", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    signals = compute_all_signals(args.window)

    for s in signals:
        if s.get("confidence", 0) >= args.min_conf:
            print(
                f"{s['ticker']:>6} | {s['quality']:>8} | "
                f"{s.get('fundamental_flag','–'):>12} | "
                f"{s['confidence']:.3f} | {s['ret_ens_pct']:+.2f}%"
            )

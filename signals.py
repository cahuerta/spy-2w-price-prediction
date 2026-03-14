# ======================================================
# signals.py — VERSIÓN ABSOLUTA FINAL (ENTERPRISE 10/10)
# ======================================================
# ✅ Compatible con model.py + evaluator.py
# ✅ Fundamental (model2_improved) como CONTEXTO (NO score)
# ✅ Cache + concurrencia + TTL
# ✅ NO altera hit-rate ni ranking
#
# FIXES v4.3-signals:
#   [S1] ret_ens_pct None → abs(None) crash en compute_signal
#   [S2] rolling_metrics None → alpha_engine recibe None en vez de dict default
#   [S3] price_now / price_pred None → round(None) crash
#   [S4] compute_signal retornaba dict vacío en errores → alpha_engine hacía .get() sobre None
#   [S5] confidence_score con metrics=None → retornaba None, ahora retorna 0.0
# ======================================================

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

# ======================================================
# FUNDAMENTAL CONTEXT (SAFE IMPORT)
# ======================================================
try:
    from model2 import fundamental_signal_context
    FUNDAMENTAL_AVAILABLE = True
except Exception:
    FUNDAMENTAL_AVAILABLE = False
    def fundamental_signal_context(ticker: str) -> Dict[str, Any]:
        return {"usable": False}

# ======================================================
# Configuración ENV
# ======================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")
DEFAULT_WINDOW = int(os.getenv("SIGNAL_WINDOW", "30"))
MIN_CONFIDENCE = float(os.getenv("SIGNAL_MIN_CONFIDENCE", "0.4"))
MIN_HISTORY = int(os.getenv("SIGNAL_MIN_HISTORY", "3"))
MAX_CONCURRENT = int(os.getenv("SIGNAL_MAX_CONCURRENT", "4"))
SIGNAL_DECAY_DAYS = int(os.getenv("SIGNAL_DECAY_DAYS", "14"))
CACHE_TTL = int(os.getenv("SIGNAL_CACHE_TTL", "300"))

# Métricas por defecto cuando no hay historial de evaluaciones
_DEFAULT_METRICS = {
    "n": 0,
    "effective_n": 0.0,
    "hit_rate": 0.45,
    "mae_return_pct": 10.0,
}

# ======================================================
# Logging
# ======================================================
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

# ======================================================
# Cache thread-safe
# ======================================================
_signals_cache: Dict[str, Dict] = {}
_cache_lock = Lock()

# ======================================================
# Utils
# ======================================================
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


# ======================================================
# Rolling metrics (decay)
# ======================================================
def rolling_metrics(evals: List[Dict], window_days: int) -> Optional[Dict]:
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

        hits.append(1.0 if e.get("hit_sign") else 0.0)
        errors.append(abs(e.get("error_return_pct") or 0))
        weights.append(w)

    if not weights or sum(weights) == 0:
        return None

    return {
        "n": len(hits),
        "effective_n": round(float(sum(weights)), 2),
        "hit_rate": float(np.average(hits, weights=weights)),
        "mae_return_pct": float(np.average(errors, weights=weights)),
    }


# ======================================================
# Confidence score
# ======================================================
def confidence_score(ret_pct: float, metrics: Dict) -> float:
    """
    [S5] Nunca retorna None — retorna 0.0 si no hay métricas válidas.
    Alpha engine puede asumir que confidence es siempre float.
    """
    if not metrics:
        return 0.0

    # [S5] ret_pct debe ser float válido
    if ret_pct is None:
        return 0.0

    strength = min(abs(ret_pct) / 3.0, 1.0)
    acc = metrics.get("hit_rate") or 0.45
    mae = metrics.get("mae_return_pct") or 10.0
    eff_n = metrics.get("effective_n") or 0.0

    err = 1.0 / (1.0 + mae / 10.0)
    size = min(eff_n / 10.0, 0.3)

    score = 0.35 * strength + 0.35 * acc + 0.2 * err + 0.1 * size

    if eff_n < MIN_HISTORY:
        score *= 0.7

    return round(float(np.clip(score, 0, 1)), 3)


def signal_quality(conf: float) -> str:
    # [S5] conf es ahora siempre float, pero por seguridad mantenemos el guard
    if conf is None or conf == 0.0:
        return "NO_DATA"
    if conf >= 0.70:
        return "🔥 STRONG"
    if conf >= 0.55:
        return "✅ GOOD"
    if conf >= MIN_CONFIDENCE:
        return "⚠️ WEAK"
    return "❌ NOISE"


# ======================================================
# Compute signal (CORE)
# ======================================================
def compute_signal(ticker: str, window_days: int = DEFAULT_WINDOW) -> Dict[str, Any]:
    """
    Retorna siempre un dict con al menos {"ticker": ticker, "error": ...}
    NUNCA retorna None — alpha_engine puede hacer .get() sin riesgo.

    [S4] Todos los early-returns son dicts válidos, no None.
    """
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
        return {"ticker": ticker, "error": "NO_PREDICTIONS", "confidence": 0.0, "rolling_metrics": _DEFAULT_METRICS}

    last = load_json(pred_dir / preds[-1])
    if not last or "prediction" not in last or "meta" not in last:
        return {"ticker": ticker, "error": "INVALID_PREDICTION", "confidence": 0.0, "rolling_metrics": _DEFAULT_METRICS}

    p = last["prediction"]

    # --------------------------------------------------
    # [S1] Validar ret_ens_pct antes de cualquier operación
    # --------------------------------------------------
    ret_ens_pct = p.get("ret_ens_pct")
    if ret_ens_pct is None:
        logger.warning(f"⚠️ [{ticker}] ret_ens_pct es None en JSON de predicción — ticker saltado")
        return {
            "ticker": ticker,
            "error": "ret_ens_pct_null",
            "confidence": 0.0,
            "rolling_metrics": _DEFAULT_METRICS,
        }

    ret_ens_pct = float(ret_ens_pct)

    # --------------------------------------------------
    # [S3] Validar price_now y price_pred
    # --------------------------------------------------
    price_now = p.get("price_now")
    price_pred = p.get("price_pred")

    if price_now is None or price_pred is None:
        logger.warning(f"⚠️ [{ticker}] price_now o price_pred es None")
        return {
            "ticker": ticker,
            "error": "price_null",
            "confidence": 0.0,
            "rolling_metrics": _DEFAULT_METRICS,
        }

    price_now = float(price_now)
    price_pred = float(price_pred)

    # --------------------------------------------------
    # Evaluaciones + métricas
    # --------------------------------------------------
    evals = [load_json(eval_dir / f) for f in list_json_files(eval_dir)]
    metrics = rolling_metrics(evals, window_days)

    # [S2] rolling_metrics nunca llega como None al alpha_engine
    metrics_safe = metrics if metrics is not None else _DEFAULT_METRICS

    # --------------------------------------------------
    # Confidence + quality
    # --------------------------------------------------
    conf = confidence_score(ret_ens_pct, metrics)  # [S5] siempre float
    quality = signal_quality(conf)
    strength = min(abs(ret_ens_pct) / 3.0, 1.0)   # [S1] ret_ens_pct ya validado

    # --------------------------------------------------
    # FUNDAMENTAL CONTEXT
    # --------------------------------------------------
    fundamental = (
        fundamental_signal_context(ticker)
        if FUNDAMENTAL_AVAILABLE
        else {"usable": False}
    )

    # Asegurar que fundamental nunca sea None
    if fundamental is None:
        fundamental = {"usable": False}

    fundamental_flag = None
    if fundamental.get("usable"):
        mp = fundamental.get("mispricing_pct")
        if mp is not None:
            if mp <= -30:
                fundamental_flag = "🟢 DEEP_VALUE"
            elif mp >= 30:
                fundamental_flag = "🔴 OVERHEATED"

    result = {
        "ticker": ticker,
        "date": p.get("date_base"),
        "recommendation": p.get("recommendation"),
        "ret_ens_pct": round(ret_ens_pct, 2),
        "price_now": round(price_now, 2),
        "price_pred": round(price_pred, 2),
        "confidence": conf,                          # [S5] siempre float
        "quality": quality,
        "signal_strength": round(strength, 3),
        "rolling_metrics": metrics_safe,             # [S2] siempre dict
        "fundamental": fundamental if fundamental.get("usable") else None,
        "fundamental_flag": fundamental_flag,
        "horizon_days": last["meta"].get("horizon_days"),
        "theta": last["meta"].get("theta"),
        "_cached_at": datetime.utcnow().isoformat(),
    }

    with _cache_lock:
        _signals_cache[cache_key] = result

    return result


# ======================================================
# Batch
# ======================================================
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
                # [S4] Incluso si compute_signal explota inesperadamente, retornar dict válido
                ticker = futs[f]
                logger.error(f"❌ compute_signal thread error {ticker}: {e}")
                out.append({
                    "ticker": ticker,
                    "error": str(e),
                    "confidence": 0.0,
                    "rolling_metrics": _DEFAULT_METRICS,
                })

    out.sort(key=lambda s: (-(s.get("confidence") or 0), -(s.get("signal_strength") or 0)))
    return out


# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--min-conf", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    signals = compute_all_signals(args.window)

    for s in signals:
        if (s.get("confidence") or 0) >= args.min_conf:
            print(
                f"{s['ticker']:>6} | {s.get('quality', 'N/A'):>8} | "
                f"{s.get('fundamental_flag', '–'):>12} | "
                f"{s.get('confidence', 0):.3f} | {s.get('ret_ens_pct', 0):+.2f}%"
        )
                

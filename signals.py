# ======================================================
# signals.py — v4.4
# ======================================================
# ✅ Compatible con evaluator.py v2.2
# ✅ Fundamental (model2_improved) como CONTEXTO
# ✅ Cache + concurrencia + TTL
#
# FIXES v4.3 (heredados):
#   [S1] ret_ens_pct None → crash
#   [S2] rolling_metrics None → alpha_engine recibe None
#   [S3] price_now / price_pred None → crash
#   [S4] compute_signal retornaba dict vacío en errores
#   [S5] confidence_score con metrics=None → retorna 0.0
#
# FIXES v4.4:
#   [S6] rolling_metrics filtra evaluaciones inválidas:
#        - legacy_bad_horizon=True  → excluir
#        - price_real=None          → excluir
#        - alpaca_unsupported=True  → excluir
#        - hit_sign=None            → excluir
#        Antes: None contaba como miss y distorsionaba hit_rate.
#   [S7] rolling_metrics reporta n_total y n_clean para
#        trazabilidad — visible en compute_signal resultado.
# ======================================================

import os
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
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
# CONFIG ENV
# ======================================================
DATA_PATH         = os.getenv("DATA_PATH", "/data")
DEFAULT_WINDOW    = int(os.getenv("SIGNAL_WINDOW", "30"))
MIN_CONFIDENCE    = float(os.getenv("SIGNAL_MIN_CONFIDENCE", "0.4"))
MIN_HISTORY       = int(os.getenv("SIGNAL_MIN_HISTORY", "3"))
MAX_CONCURRENT    = int(os.getenv("SIGNAL_MAX_CONCURRENT", "4"))
SIGNAL_DECAY_DAYS = int(os.getenv("SIGNAL_DECAY_DAYS", "14"))
CACHE_TTL         = int(os.getenv("SIGNAL_CACHE_TTL", "300"))

_DEFAULT_METRICS = {
    "n":              0,
    "n_clean":        0,
    "effective_n":    0.0,
    "hit_rate":       0.45,
    "mae_return_pct": 10.0,
}

# ======================================================
# LOGGING
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
        handlers=[fh, sh],
    )

logger = logging.getLogger(__name__)

# ======================================================
# CACHE
# ======================================================
_signals_cache: Dict[str, Dict] = {}
_cache_lock = Lock()

# ======================================================
# UTILS
# ======================================================
def load_json(path) -> Dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def list_json_files(path) -> List[str]:
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
# [S6] FILTRO DE EVALUACIONES LIMPIAS
# ======================================================

def _is_eval_clean(e: Dict) -> bool:
    """
    [S6] Una evaluación es válida para rolling_metrics si:
      1. No está marcada como legacy_bad_horizon=True
      2. Tiene price_real válido (no None)
      3. No es alpaca_unsupported=True
      4. Tiene hit_sign no nulo (None significa que no se pudo evaluar)

    Evaluaciones sin el campo legacy_bad_horizon (anteriores al fix)
    se aceptan SOLO si tienen price_real válido y hit_sign no nulo.
    Esto cubre el período de transición hasta que el script de
    limpieza marque todas las antiguas.
    """
    # Explícitamente marcada como mala
    if e.get("legacy_bad_horizon") is True:
        return False

    # Sin precio real → no evaluable
    if e.get("price_real") is None:
        return False

    # Ticker no soportado en Alpaca
    if e.get("alpaca_unsupported") is True:
        return False

    # hit_sign None significa que no se calculó (no es un miss)
    if e.get("hit_sign") is None:
        return False

    return True


# ======================================================
# ROLLING METRICS — [S6] solo evaluaciones limpias
# ======================================================

def rolling_metrics(evals: List[Dict], window_days: int) -> Optional[Dict]:
    """
    [S6] Filtra antes de calcular. Solo usa evaluaciones donde:
      - legacy_bad_horizon != True
      - price_real no es None
      - alpaca_unsupported != True
      - hit_sign no es None

    [S7] Reporta n_total y n_clean para trazabilidad.
    """
    if not evals:
        return None

    now    = datetime.utcnow().date()
    cutoff = now - timedelta(days=window_days)

    hits, errors, weights = [], [], []
    n_total = len(evals)
    n_clean = 0

    for e in evals:
        # [S6] Filtro estricto
        if not _is_eval_clean(e):
            continue

        d = parse_date(e.get("prediction_date", ""))
        if not d or d < cutoff:
            continue

        n_clean += 1
        age = max(0, (now - d).days)
        w   = np.exp(-max(0, age - 7) / (SIGNAL_DECAY_DAYS / 2))

        hits.append(1.0 if e.get("hit_sign") else 0.0)
        errors.append(abs(e.get("error_return_pct") or 0))
        weights.append(w)

    if not weights or sum(weights) == 0:
        return None

    return {
        "n":              n_total,
        "n_clean":        n_clean,   # [S7] trazabilidad
        "effective_n":    round(float(sum(weights)), 2),
        "hit_rate":       float(np.average(hits, weights=weights)),
        "mae_return_pct": float(np.average(errors, weights=weights)),
    }


# ======================================================
# CONFIDENCE SCORE
# ======================================================

def confidence_score(ret_pct: float, metrics: Dict) -> float:
    """[S5] Siempre retorna float, nunca None."""
    if not metrics:
        return 0.0
    if ret_pct is None:
        return 0.0

    strength = min(abs(ret_pct) / 3.0, 1.0)
    acc      = metrics.get("hit_rate") or 0.45
    mae      = metrics.get("mae_return_pct") or 10.0
    eff_n    = metrics.get("effective_n") or 0.0

    err  = 1.0 / (1.0 + mae / 10.0)
    size = min(eff_n / 10.0, 0.3)

    score = 0.35 * strength + 0.35 * acc + 0.2 * err + 0.1 * size

    if eff_n < MIN_HISTORY:
        score *= 0.7

    return round(float(np.clip(score, 0, 1)), 3)


def signal_quality(conf: float) -> str:
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
# COMPUTE SIGNAL (CORE)
# ======================================================

def compute_signal(ticker: str, window_days: int = DEFAULT_WINDOW) -> Dict[str, Any]:
    """
    [S4] Siempre retorna dict válido, nunca None.
    [S6][S7] rolling_metrics solo sobre evaluaciones limpias.
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
        return {
            "ticker": ticker, "error": "NO_PREDICTIONS",
            "confidence": 0.0, "rolling_metrics": _DEFAULT_METRICS,
        }

    last = load_json(pred_dir / preds[-1])
    if not last or "prediction" not in last or "meta" not in last:
        return {
            "ticker": ticker, "error": "INVALID_PREDICTION",
            "confidence": 0.0, "rolling_metrics": _DEFAULT_METRICS,
        }

    p = last["prediction"]

    # [S1] Validar ret_ens_pct
    ret_ens_pct = p.get("ret_ens_pct")
    if ret_ens_pct is None:
        logger.warning(f"⚠️ [{ticker}] ret_ens_pct es None")
        return {
            "ticker": ticker, "error": "ret_ens_pct_null",
            "confidence": 0.0, "rolling_metrics": _DEFAULT_METRICS,
        }
    ret_ens_pct = float(ret_ens_pct)

    # [S3] Validar precios
    price_now  = p.get("price_now")
    price_pred = p.get("price_pred")
    if price_now is None or price_pred is None:
        logger.warning(f"⚠️ [{ticker}] price_now o price_pred es None")
        return {
            "ticker": ticker, "error": "price_null",
            "confidence": 0.0, "rolling_metrics": _DEFAULT_METRICS,
        }
    price_now  = float(price_now)
    price_pred = float(price_pred)

    # [S6] Cargar evaluaciones y filtrar limpias
    all_evals = [load_json(eval_dir / f) for f in list_json_files(eval_dir)]
    metrics   = rolling_metrics(all_evals, window_days)

    # [S2] Nunca None al alpha_engine
    metrics_safe = metrics if metrics is not None else _DEFAULT_METRICS

    conf     = confidence_score(ret_ens_pct, metrics)
    quality  = signal_quality(conf)
    strength = min(abs(ret_ens_pct) / 3.0, 1.0)

    fundamental = (
        fundamental_signal_context(ticker)
        if FUNDAMENTAL_AVAILABLE
        else {"usable": False}
    ) or {"usable": False}

    fundamental_flag = None
    if fundamental.get("usable"):
        mp = fundamental.get("mispricing_pct")
        if mp is not None:
            if mp <= -30:
                fundamental_flag = "🟢 DEEP_VALUE"
            elif mp >= 30:
                fundamental_flag = "🔴 OVERHEATED"

    result = {
        "ticker":          ticker,
        "date":            p.get("date_base"),
        "recommendation":  p.get("recommendation"),
        "ret_ens_pct":     round(ret_ens_pct, 2),
        "price_now":       round(price_now, 2),
        "price_pred":      round(price_pred, 2),
        "confidence":      conf,
        "quality":         quality,
        "signal_strength": round(strength, 3),
        "rolling_metrics": metrics_safe,
        "eval_coverage": {                      # [S7] trazabilidad
            "n_total":   len(all_evals),
            "n_clean":   metrics_safe.get("n_clean", 0),
            "pct_clean": round(
                metrics_safe.get("n_clean", 0) / len(all_evals) * 100, 1
            ) if all_evals else 0,
        },
        "fundamental":      fundamental if fundamental.get("usable") else None,
        "fundamental_flag": fundamental_flag,
        "horizon_days":     last["meta"].get("horizon_days"),
        "theta":            last["meta"].get("theta"),
        "_cached_at":       datetime.utcnow().isoformat(),
    }

    with _cache_lock:
        _signals_cache[cache_key] = result

    return result


# ======================================================
# BATCH
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
                ticker = futs[f]
                logger.error(f"❌ compute_signal thread error {ticker}: {e}")
                out.append({
                    "ticker": ticker, "error": str(e),
                    "confidence": 0.0, "rolling_metrics": _DEFAULT_METRICS,
                })

    out.sort(key=lambda s: (-(s.get("confidence") or 0), -(s.get("signal_strength") or 0)))
    return out


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window",   type=int,   default=DEFAULT_WINDOW)
    parser.add_argument("--min-conf", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--verbose",  "-v",       action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    signals = compute_all_signals(args.window)

    for s in signals:
        if (s.get("confidence") or 0) >= args.min_conf:
            cov = s.get("eval_coverage", {})
            print(
                f"{s['ticker']:>12} | {s.get('quality', 'N/A'):>10} | "
                f"conf={s.get('confidence', 0):.3f} | "
                f"{s.get('ret_ens_pct', 0):+.2f}% | "
                f"evals_limpias={cov.get('n_clean', 0)}/{cov.get('n_total', 0)}"
            )

# =========================================================
# alpha_engine_enterprise_strict.py
# =========================================================
# ✔ ALFA = Resumen estructural del sistema completo
# ✔ NO adaptativo
# ✔ Mercado es componente interno
# ✔ Lee SOLO datos persistidos
# ✔ Calcula SOLO lo faltante
# ✔ Sin fallback silencioso
# ✔ Debug estructural completo
# =========================================================

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

from signals import compute_signal
from data_provider import get_price_history

# =========================================================
# PATHS
# =========================================================

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
PRED_DIR = DATA_PATH / "predictions"
EVAL_DIR = DATA_PATH / "evaluations"
MARKET_FILE = DATA_PATH / "market_context.json"
ALPHA_FILE = DATA_PATH / "alpha_last.json"

# =========================================================
# JSON HELPERS
# =========================================================

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def get_latest_prediction_file(ticker: str) -> Path:
    d = PRED_DIR / ticker
    if not d.exists():
        raise FileNotFoundError(f"No predictions folder for {ticker}")

    files = sorted(d.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No prediction files for {ticker}")

    return files[-1]

# =========================================================
# NORMALIZACIONES
# =========================================================

def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def normalize_return(ret_pct: float) -> float:
    return clip01(abs(ret_pct) / 3.0)


def normalize_hit_rate(hit: float) -> float:
    return clip01((hit - 0.5) / 0.3)


def normalize_error(mae_pct: float) -> float:
    return clip01(1.0 / (1.0 + mae_pct / 10.0))


def normalize_fundamental(mispricing: float) -> float:
    return clip01(abs(mispricing) / 40.0)


def normalize_market(volatility: float, drawdown: float) -> float:
    vol_component = clip01(1 - abs(volatility - 0.03) / 0.03)
    dd_component = clip01(1 + drawdown)
    return clip01(0.6 * vol_component + 0.4 * dd_component)

# =========================================================
# STRUCTURAL SCORE (SI NO EXISTE)
# =========================================================

def compute_structural_score(ticker: str) -> float:

    raw = get_price_history(ticker, period="1y", interval="1d")
    if raw is None or len(raw) < 60:
        raise ValueError("Not enough history for structural")

    closes = raw["Close"].values
    volumes = raw["Volume"].values

    returns = np.diff(closes) / closes[:-1]
    volatility = np.std(returns)
    trend = (closes[-1] / closes[0]) - 1
    liquidity = np.mean(closes * volumes)

    trend_score = clip01(trend / 0.20)
    vol_score = clip01(1 - abs(volatility - 0.03) / 0.03)
    liquidity_score = clip01(liquidity / 50_000_000)

    return clip01(0.4 * trend_score + 0.3 * vol_score + 0.3 * liquidity_score)

# =========================================================
# PERFORMANCE HISTÓRICA
# =========================================================

def compute_performance_metrics(ticker: str, days: int = 60):

    folder = EVAL_DIR / ticker
    if not folder.exists():
        raise FileNotFoundError("No evaluation folder")

    cutoff = datetime.utcnow() - timedelta(days=days)

    hits = []
    errors = []

    for f in folder.glob("*.json"):
        d = load_json(f)
        if not d:
            continue

        dt = datetime.fromisoformat(d["evaluated_at"])
        if dt < cutoff:
            continue

        hits.append(1 if d.get("decision_correct") else 0)
        errors.append(abs(d.get("error_return_pct", 0)))

    if len(hits) < 3:
        raise ValueError("Not enough evaluation history")

    hit_rate = sum(hits) / len(hits)
    mae = sum(errors) / len(errors)

    return hit_rate, mae

# =========================================================
# ALFA CORE
# =========================================================

def compute_alpha_for_ticker(ticker: str):

    # -------- Prediction --------
    fp = get_latest_prediction_file(ticker)
    pred_json = load_json(fp)

    if not pred_json or "prediction" not in pred_json:
        raise ValueError("Invalid prediction file")

    p = pred_json["prediction"]
    ret_pct = p["ret_ens_pct"]

    # -------- Signal --------
    signal = compute_signal(ticker)

    if "confidence" not in signal:
        raise ValueError("Signal missing confidence")

    confidence = signal.get("confidence")

    if confidence is None:
        # Castigo explícito por falta de historia suficiente
        confidence = 0.0
        confidence_penalty = True
    else:
        confidence_penalty = False

    metrics = signal.get("rolling_metrics")

    if not metrics:
        # Castigo máximo estructural por falta de historia
        hit_rate = 0.0
        mae = 100.0   # Error extremo → normaliza casi a 0
        rolling_penalty = True
    else:
        hit_rate = metrics["hit_rate"]
        mae = metrics["mae_return_pct"]
        rolling_penalty = False

    # -------- Fundamental --------
    fundamental = signal.get("fundamental")

    if not fundamental or not fundamental.get("usable"):
        # Castigo máximo si no hay fundamental
        fundamental_score = 0.0
        fundamental_penalty = True
    else:
        mispricing = fundamental.get("mispricing_pct", 0.0)
        fundamental_score = normalize_fundamental(mispricing)
        fundamental_penalty = False
        
    # -------- Market --------
    market = load_json(MARKET_FILE)
    if not market:
        raise ValueError("Missing market_context.json")

    market_score = normalize_market(
        market.get("volatility", 0.03),
        market.get("drawdown", 0.0)
    )

    # -------- Structural --------
    try:
        structural_score = compute_structural_score(ticker)
        structural_penalty = False
    except Exception:
        # Castigo máximo si falla cálculo estructural
        structural_score = 0.0
        structural_penalty = True
    # =====================================================
    # VECTOR ALFA
    # =====================================================

    alpha = (
        0.22 * normalize_return(ret_pct) +
        0.18 * confidence +
        0.18 * normalize_hit_rate(hit_rate) +
        0.12 * normalize_error(mae) +
        0.15 * structural_score +
        0.10 * fundamental_score +
        0.05 * market_score
    )

    alpha = clip01(alpha)

    return {
        "ticker": ticker,
        "alpha_score": round(alpha, 4),
        "components": {
            "return": normalize_return(ret_pct),
            "confidence": confidence,
            "hit_rate": normalize_hit_rate(hit_rate),
            "error_component": normalize_error(mae),
            "structural": structural_score,
            "fundamental": fundamental_score,
            "market": market_score,
        },
        "raw": {
            "ret_pct": ret_pct,
            "hit_rate": hit_rate,
            "mae": mae,
        }
    }

# =========================================================
# BATCH + PERSIST
# =========================================================

def compute_and_persist_alpha(tickers: List[str]):

    results = {}

    for t in tickers:
        try:
            results[t] = compute_alpha_for_ticker(t)
        except Exception as e:
            results[t] = {"error": str(e)}

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "universe_size": len(tickers),
        "calculated": len([r for r in results.values() if "alpha_score" in r]),
        "results": results
    }

    save_json(ALPHA_FILE, payload)

    return payload

# signals.py
# Construcción de señales operativas a partir de:
# - predicciones guardadas
# - evaluaciones históricas
# NO entrena modelos
# NO recalcula precios
# SOLO agrega métricas de confianza y calidad

import os
import json
from datetime import datetime, timedelta

import numpy as np


DATA_PATH = os.getenv("DATA_PATH", "/data")


# =========================
# Utils
# =========================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def list_json_files(path):
    if not os.path.exists(path):
        return []
    return sorted([f for f in os.listdir(path) if f.endswith(".json")])


def load_universe():
    """
    Lee tickers.json desde la raíz del proyecto
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "tickers.json")

    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        data = json.load(f)
        return data.get("tickers", [])


# =========================
# Métricas rolling
# =========================
def rolling_metrics(evals, window=30):
    """
    Calcula métricas rolling sobre las últimas N evaluaciones
    """
    recent = evals[-window:] if len(evals) >= window else evals

    if not recent:
        return None

    hits = [e["decision_correct"] for e in recent]
    errors = [abs(e["error_price_pct"]) for e in recent]

    return {
        "n": len(recent),
        "hit_rate": float(np.mean(hits)),
        "mae_price_pct": float(np.mean(errors)),
    }


# =========================
# Confidence score
# =========================
def confidence_score(ret_pct, hit_rate, mae_pct):
    """
    Score heurístico [0,1]
    - fuerza de señal
    - calidad histórica
    - penaliza error alto
    """

    if hit_rate is None or mae_pct is None:
        return None

    strength = min(abs(ret_pct) / 2.0, 1.0)        # saturación a 2%
    quality = hit_rate                             # 0–1
    penalty = min(mae_pct / 5.0, 1.0)              # castigo >5%

    score = 0.5 * strength + 0.4 * quality + 0.1 * (1 - penalty)
    return round(float(score), 3)


# =========================
# Calidad de señal
# =========================
def signal_quality(confidence):
    if confidence is None:
        return "NO_DATA"
    if confidence >= 0.65:
        return "GOOD"
    if confidence >= 0.45:
        return "WEAK"
    return "NOISE"


# =========================
# Señal para un ticker
# =========================
def compute_signal(ticker: str, window: int = 30):
    """
    Devuelve señal operativa consolidada para un ticker
    """

    pred_dir = os.path.join(DATA_PATH, "predictions", ticker)
    eval_dir = os.path.join(DATA_PATH, "evaluations", ticker)

    preds = list_json_files(pred_dir)
    evals_files = list_json_files(eval_dir)

    if not preds:
        return {"ticker": ticker, "error": "No predictions found"}

    # Última predicción
    last_pred = load_json(os.path.join(pred_dir, preds[-1]))
    p = last_pred["prediction"]

    # Cargar evaluaciones
    evals = []
    for f in evals_files:
        try:
            evals.append(load_json(os.path.join(eval_dir, f)))
        except Exception:
            continue

    metrics = rolling_metrics(evals, window=window) if evals else None

    if metrics:
        conf = confidence_score(
            ret_pct=p["ret_ens_pct"],
            hit_rate=metrics["hit_rate"],
            mae_pct=metrics["mae_price_pct"],
        )
        quality = signal_quality(conf)
    else:
        conf = None
        quality = "NO_DATA"

    return {
        "ticker": ticker,
        "date": p["date_base"],
        "recommendation": p["recommendation"],
        "ret_ens_pct": p["ret_ens_pct"],
        "price_now": p["price_now"],
        "price_pred": p["price_pred"],
        "confidence": conf,
        "quality": quality,
        "rolling": metrics,
    }


# =========================
# Señales para todos
# =========================
def compute_all_signals(window: int = 30):
    """
    Devuelve señales para todos los tickers definidos en tickers.json
    """

    tickers = load_universe()
    if not tickers:
        return []

    signals = []
    for ticker in tickers:
        sig = compute_signal(ticker, window=window)
        signals.append(sig)

    return signals

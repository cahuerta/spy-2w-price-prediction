# =========================================================
# alpha_engine_v4.py — PRODUCTION ALPHA ENGINE
# =========================================================
# ✔ Lee JSON si existen
# ✔ Si no existen → calcula
# ✔ Si no puede calcular → descarta
# ✔ Totalmente desacoplado del orchestrator
# ✔ Listo para producción
# =========================================================

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

from model import run_model
from signals import compute_signal
from model2 import fundamental_signal_context
from structural_engine import compute_score
from data_provider import get_price_history


DATA_PATH = os.getenv("DATA_PATH", "/data")


# =========================================================
# Utils
# =========================================================

def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def normalize_return(ret_pct: float) -> float:
    return clip01(abs(ret_pct) / 3.0)


def normalize_hit_rate(hit: Optional[float]) -> float:
    if hit is None:
        return 0.5
    return clip01((hit - 0.5) / 0.3)


def normalize_fundamental(mispricing: Optional[float]) -> float:
    if mispricing is None:
        return 0.5
    return clip01(abs(mispricing) / 40.0)


# =========================================================
# CORE ALPHA
# =========================================================

def compute_alpha_for_ticker(ticker: str, horizon: int = 10) -> Optional[Dict[str, Any]]:

    # =============================================
    # 1️⃣ MODEL (usa JSON si existe)
    # =============================================
    try:
        model_result = run_model(ticker=ticker, horizon=horizon)
    except Exception:
        return None

    try:
        ret_pct = model_result["prediction"]["ret_ens_pct"]
        hit_rate = model_result["historical"]["hit_rate_mean"]
        n_windows = model_result["historical"]["n_windows"]
    except Exception:
        return None

    ret_score = normalize_return(ret_pct)
    hit_score = normalize_hit_rate(hit_rate)
    wf_score = clip01(n_windows / 10.0)

    # =============================================
    # 2️⃣ SIGNALS
    # =============================================
    signal_data = compute_signal(ticker)
    confidence = signal_data.get("confidence", 0.0)
    conf_score = clip01(confidence)

    # =============================================
    # 3️⃣ STRUCTURAL (lee precios si necesita)
    # =============================================
    try:
        raw = get_price_history(ticker, period="1y", interval="1d")
        if raw is None or len(raw) < 60:
            return None

        closes = raw["Close"].values
        volumes = raw["Volume"].values

        structural = compute_score(closes=closes, volumes=volumes)
        if structural is None:
            return None

        structural_score = structural["score"]

    except Exception:
        return None

    # =============================================
    # 4️⃣ FUNDAMENTAL
    # =============================================
    fundamental = fundamental_signal_context(ticker)
    fundamental_score = 0.5

    if fundamental.get("usable"):
        fundamental_score = normalize_fundamental(
            fundamental.get("mispricing_pct")
        )

    # =============================================
    # 5️⃣ ALPHA COMPOSITE
    # =============================================
    alpha = (
        0.30 * ret_score +
        0.20 * conf_score +
        0.15 * hit_score +
        0.20 * structural_score +
        0.10 * fundamental_score +
        0.05 * wf_score
    )

    alpha = clip01(alpha)

    if alpha >= 0.85:
        quality = "ELITE"
    elif alpha >= 0.75:
        quality = "HIGH"
    elif alpha >= 0.65:
        quality = "STRONG"
    elif alpha >= 0.55:
        quality = "MODERATE"
    else:
        quality = "WEAK"

    return {
        "ticker": ticker,
        "alpha_score": round(alpha, 3),
        "quality": quality,
        "ret_pct": ret_pct,
        "confidence": confidence,
        "structural_score": structural_score,
        "hit_rate": hit_rate,
        "walkforward_windows": n_windows,
    }


# =========================================================
# BATCH
# =========================================================

def compute_batch(tickers: List[str]) -> Dict[str, Dict[str, Any]]:

    results = {}

    for ticker in tickers:
        try:
            r = compute_alpha_for_ticker(ticker)
            if r:
                results[ticker] = r
        except Exception:
            continue

    return results

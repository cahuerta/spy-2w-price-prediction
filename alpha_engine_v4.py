# =========================================================
# alpha_engine_v4.py — PRODUCTION ALPHA ENGINE (ADAPTIVE)
# =========================================================
# ✔ Lee JSON si existen (model / signals ya lo hacen)
# ✔ Si no existen → calcula
# ✔ Si no puede calcular → descarta
# ✔ Totalmente desacoplado del orchestrator
# ✔ Sin dependencias inventadas
# ✔ NUEVO: Adaptativo según performance real (evaluator)
# =========================================================

import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from model import run_model
from signals import compute_signal
from model2 import fundamental_signal_context
from data_provider import get_price_history


DATA_PATH = os.getenv("DATA_PATH", "/data")


# =========================================================
# ================== STRUCTURAL ENGINE ====================
# =========================================================

def pct_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    prices = np.nan_to_num(prices)

    if len(prices) < 2:
        return np.array([], dtype=float)

    prev = prices[:-1]
    prev = np.where(prev == 0, 1e-9, prev)

    return np.diff(prices) / prev


def compute_rsi_wilder(prices: np.ndarray, period: int = 14) -> float:
    prices = np.asarray(prices, dtype=float)
    prices = np.nan_to_num(prices)

    if len(prices) < period + 1:
        return 50.0

    deltas = np.diff(prices)

    gains = pd.Series(np.clip(deltas, 0, None))
    losses = pd.Series(np.clip(-deltas, 0, None))

    avg_gain = gains.ewm(span=period, adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(span=period, adjust=False).mean().iloc[-1]

    if avg_loss == 0:
        return 70.0

    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def compute_max_drawdown(prices: np.ndarray) -> float:
    prices = np.asarray(prices, dtype=float)
    cumulative_max = np.maximum.accumulate(prices)
    drawdowns = (prices - cumulative_max) / cumulative_max
    return float(np.min(drawdowns))


def compute_score(
    closes: np.ndarray,
    volumes: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    rsi_period: int = 14,
    min_dollar_volume: float = 50_000_000,
) -> Optional[Dict[str, Any]]:

    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    closes = np.nan_to_num(closes)
    volumes = np.nan_to_num(volumes)

    if len(closes) < 30:
        return None

    returns = pct_returns(closes)
    if len(returns) < 20:
        return None

    volatility = float(np.std(returns))
    trend = float((closes[-1] / closes[0]) - 1)
    trend_score = float(np.clip(trend / 0.20, 0, 1))

    closes_pd = pd.Series(closes)
    ma20 = float(closes_pd.tail(20).mean())
    ma50 = float(closes_pd.tail(50).mean()) if len(closes) >= 50 else ma20

    momentum_short = max((closes[-1] - ma20) / ma20, 0) if ma20 > 0 else 0.0
    momentum_long = max((closes[-1] - ma50) / ma50, 0) if ma50 > 0 else 0.0
    momentum = float(np.clip(0.6 * momentum_short + 0.4 * momentum_long, 0, 1))

    rsi = compute_rsi_wilder(closes, rsi_period)
    rsi_score = float(1 - abs(rsi - 50) / 50)

    sharpe = float(np.mean(returns) / volatility * np.sqrt(252)) if volatility > 0 else 0.0
    sharpe_score = float(np.clip((sharpe - 0.5) / 1.5, 0, 1))

    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 5 else volatility
    sortino = float(np.mean(returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
    sortino_score = float(np.clip(sortino / 2.0, 0, 1))

    max_dd = compute_max_drawdown(closes)
    dd_score = float(np.clip(1 + max_dd, 0, 1))

    optimal_vol = 0.03
    vol_score = float(np.clip(1 - abs(volatility - optimal_vol) / optimal_vol, 0, 1))

    dollar_volume = float(np.mean(closes * volumes))
    liquidity_score = float(np.clip(dollar_volume / min_dollar_volume, 0, 1))

    score = (
        0.20 * trend_score +
        0.15 * momentum +
        0.10 * rsi_score +
        0.15 * sharpe_score +
        0.10 * sortino_score +
        0.10 * dd_score +
        0.10 * vol_score +
        0.07 * liquidity_score
    )

    return {"score": float(np.clip(score, 0, 1))}


# =========================================================
# ================= ADAPTIVE PERFORMANCE ==================
# =========================================================

def compute_performance_multiplier(ticker: str, days: int = 60) -> float:

    eval_dir = Path(DATA_PATH) / "evaluations" / ticker
    if not eval_dir.exists():
        return 1.0

    cutoff = datetime.utcnow() - timedelta(days=days)

    hits = []
    errors = []

    for f in eval_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            dt = datetime.fromisoformat(d["evaluated_at"])
            if dt < cutoff:
                continue

            hits.append(1 if d.get("decision_correct") else 0)
            errors.append(abs(d.get("error_return_pct", 0)))

        except Exception:
            continue

    if len(hits) < 5:
        return 1.0

    hit_rate = sum(hits) / len(hits)
    avg_error = sum(errors) / len(errors)

    hit_component = np.clip((hit_rate - 0.5) / 0.25, -1, 1)
    error_component = np.clip(1 - avg_error / 10.0, 0, 1)

    multiplier = 1 + 0.4 * hit_component + 0.2 * (error_component - 0.5)

    return float(np.clip(multiplier, 0.6, 1.4))


# =========================================================
# ===================== ALPHA CORE ========================
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


def compute_alpha_for_ticker(ticker: str, horizon: int = 10) -> Optional[Dict[str, Any]]:

    try:
        model_result = run_model(ticker=ticker, horizon=horizon)
        ret_pct = model_result["prediction"]["ret_ens_pct"]
        hit_rate = model_result["historical"]["hit_rate_mean"]
        n_windows = model_result["historical"]["n_windows"]
    except Exception:
        return None

    signal_data = compute_signal(ticker)
    confidence = signal_data.get("confidence", 0.0)

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

    fundamental = fundamental_signal_context(ticker)
    fundamental_score = 0.5

    if fundamental.get("usable"):
        fundamental_score = normalize_fundamental(
            fundamental.get("mispricing_pct")
        )

    alpha = (
        0.30 * normalize_return(ret_pct) +
        0.20 * clip01(confidence) +
        0.15 * normalize_hit_rate(hit_rate) +
        0.20 * structural_score +
        0.10 * fundamental_score +
        0.05 * clip01(n_windows / 10.0)
    )

    alpha = clip01(alpha)

    # =============================
    # ADAPTIVE MULTIPLIER
    # =============================
    performance_multiplier = compute_performance_multiplier(ticker)
    alpha = clip01(alpha * performance_multiplier)

    return {
        "ticker": ticker,
        "alpha_score": round(alpha, 3),
        "ret_pct": ret_pct,
        "confidence": confidence,
        "structural_score": structural_score,
        "hit_rate": hit_rate,
        "walkforward_windows": n_windows,
        "performance_multiplier": round(performance_multiplier, 3),
    }


def compute_batch(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    results = {}
    for ticker in tickers:
        r = compute_alpha_for_ticker(ticker)
        if r:
            results[ticker] = r
    return results


# =========================================================
# ================== PERSISTENCE LAYER ====================
# =========================================================

ALPHA_FILE = Path(DATA_PATH) / "alpha_last.json"


def compute_and_persist_alpha(tickers: List[str]) -> Dict[str, Any]:

    results = compute_batch(tickers)

    ranked = dict(
        sorted(
            results.items(),
            key=lambda x: x[1]["alpha_score"],
            reverse=True
        )
    )

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "universe_size": len(tickers),
        "calculated": len(ranked),
        "results": ranked
    }

    ALPHA_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALPHA_FILE.write_text(json.dumps(payload, indent=2))

    return payload

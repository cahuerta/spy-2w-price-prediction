from typing import Optional, Dict, Any
import numpy as np
import pandas as pd


# =========================================================
# Utils
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


# =========================================================
# Motor principal Enterprise
# =========================================================

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

    # =============================
    # Trend
    # =============================
    trend = float((closes[-1] / closes[0]) - 1)
    trend_score = float(np.clip(trend / 0.20, 0, 1))

    # =============================
    # Momentum
    # =============================
    closes_pd = pd.Series(closes)

    ma20 = float(closes_pd.tail(20).mean())
    ma50 = float(closes_pd.tail(50).mean()) if len(closes) >= 50 else ma20

    momentum_short = max((closes[-1] - ma20) / ma20, 0) if ma20 > 0 else 0.0
    momentum_long = max((closes[-1] - ma50) / ma50, 0) if ma50 > 0 else 0.0

    momentum = float(np.clip(0.6 * momentum_short + 0.4 * momentum_long, 0, 1))

    # =============================
    # RSI
    # =============================
    rsi = compute_rsi_wilder(closes, rsi_period)
    rsi_score = float(1 - abs(rsi - 50) / 50)

    # =============================
    # Sharpe (mejor escalado)
    # =============================
    sharpe = float(np.mean(returns) / volatility * np.sqrt(252)) if volatility > 0 else 0.0
    sharpe_score = float(np.clip((sharpe - 0.5) / 1.5, 0, 1))

    # =============================
    # Sortino
    # =============================
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 5 else volatility
    sortino = float(np.mean(returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
    sortino_score = float(np.clip(sortino / 2.0, 0, 1))

    # =============================
    # Max Drawdown
    # =============================
    max_dd = compute_max_drawdown(closes)
    dd_score = float(np.clip(1 + max_dd, 0, 1))  # menos drawdown = mejor

    # =============================
    # Volatility (zona óptima)
    # =============================
    optimal_vol = 0.03
    vol_score = float(np.clip(1 - abs(volatility - optimal_vol) / optimal_vol, 0, 1))

    # =============================
    # Liquidez
    # =============================
    dollar_volume = float(np.mean(closes * volumes))
    liquidity_score = float(np.clip(dollar_volume / min_dollar_volume, 0, 1))

    # =============================
    # Beta
    # =============================
    beta = 1.0
    beta_score = 0.0

    if benchmark_returns is not None:
        benchmark_returns = np.asarray(benchmark_returns, dtype=float)
        n = min(len(returns), len(benchmark_returns))
        if n > 20:
            r = returns[-n:]
            b = benchmark_returns[-n:]
            var_b = float(np.var(b))
            if var_b > 1e-10:
                beta = float(np.cov(r, b)[0, 1] / var_b)
                beta_score = float(np.clip(1 - abs(beta - 1.0), 0, 1))

    # =============================
    # Score final Enterprise
    # =============================
    score = (
        0.20 * trend_score +
        0.15 * momentum +
        0.10 * rsi_score +
        0.15 * sharpe_score +
        0.10 * sortino_score +
        0.10 * dd_score +
        0.10 * vol_score +
        0.07 * liquidity_score +
        0.03 * beta_score
    )

    score = float(np.clip(score, 0, 1))

    # =============================
    # Quality label
    # =============================
    if score >= 0.85:
        quality = "🚀 INSTITUTIONAL"
    elif score >= 0.70:
        quality = "✅ STRONG"
    elif score >= 0.55:
        quality = "⚠️ MODERATE"
    else:
        quality = "❌ NOISE"

    return {
        "score": round(score, 3),
        "quality": quality,
        "trend_3m_pct": round(trend * 100, 2),
        "volatility": round(volatility, 4),
        "rsi_wilder": round(rsi, 1),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "beta_spy": round(beta, 3),
        "avg_dollar_volume": int(dollar_volume),
    }

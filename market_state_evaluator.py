# =========================================================
# market_quant_context.py — V3.2 RIESGO REAL (PRODUCCIÓN)
# =========================================================

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Literal
from datetime import datetime, timedelta

import yfinance as yf


# =========================================================
# CONFIGURACIÓN
# =========================================================
MARKET_MAIN_SYMBOL = "SPY"
MARKET_CROSS_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
MARKET_LOOKBACK_DAYS = 150

VOL_LOOKBACK = 20
DD_LOOKBACK = 63
TREND_LOOKBACK = 50
CORR_LOOKBACK = 30

MARKET_HISTORY_DIR = "data/market_history"


# =========================================================
# UTILIDAD CRÍTICA — ESCALAR FORZADO (FUENTE)
# =========================================================
def _scalar(x, default: float = 0.0) -> float:
    """
    Fuerza float escalar REAL.
    Nunca retorna Series / ndarray / NaN.
    """
    try:
        if x is None:
            return default

        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            return float(np.nanmean(arr)) if arr.size else default

        if hasattr(x, "mean"):  # pandas Series
            v = x.mean()
            return float(v) if np.isfinite(v) else default

        x = float(x)
        return x if np.isfinite(x) else default

    except Exception:
        return default


# =========================================================
# DATACLASS SALIDA
# =========================================================
@dataclass
class QuantMarketContext:
    regime: Literal["growth", "neutral", "defensive"]
    volatility: float
    drawdown_rolling: float
    trend_strength: float
    cross_asset_correlation: float
    downside_risk: Literal["low", "medium", "high", "unknown"]
    n_observations: int
    corr_source: Literal["measured", "historical", "knn", "unavailable"]

    def to_dict(self) -> Dict:
        return asdict(self)


# =========================================================
# MÉTRICAS DETERMINÍSTICAS
# =========================================================
def realized_volatility(returns: pd.Series):
    if len(returns) < VOL_LOOKBACK:
        return np.nan
    return np.sqrt(252) * returns.tail(VOL_LOOKBACK).std()


def rolling_drawdown(prices: pd.Series):
    if len(prices) < DD_LOOKBACK:
        return np.nan
    w = prices.tail(DD_LOOKBACK)
    return ((w / w.cummax()) - 1).min()


def trend_strength(prices: pd.Series):
    if len(prices) < TREND_LOOKBACK:
        return np.nan
    ma = prices.rolling(TREND_LOOKBACK).mean().dropna()
    if ma.empty:
        return np.nan
    return (prices.iloc[-1] / ma.iloc[-1]) - 1.0


def cross_asset_corr(df: pd.DataFrame):
    if df.shape[1] < 2:
        return np.nan
    rets = df.pct_change().dropna()
    if len(rets) < CORR_LOOKBACK:
        return np.nan
    corr = rets.tail(CORR_LOOKBACK).corr().values
    upper = corr[np.triu_indices_from(corr, k=1)]
    return np.nanmean(upper) if upper.size else np.nan


# =========================================================
# HISTÓRICO REAL DESDE JSON
# =========================================================
def load_market_history() -> pd.DataFrame:
    if not os.path.isdir(MARKET_HISTORY_DIR):
        return pd.DataFrame()

    rows = []
    for f in os.listdir(MARKET_HISTORY_DIR):
        if not f.endswith(".json"):
            continue
        try:
            with open(os.path.join(MARKET_HISTORY_DIR, f), "r") as fh:
                rows.append(json.load(fh))
        except Exception:
            continue

    return pd.DataFrame(rows)


# =========================================================
# FALLBACK HISTÓRICO
# =========================================================
def historical_corr(history: pd.DataFrame):
    if history.empty or "cross_asset_correlation" not in history:
        return np.nan
    s = history["cross_asset_correlation"].dropna()
    return s.median() if len(s) >= 20 else np.nan


def knn_corr(history: pd.DataFrame, features: Dict[str, float], k: int = 5):
    required = ["volatility", "drawdown_rolling", "trend_strength"]

    if history.empty or not all(c in history for c in required):
        return np.nan

    if any(np.isnan(features[r]) for r in required):
        return np.nan

    hist = history.dropna(subset=required + ["cross_asset_correlation"])
    if len(hist) < k:
        return np.nan

    X = hist[required].values
    y = hist["cross_asset_correlation"].values
    x0 = np.array([features[r] for r in required])

    dists = np.linalg.norm(X - x0, axis=1)
    idx = np.argsort(dists)[:k]

    return np.mean(y[idx])


# =========================================================
# CLASIFICADORES
# =========================================================
def classify_downside(dd: float) -> Literal["low", "medium", "high", "unknown"]:
    if np.isnan(dd):
        return "unknown"
    if dd <= -0.20:
        return "high"
    if dd <= -0.10:
        return "medium"
    return "low"


def classify_regime(vol, dd, corr, trend) -> Literal["growth", "neutral", "defensive"]:
    if any(np.isnan(x) for x in [vol, dd, corr, trend]):
        return "neutral"
    if dd <= -0.25 or vol >= 0.45 or corr >= 0.90:
        return "defensive"
    if trend > 0.05 and vol <= 0.10 and corr < 0.60:
        return "growth"
    return "neutral"


# =========================================================
# LOADER YAHOO FINANCE
# =========================================================
def load_market_prices() -> tuple[pd.Series, pd.DataFrame]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=MARKET_LOOKBACK_DAYS)

    main = yf.download(
        MARKET_MAIN_SYMBOL,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    cross = yf.download(
        MARKET_CROSS_SYMBOLS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if main.empty or cross.empty:
        raise RuntimeError("Datos de mercado no disponibles")

    prices_main = main["Close"].dropna()
    prices_cross = (
        cross["Close"]
        if isinstance(cross.columns, pd.MultiIndex)
        else cross.filter(like="Close")
    ).dropna()

    return prices_main, prices_cross


# =========================================================
# CORE
# =========================================================
def run_market_state() -> QuantMarketContext:
    prices_main, prices_cross = load_market_prices()
    history = load_market_history()

    returns = prices_main.pct_change().dropna()

    vol_raw = realized_volatility(returns)
    dd_raw = rolling_drawdown(prices_main)
    trend_raw = trend_strength(prices_main)
    corr_raw = cross_asset_corr(prices_cross)

    vol = _scalar(vol_raw)
    dd = _scalar(dd_raw)
    trend = _scalar(trend_raw)
    corr = _scalar(corr_raw)

    source = "measured"

    if corr == 0.0:
        corr = _scalar(historical_corr(history))
        source = "historical"

    if corr == 0.0:
        corr = _scalar(
            knn_corr(
                history,
                {
                    "volatility": vol,
                    "drawdown_rolling": dd,
                    "trend_strength": trend,
                },
            )
        )
        source = "knn"

    if corr == 0.0:
        source = "unavailable"

    return QuantMarketContext(
        regime=classify_regime(vol, dd, corr, trend),
        volatility=vol,
        drawdown_rolling=dd,
        trend_strength=trend,
        cross_asset_correlation=corr,
        downside_risk=classify_downside(dd),
        n_observations=len(prices_main),
        corr_source=source,
    )


if __name__ == "__main__":
    print(run_market_state().to_dict())

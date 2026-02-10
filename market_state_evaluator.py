# =========================================================
# evaluador_cuantitativo_mercado.py — V3.1 RIESGO REAL (PRODUCCIÓN)
#
# MOTOR CUANTITATIVO DE ENTORNO DE MERCADO (AUTÓNOMO)
# ---------------------------------------------------------
# ✔ Determinista, auditable, reproducible
# ✔ NO predice | NO decide | SOLO mide riesgo real
# ✔ LEE Yahoo Finance directamente
# ✔ NO guarda nada en disco
# ✔ Si el mercado NO es evaluable → lo declara explícitamente
# =========================================================

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Literal
from datetime import datetime, timedelta

import yfinance as yf


# =========================================================
# CONFIGURACIÓN DE MERCADO (MACRO)
# =========================================================
MARKET_MAIN_SYMBOL = "SPY"

MARKET_CROSS_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "TLT",
    "GLD",
]

MARKET_LOOKBACK_DAYS = 150


# =========================================================
# CONFIGURACIÓN DE RIESGO (NEUTRAL)
# =========================================================
VOL_LOOKBACK = 20
DD_LOOKBACK = 63
TREND_LOOKBACK = 50
CORR_LOOKBACK = 30


# =========================================================
# DATACLASS SALIDA
# =========================================================
@dataclass
class QuantMarketContext:
    regime: Literal["growth", "neutral", "defensive"]
    volatility: float | float("nan")
    drawdown_rolling: float | float("nan")
    trend_strength: float | float("nan")
    cross_asset_correlation: float | float("nan")
    downside_risk: Literal["low", "medium", "high", "unknown"]
    n_observations: int

    def to_dict(self) -> Dict:
        return asdict(self)


# =========================================================
# MÉTRICAS ROBUSTAS (ESCALARES O NaN)
# =========================================================
def realized_volatility(returns: pd.Series, lookback: int) -> float:
    if len(returns) < lookback:
        return np.nan
    return float(np.sqrt(252) * returns.tail(lookback).std())


def rolling_drawdown(prices: pd.Series, lookback: int) -> float:
    if len(prices) < lookback:
        return np.nan
    window = prices.tail(lookback)
    return float(((window / window.cummax()) - 1).min())


def trend_strength(prices: pd.Series, lookback: int) -> float:
    if len(prices) < lookback:
        return np.nan
    ma = prices.rolling(lookback).mean().dropna()
    if ma.empty:
        return np.nan
    return float((prices.iloc[-1] / ma.iloc[-1]) - 1.0)


def cross_asset_corr(df_prices: pd.DataFrame, lookback: int) -> float:
    """
    Retorna:
      - float escalar si es estadísticamente definible
      - np.nan si NO es definible (pocos activos / datos)
    """
    if df_prices.shape[1] < 2:
        return np.nan

    returns = df_prices.pct_change().dropna()
    if len(returns) < lookback:
        return np.nan

    corr_matrix = returns.tail(lookback).corr()
    values = corr_matrix.values
    upper = values[np.triu_indices_from(values, k=1)]

    if upper.size == 0:
        return np.nan

    return float(np.nanmean(upper))


# =========================================================
# CLASIFICADORES (CON NAN AWARE)
# =========================================================
def classify_downside(dd: float) -> Literal["low", "medium", "high", "unknown"]:
    if np.isnan(dd):
        return "unknown"
    if dd <= -0.20:
        return "high"
    if dd <= -0.10:
        return "medium"
    return "low"


def classify_regime(
    vol: float,
    dd: float,
    corr: float,
    trend: float
) -> Literal["growth", "neutral", "defensive"]:

    # ⚠️ Mercado NO evaluable → neutral forzado
    if any(np.isnan(x) for x in [vol, dd, corr, trend]):
        return "neutral"

    if dd <= -0.25 or vol >= 0.45 or corr >= 0.90:
        return "defensive"

    if trend > 0.05 and vol <= 0.10 and corr < 0.60:
        return "growth"

    return "neutral"


# =========================================================
# LOADER AUTÓNOMO DE MERCADO (YAHOO FINANCE)
# =========================================================
def load_market_prices() -> tuple[pd.Series, pd.DataFrame]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=MARKET_LOOKBACK_DAYS)

    df_main = yf.download(
        MARKET_MAIN_SYMBOL,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if df_main is None or df_main.empty:
        raise RuntimeError("Datos MAIN no disponibles")

    prices_main = df_main["Close"].dropna()

    df_cross = yf.download(
        MARKET_CROSS_SYMBOLS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if df_cross is None or df_cross.empty:
        raise RuntimeError("Datos CROSS no disponibles")

    prices_cross = (
        df_cross["Close"]
        if isinstance(df_cross.columns, pd.MultiIndex)
        else df_cross.filter(like="Close")
    ).dropna()

    return prices_main, prices_cross


# =========================================================
# CORE EVALUATOR
# =========================================================
def evaluate_quant_market(
    prices_main: pd.Series,
    prices_cross: pd.DataFrame
) -> QuantMarketContext:

    returns = prices_main.pct_change().dropna()

    vol = realized_volatility(returns, VOL_LOOKBACK)
    dd = rolling_drawdown(prices_main, DD_LOOKBACK)
    trend = trend_strength(prices_main, TREND_LOOKBACK)
    corr = cross_asset_corr(prices_cross, CORR_LOOKBACK)

    downside = classify_downside(dd)
    regime = classify_regime(vol, dd, corr, trend)

    return QuantMarketContext(
        regime=regime,
        volatility=vol,
        drawdown_rolling=dd,
        trend_strength=trend,
        cross_asset_correlation=corr,
        downside_risk=downside,
        n_observations=len(prices_main),
    )


# =========================================================
# ENTRYPOINT PARA PIPELINE
# =========================================================
def run_market_state() -> QuantMarketContext:
    prices_main, prices_cross = load_market_prices()
    return evaluate_quant_market(prices_main, prices_cross)


if __name__ == "__main__":
    print(run_market_state().to_dict())

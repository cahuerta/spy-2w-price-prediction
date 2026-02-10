# =========================================================
# evaluador_cuantitativo_mercado.py — V3.0 RIESGO REAL (PRODUCCIÓN)
#
# MOTOR CUANTITATIVO DE ENTORNO DE MERCADO (AUTÓNOMO)
# ---------------------------------------------------------
# ✔ Determinista, auditable, reproducible
# ✔ NO predice | NO decide | SOLO mide riesgo real
# ✔ LEE Yahoo Finance directamente (estable)
# ✔ NO guarda nada en disco
# ✔ Entry-point único para el pipeline
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

VOL_HIGH = 0.35
VOL_LOW = 0.10
DD_HIGH = -0.20
DD_MED = -0.10


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
    downside_risk: Literal["low", "medium", "high"]
    n_observations: int

    def to_dict(self) -> Dict:
        return asdict(self)


# =========================================================
# MÉTRICAS ROBUSTAS
# =========================================================
def realized_volatility(returns: pd.Series, lookback: int) -> float:
    if len(returns) < lookback:
        return 0.0
    return float(np.sqrt(252) * returns.tail(lookback).std())


def rolling_drawdown(prices: pd.Series, lookback: int) -> float:
    if len(prices) < lookback:
        return 0.0
    window = prices.tail(lookback)
    return float(((window / window.cummax()) - 1).min())


def trend_strength(prices: pd.Series, lookback: int) -> float:
    if len(prices) < lookback:
        return 0.0
    ma = prices.rolling(lookback).mean().dropna()
    if ma.empty:
        return 0.0
    return float((prices.iloc[-1] / ma.iloc[-1]) - 1.0)


def cross_asset_corr(df_prices: pd.DataFrame, lookback: int) -> float:
    if df_prices.shape[1] < 2:
        return 0.0
    returns = df_prices.pct_change().dropna()
    if len(returns) < lookback:
        return 0.0
    recent = returns.tail(lookback)
    corr = recent.corr().values
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.nanmean(upper))


# =========================================================
# CLASIFICADORES
# =========================================================
def classify_downside(dd: float) -> Literal["low", "medium", "high"]:
    if dd <= DD_HIGH:
        return "high"
    if dd <= DD_MED:
        return "medium"
    return "low"


def classify_regime(
    vol: float,
    dd: float,
    corr: float,
    trend: float
) -> Literal["growth", "neutral", "defensive"]:

    if dd <= -0.25 or vol >= 0.45 or corr >= 0.90:
        return "defensive"

    if trend > 0.05 and vol <= VOL_LOW and corr < 0.60:
        return "growth"

    return "neutral"


# =========================================================
# LOADER AUTÓNOMO DE MERCADO (YAHOO FINANCE)
# =========================================================
def load_market_prices() -> tuple[pd.Series, pd.DataFrame]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=MARKET_LOOKBACK_DAYS)

    # --------- MAIN (SPY) ----------
    df_main = yf.download(
        MARKET_MAIN_SYMBOL,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if df_main is None or df_main.empty:
        raise RuntimeError("No se pudieron cargar datos MAIN desde Yahoo")

    prices_main = df_main["Close"].dropna()

    # --------- CROSS ASSETS ----------
    df_cross = yf.download(
        MARKET_CROSS_SYMBOLS,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if df_cross is None or df_cross.empty:
        raise RuntimeError("No se pudieron cargar datos CROSS desde Yahoo")

    # yfinance puede devolver columnas MultiIndex o simples
    if isinstance(df_cross.columns, pd.MultiIndex):
        prices_cross = df_cross["Close"].dropna()
    else:
        prices_cross = df_cross.filter(like="Close").dropna()

    # 🔒 FIX CRÍTICO: garantizar DataFrame SIEMPRE
    if isinstance(prices_cross, pd.Series):
        prices_cross = prices_cross.to_frame()

    return prices_main, prices_cross


# =========================================================
# CORE EVALUATOR (PURO)
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
        volatility=round(vol, 4),
        drawdown_rolling=round(dd, 4),
        trend_strength=round(trend, 4),
        cross_asset_correlation=round(corr, 4),
        downside_risk=downside,
        n_observations=len(prices_main),
    )


# =========================================================
# ENTRYPOINT PARA PIPELINE (MISMO CONTRATO)
# =========================================================
def run_market_state() -> QuantMarketContext:
    prices_main, prices_cross = load_market_prices()
    return evaluate_quant_market(prices_main, prices_cross)


# =========================================================
# CLI (DEBUG / MANUAL)
# =========================================================
if __name__ == "__main__":
    ctx = run_market_state()
    print(ctx.to_dict())

=========================================================
evaluador_cuantitativo_mercado.py — V2.0 RIESGO REAL

MOTOR CUANTITATIVO DE ENTORNO DE MERCADO (SIN DECISIONES)
---------------------------------------------------------
✔ Determinista, auditable, reproducible
✔ NO predice | NO decide | SOLO MIDE RIESGO REAL
✔ Thresholds neutrales → TÚ decides estrategia
✔ Edge cases robustos + logging mínimo
=========================================================

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Literal, Optional

=========================================================
CONFIGURACIÓN NEUTRAL (RIESGO REAL)
=========================================================
VOL_LOOKBACK = 20
DD_LOOKBACK = 63
TREND_LOOKBACK = 50
CORR_LOOKBACK = 30

# THRESHOLDS REALISTAS (sin bias)
VOL_HIGH = 0.35
VOL_LOW = 0.10
DD_HIGH = -0.20
DD_MED = -0.10
CORR_HIGH = 0.85

=========================================================
DATACLASS SALIDA
=========================================================
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

=========================================================
MÉTRICAS (ROBUSTAS)
=========================================================
def realized_volatility(returns: pd.Series, lookback: int) -> float:
    if len(returns) < lookback: return 0.0
    return np.sqrt(252) * returns.tail(lookback).std()

def rolling_drawdown(prices: pd.Series, lookback: int) -> float:
    if len(prices) < lookback: return 0.0
    window = prices.tail(lookback)
    return float(((window / window.cummax()) - 1).min())

def trend_strength(prices: pd.Series, lookback: int) -> float:
    if len(prices) < lookback: return 0.0
    ma = prices.rolling(lookback).mean().dropna()
    if len(ma) == 0: return 0.0
    return float((prices.iloc[-1] / ma.iloc[-1]) - 1.0)

def cross_asset_corr(df_prices: pd.DataFrame, lookback: int) -> float:
    if df_prices.shape[1] < 2: return 0.0
    returns = df_prices.pct_change().dropna()
    if len(returns) < lookback: return 0.0
    recent = returns.tail(lookback)
    corr = recent.corr().values
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.nanmean(upper))

=========================================================
CLASIFICADORES NEUTRALES
=========================================================
def classify_downside(dd: float) -> Literal["low", "medium", "high"]:
    if dd <= DD_HIGH: return "high"
    if dd <= DD_MED: return "medium"
    return "low"

def classify_regime(vol: float, dd: float, corr: float, trend: float) -> Literal["growth", "neutral", "defensive"]:
    # Solo extremos → defensive (tú decides acción)
    if dd <= -0.25 or vol >= 0.45 or corr >= 0.90: return "defensive"
    if trend > 0.05 and vol <= VOL_LOW and corr < 0.60: return "growth"
    return "neutral"

=========================================================
FUNCIÓN PRINCIPAL (TÚ LA LLAMAS)
=========================================================
def evaluate_quant_market(prices_main: pd.Series, prices_cross: pd.DataFrame) -> QuantMarketContext:
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

# USO: tú cargas datos y llamas
# context = evaluate_quant_market(spy_prices, cross_prices)

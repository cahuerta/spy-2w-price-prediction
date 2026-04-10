# =========================================================
# market_quant_context.py — V3.3 ADAPTIVE REGIME
# =========================================================
# V3.2 → V3.3:
#   [A1] classify_regime lee umbrales desde regime_thresholds.json
#        en vez de valores hardcodeados imposibles.
#        Si el archivo no existe, parte con valores realistas sensatos.
#   [A2] Umbrales iniciales realistas:
#        DEFENSIVE: dd <= -0.08 | vol >= 0.25 | corr >= 0.75
#        GROWTH:    trend > 0.05 | vol <= 0.15 | corr < 0.55
#        Antes: dd <= -0.25 → nunca se activaba DEFENSIVE
#   [A3] Guarda snapshot diario en market_history/ para que
#        regime_threshold_learner.py pueda evaluar aciertos
# =========================================================

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Literal
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

# =========================================================
# CONFIGURACIÓN
# =========================================================
MARKET_MAIN_SYMBOL   = "SPY"
MARKET_CROSS_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
MARKET_LOOKBACK_DAYS = 150

VOL_LOOKBACK   = 20
DD_LOOKBACK    = 63
TREND_LOOKBACK = 50
CORR_LOOKBACK  = 30

DATA_PATH            = Path(os.getenv("DATA_PATH", "/data"))
MARKET_HISTORY_DIR   = DATA_PATH / "market_history"
THRESHOLDS_FILE      = DATA_PATH / "regime_thresholds.json"

# =========================================================
# [A2] UMBRALES POR DEFECTO — REALISTAS
# =========================================================
DEFAULT_THRESHOLDS = {
    "defensive": {
        "dd_max":   -0.08,   # drawdown ≤ -8% → defensive
        "vol_min":   0.25,   # volatilidad ≥ 25% → defensive
        "corr_min":  0.75,   # correlación ≥ 75% → defensive
    },
    "growth": {
        "trend_min": 0.05,   # tendencia > 5% → candidato growth
        "vol_max":   0.15,   # volatilidad ≤ 15% → growth posible
        "corr_max":  0.55,   # correlación < 55% → growth posible
    },
    "meta": {
        "version":        1,
        "created_at":     datetime.utcnow().isoformat(),
        "last_updated":   datetime.utcnow().isoformat(),
        "total_evals":    0,
        "defensive_hits": 0,
        "growth_hits":    0,
        "neutral_misses": 0,
    }
}

# Límites para evitar umbrales absurdos
THRESHOLD_LIMITS = {
    "defensive": {
        "dd_max":   (-0.03, -0.20),   # entre -3% y -20%
        "vol_min":  (0.15,   0.50),   # entre 15% y 50%
        "corr_min": (0.55,   0.95),   # entre 55% y 95%
    },
    "growth": {
        "trend_min": (0.01,  0.15),   # entre 1% y 15%
        "vol_max":   (0.08,  0.25),   # entre 8% y 25%
        "corr_max":  (0.35,  0.70),   # entre 35% y 70%
    },
}


# =========================================================
# CARGA / GUARDA UMBRALES
# =========================================================

def load_thresholds() -> Dict:
    """Carga umbrales desde disco. Si no existe, crea con defaults."""
    if THRESHOLDS_FILE.exists():
        try:
            data = json.loads(THRESHOLDS_FILE.read_text())
            # Validar que tenga todas las claves necesarias
            if "defensive" in data and "growth" in data:
                return data
        except Exception:
            pass

    # Crear archivo con defaults
    save_thresholds(DEFAULT_THRESHOLDS)
    return DEFAULT_THRESHOLDS.copy()


def save_thresholds(thresholds: Dict) -> None:
    THRESHOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = THRESHOLDS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(thresholds, indent=2))
    tmp.replace(THRESHOLDS_FILE)


# =========================================================
# UTILIDAD CRÍTICA
# =========================================================

def _scalar(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            return float(np.nanmean(arr)) if arr.size else default
        if hasattr(x, "mean"):
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
    thresholds_used: Dict   # [A1] para trazabilidad

    def to_dict(self) -> Dict:
        return asdict(self)


# =========================================================
# MÉTRICAS
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
# HISTÓRICO
# =========================================================

def load_market_history() -> pd.DataFrame:
    if not MARKET_HISTORY_DIR.exists():
        return pd.DataFrame()
    rows = []
    for f in MARKET_HISTORY_DIR.iterdir():
        if not f.suffix == ".json":
            continue
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:
            continue
    return pd.DataFrame(rows)


def save_market_snapshot(ctx: QuantMarketContext, spy_price: float) -> None:
    """
    [A3] Guarda snapshot diario para que el learner evalúe aciertos.
    Incluye precio SPY para calcular retorno futuro.
    """
    MARKET_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().date().isoformat()
    path  = MARKET_HISTORY_DIR / f"{today}.json"

    if path.exists():
        return  # ya guardado hoy

    snapshot = {
        "date":                    today,
        "regime":                  ctx.regime,
        "volatility":              ctx.volatility,
        "drawdown_rolling":        ctx.drawdown_rolling,
        "trend_strength":          ctx.trend_strength,
        "cross_asset_correlation": ctx.cross_asset_correlation,
        "spy_price":               round(float(spy_price), 4),
        "thresholds_used":         ctx.thresholds_used,
    }
    path.write_text(json.dumps(snapshot, indent=2))


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
    X  = hist[required].values
    y  = hist["cross_asset_correlation"].values
    x0 = np.array([features[r] for r in required])
    dists = np.linalg.norm(X - x0, axis=1)
    idx   = np.argsort(dists)[:k]
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


def classify_regime(
    vol: float,
    dd: float,
    corr: float,
    trend: float,
    thresholds: Dict,
) -> Literal["growth", "neutral", "defensive"]:
    """
    [A1] Usa umbrales adaptativos desde thresholds.json
    En vez de valores hardcodeados imposibles.
    """
    if any(np.isnan(x) for x in [vol, dd, corr, trend]):
        return "neutral"

    td = thresholds["defensive"]
    tg = thresholds["growth"]

    # DEFENSIVE — cualquier condición suficiente
    if dd <= td["dd_max"] or vol >= td["vol_min"] or corr >= td["corr_min"]:
        return "defensive"

    # GROWTH — todas las condiciones deben cumplirse
    if trend > tg["trend_min"] and vol <= tg["vol_max"] and corr < tg["corr_max"]:
        return "growth"

    return "neutral"


# =========================================================
# LOADER YAHOO FINANCE
# =========================================================

def load_market_prices() -> tuple:
    end   = datetime.utcnow().date()
    start = end - timedelta(days=MARKET_LOOKBACK_DAYS)

    main  = yf.download(MARKET_MAIN_SYMBOL,   start=start, end=end, auto_adjust=True, progress=False)
    cross = yf.download(MARKET_CROSS_SYMBOLS, start=start, end=end, auto_adjust=True, progress=False)

    if main.empty or cross.empty:
        raise RuntimeError("Datos de mercado no disponibles")

    prices_main  = main["Close"].dropna()
    prices_cross = (
        cross["Close"] if isinstance(cross.columns, pd.MultiIndex)
        else cross.filter(like="Close")
    ).dropna()

    return prices_main, prices_cross


# =========================================================
# CORE
# =========================================================

def run_market_state() -> QuantMarketContext:
    prices_main, prices_cross = load_market_prices()
    history    = load_market_history()
    thresholds = load_thresholds()  # [A1]

    returns  = prices_main.pct_change().dropna()
    spy_price = float(prices_main.iloc[-1])

    vol_raw   = realized_volatility(returns)
    dd_raw    = rolling_drawdown(prices_main)
    trend_raw = trend_strength(prices_main)
    corr_raw  = cross_asset_corr(prices_cross)

    vol   = _scalar(vol_raw)
    dd    = _scalar(dd_raw)
    trend = _scalar(trend_raw)
    corr  = _scalar(corr_raw)

    source = "measured"
    if corr == 0.0:
        corr   = _scalar(historical_corr(history))
        source = "historical"
    if corr == 0.0:
        corr   = _scalar(knn_corr(history, {"volatility": vol, "drawdown_rolling": dd, "trend_strength": trend}))
        source = "knn"
    if corr == 0.0:
        source = "unavailable"

    regime = classify_regime(vol, dd, corr, trend, thresholds)

    ctx = QuantMarketContext(
        regime=regime,
        volatility=vol,
        drawdown_rolling=dd,
        trend_strength=trend,
        cross_asset_correlation=corr,
        downside_risk=classify_downside(dd),
        n_observations=len(prices_main),
        corr_source=source,
        thresholds_used={
            "defensive": thresholds["defensive"],
            "growth":    thresholds["growth"],
        },
    )

    # [A3] Guardar snapshot para aprendizaje posterior
    save_market_snapshot(ctx, spy_price)

    return ctx


if __name__ == "__main__":
    ctx = run_market_state()
    print(json.dumps(ctx.to_dict(), indent=2))
        

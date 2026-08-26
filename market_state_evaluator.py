# =========================================================
# market_quant_context.py — V3.4 ADAPTIVE REGIME + MACRO
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
#
# FIX v3.3.1:
#   [A4] spy_price: yfinance puede retornar DataFrame en vez de Series
#        para un solo ticker. Usar squeeze() + scalar() para forzar float.
#
# FIX v3.3.2:
#   [A5] THRESHOLD_LIMITS.defensive.dd_max tenía el tuple invertido:
#        (-0.03, -0.20) en vez de (-0.20, -0.03). Con _clamp(lo, hi)
#        usando max(lo, min(hi, value)), ese orden forzaba SIEMPRE
#        el resultado a -0.03 sin importar el ajuste del learner —
#        por eso "defensive" se disparaba con cualquier caída de 3%
#        y "growth" nunca llegaba a evaluarse (29/30 evals = defensive).
#        Este diccionario no se usa en este archivo (solo en
#        regime_threshold_learner.py), pero se corrige aquí también
#        para que ambas copias queden consistentes hasta centralizarlo.
#
# V3.4 (2026-08-26): integración de factores macro (macro_factors.py)
#   [M1] classify_regime() recibe dos parámetros OPCIONALES nuevos
#        (macro_stress, macro_risk_off), default None — si no se
#        pasan, o si son None, el comportamiento es IDÉNTICO a V3.3.
#        Ningún llamador existente que no los use se rompe.
#   [M2] DEFAULT_THRESHOLDS gana una sección "macro" nueva, sin tocar
#        "defensive"/"growth". load_thresholds() usa .get("macro", ...)
#        con fallback — un regime_thresholds.json real ya persistido
#        en producción (sin la clave "macro" todavía) sigue
#        cargándose sin error; la sección macro se completa con
#        defaults la primera vez, sin pisar los umbrales de
#        defensive/growth ya aprendidos.
#   [M3] _load_macro_scores() lee /data/macro_context.json (generado
#        por macro_factors.py, proceso separado) — este archivo NO
#        hace llamadas nuevas a Yahoo Finance, solo lee lo que
#        macro_factors.py ya calculó y guardó.
#   [M4] QuantMarketContext gana 3 campos nuevos con default (no
#        rompe construcciones existentes): macro_stress_magnitude,
#        macro_risk_off_score, macro_source.
#   [M5] save_market_snapshot() agrega los dos scores al snapshot
#        diario — regime_threshold_learner.py los necesita para poder
#        ajustar los umbrales macro.
# =========================================================

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Literal, Optional
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

DATA_PATH          = Path(os.getenv("DATA_PATH", "/data"))
MARKET_HISTORY_DIR = DATA_PATH / "market_history"
THRESHOLDS_FILE    = DATA_PATH / "regime_thresholds.json"
MACRO_CONTEXT_FILE = DATA_PATH / "macro_context.json"  # [M3]

# =========================================================
# [A2][M2] UMBRALES POR DEFECTO — REALISTAS
# =========================================================
DEFAULT_THRESHOLDS = {
    "defensive": {
        "dd_max":  -0.08,
        "vol_min":  0.25,
        "corr_min": 0.75,
    },
    "growth": {
        "trend_min": 0.05,
        "vol_max":   0.15,
        "corr_max":  0.55,
    },
    # [M2] Sección nueva — no reemplaza nada de defensive/growth.
    "macro": {
        "stress_magnitude_min":       2.0,   # |zscore| promedio >= esto → contribuye a defensive
        "risk_off_score_min":         1.5,   # risk_off_score >= esto → contribuye a defensive
        "risk_off_score_max_growth": -0.3,   # risk_off_score >= esto → bloquea growth
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

# [A5] Fix: dd_max debe ir (lo, hi) con lo < hi, igual que los demás
THRESHOLD_LIMITS = {
    "defensive": {
        "dd_max":   (-0.20, -0.03),
        "vol_min":  (0.15,   0.50),
        "corr_min": (0.55,   0.95),
    },
    "growth": {
        "trend_min": (0.01,  0.15),
        "vol_max":   (0.08,  0.25),
        "corr_max":  (0.35,  0.70),
    },
    # [M2] mismos límites que regime_threshold_learner.py — no se usa
    # en este archivo (solo en el learner), se mantiene por
    # consistencia hasta centralizarlo, igual que ya se hacía con
    # defensive/growth (ver nota [A5] arriba).
    "macro": {
        "stress_magnitude_min":      (1.0,  3.0),
        "risk_off_score_min":        (0.5,  2.5),
        "risk_off_score_max_growth": (-1.0, 0.0),
    },
}


# =========================================================
# CARGA / GUARDA UMBRALES
# =========================================================

def load_thresholds() -> Dict:
    if THRESHOLDS_FILE.exists():
        try:
            data = json.loads(THRESHOLDS_FILE.read_text())
            if "defensive" in data and "growth" in data:
                # [M2] Compatibilidad: un archivo ya persistido sin la
                # clave "macro" se completa con el default, sin tocar
                # defensive/growth ya aprendidos.
                if "macro" not in data:
                    data["macro"] = DEFAULT_THRESHOLDS["macro"].copy()
                return data
        except Exception:
            pass
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
# [M3] CARGA DE FACTORES MACRO
# =========================================================

def _load_macro_scores() -> tuple:
    """
    [M3] Lee /data/macro_context.json (generado por macro_factors.py,
    proceso separado — este archivo no llama a Yahoo Finance).
    Retorna (macro_stress_magnitude, macro_risk_off_score, source).
    Si el archivo no existe o falta algún campo, retorna (None, None,
    "unavailable") — classify_regime() ya maneja None sin romper
    nada (ver [M1]).
    """
    if not MACRO_CONTEXT_FILE.exists():
        return None, None, "unavailable"
    try:
        data = json.loads(MACRO_CONTEXT_FILE.read_text())
        stress   = data.get("macro_stress_magnitude")
        risk_off = data.get("macro_risk_off_score")
        stress   = float(stress)   if stress   is not None else None
        risk_off = float(risk_off) if risk_off is not None else None
        source   = "measured" if (stress is not None or risk_off is not None) else "unavailable"
        return stress, risk_off, source
    except Exception:
        return None, None, "unavailable"


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
    thresholds_used: Dict
    # [M4] Campos nuevos con default — no rompe construcciones existentes.
    macro_stress_magnitude: Optional[float] = None
    macro_risk_off_score:   Optional[float] = None
    macro_source: Literal["measured", "unavailable"] = "unavailable"

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
        if f.suffix != ".json":
            continue
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:
            continue
    return pd.DataFrame(rows)


def save_market_snapshot(ctx: QuantMarketContext, spy_price: float) -> None:
    """
    [A3] Snapshot diario para aprendizaje del learner.
    [M5] Se agregan macro_stress_magnitude y macro_risk_off_score al
    snapshot — regime_threshold_learner.py los necesita para poder
    ajustar los umbrales macro. Snapshots viejos sin estas claves
    siguen siendo válidos (el learner usa .get() con default None).
    """
    MARKET_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().date().isoformat()
    path  = MARKET_HISTORY_DIR / f"{today}.json"
    if path.exists():
        return
    snapshot = {
        "date":                    today,
        "regime":                  ctx.regime,
        "volatility":              ctx.volatility,
        "drawdown_rolling":        ctx.drawdown_rolling,
        "trend_strength":          ctx.trend_strength,
        "cross_asset_correlation": ctx.cross_asset_correlation,
        "spy_price":               round(float(spy_price), 4),
        "thresholds_used":         ctx.thresholds_used,
        "macro_stress_magnitude":  ctx.macro_stress_magnitude,  # [M5]
        "macro_risk_off_score":    ctx.macro_risk_off_score,    # [M5]
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
    macro_stress: Optional[float] = None,     # [M1]
    macro_risk_off: Optional[float] = None,   # [M1]
) -> Literal["growth", "neutral", "defensive"]:
    """
    [M1] macro_stress y macro_risk_off son OPCIONALES (default None).
    Si no se pasan, o si macro_context.json no tiene datos ese día,
    el comportamiento es IDÉNTICO a V3.3 — ningún llamador existente
    que no los use se rompe.
    """
    if any(np.isnan(x) for x in [vol, dd, corr, trend]):
        return "neutral"

    td = thresholds["defensive"]
    tg = thresholds["growth"]
    tm = thresholds.get("macro", DEFAULT_THRESHOLDS["macro"])  # [M2] fallback seguro

    # [M1] El estrés macro solo puede EMPUJAR hacia defensive, nunca
    # bloquear defensive ni forzar growth por sí solo.
    macro_triggers_defensive = (
        (macro_stress   is not None and macro_stress   >= tm.get("stress_magnitude_min", 2.0))
        or (macro_risk_off is not None and macro_risk_off >= tm.get("risk_off_score_min", 1.5))
    )

    if dd <= td["dd_max"] or vol >= td["vol_min"] or corr >= td["corr_min"] or macro_triggers_defensive:
        return "defensive"

    # [M1] El estrés macro solo puede BLOQUEAR growth, nunca forzarlo.
    macro_blocks_growth = (
        macro_risk_off is not None
        and macro_risk_off >= tm.get("risk_off_score_max_growth", -0.3)
    )

    if trend > tg["trend_min"] and vol <= tg["vol_max"] and corr < tg["corr_max"] and not macro_blocks_growth:
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

    # [A4] squeeze() fuerza Series aunque yfinance retorne DataFrame
    prices_main = main["Close"].squeeze().dropna()

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
    thresholds = load_thresholds()

    returns = prices_main.pct_change().dropna()

    # [A4] _scalar() garantiza float aunque iloc[-1] sea Serie o array
    spy_price = _scalar(prices_main.iloc[-1])

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

    # [M3] Cargar factores macro — no hace llamadas nuevas a Yahoo.
    macro_stress, macro_risk_off, macro_source = _load_macro_scores()

    regime = classify_regime(vol, dd, corr, trend, thresholds, macro_stress, macro_risk_off)

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
            "macro":     thresholds.get("macro", DEFAULT_THRESHOLDS["macro"]),  # [M2]
        },
        macro_stress_magnitude=macro_stress,   # [M4]
        macro_risk_off_score=macro_risk_off,   # [M4]
        macro_source=macro_source,             # [M4]
    )

    save_market_snapshot(ctx, spy_price)
    return ctx


if __name__ == "__main__":
    ctx = run_market_state()
    print(json.dumps(ctx.to_dict(), indent=2))

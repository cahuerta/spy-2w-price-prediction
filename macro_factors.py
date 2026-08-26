# =========================================================
# macro_factors.py — FACTORES MACRO COMPARTIDOS (V2)
# =========================================================
# Módulo de SOLO DATOS — trae y calcula factores macro, no decide
# nada (ni régimen, ni features de predictor). Consumidores:
#   - market_quant_context.py → régimen más rico
#   - predictor_h1.py...h10.py → features macro nuevas por ticker
#     (pendiente de conectar)
#
# V2 (2026-08-26): se agregan dos scores agregados a nivel de todo
# el batch, calculados sobre los mismos 10 factores de V1, sin tocar
# la estructura "factors" ya existente (compatible con V1):
#
#   macro_stress_magnitude: promedio de |zscore_20d| de TODOS los
#     factores disponibles — "qué tan inusual está el mercado hoy en
#     general", sin importar dirección. None si hay menos de 3
#     factores disponibles (muestra insuficiente para promediar).
#
#   macro_risk_off_score: promedio de zscore_20d con signo, SOLO de
#     los 5 factores con dirección de "huida al refugio" clara
#     (VIX, GOLD, TLT suben en risk-off; TNX baja en risk-off, DXY
#     sube en risk-off — por eso TNX entra con peso -1). Se deja
#     fuera del compuesto direccional a OIL/NATGAS/COPPER/SILVER
#     porque su dirección ante estrés es ambigua (pueden subir tanto
#     por miedo/inflación como por crecimiento). None si hay menos de
#     2 de los 5 disponibles.
#
# Instrumentos (todos vía Yahoo Finance, mismo proveedor que ya usa
# market_quant_context.py — sin API nueva, sin costo adicional):
#   GOLD    GC=F     Oro (futuros)
#   OIL     CL=F     Petróleo WTI (futuros)
#   NATGAS  NG=F     Gas natural (futuros)
#   COPPER  HG=F     Cobre (futuros)
#   SILVER  SI=F     Plata (futuros)
#   DXY     DX-Y.NYB Índice dólar
#   TNX     ^TNX     Rendimiento bono 10 años
#   VIX     ^VIX     Volatilidad implícita
#   TLT     TLT      Bonos largos (ETF)
#   SPY     SPY      S&P 500 (referencia)
#
# Para cada instrumento se calcula:
#   - level: último valor (precio o yield, según el instrumento)
#   - ret_1d / ret_5d / ret_20d: retorno porcentual a esos horizontes
#   - zscore_20d: qué tan inusual es el retorno de 1 día de HOY
#     respecto a la distribución de retornos diarios de los últimos
#     20 días hábiles (mismo criterio de "drift local" que ya usan
#     los predictores H1-H10)
# =========================================================

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MACRO] %(message)s")
logger = logging.getLogger("macro_factors")

DATA_PATH   = Path(os.getenv("DATA_PATH", "/data"))
OUTPUT_FILE = DATA_PATH / "macro_context.json"

LOOKBACK_DAYS = int(os.getenv("MACRO_LOOKBACK_DAYS", "90"))
ZSCORE_WINDOW = int(os.getenv("MACRO_ZSCORE_WINDOW", "20"))

# Nombre lógico → símbolo Yahoo
MACRO_INSTRUMENTS: Dict[str, str] = {
    "GOLD":   "GC=F",
    "OIL":    "CL=F",
    "NATGAS": "NG=F",
    "COPPER": "HG=F",
    "SILVER": "SI=F",
    "DXY":    "DX-Y.NYB",
    "TNX":    "^TNX",
    "VIX":    "^VIX",
    "TLT":    "TLT",
    "SPY":    "SPY",
}

# [V2] Factores con dirección de "risk-off" clara y su peso de signo:
# +1 → sube en risk-off | -1 → baja en risk-off (se invierte el signo
# de su zscore antes de promediar, para que todos "sumen" en la
# misma dirección de riesgo).
RISK_OFF_WEIGHTS: Dict[str, int] = {
    "VIX":  +1,
    "GOLD": +1,
    "TLT":  +1,
    "TNX":  -1,
    "DXY":  +1,
}
MIN_STRESS_FACTORS   = 3   # mínimo de factores disponibles para macro_stress_magnitude
MIN_RISK_OFF_FACTORS = 2   # mínimo de factores disponibles para macro_risk_off_score


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _compute_factor(name: str, symbol: str) -> Optional[Dict[str, Any]]:
    """
    Trae el historial de un instrumento y calcula level/retornos/zscore.
    Retorna None si no hay datos suficientes — no rompe el batch completo.
    """
    try:
        end   = datetime.utcnow().date()
        start = end - timedelta(days=LOOKBACK_DAYS)

        data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if data is None or data.empty:
            logger.warning(f"⚠️ {name} ({symbol}): sin datos")
            return None

        closes = data["Close"].squeeze().dropna()
        if len(closes) < ZSCORE_WINDOW + 5:
            logger.warning(f"⚠️ {name} ({symbol}): historial insuficiente ({len(closes)} días)")
            return None

        returns = closes.pct_change().dropna()

        level = float(closes.iloc[-1])

        ret_1d  = float(returns.iloc[-1])                                    if len(returns) >= 1  else None
        ret_5d  = float((closes.iloc[-1] / closes.iloc[-6]  - 1))            if len(closes) >= 6   else None
        ret_20d = float((closes.iloc[-1] / closes.iloc[-21] - 1))            if len(closes) >= 21  else None

        recent_returns = returns.tail(ZSCORE_WINDOW)
        mean_ret = float(recent_returns.mean())
        std_ret  = float(recent_returns.std())
        zscore_20d = float((ret_1d - mean_ret) / std_ret) if (ret_1d is not None and std_ret > 1e-9) else 0.0

        return {
            "symbol":     symbol,
            "level":      round(level, 4),
            "ret_1d_pct":  round(ret_1d  * 100, 4) if ret_1d  is not None else None,
            "ret_5d_pct":  round(ret_5d  * 100, 4) if ret_5d  is not None else None,
            "ret_20d_pct": round(ret_20d * 100, 4) if ret_20d is not None else None,
            "zscore_20d": round(zscore_20d, 4),
            "n_observations": len(closes),
        }
    except Exception as e:
        logger.error(f"❌ {name} ({symbol}) error: {e}")
        return None


def _compute_aggregate_scores(factors: Dict[str, Dict]) -> Dict[str, Any]:
    """
    [V2] Calcula macro_stress_magnitude y macro_risk_off_score a
    partir de los factores ya calculados. No vuelve a llamar a Yahoo.
    """
    # --- macro_stress_magnitude: |zscore| promedio de TODOS ---
    all_zscores = [
        abs(f["zscore_20d"]) for f in factors.values()
        if f.get("zscore_20d") is not None
    ]
    stress_magnitude = (
        round(float(np.mean(all_zscores)), 4)
        if len(all_zscores) >= MIN_STRESS_FACTORS else None
    )

    # --- macro_risk_off_score: promedio con signo, solo direccionales ---
    risk_off_components: Dict[str, float] = {}
    weighted_values = []
    for name, weight in RISK_OFF_WEIGHTS.items():
        f = factors.get(name)
        if not f or f.get("zscore_20d") is None:
            continue
        signed = weight * f["zscore_20d"]
        risk_off_components[name] = round(signed, 4)
        weighted_values.append(signed)

    risk_off_score = (
        round(float(np.mean(weighted_values)), 4)
        if len(weighted_values) >= MIN_RISK_OFF_FACTORS else None
    )

    return {
        "macro_stress_magnitude": stress_magnitude,
        "macro_risk_off_score":   risk_off_score,
        "risk_off_components":    risk_off_components,  # [V2] trazabilidad
        "n_stress_factors":       len(all_zscores),
        "n_risk_off_factors":     len(weighted_values),
    }


def run_macro_factors() -> Dict[str, Any]:
    """
    Trae y calcula todos los factores macro configurados, más los
    dos scores agregados [V2]. Guarda en /data/macro_context.json.
    Instrumentos que fallen quedan ausentes del resultado (no
    bloquean a los demás ni a los scores agregados, que se calculan
    solo con los factores disponibles).
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    factors: Dict[str, Any] = {}

    for name, symbol in MACRO_INSTRUMENTS.items():
        result = _compute_factor(name, symbol)
        if result:
            factors[name] = result

    aggregates = _compute_aggregate_scores(factors)

    payload = {
        "generated_at": timestamp,
        "n_configured": len(MACRO_INSTRUMENTS),
        "n_available":  len(factors),
        "factors":      factors,
        **aggregates,  # [V2] macro_stress_magnitude, macro_risk_off_score, etc.
    }

    _save_json(OUTPUT_FILE, payload)
    logger.info(
        f"✅ macro_context.json guardado | {len(factors)}/{len(MACRO_INSTRUMENTS)} factores | "
        f"stress={aggregates['macro_stress_magnitude']} | "
        f"risk_off={aggregates['macro_risk_off_score']}"
    )

    return payload


if __name__ == "__main__":
    result = run_macro_factors()
    print(json.dumps(result, indent=2, ensure_ascii=False))

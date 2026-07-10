# =========================================================
# regime_threshold_learner.py — ADAPTIVE REGIME LEARNER V1.0
# =========================================================
# Evalúa si los regímenes declarados fueron correctos
# comparando con el retorno real de SPY N días después.
# Ajusta umbrales de classify_regime gradualmente.
#
# Se ejecuta como parte del pipeline diario, ANTES de
# run_market_state(), para que los umbrales actualizados
# se usen en el ciclo actual.
#
# LÓGICA:
#   DEFENSIVE declarado + SPY cayó   → acierto → umbral más sensible
#   DEFENSIVE declarado + SPY subió  → falsa alarma → umbral menos sensible
#   NEUTRAL declarado + SPY cayó >5% → perdió caída → umbral más sensible
#   GROWTH declarado + SPY subió     → acierto → umbral más sensible
#   GROWTH declarado + SPY cayó      → falsa alarma → umbral menos sensible
#
# FIX V1.1:
#   [A1] THRESHOLD_LIMITS.defensive.dd_max tenía el tuple invertido:
#        (-0.03, -0.20) en vez de (-0.20, -0.03). _clamp(value, lo, hi)
#        hace max(lo, min(hi, value)) — con lo=-0.03 y hi=-0.20 (lo > hi),
#        el resultado SIEMPRE daba -0.03 sin importar el ajuste real,
#        porque min(hi, value) nunca supera -0.20 y luego max(-0.03, ese)
#        siempre gana -0.03. Efecto: dd_max quedó congelado en -0.03
#        (cualquier caída de 3% dispara "defensive"), ignorando 29
#        ajustes acumulados de "false_alarm" que intentaban subirlo
#        a -0.20. Con "defensive" disparándose casi todos los días,
#        "growth"/"neutral" casi nunca se evaluaban, y por eso los
#        umbrales de growth nunca se movieron de sus defaults.
# =========================================================

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

logger = logging.getLogger("regime_learner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [LEARNER] %(message)s")

import os
DATA_PATH          = Path(os.getenv("DATA_PATH", "/data"))
MARKET_HISTORY_DIR = DATA_PATH / "market_history"
THRESHOLDS_FILE    = DATA_PATH / "regime_thresholds.json"

# =========================================================
# PARÁMETROS DE APRENDIZAJE
# =========================================================
EVAL_HORIZON_DAYS = 5       # evalúa retorno N días después
LEARNING_RATE     = 0.008   # ajuste por evaluación (gradual)
MISS_PENALTY      = 0.015   # ajuste extra cuando pierde una caída fuerte
STRONG_MOVE_PCT   = 0.04    # >4% = movimiento fuerte de mercado

# Límites de umbrales (no pueden salirse de rango)
# [A1] dd_max corregido: (lo, hi) con lo < hi, igual que los demás
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
}

DEFAULT_THRESHOLDS = {
    "defensive": {"dd_max": -0.08, "vol_min": 0.25, "corr_min": 0.75},
    "growth":    {"trend_min": 0.05, "vol_max": 0.15, "corr_max": 0.55},
    "meta": {
        "version": 1,
        "created_at": datetime.utcnow().isoformat(),
        "last_updated": datetime.utcnow().isoformat(),
        "total_evals": 0,
        "defensive_hits": 0,
        "defensive_misses": 0,
        "growth_hits": 0,
        "growth_misses": 0,
        "neutral_misses": 0,
    }
}


# =========================================================
# HELPERS
# =========================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def load_thresholds() -> Dict:
    if THRESHOLDS_FILE.exists():
        try:
            data = json.loads(THRESHOLDS_FILE.read_text())
            if "defensive" in data and "growth" in data:
                if "meta" not in data:
                    data["meta"] = DEFAULT_THRESHOLDS["meta"].copy()
                return data
        except Exception:
            pass
    return DEFAULT_THRESHOLDS.copy()


def save_thresholds(t: Dict) -> None:
    THRESHOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = THRESHOLDS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(t, indent=2))
    tmp.replace(THRESHOLDS_FILE)


def load_snapshot(date_str: str) -> Optional[Dict]:
    path = MARKET_HISTORY_DIR / f"{date_str}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def get_spy_return(base_date_str: str, horizon: int) -> Optional[float]:
    """
    Calcula retorno de SPY entre base_date y base_date + horizon días.
    Lee desde los snapshots en disco (ya guardados por market_quant_context).
    """
    try:
        base_date   = datetime.strptime(base_date_str, "%Y-%m-%d").date()
        target_date = base_date + timedelta(days=horizon)

        base_snap = load_snapshot(base_date_str)
        if not base_snap or "spy_price" not in base_snap:
            return None

        # Buscar snapshot del día target o el más cercano posterior
        for offset in range(0, 5):
            candidate = target_date + timedelta(days=offset)
            snap = load_snapshot(candidate.isoformat())
            if snap and "spy_price" in snap:
                base_price   = float(base_snap["spy_price"])
                target_price = float(snap["spy_price"])
                if base_price > 0:
                    return (target_price / base_price) - 1.0

        return None
    except Exception as e:
        logger.debug(f"get_spy_return error: {e}")
        return None


# =========================================================
# EVALUACIÓN DE UN SNAPSHOT
# =========================================================

def evaluate_snapshot(
    snapshot: Dict,
    spy_return: float,
    thresholds: Dict,
) -> Tuple[str, str, Dict]:
    """
    Evalúa si el régimen declarado fue correcto dado el retorno real.

    Retorna: (outcome, regime, ajustes_sugeridos)
    outcome: "hit" | "miss" | "false_alarm" | "neutral_miss"
    """
    regime     = snapshot.get("regime", "neutral")
    adjustments = {}

    if regime == "defensive":
        if spy_return < -0.01:
            # ✅ ACIERTO: declaró defensive y mercado cayó
            outcome = "hit"
            # Hacer umbral ligeramente más sensible (más fácil activar defensive)
            adjustments["defensive.dd_max"]   = +LEARNING_RATE  # menos negativo = más fácil activar
            adjustments["defensive.vol_min"]  = -LEARNING_RATE  # más bajo = más fácil activar
            adjustments["defensive.corr_min"] = -LEARNING_RATE
            logger.info(f"✅ DEFENSIVE HIT | spy_ret={spy_return:.2%} → ajustando umbrales más sensibles")
        else:
            # ❌ FALSA ALARMA: declaró defensive pero mercado subió
            outcome = "false_alarm"
            # Hacer umbral menos sensible (más difícil activar defensive)
            adjustments["defensive.dd_max"]   = -LEARNING_RATE
            adjustments["defensive.vol_min"]  = +LEARNING_RATE
            adjustments["defensive.corr_min"] = +LEARNING_RATE
            logger.info(f"❌ DEFENSIVE FALSE ALARM | spy_ret={spy_return:.2%} → ajustando umbrales menos sensibles")

    elif regime == "growth":
        if spy_return > 0.01:
            # ✅ ACIERTO: declaró growth y mercado subió
            outcome = "hit"
            adjustments["growth.trend_min"] = -LEARNING_RATE  # más fácil activar growth
            adjustments["growth.vol_max"]   = +LEARNING_RATE
            adjustments["growth.corr_max"]  = +LEARNING_RATE
            logger.info(f"✅ GROWTH HIT | spy_ret={spy_return:.2%} → ajustando umbrales más sensibles")
        else:
            # ❌ FALSA ALARMA: declaró growth pero mercado cayó
            outcome = "false_alarm"
            adjustments["growth.trend_min"] = +LEARNING_RATE
            adjustments["growth.vol_max"]   = -LEARNING_RATE
            adjustments["growth.corr_max"]  = -LEARNING_RATE
            logger.info(f"❌ GROWTH FALSE ALARM | spy_ret={spy_return:.2%} → ajustando umbrales menos sensibles")

    elif regime == "neutral":
        if spy_return < -STRONG_MOVE_PCT:
            # ❌ MISS: estaba neutral pero mercado cayó fuerte — debería haber sido DEFENSIVE
            outcome = "neutral_miss"
            # Ajuste más agresivo — el sistema perdió una caída real
            adjustments["defensive.dd_max"]   = +MISS_PENALTY
            adjustments["defensive.vol_min"]  = -MISS_PENALTY
            adjustments["defensive.corr_min"] = -MISS_PENALTY
            logger.warning(f"⚠️ NEUTRAL MISS (debió ser DEFENSIVE) | spy_ret={spy_return:.2%} → ajuste agresivo")
        elif spy_return > STRONG_MOVE_PCT:
            # ❌ MISS: estaba neutral pero mercado subió fuerte — debería haber sido GROWTH
            outcome = "neutral_miss_up"
            adjustments["growth.trend_min"] = -MISS_PENALTY
            adjustments["growth.vol_max"]   = +MISS_PENALTY
            adjustments["growth.corr_max"]  = +MISS_PENALTY
            logger.info(f"⚠️ NEUTRAL MISS (debió ser GROWTH) | spy_ret={spy_return:.2%}")
        else:
            outcome = "correct_neutral"

    else:
        outcome = "unknown"

    return outcome, regime, adjustments


# =========================================================
# APLICAR AJUSTES A UMBRALES
# =========================================================

def apply_adjustments(thresholds: Dict, adjustments: Dict) -> Dict:
    """
    Aplica ajustes graduales respetando los límites de cada umbral.
    """
    import copy
    t = copy.deepcopy(thresholds)

    for key, delta in adjustments.items():
        group, param = key.split(".", 1)
        if group not in t or param not in t[group]:
            continue

        current = t[group][param]
        limits  = THRESHOLD_LIMITS.get(group, {}).get(param)

        if limits is None:
            continue

        lo, hi = limits
        new_val = _clamp(current + delta, lo, hi)

        if abs(new_val - current) > 1e-6:
            logger.info(
                f"📐 {group}.{param}: {current:.4f} → {new_val:.4f} "
                f"(Δ={delta:+.4f}, límites=[{lo},{hi}])"
            )
            t[group][param] = round(new_val, 5)

    return t


# =========================================================
# CORE — EJECUTAR APRENDIZAJE
# =========================================================

def run_learning_cycle() -> Dict:
    """
    Evalúa snapshots de hace EVAL_HORIZON_DAYS días
    y ajusta umbrales según aciertos/errores.

    Retorna resumen del ciclo.
    """
    logger.info(f"🧠 RegimeLearner V1.0 | horizon={EVAL_HORIZON_DAYS}d")

    if not MARKET_HISTORY_DIR.exists():
        logger.warning("⚠️ Sin historial de mercado — nada que aprender todavía")
        return {"status": "no_history"}

    thresholds = load_thresholds()
    today      = datetime.utcnow().date()

    # Evaluar snapshots de hace EVAL_HORIZON_DAYS días
    eval_date = today - timedelta(days=EVAL_HORIZON_DAYS)
    snapshot  = load_snapshot(eval_date.isoformat())

    if not snapshot:
        logger.info(f"⏭ Sin snapshot para {eval_date} — saltando")
        return {"status": "no_snapshot", "date": eval_date.isoformat()}

    spy_return = get_spy_return(eval_date.isoformat(), EVAL_HORIZON_DAYS)

    if spy_return is None:
        logger.warning(f"⚠️ No se pudo calcular retorno SPY para {eval_date}")
        return {"status": "no_spy_return", "date": eval_date.isoformat()}

    outcome, regime, adjustments = evaluate_snapshot(snapshot, spy_return, thresholds)

    # Actualizar contadores
    meta = thresholds.get("meta", {})
    meta["total_evals"]    = meta.get("total_evals", 0) + 1
    meta["last_updated"]   = datetime.utcnow().isoformat()
    meta["last_eval_date"] = eval_date.isoformat()
    meta["last_outcome"]   = outcome
    meta["last_spy_return"]= round(spy_return * 100, 3)

    if outcome == "hit":
        key = f"{regime}_hits"
        meta[key] = meta.get(key, 0) + 1
    elif outcome in ("false_alarm", "neutral_miss", "neutral_miss_up"):
        if regime == "neutral":
            meta["neutral_misses"] = meta.get("neutral_misses", 0) + 1
        else:
            key = f"{regime}_misses"
            meta[key] = meta.get(key, 0) + 1

    thresholds["meta"] = meta

    # Aplicar ajustes
    if adjustments:
        thresholds = apply_adjustments(thresholds, adjustments)

    save_thresholds(thresholds)

    # Calcular accuracy acumulada
    total = meta.get("total_evals", 1)
    d_hits = meta.get("defensive_hits", 0)
    g_hits = meta.get("growth_hits", 0)

    summary = {
        "status":         "ok",
        "eval_date":      eval_date.isoformat(),
        "regime":         regime,
        "spy_return_pct": round(spy_return * 100, 2),
        "outcome":        outcome,
        "adjustments":    adjustments,
        "thresholds_now": {
            "defensive": thresholds["defensive"],
            "growth":    thresholds["growth"],
        },
        "cumulative": {
            "total_evals":      total,
            "defensive_hits":   d_hits,
            "growth_hits":      g_hits,
            "neutral_misses":   meta.get("neutral_misses", 0),
        },
    }

    logger.info(
        f"✅ Ciclo completado | regime={regime} outcome={outcome} "
        f"spy_ret={spy_return:.2%} | evals={total}"
    )
    logger.info(f"📊 Umbrales actuales: defensive.dd_max={thresholds['defensive']['dd_max']:.4f} "
                f"vol_min={thresholds['defensive']['vol_min']:.4f}")

    return summary


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    result = run_learning_cycle()
    print(json.dumps(result, indent=2))

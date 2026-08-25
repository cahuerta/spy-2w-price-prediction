# =========================================================
# alpha_engine_enterprise_strict_v4.3_fixed.py
# =========================================================
# 🔥 V4.3 = V4.2 + V6.3 THETA BONUS + PRODUCTION READY
# ✔ Time decay + Liquidity gate + Disagreement haircut
# ✔ V6.3 theta_dynamic boost (+10% si supera umbral propio)
# ✔ Institutional-grade alpha filtering
#
# FIXES v4.3-fixed:
#   [F1] std_u == 0 → ZeroDivisionError en s_ret_rel (CRASH)
#   [F2] glob vacío en pre-scan → IndexError silenciado
#   [F3] pred_data None o key faltante → KeyError / TypeError
#   [F4] save_json sin json_converter → TypeError en np types
#   [F5] bare except → ahora captura específica
#   [F6] valid_count ahora cuenta alpha_score != 0 (signed)
#
# FIXES v4.4-mem:
#   [F7] gc.collect() entre tickers en compute_and_persist_alpha
#        → libera DataFrames de precio y señales entre cada ticker
#        → evita acumulación de ~87 DataFrames en memoria simultáneos
#        → resuelve crash "Ran out of memory (used over 512MB)"
#
# FIXES v4.5 (auditoría 2026-08-24):
#   [F8] fundamental=None (clave presente con valor None, no ausente)
#        rompía `fundamental.get("usable")` con AttributeError. Esto
#        afectaba ~37/191 tickers (19.4% del universo), incluyendo
#        nombres grandes (ADBE, AMD, AMZN, BA...). Confirmado en
#        /data/alpha_last.json: 37 de 42 errores totales eran
#        exactamente 'NoneType' object has no attribute 'get'.
#        Fix: `signal.get("fundamental") or {}` en vez de
#        `signal.get("fundamental", {})` — el default de .get() solo
#        aplica si la clave está AUSENTE, no si está presente con
#        valor None.
#   [F9] except Exception genérico en compute_and_persist_alpha
#        mezclaba bugs reales (como F8) con casos esperados de "sin
#        datos" (FileNotFoundError, ValueError, KeyError — que el
#        propio código ya levanta deliberadamente más arriba). Ambos
#        terminaban con el mismo alpha_score=0.0 indistinguible entre
#        sí. Ahora se separan: los esperados se registran como
#        warning con reason="no_data", cualquier otra excepción se
#        registra como error con reason="unexpected_bug" y traceback
#        completo, para que un bug nuevo no vuelva a pasar
#        desapercibido mezclado con "sin datos".
#   [F10] `valid_alphas` contaba `alpha_score != 0`, pero el nombre
#        del campo invitaba a leerlo como "candidatos operables" —
#        confirmado en la auditoría: 119 tickers con alpha_score
#        distinto de cero, pero solo 19 superan `alpha_threshold`
#        (0.70), que es el umbral que trading_orchestrator.py
#        realmente usa para decidir aperturas. Cualquier consumidor
#        que asumiera "119 válidos = 119 operables" sobreestimaba el
#        universo real en 6x.
#        Fix: `valid_alphas` se renombra a `nonzero_alpha_count`
#        (mismo valor, nombre honesto sobre lo que mide), y se agrega
#        `above_threshold_count` — el conteo real de tickers que
#        superan `alpha_threshold` en magnitud. Se mantiene
#        `valid_alphas` como alias del mismo valor por compatibilidad
#        con cualquier consumidor existente que ya lea esa clave.
# =========================================================

import os
import gc
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from signals import compute_signal
from data_provider import get_price_history

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [ALPHA] %(message)s")
logger = logging.getLogger("alpha_engine")

# PATHS
DATA_PATH   = Path(os.getenv("DATA_PATH", "/data"))
PRED_DIR    = DATA_PATH / "predictions"
MARKET_FILE = DATA_PATH / "market_context.json"
ALPHA_FILE  = DATA_PATH / "alpha_last.json"

# CONSTANTES INSTITUCIONALES
MAX_PRED_AGE_HOURS       = 240
MIN_STRUCTURAL_LIQUIDITY = 0.20
DISAGREEMENT_HAIRCUT     = 0.75
V6_3_THETA_BONUS         = 1.10

# [F10] Umbral real de "candidato operable" — el mismo que
# trading_orchestrator.py usa para decidir aperturas. Antes estaba
# hardcodeado como 0.70 solo dentro del payload de salida, sin
# usarse en ningún cálculo de este archivo — ahora es la fuente
# única para calcular above_threshold_count.
ALPHA_THRESHOLD = 0.70

# [F9] Excepciones "esperadas" — el propio código las levanta
# deliberadamente para señalar "sin datos suficientes", no un bug.
_EXPECTED_NO_DATA_ERRORS = (FileNotFoundError, ValueError, KeyError)


# =========================================================
# HELPERS ATÓMICOS
# =========================================================

def _json_converter(obj: Any) -> Any:
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"⚠️ load_json falló en {path}: {e}")
        return None


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=_json_converter))
    tmp.replace(path)


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _get_latest_pred_path(ticker: str) -> Optional[Path]:
    ticker_dir = PRED_DIR / ticker
    if not ticker_dir.exists():
        return None
    candidates = sorted(ticker_dir.glob("*.json"))
    return candidates[-1] if candidates else None


# =========================================================
# ESTRUCTURAL
# =========================================================

def compute_structural_score(ticker: str) -> Dict[str, float]:
    raw = get_price_history(ticker, period="1y", interval="1d")
    if raw is None or len(raw) < 60:
        raise ValueError(f"[{ticker}] Insuficiente historial estructural (mínimo 60 días)")

    closes    = raw["Close"].values
    volumes   = raw["Volume"].values
    returns   = np.diff(closes) / closes[:-1]

    volatility = float(np.std(returns))
    trend      = float((closes[-1] / closes[0]) - 1)
    liquidity  = float(np.mean(closes * volumes))

    s_trend = clip01(trend / 0.15)
    s_vol   = clip01(1 - abs(volatility - 0.02) / 0.04)
    s_liq   = clip01(liquidity / 75_000_000)

    final_struct = clip01(0.4 * s_trend + 0.3 * s_vol + 0.3 * s_liq)

    # [F7] Liberar el DataFrame de precio inmediatamente
    del raw, closes, volumes, returns

    return {
        "score":      final_struct,
        "is_uptrend": trend > 0,
        "liquidity":  s_liq,
        "volatility": volatility,
        "trend_pct":  trend,
    }


# =========================================================
# 🔥 V4.3 CORE ENGINE
# =========================================================

def compute_alpha_for_ticker(
    ticker: str,
    market_ctx: Dict,
    universe_returns: List[float],
) -> Dict[str, Any]:

    # 1. PREDICTION + TIME DECAY
    pred_path = _get_latest_pred_path(ticker)
    if pred_path is None:
        raise FileNotFoundError(f"No prediction file for {ticker}")

    pred_data = load_json(pred_path)
    if not pred_data or "prediction" not in pred_data:
        raise ValueError(f"[{ticker}] JSON de predicción inválido o corrupto")

    prediction = pred_data["prediction"]
    if "ret_ens_pct" not in prediction or "theta_dynamic_pct" not in prediction:
        raise KeyError(f"[{ticker}] Faltan claves 'ret_ens_pct' o 'theta_dynamic_pct'")

    file_time     = datetime.fromtimestamp(pred_path.stat().st_mtime, tz=timezone.utc)
    age_hours     = (datetime.now(timezone.utc) - file_time).total_seconds() / 3600
    time_decay    = clip01(1.0 - (age_hours / MAX_PRED_AGE_HOURS))
    pred_ret      = float(prediction["ret_ens_pct"])
    theta_dynamic = float(prediction["theta_dynamic_pct"])
    v6_3_bonus    = V6_3_THETA_BONUS if abs(pred_ret) >= theta_dynamic else 1.0

    # [F7] Liberar pred_data — ya extraímos lo que necesitamos
    del pred_data

    # 2. SIGNAL & HISTÓRICO
    signal     = compute_signal(ticker)
    confidence = float(signal.get("confidence", 0.0))
    metrics    = signal.get("rolling_metrics", {"hit_rate": 0.45, "mae_return_pct": 10.0})
    s_hit      = clip01((metrics["hit_rate"] - 0.45) / 0.30)
    s_mae      = clip01(1.0 / (1.0 + metrics["mae_return_pct"] / 5.0))

    # [F8] fundamental puede venir con la clave PRESENTE pero valor
    # None (signals.py: "fundamental": fundamental if usable else None).
    # `.get("fundamental", {})` NO aplica el default en ese caso, porque
    # el default de dict.get() solo se usa si la clave está AUSENTE.
    # `or {}` cubre ambos casos: clave ausente Y clave presente con None.
    fundamental = signal.get("fundamental") or {}
    mispricing  = fundamental.get("mispricing_pct", 0.0) if fundamental.get("usable") else 0.0
    s_fund      = clip01(abs(mispricing) / 25.0)

    # [F7] Liberar signal
    del signal, fundamental

    # 3. LIQUIDITY GATE
    struct = compute_structural_score(ticker)
    if struct["liquidity"] < MIN_STRUCTURAL_LIQUIDITY:
        logger.warning(
            f"🚫 {ticker} BLOCKED: liquidity {struct['liquidity']:.3f} < {MIN_STRUCTURAL_LIQUIDITY}"
        )
        return {
            "ticker":      ticker,
            "alpha_score": 0.0,
            "reason":      "liquidity_gate_triggered",
            "liquidity":   struct["liquidity"],
        }

    # 4. MARKET
    m_raw    = market_ctx.get("source", {}).get("raw_quant", {})
    vol_m    = clip01(1 - abs(m_raw.get("volatility", 0.03) - 0.02) / 0.05)
    s_market = clip01(0.6 * vol_m + 0.4 * (1 + m_raw.get("drawdown_rolling", 0.0)))

    # 5. Z-SCORE RELATIVO
    avg_u    = float(np.mean(universe_returns)) if universe_returns else 0.0
    std_u    = float(np.std(universe_returns))  if universe_returns else 1.0
    if std_u == 0.0:
        std_u = 1.0
    s_ret_rel = clip01(0.5 + (pred_ret - avg_u) / (2 * std_u))

    # 🏆 ENSAMBLE
    base_alpha = (
        0.25 * struct["score"] +
        0.20 * confidence      +
        0.15 * s_ret_rel       +
        0.15 * s_hit           +
        0.10 * s_mae           +
        0.10 * s_fund          +
        0.05 * s_market
    )

    is_pred_positive = pred_ret > 0
    disagreement     = is_pred_positive != struct["is_uptrend"]
    if disagreement:
        base_alpha *= DISAGREEMENT_HAIRCUT
        logger.info(f"⚠️ {ticker} haircut: pred vs trend mismatch")

    unsigned_alpha = clip01(base_alpha * v6_3_bonus * time_decay)
    direction      = float(np.sign(pred_ret))
    signed_alpha   = float(unsigned_alpha * direction)

    return {
        "ticker":      ticker,
        "alpha_score": round(signed_alpha, 4),
        "components": {
            "structural":      round(struct["score"], 3),
            "relative_return": round(s_ret_rel, 3),
            "confidence":      round(confidence, 3),
            "v6_3_bonus":      round(v6_3_bonus, 3),
            "time_decay":      round(time_decay, 3),
            "hit_rate":        round(s_hit, 3),
            "fundamental":     round(s_fund, 3),
        },
        "flags": {
            "disagreement_penalty":  bool(disagreement),
            "liquidity_gate":        False,
            "v6_3_theta_cleared":    bool(abs(pred_ret) >= theta_dynamic),
            "age_hours":             round(age_hours, 1),
            "theta_dynamic_pct":     round(theta_dynamic, 3),
        },
        "debug": {
            "pred_ret_pct":  round(pred_ret, 3),
            "struct_trend":  round(struct["trend_pct"], 3),
        },
    }


# =========================================================
# 🚀 BATCH PRODUCTION
# =========================================================

def compute_and_persist_alpha(tickers: List[str]) -> Dict[str, Any]:
    """Entrada: tickers.json → Salida: /data/alpha_last.json"""

    logger.info(f"🔬 AlphaEngine V4.6 iniciado | {len(tickers)} tickers")

    market_ctx = load_json(MARKET_FILE) or {}

    # Pre-scan universo para Z-score
    universe_preds: List[float] = []
    for t in tickers:
        try:
            path = _get_latest_pred_path(t)
            if path is None:
                continue
            d = load_json(path)
            if d and "prediction" in d and "ret_ens_pct" in d["prediction"]:
                universe_preds.append(float(d["prediction"]["ret_ens_pct"]))
            del d  # [F7] liberar inmediatamente
        except (OSError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Pre-scan skip {t}: {e}")
            continue

    results:     Dict[str, Any] = {}
    nonzero_alpha_count: int    = 0   # [F10] antes "valid_count" → payload "valid_alphas"
    above_threshold_count: int  = 0   # [F10] nuevo — candidatos realmente operables
    no_data_count:  int         = 0
    unexpected_count: int       = 0

    for t in tickers:
        try:
            result = compute_alpha_for_ticker(t, market_ctx, universe_preds)
            results[t] = result
            if result["alpha_score"] != 0:
                nonzero_alpha_count += 1
            if abs(result["alpha_score"]) >= ALPHA_THRESHOLD:
                above_threshold_count += 1
        except _EXPECTED_NO_DATA_ERRORS as e:
            # [F9] Caso esperado: el propio código levantó esto para
            # señalar "sin datos suficientes" (sin predicción, historial
            # insuficiente, JSON corrupto o incompleto). No es un bug.
            logger.warning(f"⚠️ {t}: sin datos suficientes: {e}")
            results[t] = {
                "error":       str(e),
                "error_type":  type(e).__name__,
                "reason":      "no_data",
                "alpha_score": 0.0,
            }
            no_data_count += 1
        except Exception as e:
            # [F9] Cualquier otra excepción es un bug real e inesperado
            # (como F8 antes de corregirse) — se registra con traceback
            # completo para que no quede invisible mezclado con "sin
            # datos".
            logger.error(f"❌ {t}: BUG INESPERADO: {e}", exc_info=True)
            results[t] = {
                "error":       str(e),
                "error_type":  type(e).__name__,
                "reason":      "unexpected_bug",
                "alpha_score": 0.0,
            }
            unexpected_count += 1
        finally:
            # [F7] Liberar memoria entre tickers
            # compute_structural_score descarga 1 año de precios por ticker
            # compute_signal también carga datos históricos
            # Sin gc.collect() se acumulan ~87 DataFrames → crash 512MB
            gc.collect()

    payload: Dict[str, Any] = {
        "timestamp":              datetime.now(timezone.utc).isoformat(),
        "version":                "4.6",
        "universe_size":          len(tickers),
        "nonzero_alpha_count":    nonzero_alpha_count,     # [F10] antes "valid_alphas"
        "above_threshold_count":  above_threshold_count,   # [F10] candidatos realmente operables
        "valid_alphas":           nonzero_alpha_count,     # [F10] alias por compatibilidad — mismo valor, mismo significado que antes
        "no_data_count":          no_data_count,      # [F9] trazabilidad
        "unexpected_errors":      unexpected_count,   # [F9] trazabilidad — debería ser 0
        "alpha_threshold":        ALPHA_THRESHOLD,
        "results":                results,
    }

    save_json(ALPHA_FILE, payload)
    logger.info(
        f"✅ Alpha V4.6 COMPLETADO | "
        f"{nonzero_alpha_count}/{len(tickers)} con score≠0 | "
        f"{above_threshold_count} sobre umbral {ALPHA_THRESHOLD} (operables reales) | "
        f"sin_datos={no_data_count} | bugs_inesperados={unexpected_count}"
    )

    return payload

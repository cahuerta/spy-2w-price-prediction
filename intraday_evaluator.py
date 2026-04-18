# =========================================================
# intraday_evaluator.py — INTRADAY EVALUATOR v1.0
# =========================================================
# Corre una vez al día dentro del pipeline (paso 4.6).
# Lee snapshots del día anterior y evalúa si las decisiones
# del tracker fueron correctas.
#
# Aprende dos cosas:
#
# 1. ENTRY TIMING — ¿fue buena hora para entrar?
#    Compara precio al momento de "entrar_ahora: true/false"
#    vs precio de cierre del día.
#    → Si dijo esperar y el precio bajó → acertó
#    → Si dijo entrar y el precio subió → acertó
#
# 2. POSITION TRACKING — ¿fue correcto el curve_status?
#    Compara curve_status asignado vs movimiento real posterior.
#    → Si dijo "ahead" y el precio siguió subiendo → acertó
#    → Si dijo "diverging" y siguió bajando → acertó
#
# Guarda aprendizaje en /data/intraday_learning.json
# El tracker lee ese archivo para ajustar sus umbrales.
# =========================================================

import os
import json
import logging
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger("intraday_evaluator")
logging.basicConfig(level=logging.INFO)

# =========================================================
# CONFIG
# =========================================================

DATA_PATH      = Path(os.getenv("DATA_PATH", "/data"))
INTRADAY_DIR   = DATA_PATH / "intraday"
LEARNING_FILE  = DATA_PATH / "intraday_learning.json"
ALPACA_KEY     = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET  = os.getenv("ALPACA_SECRET_KEY")

# Mínimo de evaluaciones antes de ajustar umbrales
MIN_SAMPLES_ADJUST = int(os.getenv("INTRADAY_MIN_SAMPLES", "10"))

_alpaca_client: Optional[StockHistoricalDataClient] = None


# =========================================================
# HELPERS
# =========================================================

def _get_alpaca_client() -> StockHistoricalDataClient:
    global _alpaca_client
    if _alpaca_client is None:
        _alpaca_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    return _alpaca_client


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"⚠️ load_json {path}: {e}")
        return {}


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# =========================================================
# PRECIO HISTÓRICO
# =========================================================

def _get_closing_price(ticker: str, fecha: date) -> Optional[float]:
    """Precio de cierre de un día específico."""
    try:
        client  = _get_alpaca_client()
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=fecha,
            end=fecha + timedelta(days=1),
        )
        bars = client.get_stock_bars(request).df
        if bars is None or bars.empty:
            return None
        bars = bars.reset_index()
        if "close" not in bars.columns:
            return None
        return float(bars["close"].iloc[-1])
    except Exception as e:
        logger.warning(f"⚠️ closing price {ticker} {fecha}: {e}")
        return None


def _get_price_at_hour(ticker: str, fecha: date, hora_str: str) -> Optional[float]:
    """
    Precio aproximado en una hora específica del día.
    hora_str formato: "HH:MM" UTC
    """
    try:
        client = _get_alpaca_client()
        hora   = int(hora_str.split(":")[0])
        minuto = int(hora_str.split(":")[1])

        dt_start = datetime(fecha.year, fecha.month, fecha.day,
                           hora, minuto, tzinfo=timezone.utc)
        dt_end   = dt_start + timedelta(minutes=10)

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=dt_start,
            end=dt_end,
        )
        bars = client.get_stock_bars(request).df
        if bars is None or bars.empty:
            return None
        bars = bars.reset_index()
        if "close" not in bars.columns:
            return None
        return float(bars["close"].iloc[-1])
    except Exception as e:
        logger.warning(f"⚠️ price at hour {ticker} {fecha} {hora_str}: {e}")
        return None


# =========================================================
# EVALUAR DECISIONES DE ENTRY TIMING
# =========================================================

def _evaluate_entry_decisions(
    snapshot: Dict,
    fecha: date,
) -> List[Dict]:
    """
    Para cada señal del día, evalúa si la decisión fue correcta.

    Lógica:
    - Para cada hora con señal, obtener precio en ese momento
    - Comparar vs precio de cierre del día
    - "entrar_ahora: true" → correcto si precio subió hacia cierre
    - "entrar_ahora: false" → correcto si precio bajó hacia cierre
      (es decir, era mejor esperar)
    """
    evaluaciones = []
    precio_cierre_cache: Dict[str, float] = {}

    for hora_str, data in snapshot.items():
        if not isinstance(data, dict):
            continue

        for senal in data.get("senales", []):
            ticker       = senal.get("ticker")
            entrar_ahora = senal.get("entrar_ahora", False)
            entry_score  = senal.get("entry_score", 0)
            precio_señal = senal.get("precio_actual")

            if not ticker or not precio_señal:
                continue

            # Precio de cierre (cacheado por ticker)
            if ticker not in precio_cierre_cache:
                cierre = _get_closing_price(ticker, fecha)
                if cierre:
                    precio_cierre_cache[ticker] = cierre
                else:
                    continue

            precio_cierre = precio_cierre_cache[ticker]
            ret_desde_señal = (precio_cierre / precio_señal - 1) * 100

            # ¿Fue correcta la decisión?
            if entrar_ahora:
                # Dijo entrar → correcto si precio subió
                fue_correcto = ret_desde_señal > 0
            else:
                # Dijo esperar → correcto si precio bajó
                # (hubiera sido mejor esperar)
                fue_correcto = ret_desde_señal < 0

            evaluaciones.append({
                "ticker":          ticker,
                "hora":            hora_str,
                "fecha":           fecha.isoformat(),
                "entrar_ahora":    entrar_ahora,
                "entry_score":     entry_score,
                "precio_señal":    round(precio_señal, 4),
                "precio_cierre":   round(precio_cierre, 4),
                "ret_pct":         round(ret_desde_señal, 4),
                "fue_correcto":    fue_correcto,
                "uso_bandas":      senal.get("uso_bandas", False),
            })

    return evaluaciones


# =========================================================
# EVALUAR DECISIONES DE POSITION TRACKING
# =========================================================

def _evaluate_position_decisions(
    snapshot: Dict,
    fecha: date,
) -> List[Dict]:
    """
    Para cada posición monitoreada, evalúa si el curve_status
    fue correcto comparando con el movimiento posterior.

    ahead     → correcto si siguió subiendo al cierre
    on_track  → correcto si no divergió significativamente
    lagging   → correcto si siguió bajando o no recuperó
    diverging → correcto si siguió bajando
    """
    evaluaciones = []
    precio_cierre_cache: Dict[str, float] = {}

    for hora_str, data in snapshot.items():
        if not isinstance(data, dict):
            continue

        for pos in data.get("monitor_posiciones", []):
            ticker       = pos.get("ticker")
            curve_status = pos.get("curve_status")
            precio_pos   = pos.get("precio_actual")
            dia_actual   = pos.get("dia_actual")

            if not ticker or not curve_status or not precio_pos:
                continue

            if ticker not in precio_cierre_cache:
                cierre = _get_closing_price(ticker, fecha)
                if cierre:
                    precio_cierre_cache[ticker] = cierre
                else:
                    continue

            precio_cierre   = precio_cierre_cache[ticker]
            ret_desde_señal = (precio_cierre / precio_pos - 1) * 100

            # ¿El status fue correcto?
            if curve_status == "ahead":
                fue_correcto = ret_desde_señal > 0
            elif curve_status == "on_track":
                fue_correcto = abs(ret_desde_señal) < 1.5
            elif curve_status == "lagging":
                fue_correcto = ret_desde_señal <= 0
            elif curve_status == "diverging":
                fue_correcto = ret_desde_señal < -0.5
            else:
                fue_correcto = None

            evaluaciones.append({
                "ticker":        ticker,
                "hora":          hora_str,
                "fecha":         fecha.isoformat(),
                "dia_posicion":  dia_actual,
                "curve_status":  curve_status,
                "precio_señal":  round(precio_pos, 4),
                "precio_cierre": round(precio_cierre, 4),
                "ret_pct":       round(ret_desde_señal, 4),
                "fue_correcto":  fue_correcto,
                "uso_bandas":    pos.get("uso_bandas", False),
            })

    return evaluaciones


# =========================================================
# ACTUALIZAR APRENDIZAJE
# =========================================================

def _update_learning(
    entry_evals: List[Dict],
    position_evals: List[Dict],
) -> Dict:
    """
    Lee el archivo de aprendizaje actual y lo actualiza
    con las nuevas evaluaciones.

    Estructura de aprendizaje:
    {
      "entry_timing": {
        "total": 150,
        "correct": 89,
        "hit_rate": 0.593,
        "by_score_bucket": {
          "0.5-0.6": {"total": 30, "correct": 14},
          "0.6-0.7": {"total": 60, "correct": 38},
          "0.7-0.8": {"total": 40, "correct": 28},
          "0.8-1.0": {"total": 20, "correct": 17},
        },
        "by_hour": {
          "11": {"total": 20, "correct": 12},
          ...
        },
        "adjusted_thresholds": {
          "min_score":        0.55,   # ajustado desde evaluaciones
          "min_score_updated": "2026-04-18",
        }
      },
      "position_tracking": {
        "total": 80,
        "correct": 61,
        "hit_rate": 0.763,
        "by_status": {
          "ahead":     {"total": 15, "correct": 11},
          "on_track":  {"total": 40, "correct": 32},
          "lagging":   {"total": 18, "correct": 13},
          "diverging": {"total": 7,  "correct": 5},
        }
      },
      "last_updated": "2026-04-18"
    }
    """
    learning = _load_json(LEARNING_FILE)
    if not learning:
        learning = {
            "entry_timing": {
                "total": 0, "correct": 0, "hit_rate": None,
                "by_score_bucket": {
                    "0.5-0.6": {"total": 0, "correct": 0},
                    "0.6-0.7": {"total": 0, "correct": 0},
                    "0.7-0.8": {"total": 0, "correct": 0},
                    "0.8-1.0": {"total": 0, "correct": 0},
                },
                "by_hour":            {},
                "adjusted_thresholds": {
                    "min_score":         0.55,
                    "min_score_updated": None,
                },
            },
            "position_tracking": {
                "total": 0, "correct": 0, "hit_rate": None,
                "by_status": {
                    "ahead":     {"total": 0, "correct": 0},
                    "on_track":  {"total": 0, "correct": 0},
                    "lagging":   {"total": 0, "correct": 0},
                    "diverging": {"total": 0, "correct": 0},
                },
            },
            "last_updated": None,
        }

    # ── Actualizar entry timing ───────────────────────────
    et = learning["entry_timing"]
    for ev in entry_evals:
        et["total"]   += 1
        et["correct"] += 1 if ev["fue_correcto"] else 0

        # Por bucket de score
        score = ev["entry_score"]
        if score < 0.6:
            bucket = "0.5-0.6"
        elif score < 0.7:
            bucket = "0.6-0.7"
        elif score < 0.8:
            bucket = "0.7-0.8"
        else:
            bucket = "0.8-1.0"
        et["by_score_bucket"][bucket]["total"]   += 1
        et["by_score_bucket"][bucket]["correct"] += 1 if ev["fue_correcto"] else 0

        # Por hora
        hora = ev["hora"].split(":")[0]
        if hora not in et["by_hour"]:
            et["by_hour"][hora] = {"total": 0, "correct": 0}
        et["by_hour"][hora]["total"]   += 1
        et["by_hour"][hora]["correct"] += 1 if ev["fue_correcto"] else 0

    if et["total"] > 0:
        et["hit_rate"] = round(et["correct"] / et["total"], 4)

    # ── Ajuste automático de min_score ───────────────────
    # Si tenemos suficientes muestras, encontrar el bucket
    # con mejor hit_rate y ajustar el umbral
    if et["total"] >= MIN_SAMPLES_ADJUST:
        best_bucket     = None
        best_hit_rate   = 0.0
        for bucket, stats in et["by_score_bucket"].items():
            if stats["total"] >= 5:
                hr = stats["correct"] / stats["total"]
                if hr > best_hit_rate:
                    best_hit_rate = hr
                    best_bucket   = bucket

        if best_bucket:
            # Ajustar min_score al inicio del mejor bucket
            bucket_start = float(best_bucket.split("-")[0])
            current      = et["adjusted_thresholds"]["min_score"]
            # Movimiento suave: no cambiar más de 0.05 a la vez
            new_threshold = max(0.45, min(0.80,
                current + np.clip(bucket_start - current, -0.05, 0.05)
            ))
            if abs(new_threshold - current) > 0.01:
                et["adjusted_thresholds"]["min_score"]         = round(float(new_threshold), 3)
                et["adjusted_thresholds"]["min_score_updated"] = datetime.now(timezone.utc).date().isoformat()
                logger.info(
                    f"🎯 Umbral min_score ajustado: {current:.3f} → {new_threshold:.3f} "
                    f"(bucket {best_bucket} hit_rate={best_hit_rate:.1%})"
                )

    # ── Actualizar position tracking ─────────────────────
    pt = learning["position_tracking"]
    for ev in position_evals:
        pt["total"]   += 1
        pt["correct"] += 1 if ev["fue_correcto"] else 0

        status = ev["curve_status"]
        if status in pt["by_status"]:
            pt["by_status"][status]["total"]   += 1
            pt["by_status"][status]["correct"] += 1 if ev["fue_correcto"] else 0

    if pt["total"] > 0:
        pt["hit_rate"] = round(pt["correct"] / pt["total"], 4)

    learning["last_updated"] = datetime.now(timezone.utc).date().isoformat()
    _save_json(LEARNING_FILE, learning)

    return learning


# =========================================================
# RUN PRINCIPAL
# =========================================================

def run_intraday_evaluator() -> Dict:
    """
    Evalúa el snapshot del día anterior y actualiza el aprendizaje.
    Llamado desde pipeline_router como paso 4.6.
    """
    hoy   = datetime.now(timezone.utc).date()
    ayer  = hoy - timedelta(days=1)

    # Saltar fines de semana
    if ayer.weekday() >= 5:
        logger.info(f"⏭ Intraday evaluator SKIP — ayer fue fin de semana ({ayer})")
        return {"skipped": True, "reason": "weekend"}

    snapshot_path = INTRADAY_DIR / f"{ayer.isoformat()}.json"
    if not snapshot_path.exists():
        logger.info(f"⏭ Intraday evaluator SKIP — no hay snapshot para {ayer}")
        return {"skipped": True, "reason": "no_snapshot"}

    snapshot = _load_json(snapshot_path)
    if not snapshot:
        logger.info(f"⏭ Intraday evaluator SKIP — snapshot vacío para {ayer}")
        return {"skipped": True, "reason": "empty_snapshot"}

    logger.info(f"📊 Evaluando snapshot intraday de {ayer} | {len(snapshot)} horas")

    # Evaluar decisiones
    entry_evals    = _evaluate_entry_decisions(snapshot, ayer)
    position_evals = _evaluate_position_decisions(snapshot, ayer)

    logger.info(
        f"📋 Evaluaciones | entry={len(entry_evals)} "
        f"positions={len(position_evals)}"
    )

    # Actualizar aprendizaje
    learning = _update_learning(entry_evals, position_evals)

    et = learning["entry_timing"]
    pt = learning["position_tracking"]

    logger.info(
        f"✅ Aprendizaje actualizado | "
        f"entry hit_rate={et.get('hit_rate', 0):.1%} ({et['total']} evals) | "
        f"position hit_rate={pt.get('hit_rate', 0):.1%} ({pt['total']} evals) | "
        f"min_score={et['adjusted_thresholds']['min_score']}"
    )

    return {
        "fecha_evaluada":   ayer.isoformat(),
        "entry_evals":      len(entry_evals),
        "position_evals":   len(position_evals),
        "entry_hit_rate":   et.get("hit_rate"),
        "position_hit_rate":pt.get("hit_rate"),
        "min_score_actual": et["adjusted_thresholds"]["min_score"],
        "total_historico":  et["total"],
    }


# =========================================================
# LEER UMBRALES APRENDIDOS — usado por intraday_tracker
# =========================================================

def get_learned_thresholds() -> Dict:
    """
    Retorna los umbrales ajustados por el evaluador.
    El tracker llama esto al iniciar para usar valores calibrados.
    """
    learning = _load_json(LEARNING_FILE)
    if not learning:
        return {
            "min_score":   0.55,
            "calibrated":  False,
            "total_evals": 0,
        }

    et = learning.get("entry_timing", {})
    return {
        "min_score":   et.get("adjusted_thresholds", {}).get("min_score", 0.55),
        "calibrated":  et.get("total", 0) >= MIN_SAMPLES_ADJUST,
        "total_evals": et.get("total", 0),
        "hit_rate":    et.get("hit_rate"),
        "best_hours":  _get_best_hours(et.get("by_hour", {})),
    }


def _get_best_hours(by_hour: Dict) -> List[str]:
    """Retorna las horas con mejor hit rate (mínimo 5 evaluaciones)."""
    ranked = []
    for hora, stats in by_hour.items():
        if stats["total"] >= 5:
            hr = stats["correct"] / stats["total"]
            ranked.append((hora, hr))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [h for h, _ in ranked[:3]]  # top 3 horas

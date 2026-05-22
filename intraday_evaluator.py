# =========================================================
# intraday_evaluator.py — INTRADAY EVALUATOR v1.1
# =========================================================
# v1.1: Conecta evaluaciones de entry timing con Darwin
#       predictor genomes — actualiza hit rates de shadows
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

            if ticker not in precio_cierre_cache:
                cierre = _get_closing_price(ticker, fecha)
                if cierre:
                    precio_cierre_cache[ticker] = cierre
                else:
                    continue

            precio_cierre   = precio_cierre_cache[ticker]
            ret_desde_señal = (precio_cierre / precio_señal - 1) * 100

            if entrar_ahora:
                fue_correcto = ret_desde_señal > 0
            else:
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

    et = learning["entry_timing"]
    for ev in entry_evals:
        et["total"]   += 1
        et["correct"] += 1 if ev["fue_correcto"] else 0

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

        hora = ev["hora"].split(":")[0]
        if hora not in et["by_hour"]:
            et["by_hour"][hora] = {"total": 0, "correct": 0}
        et["by_hour"][hora]["total"]   += 1
        et["by_hour"][hora]["correct"] += 1 if ev["fue_correcto"] else 0

    if et["total"] > 0:
        et["hit_rate"] = round(et["correct"] / et["total"], 4)

    if et["total"] >= MIN_SAMPLES_ADJUST:
        best_bucket   = None
        best_hit_rate = 0.0
        for bucket, stats in et["by_score_bucket"].items():
            if stats["total"] >= 5:
                hr = stats["correct"] / stats["total"]
                if hr > best_hit_rate:
                    best_hit_rate = hr
                    best_bucket   = bucket

        if best_bucket:
            bucket_start  = float(best_bucket.split("-")[0])
            current       = et["adjusted_thresholds"]["min_score"]
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
# v1.1 — ACTUALIZAR SHADOW GENOMES CON HIT RATES REALES
# =========================================================

def _update_shadow_genome_evals(h_hit_rates: Dict[str, float], fecha: date) -> None:
    """
    Registra el hit rate del día en cada shadow genome de cada H.
    Esto permite que predictor_arena compare campeón vs shadows.

    Los shadows no corren predicciones propias todavía —
    acumulan el hit rate del campeón como baseline para comparar
    cuando generen sus propias predicciones.
    """
    try:
        from darwin_engine.predictor_genome import GENOME_BASE, PredictorGenome

        for h_key, hit_rate in h_hit_rates.items():
            h_num = int(h_key[1:])

            # Actualizar hit rate del campeón
            try:
                from darwin_engine.predictor_genome import update_genome_hit_rate
                update_genome_hit_rate(h_num, hit_rate, n_evals=0)
            except Exception as e:
                logger.warning(f"⚠️ update champion H{h_num}: {e}")

            # Registrar en carpeta shadow/evals para el arena
            shadow_eval_dir = GENOME_BASE / f"H{h_num}" / "shadow" / "evals"
            shadow_eval_dir.mkdir(parents=True, exist_ok=True)

            # Acumular hit rate diario del campeón como referencia
            eval_file = shadow_eval_dir / f"champion_baseline.json"
            existing  = {}
            if eval_file.exists():
                try:
                    existing = json.loads(eval_file.read_text())
                except Exception:
                    pass

            daily    = existing.get("daily_rates", [])
            daily.append({"fecha": fecha.isoformat(), "hit_rate": hit_rate})
            daily    = daily[-60:]  # últimos 60 días

            avg_rate = float(np.mean([d["hit_rate"] for d in daily]))

            existing.update({
                "genome_id":      f"H{h_num}_champion_baseline",
                "horizon":        h_num,
                "hit_rate":       round(avg_rate, 4),
                "n_evaluations":  len(daily),
                "daily_rates":    daily,
                "last_updated":   fecha.isoformat(),
            })

            tmp = eval_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(existing, indent=2, default=str))
            tmp.replace(eval_file)

        logger.info(f"🧬 Darwin shadow evals actualizados | {len(h_hit_rates)} H")

    except Exception as e:
        logger.warning(f"⚠️ _update_shadow_genome_evals error: {e}")


def _compute_h_hit_rates_from_evals(
    entry_evals: List[Dict],
) -> Dict[str, float]:
    """
    Calcula hit rate aproximado por H basándose en los tickers evaluados.
    Usa las evaluaciones del día anterior del evaluador existente en disco.
    """
    from collections import defaultdict

    eval_dir = DATA_PATH / "evaluations"
    if not eval_dir.exists():
        return {}

    hit_counts  = defaultdict(int)
    eval_counts = defaultdict(int)

    # Solo los tickers evaluados hoy
    tickers_hoy = {ev["ticker"] for ev in entry_evals}

    for ticker in tickers_hoy:
        ticker_dir = eval_dir / ticker.upper()
        if not ticker_dir.exists():
            continue
        files = sorted(ticker_dir.glob("*.json"))[-30:]
        for f in files:
            try:
                ev   = json.loads(f.read_text())
                diag = ev.get("models_diagnostics", {})
                for h in range(1, 11):
                    key  = f"H{h}"
                    data = diag.get(key)
                    if not isinstance(data, dict):
                        continue
                    hit_sign = data.get("hit_sign")
                    if hit_sign is None:
                        continue
                    eval_counts[key] += 1
                    if hit_sign:
                        hit_counts[key] += 1
            except Exception:
                continue

    result = {}
    for h in range(1, 11):
        key   = f"H{h}"
        total = eval_counts.get(key, 0)
        if total >= 5:
            result[key] = round(hit_counts[key] / total, 4)

    logger.info(f"📊 H hit rates calculados: {result}")
    return result


# =========================================================
# RUN PRINCIPAL
# =========================================================

def run_intraday_evaluator() -> Dict:
    """
    Evalúa el snapshot del día anterior y actualiza el aprendizaje.
    Llamado desde pipeline_router como paso 4.6.
    """
    hoy  = datetime.now(timezone.utc).date()
    ayer = hoy - timedelta(days=1)

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

    entry_evals    = _evaluate_entry_decisions(snapshot, ayer)
    position_evals = _evaluate_position_decisions(snapshot, ayer)

    logger.info(
        f"📋 Evaluaciones | entry={len(entry_evals)} "
        f"positions={len(position_evals)}"
    )

    learning = _update_learning(entry_evals, position_evals)

    # ── v1.1: Actualizar Darwin con hit rates reales ──────
    h_hit_rates = _compute_h_hit_rates_from_evals(entry_evals)
    if h_hit_rates:
        _update_shadow_genome_evals(h_hit_rates, ayer)
    # ─────────────────────────────────────────────────────

    et = learning["entry_timing"]
    pt = learning["position_tracking"]

    logger.info(
        f"✅ Aprendizaje actualizado | "
        f"entry hit_rate={et.get('hit_rate', 0):.1%} ({et['total']} evals) | "
        f"position hit_rate={pt.get('hit_rate', 0):.1%} ({pt['total']} evals) | "
        f"min_score={et['adjusted_thresholds']['min_score']}"
    )

    return {
        "fecha_evaluada":    ayer.isoformat(),
        "entry_evals":       len(entry_evals),
        "position_evals":    len(position_evals),
        "entry_hit_rate":    et.get("hit_rate"),
        "position_hit_rate": pt.get("hit_rate"),
        "min_score_actual":  et["adjusted_thresholds"]["min_score"],
        "total_historico":   et["total"],
        "darwin_h_updated":  list(h_hit_rates.keys()),
    }


# =========================================================
# LEER UMBRALES APRENDIDOS — usado por intraday_tracker
# =========================================================

def get_learned_thresholds() -> Dict:
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
    ranked = []
    for hora, stats in by_hour.items():
        if stats["total"] >= 5:
            hr = stats["correct"] / stats["total"]
            ranked.append((hora, hr))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [h for h, _ in ranked[:3]]

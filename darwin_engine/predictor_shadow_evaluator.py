"""
darwin_engine/predictor_shadow_evaluator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluador nocturno de shadow genomes para H1-H10.

Pieza que faltaba en el ciclo evolutivo: hasta ahora, predictor_arena.py
esperaba encontrar hit_rate real de cada shadow en
GENOME_BASE/HX/shadow/evals/*.json, pero nada lo generaba — los shadows
existían solo como parámetros guardados en disco, nunca se ejecutaban.

Este archivo corre de noche (separado del trading real) y hace lo que
faltaba, en dos fases:

  FASE 1 — generate_shadow_predictions():
    Para cada horizonte H1-H10, para cada shadow genome, para un
    subconjunto rotativo de tickers de /data/tickers.json:
      1. Respalda el champion.json real
      2. Escribe temporalmente los parámetros del shadow en champion.json
      3. Corre run_predictor_hX(ticker) — el predictor real, sin
         modificarlo, cree que está leyendo al campeón
      4. Guarda la predicción en shadow/predictions/{genome_id}/
      5. Restaura el champion.json real inmediatamente

  FASE 2 — evaluate_matured_shadow_predictions():
    Para cada predicción de shadow cuyo horizonte ya venció:
      1. Obtiene el precio real (reutiliza evaluator.py, sin duplicar)
      2. Calcula hit_sign con el mismo umbral que usa el evaluador real
      3. Acumula hit_rate / n_evaluations en shadow/evals/{genome_id}.json
         — el archivo exacto que predictor_arena.py ya espera leer

DISEÑO:
  [SE1] Rotación de universo: con ~158 tickers y TICKERS_PER_NIGHT=27,
        el universo completo se cubre en ~6 noches. Estado persistido
        en /data/darwin/shadow_eval_state.json (cursor simple).
  [SE2] Lock file (GENOME_BASE/.shadow_eval_lock) evita que esta corrida
        se solape con otra instancia de sí misma. Se libera solo si
        tiene más de 6h (asume crash previo).
  [SE3] El swap de champion.json se hace UNA vez por shadow (no por
        ticker) — todos los tickers de la noche corren bajo el mismo
        swap, minimizando ventanas de riesgo y I/O.
  [SE4] Ejecución serial (sin ThreadPoolExecutor) — el historial del
        proyecto tiene OOM conocido con concurrencia en Render Standard;
        se prioriza estabilidad sobre velocidad para este proceso nuevo.
  [SE5] Reutiliza get_price_at_date, nth_business_day y _calc_hit_sign
        de evaluator.py — no se duplica lógica de precios ni de umbral
        de señal.

NOTA PENDIENTE (no resuelto en este archivo):
  predictor_arena.py._evaluate_shadow_hit_rates() lee TODOS los .json
  en shadow/evals/, incluyendo champion_baseline.json (escrito por
  intraday_evaluator.py como placeholder temporal). Ese archivo no
  corresponde a un shadow real y debería excluirse de la comparación
  de "mejor shadow". Requiere un ajuste separado en predictor_arena.py.
"""

import os
import sys
import json
import time
import shutil
import logging
import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from darwin_engine.predictor_genome import PredictorGenome, GENOME_BASE, DATA_PATH
from evaluator import get_price_at_date, nth_business_day, _calc_hit_sign
from master_orchestrator import extract_price_prediction, extract_return_pct

logger = logging.getLogger("predictor_shadow_evaluator")

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════

TICKERS_FILE       = Path(DATA_PATH) / "tickers.json"
STATE_FILE         = Path(DATA_PATH) / "darwin" / "shadow_eval_state.json"
LOCK_FILE          = GENOME_BASE / ".shadow_eval_lock"
LOCK_MAX_AGE_SEC    = 6 * 3600  # 6 horas — lock más viejo se asume crash

TICKERS_PER_NIGHT  = int(os.getenv("SHADOW_EVAL_TICKERS_PER_NIGHT", "27"))


# ══════════════════════════════════════════════════════
# HELPERS JSON
# ══════════════════════════════════════════════════════

def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"⚠️ load_json {path}: {e}")
        return {}


def _save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# ══════════════════════════════════════════════════════
# [SE2] LOCK
# ══════════════════════════════════════════════════════

def _acquire_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            age_sec = time.time() - LOCK_FILE.stat().st_mtime
            if age_sec > LOCK_MAX_AGE_SEC:
                logger.warning(f"⚠️ Lock viejo ({age_sec/3600:.1f}h) → liberando forzado")
                LOCK_FILE.unlink(missing_ok=True)
            else:
                return False
        except Exception:
            return False
    try:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCK_FILE.write_text(datetime.now(timezone.utc).isoformat())
        return True
    except Exception as e:
        logger.error(f"❌ No se pudo crear lock: {e}")
        return False


def _release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════
# [SE1] UNIVERSO Y ROTACIÓN
# ══════════════════════════════════════════════════════

def _load_universe() -> List[str]:
    if not TICKERS_FILE.exists():
        return []
    try:
        data = json.loads(TICKERS_FILE.read_text())
        if isinstance(data, list):
            return sorted(t for t in data if isinstance(t, str))
    except Exception as e:
        logger.warning(f"⚠️ load_universe: {e}")
    return []


def _get_tonight_chunk(universe: List[str], chunk_size: int) -> List[str]:
    if not universe:
        return []

    n_chunks = max(1, -(-len(universe) // chunk_size))  # ceil division
    state    = _load_json(STATE_FILE) or {"cursor": 0}
    cursor   = state.get("cursor", 0) % n_chunks

    start = cursor * chunk_size
    end   = start + chunk_size
    chunk = universe[start:end]

    state["cursor"]        = cursor + 1
    state["last_run"]      = datetime.now(timezone.utc).isoformat()
    state["universe_size"] = len(universe)
    state["n_chunks"]      = n_chunks
    _save_json(STATE_FILE, state)

    logger.info(
        f"🎯 Chunk {cursor + 1}/{n_chunks} | "
        f"{len(chunk)} tickers | universo total={len(universe)}"
    )
    return chunk


# ══════════════════════════════════════════════════════
# FASE 1 — GENERAR PREDICCIONES DE SHADOWS
# ══════════════════════════════════════════════════════

def _save_shadow_prediction(
    horizon: int, genome_id: str, ticker: str, result: Dict
) -> bool:
    try:
        pred_dir = GENOME_BASE / f"H{horizon}" / "shadow" / "predictions" / genome_id
        pred_dir.mkdir(parents=True, exist_ok=True)
        today_str = datetime.now(timezone.utc).date().isoformat()
        pred_path = pred_dir / f"{ticker}_{today_str}.json"
        tmp = pred_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, default=str))
        tmp.replace(pred_path)
        return True
    except Exception as e:
        logger.error(f"❌ save_shadow_prediction {genome_id} {ticker}: {e}")
        return False


def _run_shadow_batch(horizon: int, shadow: PredictorGenome, tickers: List[str]) -> int:
    """
    [SE3] Swap de champion.json UNA vez por shadow — corre todos los
    tickers de la noche bajo el mismo swap, y restaura al final.
    """
    champion_path = GENOME_BASE / f"H{horizon}" / "champion.json"
    backup_path   = champion_path.with_suffix(".real_champion_backup")

    if not champion_path.exists():
        logger.warning(f"⚠️ champion.json no existe para H{horizon} — saltando shadow")
        return 0

    ok_count = 0
    swapped  = False
    try:
        shutil.copy2(champion_path, backup_path)
        tmp = champion_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(shadow.to_dict(), indent=2, default=str))
        tmp.replace(champion_path)
        swapped = True

        mod  = importlib.import_module(f"predictors_engine.predictor_h{horizon}")
        func = getattr(mod, f"run_predictor_h{horizon}")

        for ticker in tickers:
            try:
                result = func(ticker)
                if result is None:
                    continue
                if _save_shadow_prediction(horizon, shadow.genome_id, ticker, result):
                    ok_count += 1
            except Exception as e:
                logger.error(f"❌ shadow pred H{horizon} {shadow.genome_id} {ticker}: {e}")

    except Exception as e:
        logger.error(f"❌ swap error H{horizon} {shadow.genome_id}: {e}")

    finally:
        # SIEMPRE restaurar el champion real, pase lo que pase
        if swapped and backup_path.exists():
            backup_path.replace(champion_path)
        elif backup_path.exists():
            backup_path.unlink(missing_ok=True)

    return ok_count


def generate_shadow_predictions(chunk_size: int = TICKERS_PER_NIGHT) -> Dict:
    if not _acquire_lock():
        logger.warning("⚠️ Lock activo — otra corrida de shadow evaluator en progreso, abortando")
        return {"status": "locked"}

    results = {"status": "ok", "predictions_generated": 0, "errors": 0, "by_horizon": {}}
    try:
        universe = _load_universe()
        tickers_tonight = _get_tonight_chunk(universe, chunk_size)
        if not tickers_tonight:
            logger.warning("⚠️ Sin tickers para evaluar esta noche")
            return {"status": "no_tickers"}

        logger.info(f"🌙 Shadow evaluator | {len(tickers_tonight)} tickers esta noche")

        for h in range(1, 11):
            shadow_genomes = PredictorGenome.load_shadow_genomes(h)
            if not shadow_genomes:
                results["by_horizon"][f"H{h}"] = 0
                continue

            h_count = 0
            for shadow in shadow_genomes:
                h_count += _run_shadow_batch(h, shadow, tickers_tonight)

            results["by_horizon"][f"H{h}"] = h_count
            results["predictions_generated"] += h_count
            logger.info(f"   H{h}: {h_count} predicciones ({len(shadow_genomes)} shadows)")

    finally:
        _release_lock()

    return results


# ══════════════════════════════════════════════════════
# FASE 2 — EVALUAR PREDICCIONES MADURAS
# ══════════════════════════════════════════════════════

def _append_shadow_eval(
    horizon: int, genome_id: str, hit_sign: Optional[bool], weak_signal: bool
) -> None:
    eval_dir  = GENOME_BASE / f"H{horizon}" / "shadow" / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"{genome_id}.json"

    existing = _load_json(eval_path) or {
        "genome_id":      genome_id,
        "horizon":        horizon,
        "hit_count":      0,
        "n_evaluations":  0,
        "n_weak_skipped": 0,
        "hit_rate":       None,
        "last_updated":   None,
    }

    if weak_signal or hit_sign is None:
        existing["n_weak_skipped"] = existing.get("n_weak_skipped", 0) + 1
    else:
        existing["n_evaluations"] = existing.get("n_evaluations", 0) + 1
        if hit_sign:
            existing["hit_count"] = existing.get("hit_count", 0) + 1
        if existing["n_evaluations"] > 0:
            existing["hit_rate"] = round(
                existing["hit_count"] / existing["n_evaluations"], 4
            )

    existing["last_updated"] = datetime.now(timezone.utc).isoformat()
    _save_json(eval_path, existing)


def evaluate_matured_shadow_predictions() -> Dict:
    today   = datetime.now(timezone.utc).date()
    summary = {"evaluated": 0, "pending": 0, "errors": 0, "by_horizon": {}}

    for h in range(1, 11):
        shadow_pred_root = GENOME_BASE / f"H{h}" / "shadow" / "predictions"
        if not shadow_pred_root.exists():
            continue

        h_evaluated = 0
        for genome_dir in shadow_pred_root.iterdir():
            if not genome_dir.is_dir():
                continue
            genome_id = genome_dir.name

            for pred_file in list(genome_dir.glob("*.json")):
                try:
                    data = _load_json(pred_file)
                    if not data:
                        pred_file.unlink(missing_ok=True)
                        continue

                    stem = pred_file.stem
                    if "_" not in stem:
                        pred_file.unlink(missing_ok=True)
                        continue
                    ticker, date_str = stem.rsplit("_", 1)

                    try:
                        pred_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    except ValueError:
                        pred_file.unlink(missing_ok=True)
                        continue

                    target_date = nth_business_day(pred_date, h)
                    if target_date > today:
                        summary["pending"] += 1
                        continue  # aún no madura — se reintenta otra noche

                    price_now = (
                        data.get("price_now")
                        or data.get("price_today")
                        or (data.get("prediction") or {}).get("price_now")
                    )
                    price_pred       = extract_price_prediction(data)
                    predicted_return = extract_return_pct(data)

                    if price_now is None or price_pred is None:
                        pred_file.unlink(missing_ok=True)
                        continue

                    price_now  = float(price_now)
                    real_price = get_price_at_date(ticker, target_date)
                    if not real_price:
                        # Sin precio real disponible todavía → reintentar otra noche
                        summary["pending"] += 1
                        continue

                    real_return = (real_price / price_now - 1) * 100.0
                    hit_sign, weak_signal = _calc_hit_sign(predicted_return, real_return)

                    _append_shadow_eval(h, genome_id, hit_sign, weak_signal)

                    pred_file.unlink(missing_ok=True)
                    h_evaluated += 1
                    summary["evaluated"] += 1

                except Exception as e:
                    logger.error(f"❌ eval shadow pred {pred_file}: {e}")
                    summary["errors"] += 1

        summary["by_horizon"][f"H{h}"] = h_evaluated

    return summary


# ══════════════════════════════════════════════════════
# CICLO COMPLETO
# ══════════════════════════════════════════════════════

def run_shadow_evolution_cycle() -> Dict:
    logger.info("=" * 60)
    logger.info("🌙 SHADOW EVALUATOR — CICLO NOCTURNO")
    logger.info(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 60)

    gen_result  = generate_shadow_predictions()
    eval_result = evaluate_matured_shadow_predictions()

    summary = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "generation": gen_result,
        "evaluation": eval_result,
    }

    logger.info(
        f"✅ Shadow evaluator completado | "
        f"generadas={gen_result.get('predictions_generated', 0)} | "
        f"evaluadas={eval_result.get('evaluated', 0)} | "
        f"pendientes={eval_result.get('pending', 0)}"
    )
    return summary


# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    result = run_shadow_evolution_cycle()
    print(json.dumps(result, indent=2, default=str))

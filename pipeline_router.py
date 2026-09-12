# =========================================================
# pipeline_router.py — PIPELINE ROUTER v2.8
# =========================================================
# v2.2: [F7] Intraday tracker como paso 10.5
# v2.3: [F8] Intraday evaluator como paso 4.6
# v2.4: [F9] gc.collect() entre pasos pesados
#        Libera memoria entre model runner, evaluator y alpha engine
#        para evitar crash "Ran out of memory (used over 512MB)"
#        Cada paso corre uno a uno sin acumular en memoria.
# v2.5: [F10] Lock anti-doble ejecución
#        Evita que un reinicio de Render o un segundo trigger del
#        scheduler lancen dos pipelines en paralelo, causando
#        condiciones de carrera en /data y consumo doble de RAM.
# v2.6: [F11] Fix intraday tracker paso 10.5
#        run_intraday_tracker() eliminado en tracker v3.0.
#        Reemplazado por evaluar_posiciones_abiertas(tickers)
#        que recibe la lista exacta del portfolio actual.
# v2.7: [AUD-P4] (2026-09-04) COMMIT ya no se llama a sí mismo por
#        HTTP (httpx.post a /internal/pipeline/commit). Root cause
#        confirmada: desde que el servicio pasó a correr por Docker
#        (21-ago-2026, Dockerfile agregado para Claude Code CLI del
#        auditor), esa auto-llamada dejó de completarse — 13 días
#        sin market_context.json actualizado y sin nuevos trades de
#        Darwin, con el fallo cayendo en un except genérico sin
#        alertar a nadie. Fix: se escribe directo a disco en el
#        mismo proceso (mismo patrón .tmp + .replace() que ya usa
#        el screener en el paso 1), sin salir por red. El endpoint
#        /internal/pipeline/commit de main.py queda intacto por si
#        algo externo lo usa, pero este pipeline ya no depende de él.
# v2.8: [F12] (2026-09-11) FIX CRASH OOM CONFIRMADO EN RENDER EVENTS
#        ("Ran out of memory (used over 512MB) while running your
#        code" — varias veces al día, horarios variables, desde que
#        se bajó a plan $7/512MB el 09-sep).
#        Root cause: aunque cada paso ya tenía gc.collect() [F9],
#        TODOS corrían dentro del mismo proceso de FastAPI. Con
#        pandas/numpy/sklearn, gc.collect() libera los objetos
#        Python pero NO garantiza que el allocator de C (glibc)
#        devuelva esa memoria al sistema operativo — queda retenida
#        por el proceso, acumulándose paso a paso durante el día
#        hasta romper el límite de 512MB en cualquier punto del
#        pipeline (por eso el crash no tenía horario fijo).
#        Fix: los 3 pasos más pesados (model runner, evaluator,
#        alpha engine) ahora corren cada uno en su propio proceso
#        hijo vía multiprocessing.Process con contexto "spawn"
#        (no "fork", para no heredar el loop asyncio de FastAPI).
#        Cuando un proceso hijo termina, el sistema operativo libera
#        el 100% de su memoria de forma real — no depende de que
#        Python "decida" soltarla. Los 3 ya escriben su resultado
#        directo a disco (mismo patrón de siempre) y ninguno
#        depende de devolver un objeto complejo al proceso padre,
#        así que el cambio es transparente para el resto del
#        pipeline. El proceso padre (FastAPI) solo espera (.join())
#        y revisa exitcode — sin pipes ni colas de por medio.
#        NOTA: procesos hijos vía multiprocessing.Process comparten
#        el mismo filesystem del contenedor (incluyendo /data) —
#        esto NO es lo mismo que un Render Cron Job separado (que sí
#        no comparte el disco persistente al ser otro servicio). No
#        hay ningún problema de escritura a disco con este cambio.
# =========================================================

from fastapi import APIRouter, HTTPException, Request
from datetime import datetime, timezone
import logging
import traceback
import os
import gc
import asyncio
import multiprocessing
from pathlib import Path
import json

from screener import run_screener_async
from decider import run_decider

from market_state_evaluator import run_market_state
from market_qualitative_evaluator import evaluate_qualitative_market
from market_orchestrator import MarketOrchestrator
from trading_orchestrator import TradingOrchestrator
from regime_threshold_learner import run_learning_cycle

router = APIRouter()
logger = logging.getLogger("pipeline")

# =========================================================
# [F10] LOCK ANTI-DOBLE EJECUCIÓN
# =========================================================
# Variable global que indica si hay un pipeline corriendo.
# Si el scheduler o Render disparan un segundo request mientras
# el primero aún está en curso, se retorna status="skipped".
_pipeline_running = False


# =========================================================
# [F12] WORKERS DE PROCESO HIJO — deben ser funciones top-level
# (importables por nombre) para que multiprocessing con contexto
# "spawn" pueda picklearlas al lanzar el proceso hijo.
# Cada una importa su módulo pesado DENTRO de la función, así el
# import (y toda la memoria que carga: pandas, sklearn, modelos)
# ocurre solo dentro del proceso hijo, nunca en el padre.
# =========================================================

def _mp_run_model_runner():
    from model_runner import run_all_models
    run_all_models()


def _mp_run_evaluator():
    from evaluator import evaluate_all
    evaluate_all()


def _mp_run_alpha_engine(tickers: list):
    from alpha_engine_v4 import compute_and_persist_alpha
    compute_and_persist_alpha(tickers)


# Contexto "spawn": el proceso hijo arranca limpio, sin heredar
# el loop de asyncio ni conexiones abiertas del proceso padre
# (FastAPI/Uvicorn). Con "fork" (default en Linux) el hijo hereda
# todo el estado del padre y puede colgarse o corromper el loop.
_mp_ctx = multiprocessing.get_context("spawn")


async def _run_in_subprocess(target, args: tuple, step_name: str, timeout_sec: int = 20 * 60):
    """
    [F12] Corre `target(*args)` en un proceso hijo separado y espera
    a que termine, sin bloquear el loop de asyncio del proceso padre.
    No pasa objetos de vuelta — el target ya escribe su resultado a
    disco. Solo se revisa exitcode para saber si falló.
    """
    proc = _mp_ctx.Process(target=target, args=args, name=step_name)
    proc.start()

    loop = asyncio.get_event_loop()
    try:
        # proc.join() es bloqueante — se corre en un executor para no
        # trabar el loop de asyncio del proceso padre mientras espera.
        await asyncio.wait_for(
            loop.run_in_executor(None, proc.join, timeout_sec),
            timeout=timeout_sec + 30,
        )
    except asyncio.TimeoutError:
        logger.error(f"⏱️ [F12] {step_name} excedió {timeout_sec}s — terminando proceso hijo")
        proc.terminate()
        proc.join(10)
        raise RuntimeError(f"{step_name} timeout tras {timeout_sec}s")

    if proc.is_alive():
        # proc.join(timeout_sec) venció sin que el proceso terminara
        logger.error(f"⏱️ [F12] {step_name} no terminó a tiempo — terminando proceso hijo")
        proc.terminate()
        proc.join(10)
        raise RuntimeError(f"{step_name} no terminó dentro de {timeout_sec}s")

    if proc.exitcode != 0:
        raise RuntimeError(f"{step_name} falló en proceso hijo (exitcode={proc.exitcode})")

    logger.info(f"✅ [F12] {step_name} completado en proceso hijo (pid={proc.pid})")


# =========================================================
# SMART SKIP
# =========================================================

def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _file_is_from_today(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return mtime.strftime("%Y-%m-%d") == _today_utc()


def _screener_done_today(data_path: Path) -> bool:
    return _file_is_from_today(data_path / "screener_candidates.json")


def _decider_done_today(data_path: Path) -> bool:
    return _file_is_from_today(data_path / "tickers.json")


def _models_done_today(data_path: Path) -> bool:
    today     = _today_utc()
    pred_path = data_path / "predictions"
    if not pred_path.exists():
        return False
    return any(pred_path.glob(f"**/{today}.json"))


def _evaluator_done_today(data_path: Path) -> bool:
    today     = _today_utc()
    eval_path = data_path / "evaluations"
    if not eval_path.exists():
        return False
    by_folder = (eval_path / today).exists() and any((eval_path / today).glob("*.json"))
    by_name   = any(eval_path.glob(f"*{today}*.json"))
    return by_folder or by_name


def _alpha_done_today(data_path: Path) -> bool:
    """[F9] Smart skip para alpha engine — evita doble ejecución en pipeline cierre."""
    alpha_file = data_path / "alpha_last.json"
    if not alpha_file.exists():
        return False
    try:
        data = json.loads(alpha_file.read_text())
        ts   = data.get("timestamp", "")
        return str(ts)[:10] == _today_utc()
    except Exception:
        return False


# =========================================================
# BACKGROUND PIPELINE
# =========================================================

async def _run_pipeline_logic(request: Request):
    global _pipeline_running

    start_ts = datetime.utcnow().isoformat()
    logger.info("=" * 60)
    logger.info(f"🚀 PIPELINE START | {start_ts}")
    logger.info("=" * 60)

    try:
        DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
        today     = _today_utc()

        # ── 1. SCREENER ───────────────────────────────────
        if _screener_done_today(DATA_PATH):
            screener_file = DATA_PATH / "screener_candidates.json"
            screener_out  = json.loads(screener_file.read_text())
            logger.info(f"⚡ [1/10] Screener SKIP | candidates={screener_out.get('n_candidates')}")
        else:
            logger.info("🔍 [1/10] Screener...")
            screener_out = await run_screener_async()
            screener_file = DATA_PATH / "screener_candidates.json"
            screener_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = screener_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(screener_out, indent=2))
            tmp.replace(screener_file)
            logger.info(f"✅ Screener OK | candidates={screener_out.get('n_candidates')}")

        gc.collect()  # [F9] liberar después del screener

        # ── 2. DECIDER ────────────────────────────────────
        if _decider_done_today(DATA_PATH):
            tickers_data = json.loads((DATA_PATH / "tickers.json").read_text())
            total        = len(tickers_data) if isinstance(tickers_data, list) else tickers_data.get("total", "?")
            logger.info(f"⚡ [2/10] Decider SKIP | total={total}")
            decider_out  = {"total": total, "added": []}
        else:
            logger.info("🧠 [2/10] Decider...")
            decider_out = run_decider()
            logger.info(f"✅ Decider OK | added={len(decider_out.get('added', []))} | total={decider_out.get('total')}")

        gc.collect()  # [F9] liberar después del decider

        # ── 3. MODEL RUNNER ───────────────────────────────
        # [F12] Corre en proceso hijo separado — libera 100% de su
        # memoria (pandas/sklearn de los 10 modelos H1-H10 por ticker)
        # al sistema operativo cuando termina, sin depender de gc.collect().
        if _models_done_today(DATA_PATH):
            pred_count = sum(1 for _ in (DATA_PATH / "predictions").glob(f"**/{today}.json"))
            logger.info(f"⚡ [3/10] Models SKIP | {pred_count} predicciones de hoy")
        else:
            logger.info("📈 [3/10] Model runner (proceso hijo)...")
            await _run_in_subprocess(_mp_run_model_runner, (), "model_runner")
            logger.info("✅ Model runner OK")

        gc.collect()  # [F9] limpieza del proceso padre (liviana, el hijo ya liberó lo pesado)

        # ── 4. EVALUATOR ──────────────────────────────────
        # [F12] Proceso hijo separado — mismo motivo que el paso 3.
        if _evaluator_done_today(DATA_PATH):
            logger.info("⚡ [4/10] Evaluator SKIP — evaluaciones de hoy existen")
        else:
            logger.info("📊 [4/10] Evaluator (proceso hijo)...")
            await _run_in_subprocess(_mp_run_evaluator, (), "evaluator")
            logger.info("✅ Evaluator OK")

        gc.collect()  # [F9] limpieza del proceso padre

        # ── 4.5 REGIME LEARNER ────────────────────────────
        logger.info("🧠 [4.5/10] Regime learner...")
        try:
            loop         = asyncio.get_event_loop()
            learn_result = await loop.run_in_executor(None, run_learning_cycle)
            logger.info(f"✅ Learner OK | regime={learn_result.get('regime')} outcome={learn_result.get('outcome')}")
        except Exception as e:
            logger.warning(f"⚠️ Learner falló (no crítico): {e}")

        gc.collect()  # [F9] liberar learner

        # ── 4.6 INTRADAY EVALUATOR ────────────────────────
        logger.info("📊 [4.6/10] Intraday evaluator...")
        try:
            from intraday_evaluator import run_intraday_evaluator
            loop     = asyncio.get_event_loop()
            eval_out = await loop.run_in_executor(None, run_intraday_evaluator)
            if eval_out.get("skipped"):
                logger.info(f"⚡ Intraday evaluator SKIP | {eval_out.get('reason')}")
            else:
                logger.info(
                    f"✅ Intraday evaluator OK | "
                    f"entry_hr={eval_out.get('entry_hit_rate', 0):.1%} | "
                    f"pos_hr={eval_out.get('position_hit_rate', 0):.1%} | "
                    f"min_score={eval_out.get('min_score_actual')}"
                )
        except Exception as e:
            logger.warning(f"⚠️ Intraday evaluator falló (no crítico): {e}")

        gc.collect()  # [F9] liberar intraday evaluator

        # ── 5. MARKET QUANT ───────────────────────────────
        logger.info("📉 [5/10] Market quantitative...")
        quant_ctx = run_market_state()
        logger.info(f"✅ Market quant OK | regime={quant_ctx.regime}")

        gc.collect()  # [F9] liberar market quant

        # ── 6. MARKET QUALITATIVE ─────────────────────────
        logger.info("🧠 [6/10] Market qualitative...")
        qual_ctx = evaluate_qualitative_market()
        logger.info(f"✅ Market qual OK | impact={qual_ctx.impact_score:.3f}")

        gc.collect()  # [F9] liberar market qual

        # ── 7. MARKET ORCHESTRATOR ────────────────────────
        logger.info("🧭 [7/10] Market orchestration...")
        market_orch = MarketOrchestrator()
        market_ctx  = market_orch.evaluate(quant_ctx.to_dict(), qual_ctx.to_dict())
        logger.info(f"🎯 MARKET MODE = {market_ctx.market_mode.upper()} (conf {market_ctx.confidence:.2f})")

        # Liberar objetos intermedios ya no necesarios
        del quant_ctx, qual_ctx, market_orch
        gc.collect()  # [F9] CRÍTICO — liberar antes del alpha engine

        # ── 8. ALPHA ENGINE ───────────────────────────────
        # [F9] Smart skip: si ya corrió hoy (pipeline apertura) reutilizar resultado
        # [F12] Proceso hijo separado — mismo motivo que pasos 3 y 4.
        if _alpha_done_today(DATA_PATH):
            logger.info("⚡ [8/10] Alpha SKIP — ya corrió hoy")
            alpha_out = json.loads((DATA_PATH / "alpha_last.json").read_text())
        else:
            logger.info("🧠 [8/10] Alpha engine (proceso hijo)...")
            tickers_file = DATA_PATH / "tickers.json"
            tickers      = json.loads(tickers_file.read_text())
            await _run_in_subprocess(_mp_run_alpha_engine, (tickers,), "alpha_engine")
            alpha_out = json.loads((DATA_PATH / "alpha_last.json").read_text())
            logger.info(f"✅ Alpha OK | valid={alpha_out.get('valid_alphas')} universe={alpha_out.get('universe_size')}")

        gc.collect()  # [F9] limpieza del proceso padre

        # ── 9. TRADING ORCHESTRATOR ───────────────────────
        logger.info("🤖 [9/10] Trading orchestrator...")
        trading_orch = TradingOrchestrator()
        trade_out    = await trading_orch.run(market_ctx=market_ctx.to_dict())
        logger.info(f"✅ Trading OK | mode={trade_out.get('mode')} decisions={len(trade_out.get('decisions', []))}")

        gc.collect()  # [F9] liberar trading orchestrator

        # ── 10. COMMIT ────────────────────────────────────
        # [AUD-P4][2026-09-04] Antes: se armaba commit_payload y se
        # mandaba por httpx.post a /internal/pipeline/commit (self-HTTP).
        # Ahora: se escribe directo a disco, mismo proceso, mismo
        # patrón .tmp + .replace() que usa el screener en el paso 1.
        # No depende de que el contenedor pueda llamarse a sí mismo
        # por red — elimina el punto de falla que empezó el 21-ago
        # (cambio de deploy nativo → Docker).
        logger.info("💾 [10/10] Committing pipeline results...")

        commit_payload = {
            "screener":   screener_out,
            "market_ctx": market_ctx.to_dict(),
            "audit": {
                "timestamp":   datetime.utcnow().isoformat(),
                "market_mode": market_ctx.market_mode,
                "confidence":  market_ctx.confidence,
                "decisions":   len(trade_out.get("decisions", [])),
            },
        }

        try:
            market_ctx_file = DATA_PATH / "market_context.json"
            market_ctx_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_mc = market_ctx_file.with_suffix(".tmp")
            tmp_mc.write_text(json.dumps(commit_payload["market_ctx"], indent=2))
            tmp_mc.replace(market_ctx_file)

            screener_commit_file = DATA_PATH / "screener_candidates.json"
            tmp_sc = screener_commit_file.with_suffix(".tmp")
            tmp_sc.write_text(json.dumps(commit_payload["screener"], indent=2))
            tmp_sc.replace(screener_commit_file)

            pipeline_audit_file = DATA_PATH / "last_pipeline.json"
            tmp_pa = pipeline_audit_file.with_suffix(".tmp")
            tmp_pa.write_text(json.dumps(commit_payload, indent=2))
            tmp_pa.replace(pipeline_audit_file)
        except Exception as e:
            raise RuntimeError(f"Pipeline commit failed (escritura directa): {e}")

        logger.info("✅ Pipeline committed (escritura directa, sin HTTP)")

        gc.collect()  # [F9] liberar después del commit

        # ── 10.5 INTRADAY TRACKER ─────────────────────────
        # [F11] v3.0: run_intraday_tracker() eliminado.
        # El tracker ya no genera universo propio — recibe la lista
        # exacta de posiciones abiertas del portfolio actual.
        logger.info("📡 [10.5/10] Intraday tracker...")
        try:
            from intraday_tracker import evaluar_posiciones_abiertas
            from portfolio_store import load_positions

            loop             = asyncio.get_event_loop()
            tickers_abiertos = [p["ticker"] for p in load_positions()]

            if tickers_abiertos:
                tracker_out = await loop.run_in_executor(
                    None, evaluar_posiciones_abiertas, tickers_abiertos
                )
                cerrar   = [t for t, s in tracker_out.items() if s.get("sugerencia") == "CERRAR"]
                trailing = [t for t, s in tracker_out.items() if s.get("sugerencia") == "TRAILING"]
                mantener = [t for t, s in tracker_out.items() if s.get("sugerencia") == "MANTENER"]
                logger.info(
                    f"✅ Tracker OK | posiciones={len(tracker_out)} "
                    f"cerrar={cerrar} trailing={trailing} mantener={mantener}"
                )
            else:
                logger.info("✅ Tracker OK | sin posiciones abiertas")

        except Exception as e:
            logger.warning(f"⚠️ Intraday tracker falló (no crítico): {e}")

    except Exception as e:
        logger.error("❌ PIPELINE FAILED")
        logger.error(str(e))
        traceback.print_exc()

    finally:
        # [F10] Liberar lock siempre, incluso si el pipeline falló
        _pipeline_running = False
        gc.collect()  # [F9] limpieza final siempre
        end_ts = datetime.utcnow().isoformat()
        logger.info("=" * 60)
        logger.info(f"🏁 PIPELINE END | {end_ts}")
        logger.info("=" * 60)


# =========================================================
# ENDPOINT
# =========================================================

@router.post("/internal/pipeline/run")
async def run_pipeline(request: Request):
    global _pipeline_running

    if request.headers.get("X-PIPELINE-KEY") != os.getenv("PIPELINE_KEY"):
        raise HTTPException(403, "Invalid pipeline key")

    # [F10] Si ya hay un pipeline corriendo, ignorar el request
    if _pipeline_running:
        logger.warning("⚠️ [F10] Pipeline ya en ejecución — request ignorado")
        return {
            "status":    "skipped",
            "reason":    "pipeline_already_running",
            "timestamp": datetime.utcnow().isoformat(),
        }

    # [F10] Marcar como activo ANTES de lanzar la tarea
    _pipeline_running = True
    asyncio.create_task(_run_pipeline_logic(request))

    return {
        "status":    "accepted",
        "timestamp": datetime.utcnow().isoformat(),
    }

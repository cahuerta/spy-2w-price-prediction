# =========================================================
# pipeline_router.py — PIPELINE ROUTER v2.10
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
# v2.9: [F15] (2026-09-14) BATCHING de model_runner.
#        Con [F12] ya aislado, el timeout de 20 min del paso 3 igual
#        se cumplió una vez: model_runner se fue poniendo cada vez
#        más lento por ticker (memory thrashing DENTRO de su propio
#        proceso hijo) hasta no terminar el universo completo a
#        tiempo. gc.collect() por ticker [MR5 en model_runner.py] no
#        basta cuando el mismo proceso vive el tiempo suficiente para
#        procesar ~700+ tickers seguidos.
#        Fix: el paso 3 ya no lanza un solo proceso hijo con todo el
#        universo — lo divide en lotes de MODEL_RUNNER_BATCH_SIZE
#        tickers (default 50) y corre un proceso hijo por lote, uno
#        detrás de otro. Cada proceso hijo vive poco tiempo y el
#        sistema operativo le recupera el 100% de su memoria al
#        terminar, antes de que arranque el siguiente lote — ningún
#        proceso individual llega a acumular memoria suficiente para
#        ponerse lento. El timeout de 20 min ahora aplica POR LOTE,
#        no al total, así un lote lento no mata el progreso de los
#        lotes que ya terminaron bien.
# v2.10: [F16] (2026-09-17) AISLAR SCREENER — crash OOM seguía
#        ocurriendo en la apertura pese a [F12]/[F15], siempre en la
#        misma ventana horaria (11:30-11:40 Chile), confirmado por
#        Render Events varios días seguidos. El screener (paso 1,
#        primero del pipeline) nunca había sido aislado: hace ~100
#        llamadas secuenciales a yf.download() (yfinance) dentro del
#        proceso principal de FastAPI, sin gc.collect() alguno.
#        yfinance es conocido por retener objetos de sesión HTTP
#        entre descargas sucesivas dentro del mismo proceso, sin que
#        gc.collect() garantice que el sistema operativo recupere esa
#        memoria — mismo motivo de fondo que llevó a aislar
#        model_runner/evaluator/alpha_engine en [F12]. Se descartó
#        agregar gc.collect() periódico como fix (ya se confirmó en
#        [F12] que no resuelve el problema de fondo con
#        pandas/numpy/glibc) y se aplicó el mismo patrón que ya
#        funciona: el screener completo corre en su propio proceso
#        hijo. A diferencia de model_runner, no se dividió en lotes
#        por ahora — el screener trabaja con ~100 tickers (vs ~700+
#        de model_runner, que sí necesitó lotes) — si la evidencia
#        de próximas corridas muestra que un solo proceso sigue
#        acercándose al límite, se divide en lotes con el mismo
#        patrón de [F15].
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

def _mp_run_screener(result_path: str):
    """
    [F16] Corre el screener completo (fetch de yfinance para ~100
    tickers + engine de scoring) en un proceso hijo separado.
    yfinance es conocido por retener objetos de sesión HTTP entre
    llamadas sucesivas a yf.download() dentro del mismo proceso, sin
    que gc.collect() garantice su liberación real al sistema
    operativo — mismo motivo de fondo que llevó a aislar model_runner,
    evaluator y alpha_engine [F12]. Escribe su resultado a un JSON
    temporal, ya que run_screener_async() es async y retorna un
    objeto (no escribe a disco por su cuenta como los otros pasos).

    [MEMDIAG][2026-09-22] Mide RSS real antes/después — en vez de
    seguir adivinando por lectura de código cuál paso sube memoria.
    """
    from mem_diag import log_mem, log_mem_delta, quiet_logs
    quiet_logs()
    rss0 = log_mem("screener — inicio")
    import asyncio
    from screener import run_screener_async
    result = asyncio.run(run_screener_async())
    Path(result_path).write_text(json.dumps(result))
    log_mem_delta("screener — fin", rss0)


def _mp_run_model_runner_batch(tickers: list):
    """[F15] Corre un LOTE de tickers, no el universo completo.
    [MEMDIAG] Mide RSS real antes/después de cada lote."""
    from mem_diag import log_mem, log_mem_delta, quiet_logs
    quiet_logs()
    rss0 = log_mem(f"model_runner lote ({len(tickers)} tickers) — inicio")
    from model_runner import run_models_for_tickers
    run_models_for_tickers(tickers)
    log_mem_delta(f"model_runner lote ({len(tickers)} tickers) — fin", rss0)


def _batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _mp_run_evaluator():
    """[MEMDIAG] Mide RSS real antes/después."""
    from mem_diag import log_mem, log_mem_delta, quiet_logs
    quiet_logs()
    rss0 = log_mem("evaluator — inicio")
    from evaluator import evaluate_all
    evaluate_all()
    log_mem_delta("evaluator — fin", rss0)


def _mp_run_alpha_engine(tickers: list):
    """[MEMDIAG] Mide RSS real antes/después."""
    from mem_diag import log_mem, log_mem_delta, quiet_logs
    quiet_logs()
    rss0 = log_mem(f"alpha_engine ({len(tickers)} tickers) — inicio")
    from alpha_engine_v4 import compute_and_persist_alpha
    compute_and_persist_alpha(tickers)
    log_mem_delta(f"alpha_engine ({len(tickers)} tickers) — fin", rss0)


# Contexto "spawn": el proceso hijo arranca limpio, sin heredar
# el loop de asyncio ni conexiones abiertas del proceso padre
# (FastAPI/Uvicorn). Con "fork" (default en Linux) el hijo hereda
# todo el estado del padre y puede colgarse o corromper el loop.
_mp_ctx = multiprocessing.get_context("spawn")

# [F15] Tamaño de lote para model_runner — cada lote corre en su
# propio proceso hijo, uno detrás de otro. Ajustable por env var sin
# tocar código si 50 resulta muy alto o muy bajo para 512MB.
MODEL_RUNNER_BATCH_SIZE = int(os.getenv("MODEL_RUNNER_BATCH_SIZE", "50"))

# [F15] Timeout por LOTE (no por el total de tickers). 10 min por
# defecto — un lote de 50 tickers nunca debería acercarse a esto si
# el batching está funcionando como se espera.
MODEL_RUNNER_BATCH_TIMEOUT_SEC = int(os.getenv("MODEL_RUNNER_BATCH_TIMEOUT_SEC", str(10 * 60)))


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
        # [F16] Corre en proceso hijo separado — yfinance (100
        # tickers, fetch secuencial) retiene memoria de sesión HTTP
        # que gc.collect() no garantiza liberar dentro del mismo
        # proceso. Aislado, el sistema operativo recupera el 100% de
        # su memoria cuando el proceso hijo termina.
        if _screener_done_today(DATA_PATH):
            screener_file = DATA_PATH / "screener_candidates.json"
            screener_out  = json.loads(screener_file.read_text())
            logger.info(f"⚡ [1/10] Screener SKIP | candidates={screener_out.get('n_candidates')}")
        else:
            logger.info("🔍 [1/10] Screener (proceso hijo)...")
            screener_result_path = str(DATA_PATH / f"tmp_screener_result_{os.getpid()}.json")
            await _run_in_subprocess(_mp_run_screener, (screener_result_path,), "screener")

            screener_result_file = Path(screener_result_path)
            if not screener_result_file.exists():
                raise RuntimeError("screener: proceso hijo terminó pero no dejó resultado")
            screener_out = json.loads(screener_result_file.read_text())
            screener_result_file.unlink(missing_ok=True)

            screener_file = DATA_PATH / "screener_candidates.json"
            screener_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = screener_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(screener_out, indent=2))
            tmp.replace(screener_file)
            logger.info(f"✅ Screener OK | candidates={screener_out.get('n_candidates')}")

        gc.collect()  # [F9] limpieza del proceso padre (liviana, el hijo ya liberó lo pesado)

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
        # [F12][F15] Corre en lotes, cada uno en su propio proceso
        # hijo separado — libera 100% de su memoria al sistema
        # operativo cuando termina cada lote, antes de arrancar el
        # siguiente. Evita que un solo proceso larguísimo (~700+
        # tickers) acumule memoria hasta ponerse lento y exceder el
        # timeout, como ocurrió con un solo proceso para todo el
        # universo.
        if _models_done_today(DATA_PATH):
            pred_count = sum(1 for _ in (DATA_PATH / "predictions").glob(f"**/{today}.json"))
            logger.info(f"⚡ [3/10] Models SKIP | {pred_count} predicciones de hoy")
        else:
            logger.info("📈 [3/10] Model runner (por lotes, cada uno en proceso hijo)...")
            all_tickers = json.loads((DATA_PATH / "tickers.json").read_text())
            if isinstance(all_tickers, dict) and "tickers" in all_tickers:
                all_tickers = all_tickers["tickers"]

            lotes = list(_batched(all_tickers, MODEL_RUNNER_BATCH_SIZE))
            logger.info(f"   {len(all_tickers)} tickers en {len(lotes)} lotes de hasta {MODEL_RUNNER_BATCH_SIZE}")

            for i, lote in enumerate(lotes, start=1):
                logger.info(f"   📦 Lote {i}/{len(lotes)} ({len(lote)} tickers)...")
                try:
                    await _run_in_subprocess(
                        _mp_run_model_runner_batch, (lote,), f"model_runner_lote_{i}",
                        timeout_sec=MODEL_RUNNER_BATCH_TIMEOUT_SEC,
                    )
                except Exception as e:
                    # [F15] Un lote que falla o se cuelga no debe tumbar
                    # los lotes siguientes — se loguea y se continúa.
                    logger.error(f"❌ Lote {i}/{len(lotes)} falló: {e} — continuando con el siguiente lote")
                gc.collect()  # limpieza del proceso padre entre lotes

            logger.info("✅ Model runner OK (todos los lotes procesados)")

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

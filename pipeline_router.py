# =========================================================
# pipeline_router.py — PIPELINE ROUTER v2.6
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
# =========================================================

from fastapi import APIRouter, HTTPException, Request
from datetime import datetime, timezone
import logging
import traceback
import os
import gc
import httpx
import asyncio
from pathlib import Path
import json

from screener import run_screener_async
from decider import run_decider
from model_runner import run_all_models
from evaluator import evaluate_all
from alpha_engine_v4 import compute_and_persist_alpha

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
        if _models_done_today(DATA_PATH):
            pred_count = sum(1 for _ in (DATA_PATH / "predictions").glob(f"**/{today}.json"))
            logger.info(f"⚡ [3/10] Models SKIP | {pred_count} predicciones de hoy")
        else:
            logger.info("📈 [3/10] Model runner...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_all_models)
            logger.info("✅ Model runner OK")

        gc.collect()  # [F9] CRÍTICO — liberar los 87 modelos antes del evaluator

        # ── 4. EVALUATOR ──────────────────────────────────
        if _evaluator_done_today(DATA_PATH):
            logger.info("⚡ [4/10] Evaluator SKIP — evaluaciones de hoy existen")
        else:
            logger.info("📊 [4/10] Evaluator...")
            evaluate_all()
            logger.info("✅ Evaluator OK")

        gc.collect()  # [F9] CRÍTICO — liberar evaluaciones antes del alpha engine

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
        if _alpha_done_today(DATA_PATH):
            logger.info("⚡ [8/10] Alpha SKIP — ya corrió hoy")
            alpha_out = json.loads((DATA_PATH / "alpha_last.json").read_text())
        else:
            logger.info("🧠 [8/10] Alpha engine...")
            tickers_file = DATA_PATH / "tickers.json"
            tickers      = json.loads(tickers_file.read_text())
            alpha_out    = compute_and_persist_alpha(tickers)
            logger.info(f"✅ Alpha OK | valid={alpha_out.get('valid_alphas')} universe={alpha_out.get('universe_size')}")

        gc.collect()  # [F9] CRÍTICO — liberar alpha antes del trading

        # ── 9. TRADING ORCHESTRATOR ───────────────────────
        logger.info("🤖 [9/10] Trading orchestrator...")
        trading_orch = TradingOrchestrator()
        trade_out    = await trading_orch.run(market_ctx=market_ctx.to_dict())
        logger.info(f"✅ Trading OK | mode={trade_out.get('mode')} decisions={len(trade_out.get('decisions', []))}")

        gc.collect()  # [F9] liberar trading orchestrator

        # ── 10. COMMIT ────────────────────────────────────
        logger.info("💾 [10/10] Committing pipeline results...")
        PIPELINE_KEY = os.getenv("PIPELINE_KEY")
        BASE_URL     = str(request.base_url).rstrip("/")

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

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/internal/pipeline/commit",
                json=commit_payload,
                headers={"X-PIPELINE-KEY": PIPELINE_KEY},
                timeout=30,
            )

        if resp.status_code != 200:
            raise RuntimeError(f"Pipeline commit failed: {resp.status_code} {resp.text}")

        logger.info("✅ Pipeline committed")

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
    

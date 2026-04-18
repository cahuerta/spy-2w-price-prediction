# =========================================================
# pipeline_router.py — PIPELINE ROUTER v2.2
# =========================================================
# v2.2:
#   [F7] Intraday tracker integrado como paso 10.5
#        Corre DESPUÉS del commit — siempre tiene predicciones
#        frescas. No es crítico: si falla el pipeline continúa.
# =========================================================

from fastapi import APIRouter, HTTPException, Request
from datetime import datetime, timezone
import logging
import traceback
import os
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


# =========================================================
# BACKGROUND PIPELINE
# =========================================================

async def _run_pipeline_logic(request: Request):

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

        # ── 3. MODEL RUNNER ───────────────────────────────
        if _models_done_today(DATA_PATH):
            pred_count = sum(1 for _ in (DATA_PATH / "predictions").glob(f"**/{today}.json"))
            logger.info(f"⚡ [3/10] Models SKIP | {pred_count} predicciones de hoy")
        else:
            logger.info("📈 [3/10] Model runner...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_all_models)
            logger.info("✅ Model runner OK")

        # ── 4. EVALUATOR ──────────────────────────────────
        if _evaluator_done_today(DATA_PATH):
            logger.info("⚡ [4/10] Evaluator SKIP — evaluaciones de hoy existen")
        else:
            logger.info("📊 [4/10] Evaluator...")
            evaluate_all()
            logger.info("✅ Evaluator OK")

        # ── 4.5 REGIME LEARNER ────────────────────────────
        logger.info("🧠 [4.5/10] Regime learner...")
        try:
            loop         = asyncio.get_event_loop()
            learn_result = await loop.run_in_executor(None, run_learning_cycle)
            logger.info(f"✅ Learner OK | regime={learn_result.get('regime')} outcome={learn_result.get('outcome')}")
        except Exception as e:
            logger.warning(f"⚠️ Learner falló (no crítico): {e}")

        # ── 5. MARKET QUANT ───────────────────────────────
        logger.info("📉 [5/10] Market quantitative...")
        quant_ctx = run_market_state()
        logger.info(f"✅ Market quant OK | regime={quant_ctx.regime}")

        # ── 6. MARKET QUALITATIVE ─────────────────────────
        logger.info("🧠 [6/10] Market qualitative...")
        qual_ctx = evaluate_qualitative_market()
        logger.info(f"✅ Market qual OK | impact={qual_ctx.impact_score:.3f}")

        # ── 7. MARKET ORCHESTRATOR ────────────────────────
        logger.info("🧭 [7/10] Market orchestration...")
        market_orch = MarketOrchestrator()
        market_ctx  = market_orch.evaluate(quant_ctx.to_dict(), qual_ctx.to_dict())
        logger.info(f"🎯 MARKET MODE = {market_ctx.market_mode.upper()} (conf {market_ctx.confidence:.2f})")

        # ── 8. ALPHA ENGINE ───────────────────────────────
        logger.info("🧠 [8/10] Alpha engine...")
        tickers_file = DATA_PATH / "tickers.json"
        tickers      = json.loads(tickers_file.read_text())
        alpha_out    = compute_and_persist_alpha(tickers)
        logger.info(f"✅ Alpha OK | valid={alpha_out.get('valid_alphas')} universe={alpha_out.get('universe_size')}")

        # ── 9. TRADING ORCHESTRATOR ───────────────────────
        logger.info("🤖 [9/10] Trading orchestrator...")
        trading_orch = TradingOrchestrator()
        trade_out    = await trading_orch.run(market_ctx=market_ctx.to_dict())
        logger.info(f"✅ Trading OK | mode={trade_out.get('mode')} decisions={len(trade_out.get('decisions', []))}")

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

        # ── 10.5 INTRADAY TRACKER ─────────────────────────
        # [F7] Corre DESPUÉS del commit — predicciones ya están en disco.
        # No es crítico: si falla el pipeline ya terminó correctamente.
        logger.info("📡 [10.5/10] Intraday tracker...")
        try:
            from intraday_tracker import run_intraday_tracker
            loop         = asyncio.get_event_loop()
            tracker_out  = await loop.run_in_executor(None, run_intraday_tracker)
            entrar       = tracker_out.get("entrar_ahora", [])
            n_candidatos = tracker_out.get("candidatos", 0)
            n_posiciones = len(tracker_out.get("monitor_posiciones", []))
            logger.info(
                f"✅ Tracker OK | candidatos={n_candidatos} "
                f"entrar={entrar} posiciones={n_posiciones}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Intraday tracker falló (no crítico): {e}")

    except Exception as e:
        logger.error("❌ PIPELINE FAILED")
        logger.error(str(e))
        traceback.print_exc()

    finally:
        end_ts = datetime.utcnow().isoformat()
        logger.info("=" * 60)
        logger.info(f"🏁 PIPELINE END | {end_ts}")
        logger.info("=" * 60)


# =========================================================
# ENDPOINT
# =========================================================

@router.post("/internal/pipeline/run")
async def run_pipeline(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != os.getenv("PIPELINE_KEY"):
        raise HTTPException(403, "Invalid pipeline key")

    asyncio.create_task(_run_pipeline_logic(request))

    return {
        "status":    "accepted",
        "timestamp": datetime.utcnow().isoformat(),
        }
        

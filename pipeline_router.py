# =========================================================
# pipeline_router.py — PIPELINE ROUTER (PRODUCCIÓN)
# =========================================================
# ✔ Responde inmediato al cron
# ✔ Ejecuta pipeline en background
# ✔ NO rompe lógica existente
# ✔ Trading intacto
# =========================================================

from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
import logging
import traceback
import os
import httpx
import asyncio
from pathlib import Path
import json
import os

# =========================
# IMPORTS — CAPAS REALES
# =========================
from screener import run_screener_async
from decider import run_decider
from model_runner import run_all_models
from evaluator import evaluate_all
from alpha_engine_v4 import compute_and_persist_alpha

from market_state_evaluator import run_market_state
from market_qualitative_evaluator import evaluate_qualitative_market
from market_orchestrator import MarketOrchestrator
from trading_orchestrator import TradingOrchestrator

# =========================
# ROUTER
# =========================
router = APIRouter()
logger = logging.getLogger("pipeline")


# =========================================================
# BACKGROUND PIPELINE LOGIC
# =========================================================
async def _run_pipeline_logic(request: Request):

    start_ts = datetime.utcnow().isoformat()
    logger.info("=" * 60)
    logger.info(f"🚀 PIPELINE START (BG) | {start_ts}")
    logger.info("=" * 60)

    try:
        # -------------------------------------------------
        # 1️⃣ SCREENER
        # -------------------------------------------------
        logger.info("🔍 [1/8] Screener...")
        screener_out = await run_screener_async()
        # ✅ CREAR ARCHIVO REQUERIDO POR DECIDER

        DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
        screener_file = DATA_PATH / "screener_candidates.json"

        screener_file.parent.mkdir(parents=True, exist_ok=True)

        tmp = screener_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(screener_out, indent=2))
        tmp.replace(screener_file)
        logger.info(
            f"✅ Screener OK | candidates={screener_out.get('n_candidates')}"
        )

        # -------------------------------------------------
        # 2️⃣ DECIDER
        # -------------------------------------------------
        logger.info("🧠 [2/8] Decider...")
        decider_out = run_decider()
        logger.info(
            f"✅ Decider OK | added={len(decider_out.get('added', []))} "
            f"| total={decider_out.get('total')}"
        )

        # -------------------------------------------------
        # 3️⃣ MODEL RUNNER
        # -------------------------------------------------
        logger.info("📈 [3/8] Model runner...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_all_models)
        logger.info("✅ Model runner OK")

        # -------------------------------------------------
        # 4️⃣ EVALUATOR
        # -------------------------------------------------
        logger.info("📊 [4/8] Evaluator...")
        evaluate_all()
        logger.info("ℹ️ Evaluator executed")

     
        # -------------------------------------------------
        # 5️⃣ MARKET QUANT
        # -------------------------------------------------
        logger.info("📉 [5/8] Market quantitative context...")
        quant_ctx = run_market_state()
        logger.info(f"✅ Market quant OK | regime={quant_ctx.regime}")

        # -------------------------------------------------
        # 6️⃣ MARKET QUALITATIVE
        # -------------------------------------------------
        logger.info("🧠 [6/8] Market qualitative context...")
        qual_ctx = evaluate_qualitative_market()
        logger.info(
            f"✅ Market qual OK | impact={qual_ctx.impact_score:.3f} "
            f"| conf={qual_ctx.aggregated_confidence:.2f}"
        )
        
        
        # -------------------------------------------------
        # 7️⃣ MARKET ORCHESTRATOR
        # -------------------------------------------------
        logger.info("🧭 [7/8] Market orchestration...")
        market_orch = MarketOrchestrator()
        market_ctx = market_orch.evaluate(
            quant_ctx.to_dict(),
            qual_ctx.to_dict(),
        )

        logger.info(
            f"🎯 MARKET MODE = {market_ctx.market_mode.upper()} "
            f"(conf {market_ctx.confidence:.2f})"
        )


        # -------------------------------------------------
        # 5️⃣ ALPHA ENGINE
        # -------------------------------------------------
        logger.info("🧠 [5/9] Alpha engine...")

        tickers_file = DATA_PATH / "tickers.json"
        tickers = json.loads(tickers_file.read_text())

        alpha_out = compute_and_persist_alpha(tickers)

        logger.info(
            f"✅ Alpha engine OK | valid_alphas={alpha_out.get('valid_alphas')} "
            f"| universe={alpha_out.get('universe_size')}"
        )   
        
        # -------------------------------------------------
        # 8️⃣ TRADING ORCHESTRATOR
        # -------------------------------------------------
        logger.info("🤖 [8/8] Trading orchestrator...")
        trading_orch = TradingOrchestrator()

        trade_out = await trading_orch.run(
            market_ctx=market_ctx.to_dict()
        )

        logger.info(
            f"✅ Trading OK | mode={trade_out.get('mode')} "
            f"| decisions={len(trade_out.get('decisions', []))}"
        )

        # -------------------------------------------------
        # 9️⃣ COMMIT A DISCO
        # -------------------------------------------------
        logger.info("💾 [9/9] Committing pipeline results...")

        PIPELINE_KEY = os.getenv("PIPELINE_KEY")
        BASE_URL = str(request.base_url).rstrip("/")

        commit_payload = {
            "screener": screener_out,
            "market_ctx": market_ctx.to_dict(),
            "audit": {
                "timestamp": datetime.utcnow().isoformat(),
                "market_mode": market_ctx.market_mode,
                "confidence": market_ctx.confidence,
                "decisions": len(trade_out.get("decisions", [])),
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
            raise RuntimeError(
                f"Pipeline commit failed: {resp.status_code} {resp.text}"
            )

        logger.info("✅ Pipeline committed to disk")

    except Exception as e:
        logger.error("❌ PIPELINE FAILED")
        logger.error(str(e))
        traceback.print_exc()

    finally:
        end_ts = datetime.utcnow().isoformat()
        logger.info("=" * 60)
        logger.info(f"🏁 PIPELINE END (BG) | {end_ts}")
        logger.info("=" * 60)


# =========================================================
# PIPELINE ENDPOINT (RESPUESTA INMEDIATA)
# =========================================================
@router.post("/internal/pipeline/run")
async def run_pipeline(request: Request):

    if request.headers.get("X-PIPELINE-KEY") != os.getenv("PIPELINE_KEY"):
        raise HTTPException(403, "Invalid pipeline key")

    # Lanza pipeline en background
    asyncio.create_task(_run_pipeline_logic(request))

    # Responde inmediato al cron
    return {
        "status": "accepted",
        "timestamp": datetime.utcnow().isoformat()
        }

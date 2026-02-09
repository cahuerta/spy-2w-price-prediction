# =========================================================
# pipeline_router.py — PIPELINE ROUTER (PRODUCCIÓN)
# =========================================================
# ✔ Vive dentro de Main (FastAPI)
# ✔ Ejecuta pipeline completo vía HTTP
# ✔ Permisos de escritura heredados de Main
# ✔ Orden EXACTO del pipeline_daily.py
# ✔ Un solo punto de activación (cron → HTTP)
# =========================================================

from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
import logging
import traceback

# =========================
# IMPORTS — CAPAS REALES
# =========================
from screener import run_screener_async
from decider import run_decider
from model_runner import run_all_models
from evaluator import evaluate_all

from market_state_evaluator import evaluate_quant_market
from market_qualitative_evaluator import evaluate_qualitative_market
from market_orchestrator import MarketOrchestrator
from trading_orchestrator import TradingOrchestrator

# =========================
# ROUTER
# =========================
router = APIRouter()
logger = logging.getLogger("pipeline")

# =========================
# PIPELINE ENDPOINT
# =========================
@router.post("/internal/pipeline/run")
async def run_pipeline(request: Request):
    """
    Ejecuta el pipeline completo.
    Cron llama a ESTE endpoint.
    Main es el dueño de permisos y disco.
    """

    start_ts = datetime.utcnow().isoformat()
    logger.info("=" * 60)
    logger.info(f"🚀 PIPELINE START (HTTP) | {start_ts}")
    logger.info("=" * 60)

    try:
        # -------------------------------------------------
        # 1️⃣ SCREENER
        # -------------------------------------------------
        logger.info("🔍 [1/8] Screener...")
        screener_out = await run_screener_async()
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
        run_all_models()
        logger.info("✅ Model runner OK")

        # 4️⃣ EVALUATOR (side-effect only)
logger.info("📊 [4/8] Evaluator (background)...")
evaluate_all()
logger.info("ℹ️ Evaluator executed (no downstream dependency)") 

        # -------------------------------------------------
        # 5️⃣ MARKET QUANT
        # -------------------------------------------------
        logger.info("📉 [5/8] Market quantitative context...")
        quant_ctx = evaluate_quant_market(
            prices_main=eval_out.get("prices_main"),
            prices_cross=eval_out.get("prices_cross"),
        )
        logger.info(f"✅ Market quant OK | regime={quant_ctx.regime}")

        # -------------------------------------------------
        # 6️⃣ MARKET QUALITATIVE (IA)
        # -------------------------------------------------
        logger.info("🧠 [6/8] Market qualitative context...")
        qual_ctx = evaluate_qualitative_market(
            quant_ctx.to_dict(),
            news_summary=eval_out.get("news_summary"),
        )
        logger.info(
            f"✅ Market qual OK | bias={qual_ctx.macro_bias} "
            f"| conf={qual_ctx.confidence:.2f}"
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
        # 8️⃣ TRADING ORCHESTRATOR
        # -------------------------------------------------
        logger.info("🤖 [8/8] Trading orchestrator...")
        trading_orch = TradingOrchestrator()

        trade_out = trading_orch.run(
            market_ctx=market_ctx.to_dict(),
            eval_out=eval_out,
        )

        logger.info(
            f"✅ Trading OK | mode={trade_out.get('mode')} "
            f"| decisions={len(trade_out.get('decisions', []))}"
        )

        # -------------------------------------------------
        # RESPONSE (MAIN DECIDE QUÉ PERSISTIR)
        # -------------------------------------------------
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "market_mode": market_ctx.market_mode,
            "confidence": market_ctx.confidence,
            "decisions": len(trade_out.get("decisions", [])),
        }

    except Exception as e:
        logger.error("❌ PIPELINE FAILED")
        logger.error(str(e))
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    finally:
        end_ts = datetime.utcnow().isoformat()
        logger.info("=" * 60)
        logger.info(f"🏁 PIPELINE END | {end_ts}")
        logger.info("=" * 60)

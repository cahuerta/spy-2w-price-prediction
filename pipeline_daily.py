# =========================================================
# pipeline_daily.py — DAILY SYSTEM PIPELINE (PRODUCCIÓN)
# =========================================================
# ✔ Flujo secuencial y bloqueante
# ✔ Un solo cron diario
# ✔ NO escribe a disco
# ✔ Backend (Main) = fuente de verdad
# ✔ Fallo corta pipeline
# ✔ Trading 100% delegado al TradingOrchestrator
# =========================================================

import traceback
from datetime import datetime
import logging

# =========================
# IMPORTS — CAPAS REALES
# =========================
from screener import run_screener
from decider import run_decider
from model_runner import run_all_models
from evaluator import evaluate_all

from market_state_evaluator import evaluate_quant_market
from market_qualitative_evaluator import evaluate_qualitative_market
from market_orchestrator import MarketOrchestrator

from trading_orchestrator import TradingOrchestrator

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s"
)
logger = logging.getLogger("pipeline")

# =========================
# PIPELINE
# =========================
def main():
    start_ts = datetime.utcnow().isoformat()
    logger.info("=" * 60)
    logger.info(f"🚀 PIPELINE DAILY START | {start_ts}")
    logger.info("=" * 60)

    try:
        # -------------------------------------------------
        # 1️⃣ SCREENER
        # -------------------------------------------------
        logger.info("🔍 [1/8] Screener...")
        screener_out = run_screener()
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

        # -------------------------------------------------
        # 4️⃣ EVALUATOR
        # -------------------------------------------------
        logger.info("📊 [4/8] Evaluator...")
        eval_out = evaluate_all()
        logger.info(
            f"✅ Evaluator OK | evaluated={len(eval_out.get('evaluated', []))}"
        )

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
        # RETURN PAYLOAD (PADRE DECIDE QUÉ GRABAR)
        # -------------------------------------------------
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "screener": screener_out,
            "decider": decider_out,
            "evaluator": eval_out,
            "market_ctx": market_ctx.to_dict(),
            "trading": trade_out,
        }

    except Exception as e:
        logger.error("❌ PIPELINE FAILED")
        logger.error(str(e))
        traceback.print_exc()

        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }

    finally:
        end_ts = datetime.utcnow().isoformat()
        logger.info("=" * 60)
        logger.info(f"🏁 PIPELINE DAILY END | {end_ts}")
        logger.info("=" * 60)


# =========================
# ENTRYPOINT (CRON DIRECTO)
# =========================
if __name__ == "__main__":
    main()

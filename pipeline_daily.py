# =========================================================
# pipeline_daily.py — DAILY SYSTEM PIPELINE (PRODUCCIÓN)
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

from evaluador_cuantitativo_mercado import evaluate_quant_market
from market_qualitative_evaluator import evaluate_qualitative_market
from market_orchestrator import MarketOrchestrator
from orchestrator import run_orchestrator

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
            f"✅ Decider OK | added={len(decider_out['added'])} | total={decider_out['total']}"
        )

        # -------------------------------------------------
        # 3️⃣ MODEL RUNNER
        # -------------------------------------------------
        logger.info("📈 [3/8] Model runner...")
        run_all_models()
        logger.info("✅ Model OK | predictions guardadas en /data/predictions")

        # -------------------------------------------------
        # 4️⃣ EVALUATOR (DIRECTO, SIN WRAPPER)
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
            prices_cross=eval_out.get("prices_cross")
        )
        logger.info(
            f"✅ Market quant OK | regime={quant_ctx.regime}"
        )

        # -------------------------------------------------
        # 6️⃣ MARKET QUALITATIVE
        # -------------------------------------------------
        logger.info("🧠 [6/8] Market qualitative context...")
        qual_ctx = evaluate_qualitative_market(
            quant_ctx.to_dict(),
            news_summary=eval_out.get("news_summary")
        )
        logger.info(
            f"✅ Market qual OK | bias={qual_ctx.macro_bias} | conf={qual_ctx.confidence:.2f}"
        )

        # -------------------------------------------------
        # 7️⃣ MARKET ORCHESTRATOR
        # -------------------------------------------------
        logger.info("🧭 [7/8] Market orchestration...")
        market_orch = MarketOrchestrator()
        market_ctx = market_orch.evaluate(
            quant_ctx.to_dict(),
            qual_ctx.to_dict()
        )
        logger.info(
            f"🎯 MARKET MODE = {market_ctx.market_mode.upper()} "
            f"(conf {market_ctx.confidence:.2f})"
        )

        # -------------------------------------------------
        # 8️⃣ TRADING ORCHESTRATOR
        # -------------------------------------------------
        if market_ctx.market_mode != "defensive":
            logger.info("🤖 [8/8] Trading orchestrator...")
            run_orchestrator()
            logger.info("✅ Trading orchestrator OK")
        else:
            logger.warning("⛔ Trading BLOQUEADO por market_mode=DEFENSIVE")

    except Exception as e:
        logger.error("❌ PIPELINE FAILED")
        logger.error(str(e))
        traceback.print_exc()
        return

    end_ts = datetime.utcnow().isoformat()
    logger.info("=" * 60)
    logger.info(f"🏁 PIPELINE DAILY END | {end_ts}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

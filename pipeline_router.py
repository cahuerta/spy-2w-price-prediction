# =========================================================
# pipeline_router.py — PIPELINE ROUTER (PRODUCCIÓN)
# =========================================================
# ✔ Responde inmediato al cron
# ✔ Ejecuta pipeline en background
# ✔ NO rompe lógica existente
# ✔ Trading intacto
#
# FIXES:
#   [F1] run_all_models() ahora se awaita correctamente
#   [F2] Alpha engine movido DESPUÉS del Market Orchestrator
#   [F3] import os duplicado removido
#   [F4] Numeración de pasos corregida en logs
#
# FIX v2.0 — SMART SKIP:
#   [F5] Pasos 1-4 se saltean si ya existe output del día.
#
# FIX v2.1 — ADAPTIVE REGIME:
#   [F6] regime_threshold_learner.run_learning_cycle() se ejecuta
#        ANTES del paso 5 (market quant), para que los umbrales
#        actualizados se usen en el ciclo actual.
#   [F7] Import cambiado: market_quant_context (V3.3 adaptativo)
#        en vez de market_state_evaluator
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

# =========================
# IMPORTS — CAPAS REALES
# =========================
from screener import run_screener_async
from decider import run_decider
from model_runner import run_all_models
from evaluator import evaluate_all
from alpha_engine_v4 import compute_and_persist_alpha

from market_quant_context import run_market_state          # [F7] V3.3 adaptativo
from market_qualitative_evaluator import evaluate_qualitative_market
from market_orchestrator import MarketOrchestrator
from trading_orchestrator import TradingOrchestrator
from regime_threshold_learner import run_learning_cycle    # [F6] aprendizaje adaptativo

# =========================
# ROUTER
# =========================
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
# BACKGROUND PIPELINE LOGIC
# =========================================================
async def _run_pipeline_logic(request: Request):

    start_ts = datetime.utcnow().isoformat()
    logger.info("=" * 60)
    logger.info(f"🚀 PIPELINE START (BG) | {start_ts}")
    logger.info("=" * 60)

    try:
        DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
        today = _today_utc()

        # -------------------------------------------------
        # 1️⃣ SCREENER
        # -------------------------------------------------
        if _screener_done_today(DATA_PATH):
            screener_file = DATA_PATH / "screener_candidates.json"
            screener_out  = json.loads(screener_file.read_text())
            logger.info(
                f"⚡ [1/11] Screener SKIP — output de hoy existe "
                f"| candidates={screener_out.get('n_candidates')}"
            )
        else:
            logger.info("🔍 [1/11] Screener...")
            screener_out  = await run_screener_async()
            screener_file = DATA_PATH / "screener_candidates.json"
            screener_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = screener_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(screener_out, indent=2))
            tmp.replace(screener_file)
            logger.info(f"✅ Screener OK | candidates={screener_out.get('n_candidates')}")

        # -------------------------------------------------
        # 2️⃣ DECIDER
        # -------------------------------------------------
        if _decider_done_today(DATA_PATH):
            tickers_data = json.loads((DATA_PATH / "tickers.json").read_text())
            total = len(tickers_data) if isinstance(tickers_data, list) else tickers_data.get("total", "?")
            logger.info(f"⚡ [2/11] Decider SKIP — tickers.json de hoy existe | total={total}")
            decider_out = {"total": total, "added": []}
        else:
            logger.info("🧠 [2/11] Decider...")
            decider_out = run_decider()
            logger.info(
                f"✅ Decider OK | added={len(decider_out.get('added', []))} "
                f"| total={decider_out.get('total')}"
            )

        # -------------------------------------------------
        # 3️⃣ MODEL RUNNER
        # -------------------------------------------------
        if _models_done_today(DATA_PATH):
            pred_count = sum(1 for _ in (DATA_PATH / "predictions").glob(f"**/{today}.json"))
            logger.info(f"⚡ [3/11] Models SKIP — {pred_count} predicciones de hoy ya existen")
        else:
            logger.info("📈 [3/11] Model runner...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_all_models)
            logger.info("✅ Model runner OK")

        # -------------------------------------------------
        # 4️⃣ EVALUATOR
        # -------------------------------------------------
        if _evaluator_done_today(DATA_PATH):
            logger.info("⚡ [4/11] Evaluator SKIP — evaluaciones de hoy ya existen")
        else:
            logger.info("📊 [4/11] Evaluator...")
            evaluate_all()
            logger.info("✅ Evaluator executed")

        # -------------------------------------------------
        # 5️⃣ REGIME LEARNER — [F6] NUEVO
        # Aprende de los últimos 5 días ANTES de calcular
        # el régimen actual, para usar umbrales actualizados
        # -------------------------------------------------
        logger.info("🧠 [5/11] Regime threshold learner...")
        try:
            loop = asyncio.get_event_loop()
            learn_result = await loop.run_in_executor(None, run_learning_cycle)
            outcome  = learn_result.get("outcome", "n/a")
            regime_l = learn_result.get("regime",  "n/a")
            spy_ret  = learn_result.get("spy_return_pct", 0)
            logger.info(
                f"✅ Learner OK | regime={regime_l} outcome={outcome} "
                f"spy_ret={spy_ret}%"
            )
        except Exception as e:
            # Learner nunca debe romper el pipeline
            logger.warning(f"⚠️ Learner falló (no crítico): {e}")

        # -------------------------------------------------
        # 6️⃣ MARKET QUANT — siempre corre, usa umbrales actualizados
        # -------------------------------------------------
        logger.info("📉 [6/11] Market quantitative context (V3.3 adaptive)...")
        quant_ctx = run_market_state()
        logger.info(
            f"✅ Market quant OK | regime={quant_ctx.regime} "
            f"| dd={quant_ctx.drawdown_rolling:.3f} "
            f"| vol={quant_ctx.volatility:.3f}"
        )

        # -------------------------------------------------
        # 7️⃣ MARKET QUALITATIVE — siempre corre
        # -------------------------------------------------
        logger.info("🧠 [7/11] Market qualitative context...")
        qual_ctx = evaluate_qualitative_market()
        logger.info(
            f"✅ Market qual OK | impact={qual_ctx.impact_score:.3f} "
            f"| conf={qual_ctx.aggregated_confidence:.2f}"
        )

        # -------------------------------------------------
        # 8️⃣ MARKET ORCHESTRATOR — siempre corre
        # -------------------------------------------------
        logger.info("🧭 [8/11] Market orchestration...")
        market_orch = MarketOrchestrator()
        market_ctx  = market_orch.evaluate(
            quant_ctx.to_dict(),
            qual_ctx.to_dict(),
        )
        logger.info(
            f"🎯 MARKET MODE = {market_ctx.market_mode.upper()} "
            f"(conf {market_ctx.confidence:.2f})"
        )

        # -------------------------------------------------
        # 9️⃣ ALPHA ENGINE — siempre corre
        # -------------------------------------------------
        logger.info("🔬 [9/11] Alpha engine...")
        tickers_file = DATA_PATH / "tickers.json"
        tickers      = json.loads(tickers_file.read_text())
        alpha_out    = compute_and_persist_alpha(tickers)
        logger.info(
            f"✅ Alpha engine OK | valid_alphas={alpha_out.get('valid_alphas')} "
            f"| universe={alpha_out.get('universe_size')}"
        )

        # -------------------------------------------------
        # 🔟 TRADING ORCHESTRATOR — siempre corre
        # -------------------------------------------------
        logger.info("🤖 [10/11] Trading orchestrator...")
        trading_orch = TradingOrchestrator()
        trade_out    = await trading_orch.run(
            market_ctx=market_ctx.to_dict()
        )
        logger.info(
            f"✅ Trading OK | mode={trade_out.get('mode')} "
            f"| decisions={len(trade_out.get('decisions', []))} "
            f"| capital=${trade_out.get('real_capital', 0):,.0f}"
        )

        # -------------------------------------------------
        # 1️⃣1️⃣ COMMIT A DISCO
        # -------------------------------------------------
        logger.info("💾 [11/11] Committing pipeline results...")

        PIPELINE_KEY = os.getenv("PIPELINE_KEY")
        BASE_URL     = str(request.base_url).rstrip("/")

        commit_payload = {
            "screener":   screener_out,
            "market_ctx": market_ctx.to_dict(),
            "audit": {
                "timestamp":    datetime.utcnow().isoformat(),
                "market_mode":  market_ctx.market_mode,
                "confidence":   market_ctx.confidence,
                "decisions":    len(trade_out.get("decisions", [])),
                "real_capital": trade_out.get("real_capital"),
                "regime_thresholds": {
                    "dd_max":  quant_ctx.thresholds_used.get("defensive", {}).get("dd_max"),
                    "vol_min": quant_ctx.thresholds_used.get("defensive", {}).get("vol_min"),
                },
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
# PIPELINE ENDPOINT
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
        

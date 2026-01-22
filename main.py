# =========================================================
# main.py — TRADING SUITE ENTERPRISE v2.8.2 PRODUCCIÓN ✅
# =========================================================
# ORQUESTADOR COMPLETO: Market → PM → Broker → PortfolioStore
# 🔒 ASYNC + RETRY + LIMITES + MONITORING MEJORADO
# =========================================================

import os
import json
import logging
import time
import asyncio
import signal
import httpx
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from tenacity import retry, stop_after_attempt, wait_exponential

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# MÓDULOS CORE
# =========================================================
from market_state_evaluator import evaluate_quant_market
from market_qualitative_evaluator import evaluate_qualitative_market
from market_orchestrator import MarketOrchestrator, MarketOrchestrationContext

from pm_growth import PMGrowth
from pm_neutral import PMNeutral
from pm_defensive import PMDefensive

from portfolio_store import (
    load_positions,
    register_open,
    register_close,
    register_rotate,
    portfolio_summary,
)

# =========================================================
# CONFIG PRODUCCIÓN
# =========================================================
class Config:
    DATA_PATH = os.getenv("DATA_PATH", "/data")
    PORT = int(os.getenv("PORT", "8000"))
    BROKER_EXEC_URL = os.getenv("BROKER_URL", "http://localhost:8001/trading/execute")
    PIPELINE_KEY = os.getenv("PIPELINE_KEY")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "10"))
    BROKER_TIMEOUT = int(os.getenv("BROKER_TIMEOUT", "15"))

config = Config()

# =========================================================
# LOGGING PRO
# =========================================================
def setup_logging():
    Path(config.DATA_PATH).mkdir(parents=True, exist_ok=True)
    log_file = Path(config.DATA_PATH) / "trading_suite.log"

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("trading_suite")

logger = setup_logging()

# =========================================================
# SINGLETONS (THREAD SAFE)
# =========================================================
import threading
_singleton_lock = threading.Lock()
_market_orchestrator = None
_pm_growth = None
_pm_neutral = None
_pm_defensive = None

def get_market_orchestrator():
    global _market_orchestrator
    with _singleton_lock:
        if _market_orchestrator is None:
            _market_orchestrator = MarketOrchestrator()
        return _market_orchestrator

def get_pm_growth():
    global _pm_growth
    with _singleton_lock:
        if _pm_growth is None:
            _pm_growth = PMGrowth()
        return _pm_growth

def get_pm_neutral():
    global _pm_neutral
    with _singleton_lock:
        if _pm_neutral is None:
            _pm_neutral = PMNeutral()
        return _pm_neutral

def get_pm_defensive():
    global _pm_defensive
    with _singleton_lock:
        if _pm_defensive is None:
            _pm_defensive = PMDefensive()
        return _pm_defensive

def resolve_pm(mode: str):
    if mode == "growth":
        return get_pm_growth()
    if mode == "neutral":
        return get_pm_neutral()
    return get_pm_defensive()

# =========================================================
# BROKER CLIENT (RETRY + BACKOFF)
# =========================================================
async def execute_broker(decision: Dict[str, Any], market_mode: str) -> Optional[Dict[str, Any]]:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _call():
        async with httpx.AsyncClient(timeout=config.BROKER_TIMEOUT) as client:
            r = await client.post(
                config.BROKER_EXEC_URL,
                json=decision,
                headers={
                    "X-MARKET-MODE": market_mode,
                    "X-PM-ACTIVE": decision.get("pm", "unknown"),
                    "Content-Type": "application/json",
                },
            )
            r.raise_for_status()
            return r.json()

    try:
        return await _call()
    except Exception as e:
        logger.error(f"❌ Broker error {decision.get('ticker', 'N/A')}: {e}")
        return None

# =========================================================
# FASTAPI APP v2.8.2
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Trading Suite v2.8.2 iniciando - PRODUCTION READY")
    yield
    logger.info("🛑 Trading Suite v2.8.2 shutdown graceful")

app = FastAPI(
    title="Trading Suite Enterprise",
    version="2.8.2",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# DAILY RUN (MEJORADO)
# =========================================================
@app.post("/internal/system/daily-run")
async def daily_system_run(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    logger.info("🧠 DAILY RUN v2.8.2 START")

    # -------------------------------
    # 1. MARKET EVALUATION
    # -------------------------------
    try:
        spy_path = Path(config.DATA_PATH) / "market/spy_prices.json"
        cross_path = Path(config.DATA_PATH) / "market/cross_prices.json"
        
        spy = json.loads(spy_path.read_text())
        cross = json.loads(cross_path.read_text())

        quant = evaluate_quant_market(spy["prices"], cross["prices"])
        qual = evaluate_qualitative_market(quant.to_dict())

        orchestrator = get_market_orchestrator()
        market_ctx = orchestrator.evaluate(quant.to_dict(), qual.to_dict())
        market_mode = market_ctx.market_mode

        Path(config.DATA_PATH, "market").mkdir(exist_ok=True)
        (Path(config.DATA_PATH) / "market/current_state.json").write_text(
            json.dumps(market_ctx.to_dict(), indent=2)
        )

        logger.info(f"🌍 MARKET MODE: {market_mode.upper()} | confidence: {market_ctx.confidence:.1f}")

    except Exception as e:
        logger.error(f"❌ Market eval failed → FALLBACK DEFENSIVE: {e}")
        market_ctx = MarketOrchestrationContext(
            market_mode="defensive",
            confidence=0.0,
            reason="fallback_error",
            timestamp=datetime.utcnow().isoformat(),
            source={},
        )
        market_mode = "defensive"

    # -------------------------------
    # 2. PM RESOLUTION
    # -------------------------------
    pm = resolve_pm(market_mode)
    logger.info(f"📦 PM ACTIVO: {pm.__class__.__name__}")

    # -------------------------------
    # 3. PORTFOLIO + LIMITS
    # -------------------------------
    positions = load_positions()
    summary = portfolio_summary()
    
    logger.info(f"📊 Portfolio: {len(positions)}/{config.MAX_POSITIONS} | Anchors: {summary['anchors']}")
    
    # Portfolio limits dinámicos
    if len(positions) >= config.MAX_POSITIONS:
        allow_actions = ["CLOSE", "ROTATE"]
        logger.warning(f"⚠️ PORTFOLIO LLENO {len(positions)}/{config.MAX_POSITIONS} → Solo CLOSES/ROTATES")
    else:
        allow_actions = ["OPEN", "CLOSE", "ROTATE"]

    # -------------------------------
    # 4. POSITION EVALUATION
    # -------------------------------
    decisions: List[Dict[str, Any]] = []
    for pos in positions:
        decision = pm.evaluate_position(pos).to_dict()
        decision["pm"] = pm.__class__.__name__
        
        if decision["action"] in allow_actions:
            decisions.append(decision)
        else:
            logger.debug(f"⏭️ Skip {decision['action']} {decision.get('ticker')} (portfolio limits)")

    logger.info(f"🎯 Decisiones válidas: {len(decisions)}")

    # -------------------------------
    # 5. ASYNC EXECUTION
    # -------------------------------
    async def exec_one(decision: Dict[str, Any]) -> bool:
        try:
            fill = await execute_broker(decision, market_mode)
            if not fill:
                logger.warning(f"⚠️ No fill {decision.get('ticker')}")
                return False

            action = decision["action"]
            if action == "OPEN":
                success = register_open(decision, fill, market_ctx.to_dict())
            elif action == "CLOSE":
                success = register_close(decision, fill)
            elif action == "ROTATE":
                success = register_rotate(
                    decision,
                    broker_close_fill=fill.get("close", {}),
                    broker_open_fill=fill.get("open", {}),
                    market_ctx=market_ctx.to_dict(),
                )
            else:
                return False

            if success:
                logger.info(f"✅ {action} {decision.get('ticker')} | ${fill.get('fill_price', 0):.2f}")
            else:
                logger.error(f"❌ {action} failed {decision.get('ticker')}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Exec error {decision.get('ticker')}: {e}")
            return False

    # Ejecutar todas las decisiones concurrentemente
    if decisions:
        results = await asyncio.gather(*(exec_one(d) for d in decisions), return_exceptions=True)
        executed = sum(1 for r in results if r is True)
    else:
        executed = 0

    # -------------------------------
    # 6. SUMMARY + PERSIST
    # -------------------------------
    final_summary = portfolio_summary()
    run_summary = {
        "market_mode": market_mode,
        "pm": pm.__class__.__name__,
        "positions_before": len(positions),
        "positions_after": final_summary["positions"],
        "max_positions": config.MAX_POSITIONS,
        "anchors": final_summary["anchors"],
        "total_value": final_summary["total_value"],
        "unrealized_pnl_pct": final_summary["unrealized_pnl_pct"],
        "decisions": len(decisions),
        "executed": executed,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Persist run
    Path(config.DATA_PATH, "daily_runs").mkdir(exist_ok=True)
    run_file = Path(config.DATA_PATH) / f"daily_runs/run_{int(time.time())}.json"
    run_file.write_text(json.dumps(run_summary, indent=2))

    logger.info(f"🏁 DAILY RUN v2.8.2 COMPLETE | {executed}/{len(decisions)} executed | Value: ${final_summary['total_value']:,.0f}")
    return run_summary

# =========================================================
# MONITORING ENDPOINTS
# =========================================================
@app.get("/internal/portfolio/summary")
async def portfolio_state():
    """Portfolio completo"""
    return portfolio_summary()

@app.get("/health")
async def health():
    """Healthcheck production"""
    summary = portfolio_summary()
    return {
        "status": "healthy",
        "version": "2.8.2",
        "positions": summary["positions"],
        "anchors": summary["anchors"],
        "total_value": summary["total_value"],
        "unrealized_pnl_pct": summary["unrealized_pnl_pct"],
        "max_positions": config.MAX_POSITIONS,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/config")
async def get_config():
    """Runtime config (sin secrets)"""
    return {
        "version": "2.8.2",
        "max_positions": config.MAX_POSITIONS,
        "broker_timeout": config.BROKER_TIMEOUT,
        "broker_url": config.BROKER_EXEC_URL.replace(config.PIPELINE_KEY or "", "***"),
    }

# =========================================================
# GRACEFUL SHUTDOWN
# =========================================================
def handle_shutdown(signum, frame):
    logger.info("🛑 SIGTERM/SIGINT recibido - Graceful shutdown")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Trading Suite v2.8.2 START | PORT={config.PORT} | MAX_POS={config.MAX_POSITIONS}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=config.PORT,
        log_level="error"
    )

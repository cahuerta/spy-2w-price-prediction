# =========================================================
# main.py — TRADING SUITE ENTERPRISE v2.8.3 PRODUCCIÓN
# =========================================================
# ✔ Runtime de trading (NO batch)
# ✔ Consume market_context.json (fuente única)
# ✔ PM decide, Broker ejecuta, Portfolio persiste
# ✔ Pipeline externo dispara ejecución
# =========================================================

import os
import json
import logging
import signal
import httpx
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# DASHBOARD (SOLO ROUTER)
# =========================================================
from dashboard import router as dashboard_router

# =========================================================
# CORE
# =========================================================
from market_orchestrator import MarketOrchestrationContext

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
# CONFIG
# =========================================================
class Config:
    DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
    PORT = int(os.getenv("PORT", "8000"))
    PIPELINE_KEY = os.getenv("PIPELINE_KEY")
    BROKER_EXEC_URL = os.getenv(
        "BROKER_URL", "http://localhost:8001/trading/execute"
    )
    BROKER_TIMEOUT = int(os.getenv("BROKER_TIMEOUT", "20"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

config = Config()

MARKET_CTX_FILE = config.DATA_PATH / "market_context.json"

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trading_suite")

# =========================================================
# SINGLETON PMs
# =========================================================
_pm_growth = PMGrowth()
_pm_neutral = PMNeutral()
_pm_defensive = PMDefensive()

def resolve_pm(mode: str):
    if mode == "growth":
        return _pm_growth
    if mode == "neutral":
        return _pm_neutral
    return _pm_defensive

# =========================================================
# MARKET CONTEXT LOADER (FUENTE ÚNICA)
# =========================================================
def load_market_context() -> MarketOrchestrationContext:
    if not MARKET_CTX_FILE.exists():
        raise RuntimeError("market_context.json no existe")

    data = json.loads(MARKET_CTX_FILE.read_text())
    return MarketOrchestrationContext(**data)

# =========================================================
# TRADING CYCLE (CORE)
# =========================================================
async def run_trading_cycle() -> Dict[str, Any]:
    """
    Ejecuta ciclo completo:
    - Lee market_context.json
    - Resuelve PM
    - PM decide
    - Broker ejecuta
    - Portfolio se actualiza
    """

    market_ctx = load_market_context()
    pm = resolve_pm(market_ctx.market_mode)

    logger.info(
        f"▶ Trading cycle | mode={market_ctx.market_mode} "
        f"conf={market_ctx.confidence:.2f}"
    )

    positions = load_positions()
    decisions = pm.evaluate_portfolio(positions)

    results = []

    async with httpx.AsyncClient(timeout=config.BROKER_TIMEOUT) as client:
        for decision in decisions:
            try:
                r = await client.post(
                    config.BROKER_EXEC_URL,
                    json=decision,
                )
                broker_res = r.json()
                results.append(broker_res)

                # ----------------------------------
                # PORTFOLIO SYNC
                # ----------------------------------
                if decision["action"] == "OPEN" and broker_res["status"] == "executed":
                    register_open(decision, broker_res, market_ctx.to_dict())

                elif decision["action"] == "CLOSE" and broker_res["status"] == "executed":
                    register_close(decision, broker_res)

                elif decision["action"] == "ROTATE" and broker_res["status"] == "executed":
                    register_rotate(
                        decision,
                        broker_res,
                        broker_res,
                        market_ctx.to_dict(),
                    )

            except Exception as e:
                logger.error(f"❌ Broker error: {e}")

    return {
        "market_mode": market_ctx.market_mode,
        "decisions": len(decisions),
        "portfolio": portfolio_summary(),
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# FASTAPI APP
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Trading Suite v2.8.3 START")
    yield
    logger.info("Trading Suite v2.8.3 STOP")

app = FastAPI(
    title="Trading Suite Enterprise",
    version="2.8.3",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)

# =========================================================
# PIPELINE TRIGGER (ÚNICO)
# =========================================================
@app.post("/internal/trading/run")
async def trading_run(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    logger.info("🔔 Trading run triggered by pipeline")
    return await run_trading_cycle()

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
async def health():
    return {"status": "ok"}

# =========================================================
# SHUTDOWN
# =========================================================
def handle_shutdown(signum, frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.PORT,
        log_level="error",
)

# =========================================================
# main.py — TRADING SUITE ENTERPRISE v2.9.0 PRODUCCIÓN
# =========================================================
# ✔ Runtime de trading (NO batch)
# ✔ NO decide mercado
# ✔ NO ejecuta modelos
# ✔ NO ejecuta PMs directamente
# ✔ Consume market_context.json (fuente única)
# ✔ Expone endpoints para PIPELINE y TRADING
# =========================================================

import os
import json
import logging
import signal
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# ROUTERS
# =========================================================
from dashboard import router as dashboard_router

# =========================================================
# CORE
# =========================================================
from market_orchestrator import MarketOrchestrationContext
from trading_orchestrator import TradingOrchestrator
from pipeline_daily import main as run_pipeline  # 🔴 IMPORT REAL

# =========================================================
# CONFIG
# =========================================================
class Config:
    DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
    PORT = int(os.getenv("PORT", "8000"))
    PIPELINE_KEY = os.getenv("PIPELINE_KEY")
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
# HELPERS
# =========================================================
def load_market_context() -> MarketOrchestrationContext:
    """
    Fuente ÚNICA de verdad.
    Generada por pipeline_daily.py
    """
    if not MARKET_CTX_FILE.exists():
        raise RuntimeError("market_context.json no existe")

    data = json.loads(MARKET_CTX_FILE.read_text())
    return MarketOrchestrationContext(**data)

# =========================================================
# FASTAPI APP
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Trading Suite v2.9.0 START")
    yield
    logger.info("Trading Suite v2.9.0 STOP")

app = FastAPI(
    title="Trading Suite Enterprise",
    version="2.9.0",
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
# PIPELINE ENDPOINT (CRON → HTTP)
# =========================================================
@app.post("/internal/pipeline/run")
async def pipeline_run(request: Request):
    """
    Ejecuta pipeline_daily.main() DENTRO del web service.
    Render permite escribir en /data aquí.
    """
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    logger.info("🚀 Pipeline triggered via HTTP")
    run_pipeline()

    return {
        "status": "ok",
        "message": "pipeline executed",
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# TRADING ENDPOINT (PIPELINE → TRADING)
# =========================================================
@app.post("/internal/trading/run")
async def trading_run(request: Request):
    """
    Ejecuta SOLO trading, usando market_context.json
    """
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    logger.info("🔔 Trading run triggered")

    market_ctx = load_market_context()
    orchestrator = TradingOrchestrator()
    result = await orchestrator.run(market_ctx.to_dict())

    return {
        "status": "ok",
        "market_mode": market_ctx.market_mode,
        "result": result,
        "timestamp": datetime.utcnow().isoformat(),
    }

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

# =========================================================
# main.py — TRADING SUITE ENTERPRISE v2.8.2 PRODUCCIÓN
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
# DASHBOARD (SOLO ROUTER, NO APP)
# =========================================================
from dashboard import router as dashboard_router

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
# LOGGING
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
# SINGLETONS
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
# FASTAPI APP (ÚNICA)
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Trading Suite v2.8.2 starting")
    yield
    logger.info("Trading Suite v2.8.2 shutdown")

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

# 🔑 DASHBOARD MONTADO AQUÍ
app.include_router(dashboard_router)

# =========================================================
# CRON ENDPOINT — NO TOCAR
# =========================================================
@app.post("/internal/system/daily-run")
async def daily_system_run(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    logger.info("DAILY RUN START")

    # TODO: (tu lógica existente intacta)
    # ⬇️ NO CAMBIADA ⬇️

    return {"status": "ok"}

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
async def health():
    return {"status": "healthy"}

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

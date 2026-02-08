# =========================================================
# main.py — TRADING SUITE ENTERPRISE v2.8.2 PRODUCCION
# =========================================================
# ORQUESTADOR COMPLETO: Market -> PM -> Broker -> PortfolioStore
# ASYNC + RETRY + LIMITES + MONITORING
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
# MODULOS CORE (CEREBRO)
# =========================================================
from market_state_evaluator import evaluate_quant_market
from market_qualitative_evaluator import evaluate_qualitative_market
from market_orchestrator import (
    MarketOrchestrator,
    MarketOrchestrationContext,
)

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
# CONFIG PRODUCCION
# =========================================================
class Config:
    DATA_PATH = os.getenv("DATA_PATH", "/data")
    PORT = int(os.getenv("PORT", "8000"))
    BROKER_EXEC_URL = os.getenv(
        "BROKER_URL",
        "http://localhost:8001/trading/execute",
    )
    PIPELINE_KEY = os.getenv("PIPELINE_KEY")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "10"))
    BROKER_TIMEOUT = int(os.getenv("BROKER_TIMEOUT", "15"))

config = Config()

# =========================================================
# LOGGING (AISLADO DEL DASHBOARD)
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
_market_orchestrator: Optional[MarketOrchestrator] = None
_pm_growth: Optional[PMGrowth] = None
_pm_neutral: Optional[PMNeutral] = None
_pm_defensive: Optional[PMDefensive] = None

def get_market_orchestrator() -> MarketOrchestrator:
    global _market_orchestrator
    with _singleton_lock:
        if _market_orchestrator is None:
            _market_orchestrator = MarketOrchestrator()
        return _market_orchestrator

def get_pm_growth() -> PMGrowth:
    global _pm_growth
    with _singleton_lock:
        if _pm_growth is None:
            _pm_growth = PMGrowth()
        return _pm_growth

def get_pm_neutral() -> PMNeutral:
    global _pm_neutral
    with _singleton_lock:
        if _pm_neutral is None:
            _pm_neutral = PMNeutral()
        return _pm_neutral

def get_pm_defensive() -> PMDefensive:
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
async def execute_broker(
    decision: Dict[str, Any],
    market_mode: str,
) -> Optional[Dict[str, Any]]:

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _call():
        async with httpx.AsyncClient(
            timeout=config.BROKER_TIMEOUT
        ) as client:
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
        logger.error(
            f"Broker error {decision.get('ticker', 'N/A')}: {e}"
        )
        return None

# =========================================================
# FASTAPI APP (CEREBRO)
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

# =========================================================
# DAILY RUN (PIPELINE PRINCIPAL)
# =========================================================
@app.post("/internal/system/daily-run")
async def daily_system_run(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(status_code=403, detail="Invalid pipeline key")

    logger.info("DAILY RUN START")

    # --------------------------------------------------
    # 1. MARKET EVALUATION
    # --------------------------------------------------
    try:
        spy_path = Path(config.DATA_PATH) / "market/spy_prices.json"
        cross_path = Path(config.DATA_PATH) / "market/cross_prices.json"

        spy = json.loads(spy_path.read_text())
        cross = json.loads(cross_path.read_text())

        quant = evaluate_quant_market(
            spy["prices"],
            cross["prices"],
        )
        qual = evaluate_qualitative_market(quant.to_dict())

        orchestrator = get_market_orchestrator()
        market_ctx = orchestrator.evaluate(
            quant.to_dict(),
            qual.to_dict(),
        )
        market_mode = market_ctx.market_mode

        Path(config.DATA_PATH, "market").mkdir(exist_ok=True)
        (Path(config.DATA_PATH) / "market/current_state.json").write_text(
            json.dumps(market_ctx.to_dict(), indent=2)
        )

        logger.info(
            f"MARKET MODE: {market_mode.upper()} "
            f"(conf {market_ctx.confidence:.2f})"
        )

    except Exception as e:
        logger.error(f"Market eval failed -> fallback defensive: {e}")
        market_ctx = MarketOrchestrationContext(
            market_mode="defensive",
            confidence=0.0,
            reason="fallback_error",
            timestamp=datetime.utcnow().isoformat(),
            source={},
        )
        market_mode = "defensive"

    # --------------------------------------------------
    # 2. PM RESOLUTION
    # --------------------------------------------------
    pm = resolve_pm(market_mode)
    logger.info(f"PM ACTIVO: {pm.__class__.__name__}")

    # --------------------------------------------------
    # 3. PORTFOLIO + LIMITS
    # --------------------------------------------------
    positions = load_positions()
    summary = portfolio_summary()

    if len(positions) >= config.MAX_POSITIONS:
        allow_actions = ["CLOSE", "ROTATE"]
        logger.warning("PORTFOLIO FULL -> only CLOSE / ROTATE")
    else:
        allow_actions = ["OPEN", "CLOSE", "ROTATE"]

    # --------------------------------------------------
    # 4. POSITION EVALUATION
    # --------------------------------------------------
    decisions: List[Dict[str, Any]] = []

    for pos in positions:
        decision = pm.evaluate_position(pos).to_dict()
        decision["pm"] = pm.__class__.__name__

        if decision["action"] in allow_actions:
            decisions.append(decision)

    logger.info(f"Valid decisions: {len(decisions)}")

    # --------------------------------------------------
    # 5. ASYNC EXECUTION
    # --------------------------------------------------
    async def exec_one(decision: Dict[str, Any]) -> bool:
        fill = await execute_broker(decision, market_mode)
        if not fill:
            return False

        action = decision["action"]

        if action == "OPEN":
            return register_open(decision, fill, market_ctx.to_dict())
        if action == "CLOSE":
            return register_close(decision, fill)
        if action == "ROTATE":
            return register_rotate(
                decision,
                broker_close_fill=fill.get("close", {}),
                broker_open_fill=fill.get("open", {}),
                market_ctx=market_ctx.to_dict(),
            )
        return False

    if decisions:
        results = await asyncio.gather(
            *(exec_one(d) for d in decisions),
            return_exceptions=True,
        )
        executed = sum(1 for r in results if r is True)
    else:
        executed = 0

    # --------------------------------------------------
    # 6. SUMMARY + PERSIST
    # --------------------------------------------------
    final_summary = portfolio_summary()

    run_summary = {
        "market_mode": market_mode,
        "pm": pm.__class__.__name__,
        "positions_before": len(positions),
        "positions_after": final_summary["positions"],
        "anchors": final_summary["anchors"],
        "total_value": final_summary["total_value"],
        "unrealized_pnl_pct": final_summary["unrealized_pnl_pct"],
        "decisions": len(decisions),
        "executed": executed,
        "timestamp": datetime.utcnow().isoformat(),
    }

    Path(config.DATA_PATH, "daily_runs").mkdir(exist_ok=True)
    run_file = Path(config.DATA_PATH) / f"daily_runs/run_{int(time.time())}.json"
    run_file.write_text(json.dumps(run_summary, indent=2))

    logger.info(
        f"DAILY RUN COMPLETE: {executed}/{len(decisions)} executed"
    )

    return run_summary

# =========================================================
# MONITORING ENDPOINTS (INTERNOS)
# =========================================================
@app.get("/health")
async def health():
    summary = portfolio_summary()
    return {
        "status": "healthy",
        "version": "2.8.2",
        "positions": summary["positions"],
        "anchors": summary["anchors"],
        "total_value": summary["total_value"],
        "unrealized_pnl_pct": summary["unrealized_pnl_pct"],
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/internal/portfolio/summary")
async def portfolio_state():
    return portfolio_summary()

@app.get("/config")
async def get_config():
    return {
        "version": "2.8.2",
        "max_positions": config.MAX_POSITIONS,
        "broker_timeout": config.BROKER_TIMEOUT,
    }

# =========================================================
# GRACEFUL SHUTDOWN
# =========================================================
def handle_shutdown(signum, frame):
    logger.info("SIGTERM/SIGINT received")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    logger.info(
        f"Trading Suite v2.8.2 START | PORT={config.PORT}"
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.PORT,
        log_level="error",
    )

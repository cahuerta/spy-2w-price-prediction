# =========================================================
# main_engine.py — TRADING SUITE ENTERPRISE v2.8.2
# =========================================================
# CEREBRO PURO (NO FASTAPI)
# Ejecutado por CRON / JOB
# =========================================================

import os
import json
import logging
import time
import asyncio
import httpx
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

from tenacity import retry, stop_after_attempt, wait_exponential

# =========================================================
# MODULOS CORE
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
# CONFIG
# =========================================================
class Config:
    DATA_PATH = os.getenv("DATA_PATH", "/data")
    BROKER_EXEC_URL = os.getenv(
        "BROKER_URL",
        "http://localhost:8001/trading/execute",
    )
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "10"))
    BROKER_TIMEOUT = int(os.getenv("BROKER_TIMEOUT", "15"))

config = Config()

# =========================================================
# LOGGING
# =========================================================
Path(config.DATA_PATH).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            Path(config.DATA_PATH) / "trading_suite.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("trading_engine")

# =========================================================
# SINGLETONS
# =========================================================
_market_orchestrator = MarketOrchestrator()
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
# BROKER CLIENT
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
# MAIN DAILY RUN (CEREBRO)
# =========================================================
async def daily_run():
    logger.info("DAILY RUN START")

    # -------------------------------
    # 1. MARKET EVALUATION
    # -------------------------------
    try:
        spy = json.loads(
            (Path(config.DATA_PATH) / "market/spy_prices.json").read_text()
        )
        cross = json.loads(
            (Path(config.DATA_PATH) / "market/cross_prices.json").read_text()
        )

        quant = evaluate_quant_market(
            spy["prices"],
            cross["prices"],
        )
        qual = evaluate_qualitative_market(quant.to_dict())

        market_ctx = _market_orchestrator.evaluate(
            quant.to_dict(),
            qual.to_dict(),
        )
        market_mode = market_ctx.market_mode

        (Path(config.DATA_PATH) / "market/current_state.json").write_text(
            json.dumps(market_ctx.to_dict(), indent=2)
        )

        logger.info(
            f"MARKET MODE: {market_mode.upper()} "
            f"(conf {market_ctx.confidence:.2f})"
        )

    except Exception as e:
        logger.error(f"Market eval failed → DEFENSIVE: {e}")
        market_ctx = MarketOrchestrationContext(
            market_mode="defensive",
            confidence=0.0,
            reason="fallback_error",
            timestamp=datetime.utcnow().isoformat(),
            source={},
        )
        market_mode = "defensive"

    # -------------------------------
    # 2. PM
    # -------------------------------
    pm = resolve_pm(market_mode)
    logger.info(f"PM ACTIVO: {pm.__class__.__name__}")

    # -------------------------------
    # 3. PORTFOLIO
    # -------------------------------
    positions = load_positions()

    allow_actions = (
        ["CLOSE", "ROTATE"]
        if len(positions) >= config.MAX_POSITIONS
        else ["OPEN", "CLOSE", "ROTATE"]
    )

    # -------------------------------
    # 4. DECISIONS
    # -------------------------------
    decisions: List[Dict[str, Any]] = []

    for pos in positions:
        d = pm.evaluate_position(pos).to_dict()
        d["pm"] = pm.__class__.__name__
        if d["action"] in allow_actions:
            decisions.append(d)

    logger.info(f"Valid decisions: {len(decisions)}")

    # -------------------------------
    # 5. EXECUTION
    # -------------------------------
    async def exec_one(d: Dict[str, Any]) -> bool:
        fill = await execute_broker(d, market_mode)
        if not fill:
            return False

        if d["action"] == "OPEN":
            return register_open(d, fill, market_ctx.to_dict())
        if d["action"] == "CLOSE":
            return register_close(d, fill)
        if d["action"] == "ROTATE":
            return register_rotate(
                d,
                broker_close_fill=fill.get("close", {}),
                broker_open_fill=fill.get("open", {}),
                market_ctx=market_ctx.to_dict(),
            )
        return False

    executed = 0
    if decisions:
        results = await asyncio.gather(*(exec_one(d) for d in decisions))
        executed = sum(1 for r in results if r)

    # -------------------------------
    # 6. SUMMARY
    # -------------------------------
    final = portfolio_summary()

    run_summary = {
        "market_mode": market_mode,
        "pm": pm.__class__.__name__,
        "positions": final["positions"],
        "anchors": final["anchors"],
        "total_value": final["total_value"],
        "unrealized_pnl_pct": final["unrealized_pnl_pct"],
        "decisions": len(decisions),
        "executed": executed,
        "timestamp": datetime.utcnow().isoformat(),
    }

    Path(config.DATA_PATH, "daily_runs").mkdir(exist_ok=True)
    (Path(config.DATA_PATH) / f"daily_runs/run_{int(time.time())}.json").write_text(
        json.dumps(run_summary, indent=2)
    )

    logger.info("DAILY RUN COMPLETE")

# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    asyncio.run(daily_run())

import logging
from typing import Dict, Any, List
from datetime import datetime

# =========================
# CORE IMPORTS
# =========================
from portfolio_store import (
    load_positions,
    register_open,
    register_close,
    register_rotate,
    portfolio_summary,
)

from broker import get_trading_engine

from pm_growth import PMGrowth
from pm_neutral import PMNeutral
from pm_defensive import PMDefensive

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s"
)
logger = logging.getLogger("trading_orchestrator")

# =========================
# ORCHESTRATOR
# =========================
class TradingOrchestrator:
    """
    Trading Orchestrator (EJECUCIÓN REAL)

    INPUT:
      - market_ctx (MarketOrchestrationContext.to_dict())
      - signals (dict opcional, solo growth/neutral)
      - anchor_universe (opcional, defensive)

    RESPONSABILIDAD ÚNICA:
      Ejecutar decisiones de trading y persistir estado
    """

    def __init__(self):
        self.broker = get_trading_engine()
        logger.info("🧭 TradingOrchestrator inicializado")

    # --------------------------------------------------
    # ENTRYPOINT PRINCIPAL
    # --------------------------------------------------
    async def run(
        self,
        market_ctx: Dict[str, Any],
        signals: Dict[str, Dict[str, Any]] | None = None,
        anchor_universe: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:

        mode = market_ctx.get("market_mode")
        logger.info(f"🚦 TRADING MODE = {mode.upper()}")

        positions = load_positions()
        decisions: List[Dict[str, Any]] = []

        # =========================
        # SELECCIÓN PM
        # =========================
        if mode == "growth":
            pm = PMGrowth()
            decisions = self._run_growth(pm, positions, signals or {})

        elif mode == "neutral":
            pm = PMNeutral()
            decisions = self._run_neutral(pm, positions, signals or {})

        elif mode == "defensive":
            pm = PMDefensive()
            decisions = self._run_defensive(pm, positions, anchor_universe or [])

        else:
            logger.warning(f"Modo desconocido: {mode}")
            return {"status": "skipped", "reason": "invalid_market_mode"}

        # =========================
        # EJECUCIÓN + PERSISTENCIA
        # =========================
        executed = []
        for d in decisions:
            result = await self._execute_and_persist(d, market_ctx)
            executed.append(result)

        summary = portfolio_summary()

        return {
            "status": "ok",
            "mode": mode,
            "decisions": decisions,
            "executed": executed,
            "portfolio": summary,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # --------------------------------------------------
    # PM HANDLERS
    # --------------------------------------------------
    def _run_growth(self, pm: PMGrowth, positions, signals):
        out = []
        for pos in positions:
            sig = signals.get(pos["ticker"])
            d = pm.evaluate_position(pos, sig)
            out.append({**d, "pm": "PMGROWTH"})
        return out

    def _run_neutral(self, pm: PMNeutral, positions, signals):
        result = pm.evaluate_portfolio(positions)
        return [{**d, "pm": "PMNEUTRAL"} for d in result["decisions"]]

    def _run_defensive(self, pm: PMDefensive, positions, anchor_universe):
        decisions = pm.evaluate_portfolio(positions, anchor_universe)
        return [{**d.to_dict(), "pm": "PMDEFENSIVE"} for d in decisions]

    # --------------------------------------------------
    # EJECUCIÓN REAL + STORE
    # --------------------------------------------------
    async def _execute_and_persist(
        self,
        decision: Dict[str, Any],
        market_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:

        action = decision.get("action")
        logger.info(f"▶️ {action} {decision.get('ticker')}")

        result = await self.broker.execute_decision(decision)

        # -------------------------
        # PERSISTENCIA
        # -------------------------
        if result.status != "executed":
            logger.warning(f"⛔ NO ejecutado: {result.reason}")
            return {"decision": decision, "result": result.model_dump()}

        if action == "OPEN":
            register_open(
                decision,
                broker_fill=result.model_dump(),
                market_ctx=market_ctx,
            )

        elif action == "CLOSE":
            register_close(
                decision,
                broker_fill=result.model_dump(),
            )

        elif action == "ROTATE":
            register_rotate(
                decision,
                broker_close_fill=result.model_dump(),
                broker_open_fill=result.model_dump(),
                market_ctx=market_ctx,
            )

        return {
            "decision": decision,
            "result": result.model_dump(),
        }

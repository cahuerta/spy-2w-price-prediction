# =========================================================
# trading_orchestrator.py — V2.1 CRON_DAILY PRODUCTION
# =========================================================

import logging
import asyncio
import os
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

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

from capital_governor import CapitalGovernor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] 🧭 %(message)s"
)

logger = logging.getLogger("trading_orchestrator")


class TradingOrchestrator:

    def __init__(self):
        self.broker = get_trading_engine()
        self._last_run_file = Path("/tmp/trading_last_run.flag")
        self._daily_limit = int(os.getenv("MAX_ORDERS_DAY", "10"))
        self._executed_today = 0
        logger.info("🧭 TradingOrchestrator V2.1 CRON_DAILY iniciado")

    def _check_daily_run(self) -> bool:
        now = datetime.utcnow()

        if self._last_run_file.exists():
            try:
                last_run = datetime.fromisoformat(
                    self._last_run_file.read_text().strip()
                )
                if last_run.date() == now.date():
                    logger.info("⏭️ SKIP: Ya ejecutado hoy")
                    return False
            except Exception:
                logger.warning("⚠️ Flag corrupto, permitiendo ejecución")

        return True

    async def run(
        self,
        market_ctx: Dict[str, Any],
        signals: Dict[str, Dict[str, Any]] | None = None,
        anchor_universe: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:

        if not self._check_daily_run():
            return {
                "status": "skipped_daily",
                "reason": "already_executed_today",
                "timestamp": datetime.utcnow().isoformat(),
            }

        if not market_ctx or "market_mode" not in market_ctx:
            raise ValueError("❌ market_ctx inválido")

        mode = market_ctx["market_mode"]
        logger.info(f"🚦 MARKET MODE = {mode.upper()} | Ejecutando cron diario")

        try:

            # 1️⃣ LOAD PORTFOLIO
            positions = load_positions()
            logger.info(f"📊 Portfolio: {len(positions)} posiciones")

            # 2️⃣ CAPITAL GOVERNOR (FIX ARRANQUE VACÍO)
            if positions:
                governor = CapitalGovernor()
                capital_state = governor.evaluate(positions)

                logger.info(
                    f"🏛 RISK | Vol={capital_state.volatility_annual:.2%} | "
                    f"VaR95={capital_state.var_95_annual:.2%} | "
                    f"ES95={capital_state.expected_shortfall_95_annual:.2%} | "
                    f"Beta={capital_state.beta_vs_spy:.2f}"
                )
            else:
                logger.info("📭 Portfolio vacío → riesgo inicial neutral")
                capital_state = None

            # 3️⃣ HARD RISK CONTROLS
            signals = signals or {}

            if capital_state:
                if capital_state.expected_shortfall_95_annual > 0.45:
                    logger.warning("🛑 Expected Shortfall crítico → solo CIERRES")
                    signals = {}

                if mode == "defensive" and capital_state.beta_vs_spy > 1.20:
                    logger.warning("🛑 Beta alto en DEFENSIVE → bloqueando aperturas")
                    signals = {}

                if capital_state.volatility_annual > 0.45:
                    logger.warning("🛑 Volatilidad extrema → bloqueando nuevas posiciones")
                    signals = {}

            # 4️⃣ PM SEGÚN REGIME
            decisions: List[Dict[str, Any]] = []

            if mode == "growth":
                pm = PMGrowth()
                decisions = self._run_growth(pm, positions, signals)

            elif mode == "neutral":
                pm = PMNeutral()
                decisions = self._run_neutral(pm, positions)

            elif mode == "defensive":
                pm = PMDefensive()
                decisions = self._run_defensive(
                    pm, positions, anchor_universe or []
                )

            else:
                raise ValueError(f"❌ market_mode desconocido: {mode}")

            logger.info(f"🤖 PM generó {len(decisions)} decisiones")

            # 5️⃣ EJECUCIÓN
            executed = []

            for i, decision in enumerate(decisions[:self._daily_limit]):

                if signals == {} and decision.get("action") == "OPEN":
                    logger.info(f"⛔ OPEN bloqueado: {decision.get('ticker')}")
                    executed.append({
                        "decision": decision,
                        "result": {
                            "status": "blocked_risk",
                            "reason": "high_risk_mode",
                        }
                    })
                    continue

                logger.info(
                    f"▶️ [{i+1}/{min(len(decisions), self._daily_limit)}] "
                    f"{decision.get('action')} {decision.get('ticker')}"
                )

                result = await self._execute_and_persist(
                    decision, market_ctx
                )

                executed.append(result)
                self._executed_today += 1
                await asyncio.sleep(1.0)

            summary = portfolio_summary()

            # 6️⃣ FLAG
            self._last_run_file.write_text(
                datetime.utcnow().isoformat()
            )

            return {
                "status": "ok",
                "mode": mode,
                "decisions": decisions,
                "executed": executed,
                "executed_count": self._executed_today,
                "daily_limit": self._daily_limit,
                "portfolio": summary,
                "capital_state": (
                    capital_state.to_dict()
                    if capital_state else None
                ),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"💥 Orquestador falló: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    # --------------------------------------------------

    def _run_growth(self, pm: PMGrowth, positions, signals):
        out = []
        for pos in positions:
            sig = signals.get(pos["ticker"])
            d = pm.evaluate_position(pos, sig)
            out.append({**d, "pm": "PMGROWTH"})
        return out

    def _run_neutral(self, pm: PMNeutral, positions):
        result = pm.evaluate_portfolio(positions)
        return [{**d, "pm": "PMNEUTRAL"} for d in result["decisions"]]

    def _run_defensive(self, pm: PMDefensive, positions, anchor_universe):
        decisions = pm.evaluate_portfolio(positions, anchor_universe)
        return [{**d.to_dict(), "pm": "PMDEFENSIVE"} for d in decisions]

    # --------------------------------------------------

    async def _execute_and_persist(
        self,
        decision: Dict[str, Any],
        market_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:

        action = decision.get("action")
        ticker = decision.get("ticker")

        client_order_id = (
            f"{action}_{ticker}_{int(datetime.utcnow().timestamp())}"
        )
        decision["client_order_id"] = client_order_id

        try:
            result = await asyncio.wait_for(
                self.broker.execute_decision(decision),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout 60s: {ticker}")
            return {
                "decision": decision,
                "result": {
                    "status": "timeout",
                    "reason": "execution_timeout_60s"
                }
            }
        except Exception as e:
            logger.error(f"❌ Broker error {ticker}: {e}")
            return {
                "decision": decision,
                "result": {
                    "status": "error",
                    "reason": str(e)
                }
            }

        if result.status != "executed":
            logger.warning(f"⛔ Broker rechazó: {result.reason}")
            return {
                "decision": decision,
                "result": result.model_dump()
            }

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

        logger.info(f"✅ {action} {ticker} ejecutado")

        return {
            "decision": decision,
            "result": result.model_dump(),
            }

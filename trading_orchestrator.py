# =========================================================
# trading_orchestrator.py — V2.2 CRON_DAILY + ALPHA FILTER
# =========================================================

import logging
import asyncio
import os
import json

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
from capital_governor import CapitalGovernor

from pm_growth import PMGrowth
from pm_neutral import PMNeutral
from pm_defensive import PMDefensive

# 🆕 ALPHA ENGINE
from alpha_engine_v4 import compute_and_persist_alpha


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
        self._alpha_threshold = float(os.getenv("ALPHA_THRESHOLD", "0.70"))
        self._executed_today = 0

        logger.info("🧭 TradingOrchestrator V2.2 ALPHA integrado iniciado")

    # =====================================================
    # DAILY CHECK
    # =====================================================
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

    # =====================================================
    # MAIN RUN
    # =====================================================
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
        logger.info(f"🚦 MARKET MODE = {mode.upper()}")

        try:

            # =====================================================
            # 1️⃣ PORTFOLIO
            # =====================================================
            positions = load_positions()
            logger.info(f"📊 Portfolio: {len(positions)} posiciones")

            # =====================================================
            # 2️⃣ ALPHA ENGINE (PRE-FILTER)
            # =====================================================
            
DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
TICKERS_FILE = DATA_PATH / "tickers.json"

if TICKERS_FILE.exists():
    universe = json.loads(TICKERS_FILE.read_text())
else:
    universe = []

if universe:
    logger.info("🧠 Calculando AlphaEngine...")
    payload = await asyncio.to_thread(
        compute_and_persist_alpha,
        universe
    )

    alpha_results = payload["results"]

    alpha_filtered = {
        t: a for t, a in alpha_results.items()
        if a and a["alpha_score"] >= self._alpha_threshold
    }

    logger.info(
        f"⭐ {len(alpha_filtered)} tickers superan alpha "
        f"{self._alpha_threshold}"
    )
else:
    logger.warning("⚠️ tickers.json vacío o inexistente")
    alpha_filtered = {}
            if universe:
                logger.info("🧠 Calculando AlphaEngine...")
                payload = await asyncio.to_thread(
                    compute_and_persist_alpha,
                    universe
                )

                alpha_results = payload["results"]

                alpha_filtered = {
                    t: a for t, a in alpha_results.items()
                    if a and a["alpha_score"] >= self._alpha_threshold
                }

                logger.info(
                    f"⭐ {len(alpha_filtered)} tickers superan alpha "
                    f"{self._alpha_threshold}"
                )
            else:
                logger.warning("⚠️ Universe vacío en market_ctx")
                alpha_filtered = {}

            # =====================================================
            # 3️⃣ CAPITAL GOVERNOR
            # =====================================================
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
                capital_state = None
                logger.info("📭 Portfolio vacío → riesgo neutral")

            # =====================================================
            # 4️⃣ HARD RISK CONTROLS
            # =====================================================
            signals = signals or {}

            if capital_state:
                if capital_state.expected_shortfall_95_annual > 0.45:
                    logger.warning("🛑 ES crítico → solo CIERRES")
                    signals = {}

                if mode == "defensive" and capital_state.beta_vs_spy > 1.20:
                    logger.warning("🛑 Beta alto DEFENSIVE → bloqueando OPEN")
                    signals = {}

                if capital_state.volatility_annual > 0.45:
                    logger.warning("🛑 Vol extrema → bloqueando OPEN")
                    signals = {}

            # =====================================================
            # 5️⃣ PM SEGÚN REGIME
            # =====================================================
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

            # =====================================================
            # 6️⃣ EJECUCIÓN CON FILTRO ALPHA
            # =====================================================
            executed = []

            for i, decision in enumerate(decisions[:self._daily_limit]):

                ticker = decision.get("ticker")
                action = decision.get("action")

                # 🔥 ALPHA FILTER SOLO PARA OPEN
                if action == "OPEN":

                    alpha_info = alpha_filtered.get(ticker)

                    if not alpha_info:
                        logger.info(
                            f"⛔ OPEN bloqueado por Alpha: {ticker}"
                        )
                        executed.append({
                            "decision": decision,
                            "result": {
                                "status": "blocked_alpha",
                                "reason": "alpha_below_threshold"
                            }
                        })
                        continue

                    logger.info(
                        f"🧠 Alpha OK {ticker} | "
                        f"{alpha_info['alpha_score']} "
                        
                    )

                logger.info(
                    f"▶️ [{i+1}/{min(len(decisions), self._daily_limit)}] "
                    f"{action} {ticker}"
                )

                result = await self._execute_and_persist(
                    decision,
                    market_ctx
                )

                executed.append(result)

                if result["result"]["status"] == "executed":
                    self._executed_today += 1

                await asyncio.sleep(1.0)

            summary = portfolio_summary()

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
            logger.error("💥 Orquestador falló", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    # =====================================================
    # PM WRAPPERS
    # =====================================================

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

    # =====================================================
    # EXECUTION
    # =====================================================

    async def _execute_and_persist(
        self,
        decision: Dict[str, Any],
        market_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:

        action = decision.get("action")
        ticker = decision.get("ticker")

        decision["client_order_id"] = (
            f"{action}_{ticker}_{int(datetime.utcnow().timestamp())}"
        )

        try:
            result = await asyncio.wait_for(
                self.broker.execute_decision(decision),
                timeout=60.0
            )
        except Exception as e:
            return {
                "decision": decision,
                "result": {
                    "status": "error",
                    "reason": str(e)
                }
            }

        if result.status != "executed":
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

# =========================================================
# trading_orchestrator.py — V2.6 HEDGE FUND EDITION
# ALPACA = SINGLE SOURCE OF TRUTH
# =========================================================

import logging
import asyncio
import os
import json
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from portfolio_store import load_positions
from broker import get_trading_engine
from capital_governor import CapitalGovernor

# PMs
from pm_growth import PMGrowth
from pm_neutral import PMNeutral
from pm_defensive import PMDefensive

# Alpha Engine
from alpha_engine_v4 import compute_and_persist_alpha

logger = logging.getLogger("trading_orchestrator")


class TradingOrchestrator:
    def __init__(self):
        self.broker = get_trading_engine()
        self._last_run_file = Path("/tmp/trading_last_run.flag")

        self.fixed_capital = float(os.getenv("FIXED_CAPITAL", "1000000"))
        self._daily_limit = int(os.getenv("MAX_ORDERS_DAY", "10"))
        self._alpha_threshold = float(os.getenv("ALPHA_THRESHOLD", "0.70"))

        self.governor = CapitalGovernor(fixed_capital=self.fixed_capital)

        logger.info(
            f"🧭 Orchestrator v2.6 Online | Capital: ${self.fixed_capital:,.0f}"
        )

    # =========================================================
    # MAIN RUN
    # =========================================================
    async def run(
        self,
        market_ctx: Dict[str, Any],
        signals: Dict[str, Dict[str, Any]] | None = None,
        anchor_universe: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:

        if not self._check_daily_run():
            return {
                "status": "skipped_daily",
                "timestamp": datetime.utcnow().isoformat(),
            }

        # 🔄 1) SINCRONIZAR ESTADO REAL DESDE ALPACA
        self.broker.sync_positions_from_broker()

        positions = load_positions()
        mode = market_ctx.get("market_mode", "neutral")
        signals = signals or {}

        logger.info(
            f"🚦 MODO: {mode.upper()} | Posiciones actuales: {len(positions)}"
        )

        # 🔍 2) Alpha filtering
        alpha_filtered_tickers = await self._get_alpha_filtered_tickers()

        # 🧠 3) PM decisions
        raw_decisions = self._collect_pm_decisions(
            mode, positions, signals, anchor_universe
        )

        # 📦 4) Segregación
        priority_closes = [
            d for d in raw_decisions if d.get("action") == "CLOSE"
        ]

        investment_candidates = [
            d for d in raw_decisions
            if d.get("action") in ["OPEN", "ROTATE"]
        ]

        # 🏛 5) Governor sizing
        validated = self.governor.adjust_sizing(
            positions, investment_candidates
        )

        # 🎯 6) Consolidar final queue
        final_queue = priority_closes.copy()

        for cmd in validated:
            ticker = cmd.get("ticker")

            if (
                cmd.get("action") == "OPEN"
                and ticker not in alpha_filtered_tickers
            ):
                logger.info(
                    f"⛔ {ticker} bloqueado por Alpha (< {self._alpha_threshold})"
                )
                continue

            final_queue.append(cmd)

        # 🤖 7) Ejecución serial
        results = []
        executed_count = 0

        for order in final_queue[: self._daily_limit]:
            res = await self._execute(order)
            results.append(res)

            if res["result"].get("status") == "executed":
                executed_count += 1

            await asyncio.sleep(0.5)

        self._last_run_file.write_text(
            datetime.utcnow().isoformat()
        )

        # 🔄 Estado final actualizado
        self.broker.sync_positions_from_broker()

        return {
            "status": "ok",
            "executed_count": executed_count,
            "capital_state": self.governor.evaluate(
                load_positions()
            ).to_dict(),
            "results": results,
        }

    # =========================================================
    # PM DECISION COLLECTION
    # =========================================================
    def _collect_pm_decisions(
        self, mode, positions, signals, anchor_universe
    ) -> List[Dict]:

        if mode == "growth":
            pm = PMGrowth(fixed_capital=self.fixed_capital)

            return [
                {
                    **pm.evaluate_position(
                        p, signals.get(p["ticker"])
                    ),
                    "pm": "GROWTH",
                }
                for p in positions
            ]

        elif mode == "defensive":
            pm = PMDefensive()

            decisions = pm.evaluate_portfolio(
                positions,
                anchor_universe,
                total_capital=self.fixed_capital,
            )

            return [
                {**d.to_dict(), "pm": "DEFENSIVE"}
                for d in decisions
            ]

        else:
            pm = PMNeutral()
            res = pm.evaluate_portfolio(positions)

            return [
                {**d, "pm": "NEUTRAL"}
                for d in res.get("decisions", [])
            ]

    # =========================================================
    # ALPHA ENGINE
    # =========================================================
    async def _get_alpha_filtered_tickers(self) -> List[str]:
        try:
            path = Path(os.getenv("DATA_PATH", "/data")) / "tickers.json"

            if not path.exists():
                return []

            universe = json.loads(path.read_text())

            payload = await asyncio.to_thread(
                compute_and_persist_alpha, universe
            )

            scores = payload.get("results", {})

            return [
                t
                for t, v in scores.items()
                if isinstance(v, dict)
                and v.get("alpha_score", 0)
                >= self._alpha_threshold
            ]

        except Exception as e:
            logger.error(f"Alpha Engine error: {e}")
            return []

    # =========================================================
    # EXECUTION (NO JSON WRITE)
    # =========================================================
    async def _execute(self, decision: Dict) -> Dict:

        ticker = (
            decision.get("ticker")
            or decision.get("open_ticker")
        )

        try:
            broker_res = await asyncio.wait_for(
                self.broker.execute_decision(decision),
                timeout=45,
            )

            # 🔄 Si ejecuta → sincronizamos espejo
            if broker_res.status == "executed":
                self.broker.sync_positions_from_broker()
                logger.info(
                    f"🔄 Sync post {decision.get('action')} {ticker}"
                )

            return {
                "decision": decision,
                "result": broker_res.model_dump(),
            }

        except Exception as e:
            logger.error(f"❌ Error ejecución {ticker}: {e}")

            return {
                "decision": decision,
                "result": {
                    "status": "error",
                    "reason": str(e),
                },
            }

    # =========================================================
    # DAILY RUN CHECK
    # =========================================================
    def _check_daily_run(self) -> bool:
        if not self._last_run_file.exists():
            return True

        last_date = datetime.fromisoformat(
            self._last_run_file.read_text().strip()
        ).date()

        return last_date != datetime.utcnow().date()

    # =========================================================
    # PREVIEW (NO BROKER CALL)
    # =========================================================
    async def preview_executability(
        self,
        market_ctx: Dict[str, Any],
        signals: Dict[str, Dict[str, Any]] | None = None,
        anchor_universe: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:

        signals = signals or {}

        self.broker.sync_positions_from_broker()
        positions = load_positions()

        mode = market_ctx.get("market_mode", "neutral")

        alpha_filtered = await self._get_alpha_filtered_tickers()

        raw_decisions = self._collect_pm_decisions(
            mode, positions, signals, anchor_universe
        )

        results = {}

        for cmd in raw_decisions:

            ticker = cmd.get("ticker")
            action = cmd.get("action")

            if action == "CLOSE":
                results[ticker] = {
                    "action": "CLOSE",
                    "executable": True,
                    "reason": None,
                }

        investment_candidates = [
            d for d in raw_decisions
            if d.get("action") in ["OPEN", "ROTATE"]
        ]

        validated = self.governor.adjust_sizing(
            positions, investment_candidates
        )

        for cmd in validated:

            ticker = cmd.get("ticker")
            action = cmd.get("action")

            if (
                action == "OPEN"
                and ticker not in alpha_filtered
            ):
                results[ticker] = {
                    "action": action,
                    "executable": False,
                    "reason": "alpha_below_threshold",
                }
                continue

            if cmd.get("shares", 0) <= 0:
                results[ticker] = {
                    "action": action,
                    "executable": False,
                    "reason": "sizing_block",
                }
                continue

            results[ticker] = {
                "action": action,
                "executable": True,
                "reason": None,
            }

        return {
            "results": results,
            "capital_state": self.governor.evaluate(
                positions
            ).to_dict(),
        }

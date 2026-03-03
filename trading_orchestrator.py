# =========================================================
# trading_orchestrator.py — V2.9 ALPHA-DICTATOR PRODUCTION
# ALPHA MANDATORY | PM ADVISORY | GOVERNOR RISK CONTROL
# =========================================================

import logging
import asyncio
import os
import json
from typing import Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path

from portfolio_store import load_positions
from broker import get_trading_engine
from capital_governor import CapitalGovernor
from alpha_engine_enterprise_strict_v4_3 import compute_and_persist_alpha

from pm_growth import PMGrowth
from pm_neutral import PMNeutral
from pm_defensive import PMDefensive

logger = logging.getLogger("trading_orchestrator")
logging.basicConfig(level=logging.INFO)


class TradingOrchestrator:
    def __init__(self):

        self.broker = get_trading_engine()
        self.flag_file = Path("/tmp/trading_last_run.flag")

        self.fixed_capital = float(os.getenv("FIXED_CAPITAL", "1000000"))
        self.daily_limit = int(os.getenv("MAX_ORDERS_DAY", "15"))

        # Alpha thresholds
        self.alpha_growth = float(os.getenv("ALPHA_GROWTH", "0.65"))
        self.alpha_neutral = float(os.getenv("ALPHA_NEUTRAL", "0.75"))
        self.alpha_defensive = float(os.getenv("ALPHA_DEFENSIVE", "0.85"))

        self.alpha_elite = float(os.getenv("ALPHA_ELITE", "0.88"))
        self.alpha_hold_shield = float(os.getenv("ALPHA_SHIELD", "0.75"))
        self.alpha_kill = float(os.getenv("ALPHA_KILL", "-0.40"))

        self.governor = CapitalGovernor(self.fixed_capital)

        logger.info(f"🚀 v2.9 ALPHA-DICTATOR PRODUCTION | Capital ${self.fixed_capital:,.0f}")

    # =========================================================
    # MAIN RUN
    # =========================================================
    async def run(self, market_ctx: Dict[str, Any], signals: Dict = None, anchor_universe: List = None) -> Dict:

        if not self._daily_flag_check():
            return {"status": "skipped_daily_limit"}

        # 1️⃣ SYNC
        if hasattr(self.broker, "sync_positions_from_broker"):
            sync_result = self.broker.sync_positions_from_broker()
            if asyncio.iscoroutine(sync_result):
                await sync_result

        positions = load_positions()
        portfolio_tickers = {p["ticker"].upper() for p in positions}
        mode = market_ctx.get("market_mode", "neutral")

        logger.info(f"📊 Portfolio: {len(positions)} | Mode: {mode}")

        # 2️⃣ FULL ALPHA
        alpha_data = await self._compute_full_alpha()
        alpha_map = {
            t.upper(): d
            for t, d in alpha_data.get("results", {}).items()
            if isinstance(d, dict)
        }

        # 3️⃣ PM DECISIONS
        pm_decisions = await self._get_pm_decisions(
            mode,
            positions,
            signals or {},
            anchor_universe
        )

        # 4️⃣ PRIORITY ENGINE
        closes = []
        opens = []

        # 🔒 SHIELD + PM CLOSE
        for decision in pm_decisions:
            if decision.get("action") == "CLOSE":
                ticker = decision.get("ticker", "").upper()
                alpha_score = alpha_map.get(ticker, {}).get("alpha_score", 0)

                if alpha_score >= self.alpha_hold_shield:
                    logger.info(f"🛡 SHIELD BLOCK {ticker} | alpha={alpha_score:.3f}")
                    continue

                closes.append({
                    "action": "CLOSE",
                    "ticker": ticker,
                    "reason": decision.get("reason", "PM_CLOSE")
                })

        # 💀 KILL SWITCH
        for ticker in portfolio_tickers:
            alpha_score = alpha_map.get(ticker, {}).get("alpha_score", 0)
            if alpha_score <= self.alpha_kill:
                closes.append({
                    "action": "CLOSE",
                    "ticker": ticker,
                    "reason": f"ALPHA_KILL_{alpha_score:.3f}"
                })
                logger.warning(f"💀 KILL {ticker} | alpha={alpha_score:.3f}")

        # 🔁 Deduplicate CLOSES
        closes = list({c["ticker"]: c for c in closes}.values())

        # 📈 PM OPENS
        for decision in pm_decisions:
            if decision.get("action") in ["OPEN", "ROTATE"]:
                opens.append(decision)

        # 🚀 ALPHA INJECTION
        threshold = self._alpha_threshold(mode)

        for ticker, data in alpha_map.items():
            score = data.get("alpha_score", 0)

            if ticker not in portfolio_tickers and score >= threshold:
                opens.append({
                    "action": "OPEN",
                    "ticker": ticker,
                    "shares": 0,
                    "reason": f"ALPHA_INJECT_{score:.3f}",
                    "alpha": score
                })

        # 🔁 Deduplicate OPENS (Alpha overrides PM)
        opens_dict = {}
        for o in opens:
            opens_dict[o["ticker"].upper()] = o
        unique_opens = list(opens_dict.values())

        # 5️⃣ GOVERNOR
        sized_opens = self.governor.adjust_sizing(positions, unique_opens)

        # 6️⃣ FINAL FILTER
        final_queue = closes[:]
        elite_count = 0

        for cmd in sized_opens:
            ticker = cmd["ticker"].upper()
            score = alpha_map.get(ticker, {}).get("alpha_score", 0)

            if score >= self.alpha_elite:
                logger.info(f"🔥 ELITE {ticker} {score:.3f}")
                elite_count += 1
                final_queue.append(cmd)

            elif score >= threshold:
                final_queue.append(cmd)

            else:
                logger.warning(f"⛔ REJECT {ticker} alpha={score:.3f} < {threshold}")

        # 7️⃣ EXECUTION
        results = []
        executed = 0

        for order in final_queue[:self.daily_limit]:
            try:
                res = await asyncio.wait_for(
                    self.broker.execute_decision(order),
                    timeout=30
                )

                results.append({
                    "ticker": order["ticker"],
                    "status": "success",
                })
                executed += 1

                await asyncio.sleep(0.8)

            except Exception as e:
                results.append({
                    "ticker": order.get("ticker"),
                    "status": "error",
                    "error": str(e)
                })

        self.flag_file.write_text(datetime.now(timezone.utc).isoformat())

        logger.info(
            f"✅ EXECUTED {executed}/{len(final_queue)} | Elite executed: {elite_count}"
        )

        return {
            "status": "success",
            "mode": mode,
            "executed": executed,
            "elite_executed": elite_count,
            "queue_size": len(final_queue),
            "results": results
        }

    # =========================================================
    # HELPERS
    # =========================================================
    async def _compute_full_alpha(self) -> Dict:
        tickers_path = Path(os.getenv("DATA_PATH", "/data")) / "tickers.json"
        universe = json.loads(tickers_path.read_text()) if tickers_path.exists() else []
        return await asyncio.to_thread(compute_and_persist_alpha, universe)

    def _alpha_threshold(self, mode: str) -> float:
        return {
            "growth": self.alpha_growth,
            "defensive": self.alpha_defensive
        }.get(mode, self.alpha_neutral)

    async def _get_pm_decisions(self, mode: str, positions: List[Dict], signals: Dict, anchor: List) -> List[Dict]:

        decisions = []

        try:
            if mode == "growth":
                pm = PMGrowth(self.fixed_capital)
                for pos in positions:
                    d = pm.evaluate_position(pos, signals.get(pos["ticker"]))
                    if isinstance(d, dict):
                        decisions.append(d)

            elif mode == "defensive":
                pm = PMDefensive()
                raw = pm.evaluate_portfolio(positions, anchor, self.fixed_capital)

                if isinstance(raw, list):
                    decisions.extend([
                        r if isinstance(r, dict) else r.to_dict()
                        for r in raw
                    ])
                elif isinstance(raw, dict):
                    decisions.extend(raw.get("decisions", []))

            else:
                pm = PMNeutral()
                raw = pm.evaluate_portfolio(positions)
                decisions.extend(raw.get("decisions", []))

        except Exception as e:
            logger.error(f"PM error: {e}")

        return decisions

    def _daily_flag_check(self) -> bool:
        if not self.flag_file.exists():
            return True
        last_run = datetime.fromisoformat(self.flag_file.read_text().strip())
        return last_run.date() != datetime.now(timezone.utc).date()

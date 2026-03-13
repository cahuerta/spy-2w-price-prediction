# =========================================================
# trading_orchestrator.py — V3.1 ALPHA-CONSUMER
# ALPHA ENGINE EXTERNO | ORCHESTRATOR SOLO LEE
# =========================================================
#
# FIX v3.1:
#   [F1] Filtro real_positions: solo intenta cerrar tickers
#        que realmente existen en el broker (evita "position not found")
#   [F2] clear_positions() tras confirmar que todos los cierres
#        fueron exitosos Y el broker confirma portfolio vacío
#
# =========================================================

import logging
import asyncio
import os
import json
from typing import Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path
from broker import get_engine

from portfolio_store import load_positions, save_positions, clear_positions

from capital_governor import CapitalGovernor

from pm_growth import PMGrowth
from pm_neutral import PMNeutral
from pm_defensive import PMDefensive

logger = logging.getLogger("trading_orchestrator")
logging.basicConfig(level=logging.INFO)


DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
ALPHA_FILE = DATA_PATH / "alpha_last.json"


class TradingOrchestrator:

    def __init__(self):

        self.broker = get_engine()
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

        logger.info(f"🚀 v3.1 ALPHA-CONSUMER | Capital ${self.fixed_capital:,.0f}")

    # =========================================================
    # MAIN RUN
    # =========================================================
    async def run(
        self,
        market_ctx: Dict[str, Any],
        signals: Dict = None,
        anchor_universe: List = None
    ) -> Dict:

        if not self._daily_flag_check():
            return {"status": "skipped_daily_limit"}

        # 1️⃣ SYNC POSITIONS
        if hasattr(self.broker, "sync_positions_from_broker"):
            sync_result = self.broker.sync_positions_from_broker()
            if asyncio.iscoroutine(sync_result):
                await sync_result

        positions = load_positions()

        if len(positions) == 0:
            logger.warning("⚠️ positions.json vacío → fallback broker")

            if hasattr(self.broker, "get_positions"):
                positions = await self.broker.get_positions()

                if positions:
                    save_positions(positions)
                    logger.info(f"🔄 Portfolio sincronizado desde broker: {len(positions)} posiciones")

        # [F1] Tickers realmente existentes en broker (fuente de verdad para cierres)
        broker_positions = load_positions()
        real_positions = {p["ticker"].upper() for p in broker_positions}

        portfolio_tickers = real_positions
        mode = market_ctx.get("market_mode", "neutral")

        logger.info(f"📊 Portfolio: {len(positions)} | Real en broker: {len(real_positions)} | Mode: {mode}")

        # 2️⃣ LOAD LAST ALPHA (NO RECOMPUTE)
        alpha_data = self._load_last_alpha()

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

        closes = []
        opens = []

        # 🔒 SHIELD + PM CLOSE
        for decision in pm_decisions:
            if decision.get("action") == "CLOSE":
                ticker = decision.get("ticker", "").upper()

                # [F1] Solo cerrar si realmente existe en broker
                if ticker not in real_positions:
                    logger.warning(f"⚠️ SKIP CLOSE {ticker} → no existe en broker")
                    continue

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

                # [F1] Solo cerrar si realmente existe en broker
                if ticker not in real_positions:
                    logger.warning(f"⚠️ SKIP KILL {ticker} → no existe en broker")
                    continue

                closes.append({
                    "action": "CLOSE",
                    "ticker": ticker,
                    "reason": f"ALPHA_KILL_{alpha_score:.3f}"
                })
                logger.warning(f"💀 KILL {ticker} | alpha={alpha_score:.3f}")

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
                    "target_pct": 0.05,
                    "reason": f"ALPHA_INJECT_{score:.3f}",
                    "alpha": score
                })

        # Deduplicate OPENS
        opens_dict = {o["ticker"].upper(): o for o in opens}
        unique_opens = list(opens_dict.values())

        # 5️⃣ GOVERNOR
        # Anclas de rotación (PM Defensivo) → ES calculado POST-cierre
        # Normales (alpha injection) → sizing estándar con portfolio actual
        anchor_opens = [
            o for o in unique_opens
            if o.get("is_anchor") or o.get("reason", "").startswith("ANCHOR_ROTATE")
        ]
        normal_opens = [o for o in unique_opens if o not in anchor_opens]

        close_tickers = [c["ticker"] for c in closes]

        sized_anchors = self.governor.adjust_sizing_after_closes(
            positions, close_tickers, anchor_opens
        ) if anchor_opens else []

        sized_normals = self.governor.adjust_sizing(
            positions, normal_opens
        ) if normal_opens else []

        sized_opens = sized_anchors + sized_normals

        logger.info(
            f"📐 Sizing | anchors={len(sized_anchors)} "
            f"normals={len(sized_normals)} total={len(sized_opens)}"
        )

        # 6️⃣ FINAL FILTER
        final_queue = closes[:]
        elite_count = 0

        for cmd in sized_opens:
            ticker = cmd["ticker"].upper()
            score = alpha_map.get(ticker, {}).get("alpha_score", 0)

            if score >= self.alpha_elite:
                elite_count += 1
                logger.info(f"🔥 ELITE {ticker} {score:.3f}")
                final_queue.append(cmd)

            elif score >= threshold:
                final_queue.append(cmd)

            else:
                logger.warning(f"⛔ REJECT {ticker} alpha={score:.3f}")

        # 7️⃣ EXECUTION
        results = []
        executed = 0
        close_successes = []

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

                if order["action"] == "CLOSE":
                    close_successes.append(order["ticker"])

                await asyncio.sleep(0.8)

            except Exception as e:
                results.append({
                    "ticker": order.get("ticker"),
                    "status": "error",
                    "error": str(e)
                })

        # [F2] Si todos los cierres fueron exitosos y broker confirma vacío → limpiar store
        if close_successes:
            all_closes_ok = len(close_successes) == len(closes)
            try:
                broker_after = self.broker.get_positions()
                if isinstance(broker_after, dict):
                    broker_after = list(broker_after.values())
                broker_empty = len(broker_after) == 0
            except Exception:
                broker_empty = False

            if all_closes_ok and broker_empty:
                logger.info("🧹 Todos los cierres OK + broker vacío → clear_positions()")
                clear_positions()
            else:
                logger.info(f"📂 Cierres parciales o broker aún tiene posiciones → manteniendo store")

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
            "alpha_timestamp": alpha_data.get("timestamp"),
            "results": results
        }

    # =========================================================
    # HELPERS
    # =========================================================

    def _load_last_alpha(self) -> Dict:
        if not ALPHA_FILE.exists():
            raise RuntimeError("❌ alpha_last.json no encontrado. Ejecuta alpha_engine primero.")

        try:
            data = json.loads(ALPHA_FILE.read_text())
        except Exception as e:
            raise RuntimeError(f"❌ alpha_last.json corrupto: {e}")

        logger.info(
            f"🧠 Alpha loaded | ts={data.get('timestamp')} "
            f"| universe={data.get('universe_size')}"
        )

        return data

    def _alpha_threshold(self, mode: str) -> float:
        return {
            "growth": self.alpha_growth,
            "defensive": self.alpha_defensive
        }.get(mode, self.alpha_neutral)

    async def _get_pm_decisions(
        self,
        mode: str,
        positions: List[Dict],
        signals: Dict,
        anchor: List
    ) -> List[Dict]:

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
            

# =========================================================
# trading_orchestrator.py — V3.3 ALPHA-CONSUMER
# ALPHA ENGINE EXTERNO | ORCHESTRATOR SOLO LEE
# =========================================================
#
# FIX v3.1:
#   [F1] Filtro real_positions: solo intenta cerrar tickers
#        que realmente existen en el broker (evita "position not found")
#   [F2] clear_positions() tras confirmar que todos los cierres
#        fueron exitosos Y el broker confirma portfolio vacío
#
# FIX v3.2:
#   [F3] Enriquecer posiciones con price_now desde caché ANTES
#        de pasarlas al PM — evita "invalid_price" cuando el broker
#        no incluye price_now en las posiciones retornadas.
#        Fallback en cascada: caché → entry_price (nunca 0).
#   [F4] Cancelar órdenes pendientes (held_for_orders) antes de
#        intentar nuevas órdenes de cierre sobre el mismo ticker.
#
# FIX v3.3:
#   [F5] _enrich_opens_with_price_and_shares(): los candidatos de
#        Alpha Injection solo traían target_pct pero NO entry_price
#        ni shares. El Governor (_size_candidates) los descartaba
#        siempre → decisions=0 → nunca hubo aperturas.
#        Fix: convertir target_pct → shares usando precio del caché
#        ANTES de pasar al Governor.
# =========================================================

import logging
import asyncio
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pathlib import Path
from broker import get_engine

from portfolio_store import load_positions, save_positions, clear_positions, update_prices

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
        self.alpha_growth    = float(os.getenv("ALPHA_GROWTH",    "0.65"))
        self.alpha_neutral   = float(os.getenv("ALPHA_NEUTRAL",   "0.75"))
        self.alpha_defensive = float(os.getenv("ALPHA_DEFENSIVE", "0.85"))

        self.alpha_elite       = float(os.getenv("ALPHA_ELITE",  "0.88"))
        self.alpha_hold_shield = float(os.getenv("ALPHA_SHIELD", "0.75"))
        self.alpha_kill        = float(os.getenv("ALPHA_KILL",   "-0.40"))

        self.governor = CapitalGovernor(self.fixed_capital)

        logger.info(f"🚀 v3.3 ALPHA-CONSUMER | Capital ${self.fixed_capital:,.0f}")

    # =========================================================
    # [F3] ENRIQUECER POSICIONES CON PRECIO ACTUAL
    # =========================================================
    def _enrich_positions_with_price(self, positions: List[Dict]) -> List[Dict]:
        """
        Agrega price_now a cada posición desde market_data_cache.
        Fallback: usar current_price del broker, luego entry_price.
        Nunca deja price_now en 0 o None — PMNeutral necesita un precio válido.
        """
        try:
            from market_data_cache import get_last_price_from_cache
        except ImportError:
            get_last_price_from_cache = None

        enriched = []
        price_map = {}

        for pos in positions:
            ticker = str(pos.get("ticker", "")).upper()
            pos = dict(pos)  # copia para no mutar el original

            # Precio ya existe y es válido
            existing = pos.get("price_now")
            if existing and float(existing) > 0:
                enriched.append(pos)
                continue

            # Intentar desde caché de mercado
            cache_price = None
            if get_last_price_from_cache:
                try:
                    cache_price = get_last_price_from_cache(ticker)
                except Exception:
                    pass

            # Intentar desde current_price del broker (Alpaca lo incluye a veces)
            broker_price = pos.get("current_price") or pos.get("lastday_price")

            # Fallback final: entry_price (nunca 0)
            entry_price = float(pos.get("entry_price", 0) or 0)

            if cache_price and float(cache_price) > 0:
                pos["price_now"] = float(cache_price)
                price_map[ticker] = float(cache_price)
                logger.debug(f"💰 {ticker} price_now desde caché: {cache_price}")
            elif broker_price and float(broker_price) > 0:
                pos["price_now"] = float(broker_price)
                price_map[ticker] = float(broker_price)
                logger.debug(f"💰 {ticker} price_now desde broker field: {broker_price}")
            elif entry_price > 0:
                pos["price_now"] = entry_price
                logger.warning(f"⚠️ {ticker} price_now fallback a entry_price: {entry_price}")
            else:
                pos["price_now"] = 1.0  # guardia absoluta — nunca llega 0 al PM
                logger.error(f"❌ {ticker} sin precio disponible — usando 1.0 como guardia")

            enriched.append(pos)

        # Persistir precios actualizados en portfolio_store
        if price_map:
            try:
                update_prices(price_map)
            except Exception as e:
                logger.warning(f"⚠️ update_prices falló: {e}")

        return enriched

    # =========================================================
    # [F5] ENRIQUECER OPENS CON PRECIO Y SHARES — FIX v3.3
    # =========================================================
    def _enrich_opens_with_price_and_shares(self, opens: List[Dict]) -> List[Dict]:
        """
        Convierte target_pct → entry_price + shares usando el caché de mercado.

        Problema raíz (v3.2): los candidatos de Alpha Injection llegaban al
        Governor solo con target_pct=0.05. El Governor (_size_candidates) espera
        entry_price y shares → los descartaba siempre → decisions=0.

        Fallback de precio: caché → entry_price/price_now ya incluido en la orden.
        Si no hay precio disponible, el ticker se salta con warning.
        """
        try:
            from market_data_cache import get_last_price_from_cache
        except ImportError:
            get_last_price_from_cache = None

        enriched = []

        for o in opens:
            ticker = o.get("ticker", "").upper()

            # Si ya viene con shares y precio válido, pasar directo
            if o.get("shares", 0) > 0 and (o.get("entry_price") or o.get("price_now")):
                enriched.append(o)
                continue

            price = None

            # 1️⃣ Intentar desde caché de mercado (precio de cierre más reciente)
            if get_last_price_from_cache:
                try:
                    price = get_last_price_from_cache(ticker)
                except Exception:
                    pass

            # 2️⃣ Fallback: precio ya incluido en la orden (pm_decisions puede traerlo)
            if not price or float(price) <= 0:
                price = o.get("entry_price") or o.get("price_now")

            if not price or float(price) <= 0:
                logger.warning(f"⚠️ {ticker} sin precio para sizing — saltado")
                continue

            price = float(price)
            target_pct = float(o.get("target_pct", 0.05))
            shares = int((self.fixed_capital * target_pct) // price)

            if shares <= 0:
                logger.warning(
                    f"⚠️ {ticker} shares=0 con price={price:.2f} "
                    f"target_pct={target_pct:.1%} capital={self.fixed_capital:,.0f} — saltado"
                )
                continue

            o = dict(o)
            o["entry_price"] = price
            o["shares"] = shares
            enriched.append(o)

            logger.info(
                f"💡 [F5] {ticker} enriquecido: {shares}s @ ${price:.2f} "
                f"(target={target_pct:.1%} → ${shares * price:,.0f})"
            )

        logger.info(f"📦 Opens enriquecidos: {len(enriched)}/{len(opens)} válidos")
        return enriched

    # =========================================================
    # [F4] CANCELAR ÓRDENES PENDIENTES
    # =========================================================
    async def _cancel_pending_orders(self, tickers: List[str]):
        """
        Cancela órdenes abiertas para los tickers que se quieren cerrar.
        Evita el error 'held_for_orders' al intentar vender qty bloqueada.
        """
        if not hasattr(self.broker, "cancel_orders_for_ticker"):
            # Si el broker no tiene el método, intentar cancel_all_orders
            if hasattr(self.broker, "cancel_all_orders"):
                try:
                    await self.broker.cancel_all_orders()
                    logger.info("🗑 Órdenes pendientes canceladas (cancel_all)")
                except Exception as e:
                    logger.warning(f"⚠️ cancel_all_orders falló: {e}")
            return

        for ticker in tickers:
            try:
                result = self.broker.cancel_orders_for_ticker(ticker)
                if asyncio.iscoroutine(result):
                    await result
                logger.info(f"🗑 Órdenes canceladas para {ticker}")
            except Exception as e:
                logger.warning(f"⚠️ cancel_orders {ticker}: {e}")

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

        # [F3] Enriquecer posiciones con precio actual ANTES de pasar al PM
        positions = self._enrich_positions_with_price(positions)

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

        # [F4] Cancelar órdenes pendientes para tickers que se quieren cerrar
        if closes:
            close_tickers = [c["ticker"] for c in closes]
            await self._cancel_pending_orders(close_tickers)
            await asyncio.sleep(0.5)  # pequeña pausa para que el broker procese

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

        # [F5] Enriquecer opens con precio y shares ANTES del Governor
        unique_opens = self._enrich_opens_with_price_and_shares(unique_opens)

        # 5️⃣ GOVERNOR
        anchor_opens = [
            o for o in unique_opens
            if o.get("is_anchor") or o.get("reason", "").startswith("ANCHOR_ROTATE")
        ]
        normal_opens = [o for o in unique_opens if o not in anchor_opens]

        close_tickers_list = [c["ticker"] for c in closes]

        sized_anchors = self.governor.adjust_sizing_after_closes(
            positions, close_tickers_list, anchor_opens
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
                logger.error(f"❌ EXECUTION ERROR for {order.get('ticker')}: {e}")
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
                logger.info("📂 Cierres parciales o broker aún tiene posiciones → manteniendo store")

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
    # PREVIEW (sin ejecutar)
    # =========================================================
    async def preview_executability(self, market_ctx: Dict) -> Dict:
        positions = self._enrich_positions_with_price(load_positions())
        alpha_data = self._load_last_alpha()
        alpha_map = {
            t.upper(): d
            for t, d in alpha_data.get("results", {}).items()
            if isinstance(d, dict)
        }
        mode = market_ctx.get("market_mode", "neutral")
        threshold = self._alpha_threshold(mode)

        rows = []
        for ticker, data in alpha_map.items():
            score = data.get("alpha_score", 0)
            rows.append({
                "ticker": ticker,
                "alpha": score,
                "executable": score >= threshold,
                "mode": mode,
                "threshold": threshold,
            })

        rows.sort(key=lambda x: x["alpha"] or 0, reverse=True)
        return {"mode": mode, "threshold": threshold, "rows": rows}

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

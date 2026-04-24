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

# ── DARWIN TRACKING ───────────────────────────────────────────────
try:
    from darwin_engine.trade_tracker import register_open, register_close
    DARWIN_TRACKING = True
except ImportError:
    DARWIN_TRACKING = False
# ─────────────────────────────────────────────────────────────────

logger = logging.getLogger("trading_orchestrator")
logging.basicConfig(level=logging.INFO)

DATA_PATH   = Path(os.getenv("DATA_PATH", "/data"))
ALPHA_FILE  = DATA_PATH / "alpha_last.json"
ANCHOR_FILE = Path(os.getenv("ANCHOR_FILE", "/opt/render/project/src/anchor_universe.json"))


class TradingOrchestrator:

    def __init__(self):
        self.broker            = get_engine()
        self.daily_limit       = int(os.getenv("MAX_ORDERS_DAY", "15"))
        self._fallback_capital = float(os.getenv("FIXED_CAPITAL", "1000000"))
        self.fixed_capital     = self._fallback_capital

        self.alpha_growth      = float(os.getenv("ALPHA_GROWTH",    "0.65"))
        self.alpha_neutral     = float(os.getenv("ALPHA_NEUTRAL",   "0.75"))
        self.alpha_defensive   = float(os.getenv("ALPHA_DEFENSIVE", "0.85"))
        self.alpha_elite       = float(os.getenv("ALPHA_ELITE",     "0.88"))
        self.alpha_hold_shield = float(os.getenv("ALPHA_SHIELD",    "0.75"))
        self.alpha_kill        = float(os.getenv("ALPHA_KILL",      "-0.40"))

        self.governor = None
        logger.info(f"🚀 v4.1 ALPHA-CONSUMER+DARWIN | fallback capital ${self._fallback_capital:,.0f}")

    async def _get_real_capital(self) -> float:
        try:
            account = await self.broker.get_account()
            equity  = float(account.equity)
            if equity > 0:
                logger.info(f"💰 Capital real Alpaca: ${equity:,.0f}")
                return equity
        except Exception as e:
            logger.warning(f"⚠️ No se pudo leer equity: {e} → fallback ${self._fallback_capital:,.0f}")
        return self._fallback_capital

    async def _refresh_governor(self) -> CapitalGovernor:
        self.fixed_capital = await self._get_real_capital()
        gov = CapitalGovernor(fixed_capital=self.fixed_capital)
        logger.info(f"🏛 Governor refrescado | capital=${self.fixed_capital:,.0f}")
        return gov

    def _load_anchor_tickers(self) -> set:
        try:
            data    = json.loads(ANCHOR_FILE.read_text())
            tickers = {a["ticker"].upper() for a in data if "ticker" in a}
            logger.info(f"⚓ Anchor universe cargado: {len(tickers)} tickers")
            return tickers
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar anchor_universe.json: {e}")
            return set()

    def _load_intraday_signals(self) -> Dict[str, Dict]:
        try:
            from intraday_tracker import get_entry_signals_today
            signals = get_entry_signals_today()
            if signals:
                listos = [t for t, s in signals.items() if s.get("entrar_ahora")]
                logger.info(
                    f"📡 Intraday signals | {len(signals)} evaluados | "
                    f"listos para entrar: {listos}"
                )
            return signals
        except Exception as e:
            logger.warning(f"⚠️ Intraday signals no disponibles: {e}")
            return {}

    def _enrich_positions_with_price(self, positions: List[Dict]) -> List[Dict]:
        broker_price_map = {}
        try:
            broker_data = self.broker.get_positions()
            if not broker_data:
                logger.warning("⚠️ Broker devolvió 0 posiciones → leyendo desde disco")
            for ticker, data in (broker_data or {}).items():
                cp = data.get("current_price") or data.get("price_now")
                if cp and float(cp) > 0:
                    broker_price_map[ticker.upper()] = float(cp)
                else:
                    avg = data.get("avg_entry_price")
                    if avg and float(avg) > 0:
                        broker_price_map[ticker.upper()] = float(avg)
                        logger.warning(
                            f"⚠️ {ticker} current_price=0 → avg_entry: {float(avg):.2f}"
                        )
            logger.info(f"💹 Precios Alpaca obtenidos: {len(broker_price_map)} tickers")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo obtener precios del broker: {e}")

        enriched  = []
        price_map = {}

        for pos in positions:
            ticker = str(pos.get("ticker", "")).upper()
            pos    = dict(pos)

            if ticker in broker_price_map:
                pos["price_now"] = broker_price_map[ticker]
                price_map[ticker] = broker_price_map[ticker]
            elif float(pos.get("entry_price", 0) or 0) > 0:
                pos["price_now"] = float(pos["entry_price"])
                logger.warning(
                    f"⚠️ {ticker} sin precio Alpaca → entry_price: {pos['price_now']:.2f}"
                )
            else:
                pos["price_now"] = 1.0
                logger.error(f"❌ {ticker} SIN PRECIO → usando 1.0")

            enriched.append(pos)

        if price_map:
            try:
                update_prices(price_map)
            except Exception as e:
                logger.warning(f"⚠️ update_prices falló: {e}")

        return enriched

    def _enrich_opens_with_price_and_shares(self, opens: List[Dict]) -> List[Dict]:
        try:
            from market_data_cache import get_last_price_from_cache
        except ImportError:
            get_last_price_from_cache = None

        enriched = []

        for o in opens:
            ticker = o.get("ticker", "").upper()

            if o.get("shares", 0) > 0 and (o.get("entry_price") or o.get("price_now")):
                enriched.append(o)
                continue

            price = None
            if get_last_price_from_cache:
                try:
                    price = get_last_price_from_cache(ticker)
                except Exception:
                    pass

            if not price or float(price) <= 0:
                price = o.get("entry_price") or o.get("price_now")

            if not price or float(price) <= 0:
                logger.warning(f"⚠️ {ticker} sin precio → saltado")
                continue

            price      = float(price)
            target_pct = float(o.get("target_pct", 0.05))
            shares     = int((self.fixed_capital * target_pct) // price)

            if shares <= 0:
                logger.warning(
                    f"⚠️ {ticker} shares=0 price={price:.2f} "
                    f"capital={self.fixed_capital:,.0f} → saltado"
                )
                continue

            o              = dict(o)
            o["entry_price"] = price
            o["shares"]      = shares
            enriched.append(o)
            logger.info(
                f"💡 {ticker} enriquecido: {shares}s @ ${price:.2f} "
                f"(target={target_pct:.1%} → ${shares * price:,.0f})"
            )

        logger.info(f"📦 Opens enriquecidos: {len(enriched)}/{len(opens)} válidos")
        return enriched

    async def _cancel_pending_orders(self, tickers: List[str]):
        if not hasattr(self.broker, "cancel_orders_for_ticker"):
            if hasattr(self.broker, "cancel_all_orders"):
                try:
                    await self.broker.cancel_all_orders()
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
    # RUN PRINCIPAL v4.1
    # =========================================================
    async def run(
        self,
        market_ctx: Dict[str, Any],
        signals: Dict = None,
        anchor_universe: List = None
    ) -> Dict:

        self.governor = await self._refresh_governor()
        logger.info(f"🚀 v4.1 ALPHA-CONSUMER+DARWIN | Capital ${self.fixed_capital:,.0f}")

        anchor_tickers = self._load_anchor_tickers()

        if hasattr(self.broker, "sync_positions_from_broker"):
            sync_result = self.broker.sync_positions_from_broker()
            if asyncio.iscoroutine(sync_result):
                await sync_result

        positions = load_positions()

        if len(positions) == 0:
            logger.warning("⚠️ positions.json vacío → fallback broker")
            if hasattr(self.broker, "get_positions"):
                raw = self.broker.get_positions()
                if isinstance(raw, dict):
                    positions = list(raw.values())
                elif isinstance(raw, list):
                    positions = raw
                else:
                    positions = []
                if positions:
                    save_positions(positions)
                    logger.info(f"🔄 Portfolio sincronizado desde broker: {len(positions)} posiciones")

        positions         = self._enrich_positions_with_price(positions)
        broker_positions  = load_positions()
        real_positions    = {p["ticker"].upper() for p in broker_positions}
        portfolio_tickers = real_positions
        mode              = market_ctx.get("market_mode", "neutral")

        logger.info(f"📊 Portfolio: {len(positions)} | Real: {len(real_positions)} | Mode: {mode}")

        alpha_data = self._load_last_alpha()
        alpha_map  = {
            t.upper(): d
            for t, d in alpha_data.get("results", {}).items()
            if isinstance(d, dict)
        }

        intraday_signals = self._load_intraday_signals()

        pm_decisions = await self._get_pm_decisions(
            mode, positions, signals or {}, anchor_universe
        )

        closes = []
        opens  = []

        for decision in pm_decisions:
            if decision.get("action") == "CLOSE":
                ticker = decision.get("ticker", "").upper()
                reason = decision.get("reason", "")

                if ticker not in real_positions:
                    logger.warning(f"⚠️ SKIP CLOSE {ticker} → no existe en broker")
                    continue

                if "hold" in reason.lower():
                    logger.info(f"🛡 SKIP CLOSE {ticker} — HOLD preventivo: {reason}")
                    continue

                alpha_score = alpha_map.get(ticker, {}).get("alpha_score", 0)
                if alpha_score >= self.alpha_hold_shield:
                    logger.info(f"🛡 SHIELD BLOCK {ticker} | alpha={alpha_score:.3f}")
                    continue

                closes.append({"action": "CLOSE", "ticker": ticker, "reason": reason})

        for ticker in portfolio_tickers:
            alpha_score = alpha_map.get(ticker, {}).get("alpha_score", 0)
            if alpha_score <= self.alpha_kill:
                if ticker not in real_positions:
                    continue
                closes.append({
                    "action": "CLOSE",
                    "ticker": ticker,
                    "reason": f"ALPHA_KILL_{alpha_score:.3f}",
                })
                logger.warning(f"💀 KILL {ticker} | alpha={alpha_score:.3f}")

        closes = list({c["ticker"]: c for c in closes}.values())

        # =========================================================
        # PASO 2: Ejecutar CIERRES
        # =========================================================
        close_successes = []
        if closes:
            await self._cancel_pending_orders([c["ticker"] for c in closes])
            await asyncio.sleep(0.5)

            for order in closes[:self.daily_limit]:
                try:
                    await asyncio.wait_for(
                        self.broker.execute_decision(order), timeout=30
                    )
                    close_successes.append(order["ticker"])
                    logger.info(f"⚰️ CLOSED {order['ticker']}")

                    # ── DARWIN: registrar cierre ──────────────────
                    if DARWIN_TRACKING:
                        try:
                            alpha_at_close = alpha_map.get(
                                order["ticker"], {}
                            ).get("alpha_score", 0.0)
                            # Obtener precio de salida desde posiciones enriquecidas
                            exit_price = next(
                                (
                                    float(p.get("price_now") or p.get("entry_price") or 0)
                                    for p in positions
                                    if str(p.get("ticker", "")).upper() == order["ticker"]
                                ),
                                0.0,
                            )
                            register_close(
                                ticker      = order["ticker"],
                                exit_price  = exit_price,
                                reason      = order.get("reason", "UNKNOWN"),
                                alpha_score = alpha_at_close,
                            )
                        except Exception as _te:
                            logger.warning(f"⚠️ darwin close {order['ticker']}: {_te}")
                    # ─────────────────────────────────────────────

                    try:
                        from positions_meta import remove_entry
                        remove_entry(order["ticker"])
                    except Exception:
                        pass
                    await asyncio.sleep(0.8)
                except Exception as e:
                    logger.error(f"❌ CLOSE ERROR {order.get('ticker')}: {e}")

        # =========================================================
        # PASO 3: Refrescar capital DESPUÉS de cierres
        # =========================================================
        self.governor = await self._refresh_governor()

        for decision in pm_decisions:
            if decision.get("action") in ["OPEN", "ROTATE"]:
                opens.append(decision)

        threshold = self._alpha_threshold(mode)

        for ticker, data in alpha_map.items():
            score = data.get("alpha_score", 0)
            if ticker not in portfolio_tickers and score >= threshold:
                opens.append({
                    "action":     "OPEN",
                    "ticker":     ticker,
                    "target_pct": 0.05,
                    "reason":     f"ALPHA_INJECT_{score:.3f}",
                    "alpha":      score,
                })

        # ── FILTRO INTRADAY ──────────────────────────────────────
        if intraday_signals:
            filtered_opens = []
            for o in opens:
                ticker    = o.get("ticker", "").upper()
                is_anchor = (
                    o.get("is_anchor")
                    or o.get("reason", "").startswith("ANCHOR")
                    or ticker in anchor_tickers
                )
                if is_anchor:
                    filtered_opens.append(o)
                    continue
                signal = intraday_signals.get(ticker)
                if signal is None:
                    filtered_opens.append(o)
                    continue
                if signal.get("entrar_ahora", False):
                    logger.info(
                        f"📡 INTRADAY PASS {ticker} | "
                        f"score={signal.get('entry_score')} "
                        f"ratio={signal.get('tracking_ratio')}"
                    )
                else:
                    logger.info(
                        f"⏳ INTRADAY ESPERA {ticker} | "
                        f"score={signal.get('entry_score')} | "
                        f"{signal.get('razon', '')}"
                    )
                filtered_opens.append(o)
            opens = filtered_opens
        # ────────────────────────────────────────────────────────

        opens_dict   = {o["ticker"].upper(): o for o in opens}
        unique_opens = self._enrich_opens_with_price_and_shares(list(opens_dict.values()))

        anchor_opens = [
            o for o in unique_opens
            if o.get("is_anchor")
            or o.get("reason", "").startswith("ANCHOR")
            or o.get("ticker", "").upper() in anchor_tickers
        ]
        normal_opens       = [o for o in unique_opens if o not in anchor_opens]
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

        final_queue = []
        elite_count = 0

        for cmd in sized_opens:
            ticker    = cmd["ticker"].upper()
            score     = alpha_map.get(ticker, {}).get("alpha_score", 0)
            is_anchor = ticker in anchor_tickers

            if score >= self.alpha_elite:
                elite_count += 1
                logger.info(f"🔥 ELITE {ticker} {score:.3f}")
                final_queue.append(cmd)
            elif is_anchor and score > 0:
                logger.info(f"⚓ ANCHOR PASS {ticker} alpha={score:.3f}")
                final_queue.append(cmd)
            elif score >= threshold:
                final_queue.append(cmd)
            else:
                logger.warning(f"⛔ REJECT {ticker} alpha={score:.3f}")

        # =========================================================
        # PASO 4: Ejecutar APERTURAS
        # =========================================================
        results          = []
        executed_opens   = 0
        remaining_orders = self.daily_limit - len(close_successes)

        for order in final_queue[:max(0, remaining_orders)]:
            try:
                result = await asyncio.wait_for(
                    self.broker.execute_decision(order), timeout=30
                )
                results.append({"ticker": order["ticker"], "status": "success"})
                executed_opens += 1

                # ── DARWIN: registrar apertura ────────────────────
                if DARWIN_TRACKING:
                    try:
                        register_open(
                            ticker              = order["ticker"],
                            entry_price         = float(order.get("entry_price") or 0),
                            shares              = int(order.get("shares") or 0),
                            reason              = order.get("reason", "UNKNOWN"),
                            alpha_score         = float(order.get("alpha") or 0),
                            executor_genome_id  = "executor_v1",
                            predictor_genome_id = "predictor_v1",
                        )
                    except Exception as _te:
                        logger.warning(f"⚠️ darwin open {order['ticker']}: {_te}")
                # ─────────────────────────────────────────────────

                if result.get("status") in ("executed", "pending"):
                    try:
                        from positions_meta import set_entry_date
                        set_entry_date(order["ticker"])
                        logger.info(f"📅 entry_date registrado: {order['ticker']}")
                    except Exception as e:
                        logger.warning(f"⚠️ entry_date no registrado {order['ticker']}: {e}")

                await asyncio.sleep(0.8)
                self.governor = await self._refresh_governor()

            except Exception as e:
                logger.error(f"❌ EXECUTION ERROR for {order.get('ticker')}: {e}")
                results.append({
                    "ticker": order.get("ticker"),
                    "status": "error",
                    "error":  str(e),
                })

        # =========================================================
        # PASO 5: Limpiar store si cierres OK
        # =========================================================
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
                logger.info("📂 Cierres parciales → manteniendo store")

        total_executed = len(close_successes) + executed_opens
        logger.info(
            f"✅ EXECUTED {total_executed} | "
            f"Cierres={len(close_successes)} Aperturas={executed_opens} | "
            f"Elite={elite_count} | Capital final=${self.fixed_capital:,.0f} | "
            f"Darwin={'ON' if DARWIN_TRACKING else 'OFF'}"
        )

        return {
            "status":             "success",
            "mode":               mode,
            "executed":           total_executed,
            "closes_executed":    len(close_successes),
            "opens_executed":     executed_opens,
            "elite_executed":     elite_count,
            "queue_size":         len(final_queue),
            "real_capital":       round(self.fixed_capital, 2),
            "alpha_timestamp":    alpha_data.get("timestamp"),
            "intraday_evaluated": len(intraday_signals),
            "darwin_tracking":    DARWIN_TRACKING,
            "results":            results,
            "decisions":          pm_decisions,
        }

    async def preview_executability(self, market_ctx: Dict) -> Dict:
        positions  = self._enrich_positions_with_price(load_positions())
        alpha_data = self._load_last_alpha()
        alpha_map  = {
            t.upper(): d
            for t, d in alpha_data.get("results", {}).items()
            if isinstance(d, dict)
        }
        mode      = market_ctx.get("market_mode", "neutral")
        threshold = self._alpha_threshold(mode)

        rows = []
        for ticker, data in alpha_map.items():
            score = data.get("alpha_score", 0)
            rows.append({
                "ticker":     ticker,
                "alpha":      score,
                "executable": score >= threshold,
                "mode":       mode,
                "threshold":  threshold,
            })

        rows.sort(key=lambda x: x["alpha"] or 0, reverse=True)
        return {"mode": mode, "threshold": threshold, "rows": rows}

    def _load_last_alpha(self) -> Dict:
        if not ALPHA_FILE.exists():
            raise RuntimeError("❌ alpha_last.json no encontrado.")
        try:
            data = json.loads(ALPHA_FILE.read_text())
        except Exception as e:
            raise RuntimeError(f"❌ alpha_last.json corrupto: {e}")
        logger.info(
            f"🧠 Alpha loaded | ts={data.get('timestamp')} | "
            f"universe={data.get('universe_size')}"
        )
        return data

    def _alpha_threshold(self, mode: str) -> float:
        return {
            "growth":    self.alpha_growth,
            "defensive": self.alpha_defensive,
        }.get(mode, self.alpha_neutral)

    async def _get_pm_decisions(
        self,
        mode: str,
        positions: List[Dict],
        signals: Dict,
        anchor: List,
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
                pm  = PMDefensive()
                raw = pm.evaluate_portfolio(positions, anchor, self.fixed_capital)
                if isinstance(raw, list):
                    decisions.extend(
                        [r if isinstance(r, dict) else r.to_dict() for r in raw]
                    )
                elif isinstance(raw, dict):
                    decisions.extend(raw.get("decisions", []))
            else:
                pm  = PMNeutral()
                raw = pm.evaluate_portfolio(positions)
                decisions.extend(raw.get("decisions", []))
        except Exception as e:
            logger.error(f"PM error: {e}")
        return decisions

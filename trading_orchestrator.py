# =========================================================
# trading_orchestrator.py — v4.8 GENOMA REAL CONECTADO (FIX AUDITORIA P2)
# =========================================================
# v4.8 [AUD-P2] (2026-09-04):
#   El SHIELD BLOCK (alpha_score >= ALPHA_SHIELD → bloquea CLOSE) y el
#   barrido ALPHA_KILL (alpha_score <= ALPHA_KILL → fuerza CLOSE) eran
#   dos umbrales fijos, completamente desconectados de ExecutorGenome
#   — el genoma que Darwin evoluciona hace 4+ meses. Cualquier campeón
#   nuevo promovido por Darwin no cambiaba en nada el trading real.
#   Fix: ambos puntos ahora consultan ExecutorGenome.load_champion()
#   .evaluate() — sus 4 reglas (profit-lock/loss-cut, días mínimos
#   según confianza de entrada, escudo anti-kill con confirmación real
#   de H, y score ponderado multi-horizonte) reemplazan los umbrales
#   fijos. Si evaluate() falla por cualquier motivo, HOLD por defecto
#   (fail-safe: no fuerza cierres por un error de lectura de datos).
#
# v4.7:
#   [AUD-P1] Fix Problema 1 de auditoría (fallos de registro de trades
#            silenciados en Darwin):
#     - register_open/register_close ya no solo loguean warning al
#       fallar: ahora es logger.error + persistencia en
#       /data/darwin/tracking_failures.json vía _record_tracking_failure()
#     - Nueva reconciliación diaria al inicio de run(): compara tickers
#       con posiciones reales en el broker contra los trades abiertos
#       hoy en darwin/trades/*.json. Si difieren, logger.error con el
#       detalle. Es un chequeo informativo — no bloquea ejecución ni
#       actúa como circuit breaker nuevo.
#
# v4.6:
#   Refactor arquitectural del flujo intraday:
#   - El tracker ya NO corre con universo propio.
#   - El PM decide primero qué comprar/vender.
#   - El tracker recibe EXACTAMENTE esas listas del PM
#     y evalúa solo timing (ENTRAR/ESPERAR/CERRAR/TRAILING).
#   - La decisión final sigue siendo del orchestrator + CG.
#
#   Cambios respecto a v4.5:
#   [IT1] _load_intraday_signals() eliminado — era universo propio
#   [IT2] Tracker se llama DESPUÉS del PM con listas pm_opens/pm_closes
#   [IT3] Filtro intraday usa sugerencia="ENTRAR" (no "entrar_ahora")
#   [IT4] Cierres: tracker puede sugerir TRAILING — orchestrator respeta
#   [IT5] Orden correcto: PM → Tracker → CG → Broker
# =========================================================

import logging
import asyncio
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, date
from pathlib import Path
from broker import get_engine

from portfolio_store import load_positions, save_positions, clear_positions, update_prices
from capital_governor import CapitalGovernor
from pm_growth import PMGrowth
from pm_neutral import PMNeutral
from pm_defensive import PMDefensive

# [AUD-P2] Genoma real del executor — reemplaza umbrales fijos ALPHA_SHIELD/ALPHA_KILL
from darwin_engine.executor_genome import ExecutorGenome
from darwin_engine.trade_tracker import _read_h_signals
from darwin_engine.pnl_fitness import _get_h_hit_rates

# ── DARWIN TRACKING ───────────────────────────────────────────────
try:
    from darwin_engine.trade_tracker import register_open, register_close
    DARWIN_TRACKING = True
except ImportError:
    DARWIN_TRACKING = False
# ─────────────────────────────────────────────────────────────────

logger = logging.getLogger("trading_orchestrator")
logging.basicConfig(level=logging.INFO)

DATA_PATH     = Path(os.getenv("DATA_PATH", "/data"))
ALPHA_FILE    = DATA_PATH / "alpha_last.json"
ANCHOR_FILE   = Path(os.getenv("ANCHOR_FILE", "/opt/render/project/src/anchor_universe.json"))
CHAMPION_FILE = DATA_PATH / "darwin" / "champion.json"

# [CB1] Archivo SOD (Start-Of-Day equity)
SOD_FILE = DATA_PATH / "circuit_breaker_sod.json"

# Umbral de drawdown diario para activar el breaker (default 5%)
CIRCUIT_BREAKER_DD_PCT = float(os.getenv("CIRCUIT_BREAKER_DD_PCT", "0.05"))

# [AUD-P1] Archivo de fallos de tracking Darwin (open/close no registrados)
TRACKING_FAILURES_FILE = DATA_PATH / "darwin" / "tracking_failures.json"

# [AUD-P1] Directorio de trades registrados por Darwin (para reconciliación)
DARWIN_TRADES_DIR = DATA_PATH / "darwin" / "trades"


def _load_active_genome_ids() -> tuple:
    try:
        if CHAMPION_FILE.exists():
            data    = json.loads(CHAMPION_FILE.read_text())
            exec_id = data.get("genome_id", "executor_v1")
            pred_id = data.get("predictor_genome_id", "predictor_v1")
            return exec_id, pred_id
    except Exception as e:
        logger.warning(f"⚠️ No se pudo leer champion.json: {e} → usando defaults")
    return "executor_v1", "predictor_v1"


def _genome_decision(
    ticker: str,
    position: Dict,
    alpha_score: float,
    genome: "ExecutorGenome",
    h_hit_rates: Dict[str, float],
) -> Dict:
    """
    [AUD-P2] Consulta al genoma campeón real (evaluate()) para decidir
    HOLD/CLOSE de una posición abierta, en vez de los umbrales fijos
    ALPHA_SHIELD/ALPHA_KILL. Fail-safe: si falta algún dato o algo
    falla, retorna HOLD (nunca fuerza un cierre por un error de lectura).
    """
    try:
        entry_date_str = position.get("entry_date")
        entry_date = (
            datetime.strptime(entry_date_str, "%Y-%m-%d").date()
            if entry_date_str else datetime.now(timezone.utc).date()
        )
        days_held = (datetime.now(timezone.utc).date() - entry_date).days

        h_signals_at_entry = _read_h_signals(ticker, entry_date)
        h_signals_now       = _read_h_signals(ticker, datetime.now(timezone.utc).date())

        entry_price   = float(position.get("avg_entry_price") or 0)
        current_price = float(
            position.get("current_price") or position.get("price_now") or entry_price
        )
        current_pnl_pct = (
            (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
        )

        return genome.evaluate(
            trade           = {"h_signals_at_entry": h_signals_at_entry},
            current_pnl_pct = current_pnl_pct,
            alpha_score     = alpha_score,
            h_signals       = h_signals_now,
            h_hit_rates     = h_hit_rates,
            days_held       = days_held,
        )
    except Exception as e:
        logger.warning(f"⚠️ _genome_decision falló para {ticker}: {e} → HOLD (fail-safe)")
        return {"action": "HOLD", "reason": "GENOME_ERROR_FAILSAFE", "confidence": 0.0}


def _record_tracking_failure(kind: str, ticker: str, error: str) -> None:
    """
    [AUD-P1] Persiste un fallo de registro de trade en Darwin (open o close)
    que antes solo se perdía en un logger.warning. Sin este registro, un
    trade real ejecutado en el broker queda invisible para el fitness/
    performance de Darwin, sesgando el reporte de rentabilidad al alza.

    No lanza excepción hacia arriba: un fallo al escribir el propio
    registro de fallos no debe interrumpir la ejecución de trading.
    """
    try:
        TRACKING_FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)

        existing: List[Dict[str, Any]] = []
        if TRACKING_FAILURES_FILE.exists():
            try:
                existing = json.loads(TRACKING_FAILURES_FILE.read_text())
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []

        existing.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind":      kind,       # "open" | "close"
            "ticker":    ticker,
            "error":     error,
        })

        tmp = TRACKING_FAILURES_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing, indent=2))
        tmp.replace(TRACKING_FAILURES_FILE)
    except Exception as e:
        logger.error(f"❌ No se pudo persistir tracking failure ({kind}/{ticker}): {e}")


def _reconcile_broker_vs_darwin(broker_tickers: set) -> None:
    """
    [AUD-P1] Reconciliación diaria: compara los tickers con posición real
    en el broker (hoy) contra los tickers que aparecen en los trades
    abiertos registrados por Darwin hoy (darwin/trades/*.json).

    Chequeo informativo — no bloquea el ciclo ni actúa como circuit
    breaker. Su único objetivo es hacer visible, vía logger.error, la
    divergencia entre "lo que el broker realmente sostiene" y "lo que
    Darwin cree que sostiene", que es la causa raíz sospechada del
    Problema 1 de la auditoría (P&L de Darwin positivo mientras la
    cuenta real cae).
    """
    try:
        if not DARWIN_TRADES_DIR.exists():
            logger.warning(f"⚠️ Reconciliación omitida: {DARWIN_TRADES_DIR} no existe")
            return

        today_str = date.today().isoformat()
        darwin_open_tickers = set()

        for trade_file in DARWIN_TRADES_DIR.glob("*.json"):
            try:
                trade = json.loads(trade_file.read_text())
            except Exception:
                continue

            opened_at = str(trade.get("opened_at", ""))
            closed_at = trade.get("closed_at")

            # Trade abierto hoy y aún sin cerrar → debería reflejarse en el broker
            if opened_at.startswith(today_str) and not closed_at:
                ticker = str(trade.get("ticker", "")).upper()
                if ticker:
                    darwin_open_tickers.add(ticker)

        broker_tickers_upper = {t.upper() for t in broker_tickers}

        missing_in_darwin = broker_tickers_upper - darwin_open_tickers
        missing_in_broker = darwin_open_tickers - broker_tickers_upper

        if missing_in_darwin or missing_in_broker:
            logger.error(
                f"🚨 RECONCILIACIÓN BROKER↔DARWIN DIVERGE | "
                f"en_broker_no_en_darwin={sorted(missing_in_darwin)} | "
                f"en_darwin_no_en_broker={sorted(missing_in_broker)} | "
                f"posible fuga de capital no capturada por darwin/trades/*.json"
            )
        else:
            logger.info(
                f"✅ Reconciliación broker↔Darwin OK | "
                f"{len(broker_tickers_upper)} posiciones coinciden"
            )
    except Exception as e:
        logger.warning(f"⚠️ Reconciliación broker↔Darwin falló: {e}")


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
        logger.info(f"🚀 v4.7 TRACKER-PM + FIX AUDITORIA P1 | fallback capital ${self._fallback_capital:,.0f}")

    # =========================================================
    # CAPITAL
    # =========================================================
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

    # =========================================================
    # [CB1] CIRCUIT BREAKER
    # =========================================================
    async def _circuit_breaker_check(self, current_equity: float) -> Dict[str, Any]:
        today_str  = date.today().isoformat()
        sod_equity = None
        try:
            if SOD_FILE.exists():
                sod_data = json.loads(SOD_FILE.read_text())
                if sod_data.get("date") == today_str:
                    sod_equity = float(sod_data["equity"])
                    logger.info(f"🔋 SOD cargado | fecha={today_str} equity=${sod_equity:,.0f}")
        except Exception as e:
            logger.warning(f"⚠️ Error leyendo SOD: {e}")

        if sod_equity is None:
            sod_equity = current_equity
            try:
                SOD_FILE.parent.mkdir(parents=True, exist_ok=True)
                tmp = SOD_FILE.with_suffix(".tmp")
                tmp.write_text(json.dumps({
                    "date":     today_str,
                    "equity":   sod_equity,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }))
                tmp.replace(SOD_FILE)
                logger.info(f"🔋 SOD nuevo | fecha={today_str} equity=${sod_equity:,.0f}")
            except Exception as e:
                logger.warning(f"⚠️ Error guardando SOD: {e}")

        daily_dd       = (sod_equity - current_equity) / sod_equity if sod_equity > 0 else 0.0
        breaker_active = daily_dd > CIRCUIT_BREAKER_DD_PCT

        if breaker_active:
            logger.error(
                f"🛑 CIRCUIT BREAKER ACTIVO | DD={daily_dd:.2%} > {CIRCUIT_BREAKER_DD_PCT:.0%} | "
                f"SOD=${sod_equity:,.0f} actual=${current_equity:,.0f} | "
                f"APERTURAS BLOQUEADAS — cierres permitidos"
            )
        else:
            logger.info(
                f"✅ Circuit breaker OK | DD={daily_dd:.2%} < {CIRCUIT_BREAKER_DD_PCT:.0%} | "
                f"SOD=${sod_equity:,.0f}"
            )

        return {
            "active":         breaker_active,
            "daily_dd":       round(daily_dd, 4),
            "threshold":      CIRCUIT_BREAKER_DD_PCT,
            "sod_equity":     sod_equity,
            "current_equity": current_equity,
        }

    # =========================================================
    # HELPERS
    # =========================================================
    def _load_anchor_tickers(self) -> set:
        try:
            data    = json.loads(ANCHOR_FILE.read_text())
            tickers = {a["ticker"].upper() for a in data if "ticker" in a}
            logger.info(f"⚓ Anchor universe cargado: {len(tickers)} tickers")
            return tickers
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar anchor_universe.json: {e}")
            return set()

    # [IT1] _load_intraday_signals() ELIMINADO — era universo propio del tracker.
    # En v4.6 el tracker recibe listas del PM via _evaluar_timing_entrada()
    # y _evaluar_posiciones_abiertas().

    def _evaluar_timing_entrada(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        [IT2] Llama al tracker con la lista exacta que el PM decidió comprar.
        Retorna dict {ticker: señal} con sugerencia ENTRAR/ESPERAR.
        """
        if not tickers:
            return {}
        try:
            from intraday_tracker import evaluar_timing_entrada
            signals = evaluar_timing_entrada(tickers)
            entrar  = [t for t, s in signals.items() if s.get("sugerencia") == "ENTRAR"]
            esperar = [t for t, s in signals.items() if s.get("sugerencia") == "ESPERAR"]
            logger.info(
                f"📡 Tracker entrada | evaluados={len(signals)} | "
                f"ENTRAR={entrar} | ESPERAR={esperar}"
            )
            return signals
        except Exception as e:
            logger.warning(f"⚠️ evaluar_timing_entrada no disponible: {e}")
            return {}

    def _evaluar_posiciones_abiertas(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        [IT2] Llama al tracker con la lista exacta de posiciones abiertas del PM.
        Retorna dict {ticker: señal} con sugerencia MANTENER/CERRAR/TRAILING.
        """
        if not tickers:
            return {}
        try:
            from intraday_tracker import evaluar_posiciones_abiertas
            signals  = evaluar_posiciones_abiertas(tickers)
            cerrar   = [t for t, s in signals.items() if s.get("sugerencia") == "CERRAR"]
            trailing = [t for t, s in signals.items() if s.get("sugerencia") == "TRAILING"]
            mantener = [t for t, s in signals.items() if s.get("sugerencia") == "MANTENER"]
            logger.info(
                f"📊 Tracker posiciones | evaluados={len(signals)} | "
                f"CERRAR={cerrar} | TRAILING={trailing} | MANTENER={mantener}"
            )
            return signals
        except Exception as e:
            logger.warning(f"⚠️ evaluar_posiciones_abiertas no disponible: {e}")
            return {}

    def _get_broker_positions_tickers(self) -> set:
        try:
            broker_data = self.broker.get_positions()
            if not broker_data:
                logger.warning("⚠️ Broker devolvió 0 posiciones directas")
                return set()
            if isinstance(broker_data, dict):
                tickers = {t.upper() for t in broker_data.keys()}
            elif isinstance(broker_data, list):
                tickers = {
                    str(p.get("ticker", p.get("symbol", ""))).upper()
                    for p in broker_data
                    if p.get("ticker") or p.get("symbol")
                }
            else:
                tickers = set()
            logger.info(f"📋 Broker positions (Alpaca directo): {sorted(tickers)}")
            return tickers
        except Exception as e:
            logger.warning(f"⚠️ _get_broker_positions_tickers error: {e}")
            return set()

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
                pos["price_now"]  = broker_price_map[ticker]
                price_map[ticker] = broker_price_map[ticker]
            elif float(pos.get("entry_price", 0) or 0) > 0:
                pos["price_now"] = float(pos["entry_price"])
                logger.warning(
                    f"⚠️ {ticker} sin precio Alpaca → entry_price: {pos['price_now']:.2f}"
                )
            else:
                # [F13] FIX: antes se usaba price_now=1.0 como placeholder.
                # pm_defensive.py calcula ret=(price/entry - 1) sobre ese
                # valor y lo interpreta como caída real de ~99%, disparando
                # "catastrophic_loss" con exit_price=1.0 — cierre real con
                # pérdida ficticia (confirmado en trades reales GILD/SO,
                # ~$4,200 USD de pérdida registrada sobre un precio falso).
                # Ahora: sin precio confiable → excluir del ciclo, no inventar.
                logger.error(f"❌ {ticker} SIN PRECIO CONFIABLE → excluyendo de este ciclo")
                pos["price_now"] = None
                pos["price_unreliable"] = True

            if pos.get("price_now") is not None:
                enriched.append(pos)
            else:
                logger.warning(f"⚠️ {ticker} excluido de evaluación por precio no confiable")

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

            try:
                broker_data = self.broker.get_positions()
                if broker_data and ticker in broker_data:
                    bp = broker_data[ticker].get("current_price") or broker_data[ticker].get("price_now")
                    if bp and float(bp) > 0:
                        price = float(bp)
            except Exception:
                pass

            if not price or float(price) <= 0:
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

            o                = dict(o)
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
    # RUN PRINCIPAL v4.7
    # =========================================================
    async def run(
        self,
        market_ctx: Dict[str, Any],
        signals: Dict = None,
        anchor_universe: List = None
    ) -> Dict:

        self.governor = await self._refresh_governor()
        logger.info(f"🚀 v4.7 TRACKER-PM + FIX AUDITORIA P1 | Capital ${self.fixed_capital:,.0f}")

        anchor_tickers = self._load_anchor_tickers()

        executor_genome_id, predictor_genome_id = _load_active_genome_ids()
        logger.info(f"🧬 Genomas activos | executor={executor_genome_id} predictor={predictor_genome_id}")

        # [CB1] Circuit breaker — chequear antes de cualquier ejecución
        cb_result      = await self._circuit_breaker_check(self.fixed_capital)
        breaker_active = cb_result["active"]

        if hasattr(self.broker, "sync_positions_from_broker"):
            sync_result = self.broker.sync_positions_from_broker()
            if asyncio.iscoroutine(sync_result):
                await sync_result

        positions = load_positions()

        if len(positions) == 0:
            logger.warning("⚠️ positions.json vacío → fallback broker")
            if hasattr(self.broker, "get_positions"):
                try:
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
                    else:
                        from pathlib import Path as _P
                        import json as _j
                        _disk = DATA_PATH / "positions.json"
                        if _disk.exists():
                            _raw = _j.loads(_disk.read_text())
                            if _raw:
                                positions = _raw if isinstance(_raw, list) else [{"ticker": t, **v} for t, v in _raw.items()]
                                logger.warning(f"⚠️ Broker vacío → usando disco stale: {len(positions)} posiciones")
                except Exception as e:
                    logger.error(f"❌ Broker get_positions falló: {e} → usando disco stale")
                    from pathlib import Path as _P
                    import json as _j
                    _disk = DATA_PATH / "positions.json"
                    if _disk.exists():
                        try:
                            _raw = _j.loads(_disk.read_text())
                            if _raw:
                                positions = _raw if isinstance(_raw, list) else [{"ticker": t, **v} for t, v in _raw.items()]
                                logger.warning(f"⚠️ Usando disco stale: {len(positions)} posiciones")
                        except Exception:
                            pass

        positions = self._enrich_positions_with_price(positions)

        broker_real_tickers = self._get_broker_positions_tickers()
        broker_positions    = load_positions()
        disk_tickers        = {p["ticker"].upper() for p in broker_positions}
        real_positions      = broker_real_tickers | disk_tickers
        portfolio_tickers   = real_positions

        # [AUD-P1] Reconciliación diaria broker↔Darwin — informativa, no bloquea.
        if DARWIN_TRACKING:
            _reconcile_broker_vs_darwin(broker_real_tickers)

        logger.info(
            f"📊 Portfolio | broker={len(broker_real_tickers)} "
            f"disco={len(disk_tickers)} union={len(real_positions)} | "
            f"Mode: {market_ctx.get('market_mode', 'neutral')}"
        )

        mode = market_ctx.get("market_mode", "neutral")

        alpha_data = self._load_last_alpha()
        alpha_map  = {
            t.upper(): d
            for t, d in alpha_data.get("results", {}).items()
            if isinstance(d, dict)
        }

        # =========================================================
        # PASO 1: PM decide primero — cierra y abre
        # [IT5] Orden correcto: PM → Tracker → CG → Broker
        # =========================================================
        pm_decisions = await self._get_pm_decisions(
            mode, positions, signals or {}, anchor_universe,
            alpha_map=alpha_map,
        )

        closes = []
        opens  = []

        # [AUD-P2] Genoma real cargado una vez por ciclo (no por ticker)
        genome      = ExecutorGenome.load_champion()
        h_hit_rates = _get_h_hit_rates()

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
                # [AUD-P2] Reemplaza el SHIELD BLOCK fijo (alpha >= 0.75)
                genome_dec = _genome_decision(
                    ticker, real_positions[ticker], alpha_score, genome, h_hit_rates
                )
                if genome_dec["action"] == "HOLD":
                    logger.info(
                        f"🧬 GENOME HOLD {ticker} | alpha={alpha_score:.3f} | "
                        f"pm_reason={reason} | genoma={genome_dec['reason']}"
                    )
                    continue

                closes.append({
                    "action": "CLOSE",
                    "ticker": ticker,
                    "reason": f"{reason}|GENOME_{genome_dec['reason']}",
                })

        # [AUD-P2] Reemplaza el barrido ALPHA_KILL fijo (alpha <= -0.40):
        # cualquier posición real que el genoma marque CLOSE, aunque el
        # PM no haya propuesto cerrarla, se cierra igual (cubre lo que
        # antes cubría KILL, ahora con las 4 reglas reales del genoma).
        ya_en_cierre = {c["ticker"] for c in closes}
        for ticker in portfolio_tickers:
            if ticker in ya_en_cierre or ticker not in real_positions:
                continue
            alpha_score = alpha_map.get(ticker, {}).get("alpha_score", 0)
            genome_dec  = _genome_decision(
                ticker, real_positions[ticker], alpha_score, genome, h_hit_rates
            )
            if genome_dec["action"] == "CLOSE":
                closes.append({
                    "action": "CLOSE",
                    "ticker": ticker,
                    "reason": f"GENOME_{genome_dec['reason']}",
                })
                logger.warning(
                    f"🧬 GENOME CLOSE {ticker} | alpha={alpha_score:.3f} | {genome_dec['reason']}"
                )

        closes = list({c["ticker"]: c for c in closes}.values())

        # =========================================================
        # [IT4] TRACKER VALIDA CIERRES — considera PnL actual
        # Sugerencia TRAILING → no cerrar posiciones muy ganadoras
        # =========================================================
        close_tickers_pm = [c["ticker"] for c in closes]
        if close_tickers_pm:
            tracker_closes = self._evaluar_posiciones_abiertas(close_tickers_pm)
            closes_filtrados = []
            for c in closes:
                ticker   = c["ticker"]
                señal    = tracker_closes.get(ticker)
                if señal and señal.get("sugerencia") == "TRAILING":
                    pnl = señal.get("pnl_actual_pct", 0)
                    logger.info(
                        f"🛡 TRAILING BLOCK {ticker} | PnL={pnl:.1f}% → "
                        f"no cerrar, activar trailing stop"
                    )
                    # Marcamos como trailing para que el orchestrator lo monitoree
                    c = {**c, "trailing": True, "pnl_actual_pct": pnl}
                    closes_filtrados.append(c)
                else:
                    closes_filtrados.append(c)
            closes = closes_filtrados

        # =========================================================
        # PASO 2: Ejecutar CIERRES (siempre — breaker no los bloquea)
        # Solo cierra los que NO tienen trailing=True
        # =========================================================
        close_successes = []
        closes_ejecutar = [c for c in closes if not c.get("trailing")]
        closes_trailing = [c for c in closes if c.get("trailing")]

        if closes_trailing:
            logger.info(
                f"⏳ TRAILING activo en: "
                f"{[c['ticker'] for c in closes_trailing]} → no se cierran"
            )

        if closes_ejecutar:
            await self._cancel_pending_orders([c["ticker"] for c in closes_ejecutar])
            await asyncio.sleep(0.5)

            for order in closes_ejecutar[:self.daily_limit]:
                try:
                    result = await asyncio.wait_for(
                        self.broker.execute_decision(order), timeout=30
                    )
                    close_successes.append(order["ticker"])
                    logger.info(f"⚰️ CLOSED {order['ticker']}")

                    if DARWIN_TRACKING:
                        try:
                            alpha_at_close = alpha_map.get(
                                order["ticker"], {}
                            ).get("alpha_score", 0.0)

                            exit_price = 0.0
                            if result and hasattr(result, "filled_avg_price") and result.filled_avg_price:
                                exit_price = float(result.filled_avg_price)
                            elif result and isinstance(result, dict):
                                exit_price = float(
                                    result.get("filled_avg_price")
                                    or result.get("fill_price")
                                    or result.get("price")
                                    or 0
                                )

                            if exit_price <= 0:
                                exit_price = next(
                                    (
                                        float(p.get("price_now") or p.get("entry_price") or 0)
                                        for p in positions
                                        if str(p.get("ticker", "")).upper() == order["ticker"]
                                    ),
                                    0.0,
                                )

                            logger.info(f"📝 Darwin close {order['ticker']} @ ${exit_price:.2f}")
                            register_close(
                                ticker      = order["ticker"],
                                exit_price  = exit_price,
                                reason      = order.get("reason", "UNKNOWN"),
                                alpha_score = alpha_at_close,
                            )
                        except Exception as _te:
                            # [AUD-P1] Antes: solo logger.warning, el fallo se perdía.
                            # Un cierre real que no se registra en Darwin deja el
                            # trade "abierto" para el fitness aunque ya no exista
                            # en el broker — sesga el cálculo de performance.
                            logger.error(f"❌ DARWIN CLOSE FALLÓ {order['ticker']}: {_te}")
                            _record_tracking_failure("close", order["ticker"], str(_te))

                    try:
                        from positions_meta import remove_entry
                        remove_entry(order["ticker"])
                    except Exception:
                        pass
                    await asyncio.sleep(0.8)
                except Exception as e:
                    logger.error(f"❌ CLOSE ERROR {order.get('ticker')}: {e}")

        # =========================================================
        # PASO 3: Refrescar capital + portfolio post-cierres
        # =========================================================
        self.governor = await self._refresh_governor()

        broker_real_post  = self._get_broker_positions_tickers()
        portfolio_tickers = broker_real_post | (disk_tickers - set(close_successes))
        logger.info(
            f"🔄 Portfolio post-cierres | "
            f"broker={len(broker_real_post)} | "
            f"efectivos={sorted(portfolio_tickers)}"
        )

        # [CB1] Si breaker activo → saltar aperturas
        if breaker_active:
            logger.error(
                f"🛑 CIRCUIT BREAKER — aperturas omitidas | "
                f"DD={cb_result['daily_dd']:.2%} | "
                f"cierres ejecutados={len(close_successes)}"
            )
            return {
                "status":             "circuit_breaker",
                "mode":               mode,
                "executed":           len(close_successes),
                "closes_executed":    len(close_successes),
                "opens_executed":     0,
                "elite_executed":     0,
                "queue_size":         0,
                "real_capital":       round(self.fixed_capital, 2),
                "circuit_breaker":    cb_result,
                "darwin_tracking":    DARWIN_TRACKING,
                "results":            [],
                "decisions":          pm_decisions,
            }

        # =========================================================
        # PASO 4: Construir cola de APERTURAS desde PM
        # =========================================================
        # [AUD-P3][2026-08-26] Se eliminó el bloque ALPHA_INJECT que
        # existía aquí: iteraba sobre TODO alpha_map (no sobre
        # decisiones del PM) y agregaba aperturas directas con
        # target_pct=0.05 fijo, sin pasar por ningún criterio de
        # calidad del Portfolio Manager, sin distinguir anclas, y con
        # un umbral (self._alpha_threshold(mode)) que ni siquiera
        # coincidía con el que alpha_engine_v4.py usa para reportar
        # "candidatos operables" (0.70 fijo vs 0.65/0.75/0.85
        # dinámico aquí). Ahora TODA apertura sale exclusivamente de
        # pm_decisions — mismo criterio de calidad y sizing para
        # cualquier posición nueva, sin vía paralela sin control.
        for decision in pm_decisions:
            if decision.get("action") in ["OPEN", "ROTATE"]:
                opens.append(decision)

        threshold = self._alpha_threshold(mode)

        # =========================================================
        # [IT2][IT3] TRACKER VALIDA TIMING DE ENTRADAS
        # Recibe EXACTAMENTE la lista que el PM decidió comprar.
        # Anclas pasan directo — tracker no bloquea anclas.
        # =========================================================
        pm_open_tickers = [
            o["ticker"].upper() for o in opens
            if not (
                o.get("is_anchor")
                or o.get("reason", "").startswith("ANCHOR")
                or o.get("ticker", "").upper() in anchor_tickers
            )
        ]

        tracker_entradas = self._evaluar_timing_entrada(pm_open_tickers)

        filtered_opens = []
        for o in opens:
            ticker    = o.get("ticker", "").upper()
            is_anchor = (
                o.get("is_anchor")
                or o.get("reason", "").startswith("ANCHOR")
                or ticker in anchor_tickers
            )
            if is_anchor:
                # Anclas siempre pasan — tracker no las bloquea
                filtered_opens.append(o)
                continue

            señal = tracker_entradas.get(ticker)

            if señal is None:
                # Sin señal del tracker → pasar (sin datos no bloqueamos)
                filtered_opens.append(o)
                continue

            if señal.get("sugerencia") == "ENTRAR":
                logger.info(
                    f"📡 INTRADAY PASS {ticker} | "
                    f"score={señal.get('entry_score')} "
                    f"ratio={señal.get('tracking_ratio')}"
                )
                filtered_opens.append(o)
            else:
                # [IT3] ESPERAR → no entra hoy
                logger.info(
                    f"⏳ INTRADAY ESPERA {ticker} | "
                    f"score={señal.get('entry_score')} | "
                    f"{señal.get('razon', '')}"
                )

        opens = filtered_opens

        # =========================================================
        # PASO 4b: CG sizing
        # =========================================================
        opens_dict   = {o["ticker"].upper(): o for o in opens}
        unique_opens = self._enrich_opens_with_price_and_shares(list(opens_dict.values()))

        anchor_opens = [
            o for o in unique_opens
            if o.get("is_anchor")
            or o.get("reason", "").startswith("ANCHOR")
            or o.get("ticker", "").upper() in anchor_tickers
        ]
        normal_opens       = [o for o in unique_opens if o not in anchor_opens]
        close_tickers_list = [c["ticker"] for c in closes_ejecutar]

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
        # PASO 5: Ejecutar APERTURAS
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

                if DARWIN_TRACKING:
                    try:
                        register_open(
                            ticker              = order["ticker"],
                            entry_price         = float(order.get("entry_price") or 0),
                            shares              = int(order.get("shares") or 0),
                            reason              = order.get("reason", "UNKNOWN"),
                            alpha_score         = float(order.get("alpha") or 0),
                            executor_genome_id  = executor_genome_id,
                            predictor_genome_id = predictor_genome_id,
                        )
                        logger.info(
                            f"📝 TRADE OPEN registrado | {order['ticker']} | "
                            f"entrada=${order.get('entry_price', 0):.2f}"
                        )
                    except Exception as _te:
                        # [AUD-P1] Antes: solo logger.warning, el fallo se perdía.
                        # Un open real que no se registra en Darwin queda
                        # totalmente invisible para pnl_fitness.py: si ese
                        # trade termina en pérdida, nunca entra al cálculo de
                        # fitness ni al performance report.
                        logger.error(f"❌ DARWIN OPEN FALLÓ {order['ticker']}: {_te}")
                        _record_tracking_failure("open", order["ticker"], str(_te))

                if result.get("status") == "executed":
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
        # PASO 6: Limpiar store si cierres OK
        # =========================================================
        if close_successes:
            all_closes_ok = len(close_successes) == len(closes_ejecutar)
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
            f"Trailing={len(closes_trailing)} | Elite={elite_count} | "
            f"Capital final=${self.fixed_capital:,.0f} | "
            f"Darwin={'ON' if DARWIN_TRACKING else 'OFF'}"
        )

        return {
            "status":             "success",
            "mode":               mode,
            "executed":           total_executed,
            "closes_executed":    len(close_successes),
            "opens_executed":     executed_opens,
            "trailing_activo":    [c["ticker"] for c in closes_trailing],
            "elite_executed":     elite_count,
            "queue_size":         len(final_queue),
            "real_capital":       round(self.fixed_capital, 2),
            "alpha_timestamp":    alpha_data.get("timestamp"),
            "intraday_evaluated": len(tracker_entradas),
            "darwin_tracking":    DARWIN_TRACKING,
            "circuit_breaker":    cb_result,
            "results":            results,
            "decisions":          pm_decisions,
        }

    # =========================================================
    # PREVIEW EXECUTABILITY
    # =========================================================
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

    # =========================================================
    # HELPERS INTERNOS
    # =========================================================
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
        alpha_map: Dict = None,
    ) -> List[Dict]:
        decisions = []
        try:
            if mode == "growth":
                pm = PMGrowth(self.fixed_capital)
                for pos in positions:
                    ticker = pos.get("ticker", "").upper()
                    # [FIX v4.7] signals siempre llegaba vacío desde run().
                    # PMGrowth.evaluate_position() quedaba con signal=None
                    # en todas las posiciones → stops de emergencia sin
                    # contexto real. Ahora construimos signal desde alpha_map.
                    alpha_entry = (alpha_map or {}).get(ticker, {})
                    # [AUD-P4] FIX: confidence vive anidado en
                    # alpha_entry["components"]["confidence"], no top-level.
                    # Antes esto siempre devolvía 0.0, disparando
                    # confidence_decay en pm_growth.py de forma sistemática
                    # e independiente de la calidad real de la señal.
                    components = alpha_entry.get("components", {})
                    signal = {
                        "confidence":  components.get("confidence", 0.0),
                        "direction":   "COMPRA" if alpha_entry.get("alpha_score", 0) > 0 else "VENDE",
                        "alpha_score": alpha_entry.get("alpha_score", 0.0),
                    } if alpha_entry else None
                    d = pm.evaluate_position(pos, signal)
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

# =========================================================
# pm_growth.py — GROWTH POSITION MANAGER v2.7.0
# =========================================================
# ✔ PM GROWTH PURO (market_mode == growth)
# ✔ Decision engine (NO ejecuta órdenes)
# ✔ REQUIERE señales predictivas (signal obligatorio)
# ✔ Integra contexto FUNDAMENTAL (Model 2) solo como lectura
# ✔ Trailing stop + rotation + sizing
# ✔ Compatible con main v2.9.3 + broker
#
# FIX v2.6.4:
#   [G1] evaluate_position: signal=None antes retornaba HOLD
#        silencioso — posiciones bajando quedaban huérfanas
#        indefinidamente si el predictor fallaba y signals={}.
#        Ahora evalúa stops de emergencia (stop_loss y
#        trailing_stop) aunque no haya signal, y loggea error
#        para visibilidad. Razones diferenciadas:
#        "stop_loss_no_signal" / "trailing_stop_no_signal" /
#        "no_signal_hold_warned".
#
# v2.7.0 — Contexto de curva futura del tracker:
#   [CF1] evaluate_position() acepta tracker_signal: Dict = None
#         Parámetro keyword opcional — retrocompatible.
#         El orchestrator lo pasa cuando tiene señales del tracker.
#   [CF2] Lógica curva futura (aplica DESPUÉS de stops de emergencia
#         y DESPUÉS de stop_loss/trailing_stop — stops tienen prioridad):
#         - pnl > 0 + slope="baja" → CLOSE "growth_close_curve_falling"
#           (tomar ganancia antes de caída esperada)
#         - pnl > 0 + slope="sube" → no aplicar time_decay ni
#           confidence_decay (destino alcista — no cortar por tiempo)
#         - en_zona_valle=True → se agrega a meta como contexto
#         Trailing stop y stop loss NUNCA se sobreescriben.
#   [CF3] Si tracker_signal es None o no tiene curva_futura,
#         comportamiento idéntico a v2.6.4 — sin regresiones.
#   [CF4] _decision() acepta curva_futura_meta opcional para
#         incluir contexto de curva en el meta del resultado.
# =========================================================

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pytz
import time

from model2 import fundamental_signal_context

# =========================================================
# ENV CONFIG (GROWTH)
# =========================================================
MAX_HOLD_DAYS          = int(os.getenv("PM_MAX_HOLD_DAYS",        "20"))
MIN_CONFIDENCE_HOLD    = float(os.getenv("PM_MIN_CONFIDENCE_HOLD", "0.55"))
ROTATION_CONF_DELTA    = float(os.getenv("PM_ROTATION_DELTA",      "0.15"))

TAKE_PROFIT_PCT        = float(os.getenv("PM_TAKE_PROFIT",         "0.10"))
STOP_LOSS_PCT          = float(os.getenv("PM_STOP_LOSS",           "0.05"))
TRAILING_STOP_PCT      = float(os.getenv("PM_TRAILING_STOP",       "0.03"))

FIXED_CAPITAL          = float(os.getenv("PM_FIXED_CAPITAL",       "100000"))
MAX_PORTFOLIO_RISK_PCT = float(os.getenv("PM_MAX_PORT_RISK",        "0.02"))
MAX_RISK_PER_TRADE_PCT = float(os.getenv("PM_RISK_PER_TRADE",       "0.01"))

CL_TIMEZONE = pytz.timezone("America/Santiago")

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pm_growth")

# =========================================================
# STRUCTS
# =========================================================
class PortfolioHealth(Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    RED    = "RED"

# =========================================================
# HELPERS
# =========================================================
def pct_change(current: float, entry: float) -> float:
    return (current / entry - 1.0) if entry > 0 else 0.0

def days_between(entry_iso: str) -> int:
    try:
        dt = datetime.fromisoformat(entry_iso.replace("Z", "+00:00"))
        return (datetime.now(CL_TIMEZONE) - dt.astimezone(CL_TIMEZONE)).days
    except Exception:
        return MAX_HOLD_DAYS

def calculate_position_size(
    risk_amount: float, entry_price: float, stop_price: float
) -> int:
    if entry_price <= 0 or stop_price >= entry_price:
        return 0
    stop_distance = (entry_price - stop_price) / entry_price
    if stop_distance <= 0:
        return 0
    shares = int(risk_amount / (entry_price * stop_distance))
    return max(1, shares)


# =========================================================
# HELPER CURVA FUTURA — v2.7.0
# =========================================================

def _leer_curva_futura(tracker_signal: Optional[Dict]) -> Optional[Dict]:
    """
    [CF1] Extrae curva_futura de la señal del tracker.
    Retorna None si no está disponible — sin efectos secundarios.
    """
    if not tracker_signal or not isinstance(tracker_signal, dict):
        return None
    return tracker_signal.get("curva_futura") or None


# =========================================================
# PM GROWTH
# =========================================================
class PMGrowth:
    """
    PM GROWTH.
    SOLO debe usarse cuando market_mode == 'growth'.
    """

    def __init__(self, fixed_capital: float = FIXED_CAPITAL):
        self.fixed_capital = fixed_capital
        self.tz = CL_TIMEZONE
        self._fundamental_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("📈 PMGrowth v2.7.0 inicializado")

    # --------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------
    def allow_new_positions(self) -> bool:
        return True

    # --------------------------------------------------
    # FUNDAMENTAL CONTEXT (CACHE 1h)
    # --------------------------------------------------
    def _get_fundamental_context(self, ticker: str) -> Dict[str, Any]:
        now = time.time()
        key = ticker.upper()

        self._fundamental_cache = {
            k: v for k, v in self._fundamental_cache.items()
            if now - v["timestamp"] < 3600
        }

        if key in self._fundamental_cache:
            return self._fundamental_cache[key]["data"]

        try:
            f = fundamental_signal_context(key)
            if not f.get("usable"):
                data = {"usable": False}
            else:
                mis   = f.get("mispricing_pct", 0.0)
                state = (
                    "UNDERVALUED" if mis < -15
                    else "OVERVALUED" if mis > 15
                    else "FAIR"
                )
                data = {
                    "usable":            True,
                    "mispricing_pct":    mis,
                    "margin_safety_pct": f.get("margin_safety_pct"),
                    "state":             state,
                    "model":             f.get("model"),
                }

            self._fundamental_cache[key] = {"data": data, "timestamp": now}
            return data

        except Exception:
            return {"usable": False}

    # --------------------------------------------------
    # POSITION EVALUATION
    # --------------------------------------------------
    def evaluate_position(
        self,
        pos:            Dict[str, Any],
        signal:         Dict[str, Any],
        tracker_signal: Optional[Dict] = None,   # [CF1] señal del tracker con curva_futura
    ) -> Dict[str, Any]:
        """
        Evalúa posición existente.
        Signal es preferible pero no bloquea stops de emergencia.

        [G1] Si signal=None: evalúa trailing_stop y stop_loss de
             emergencia antes de hacer HOLD — evita posiciones
             huérfanas cuando el predictor falla.

        [CF1] tracker_signal=None: comportamiento idéntico a v2.6.4.
              Si tiene curva_futura, se usa para refinar decisiones
              de time_decay y confidence_decay (no toca stops).
        """
        ticker = str(pos.get("ticker", "")).upper()
        entry  = float(pos.get("entry_price", 0))
        price  = float(pos.get("price_now", 0))
        peak   = float(pos.get("peak_price", entry))
        age    = days_between(str(pos.get("entry_time", "")))

        fundamental  = self._get_fundamental_context(ticker)
        curva_futura = _leer_curva_futura(tracker_signal)

        # [G1] Sin signal — evaluar stops de emergencia antes de HOLD
        if not signal:
            logger.error(
                f"🔴 {ticker} SIN SIGNAL en PMGrowth — "
                f"evaluando stops de emergencia (ret={pct_change(price, entry):+.1%})"
            )
            if entry > 0 and price > 0:
                ret        = pct_change(price, entry)
                trail_stop = peak * (1 - TRAILING_STOP_PCT)

                if ret <= -STOP_LOSS_PCT:
                    logger.warning(f"💀 {ticker} stop_loss_no_signal | ret={ret:.1%}")
                    return self._decision(
                        "CLOSE", ticker, "stop_loss_no_signal", fundamental,
                        {"ret_pct": round(ret * 100, 2), "stop_pct": -STOP_LOSS_PCT * 100},
                    )

                if price <= trail_stop:
                    logger.warning(
                        f"💀 {ticker} trailing_stop_no_signal | "
                        f"price={price:.2f} trail={trail_stop:.2f}"
                    )
                    return self._decision(
                        "CLOSE", ticker, "trailing_stop_no_signal", fundamental,
                        {"trail_price": round(trail_stop, 2)},
                    )

            return self._decision(
                "HOLD", ticker, "no_signal_hold_warned", fundamental,
                {"warning": "signal ausente — stops de emergencia evaluados"},
            )

        conf = float(signal.get("confidence", 0.0))
        ret  = pct_change(price, entry)

        # ---- TRAILING STOP — prioridad máxima, nunca se sobreescribe ----
        trail_stop = peak * (1 - TRAILING_STOP_PCT)
        if price <= trail_stop:
            return self._decision(
                "CLOSE", ticker, "trailing_stop", fundamental,
                {"trail_price": round(trail_stop, 2)},
            )

        # ---- STOP LOSS — prioridad máxima, nunca se sobreescribe ----
        if price <= entry * (1 - STOP_LOSS_PCT):
            return self._decision("CLOSE", ticker, "stop_loss", fundamental)

        # ---- TAKE PROFIT ----
        if ret >= TAKE_PROFIT_PCT:
            return self._decision("CLOSE", ticker, "take_profit", fundamental)

        # ── [CF2] Curva futura: cierre estratégico antes de caída ──────────
        # Se evalúa después de stops duros y take_profit.
        # Trailing/stop_loss nunca se tocan — tienen prioridad absoluta.
        # ───────────────────────────────────────────────────────────────────
        if curva_futura:
            slope         = curva_futura.get("slope_futura")
            en_zona_valle = curva_futura.get("en_zona_valle", False)
            ret_a_peak    = curva_futura.get("ret_desde_hoy_a_peak_pct")
            ret_a_final   = curva_futura.get("ret_desde_hoy_a_final_pct")
            dias_peak     = curva_futura.get("dias_hasta_peak")
            pnl_pct       = round(ret * 100, 2)

            # PnL positivo + curva cae → cerrar tomando ganancia
            if pnl_pct > 0 and slope == "baja":
                logger.info(
                    f"📉 {ticker} CLOSE | PnL={pnl_pct:.1f}% + curva cae "
                    f"(slope={slope} ret_final={ret_a_final}%) → tomar ganancia"
                )
                return self._decision(
                    "CLOSE", ticker, "growth_close_curve_falling", fundamental,
                    {
                        "ret_pct":         pnl_pct,
                        "slope_futura":    slope,
                        "ret_a_final_pct": ret_a_final,
                        "dias_hasta_peak": dias_peak,
                        "days_held":       age,
                        "en_zona_valle":   en_zona_valle,
                    },
                )

            # PnL positivo + curva sube → no aplicar time_decay ni confidence_decay
            # El destino es alcista — no cortar por tiempo o confianza baja
            if pnl_pct > 0 and slope == "sube":
                logger.info(
                    f"📈 {ticker} HOLD | PnL={pnl_pct:.1f}% + curva sube "
                    f"(ret_peak={ret_a_peak}% en {dias_peak}d) → omitir time/confidence"
                )
                return self._decision(
                    "HOLD", ticker, "hold_curve_recovering", fundamental,
                    {
                        "ret_pct":         pnl_pct,
                        "slope_futura":    slope,
                        "ret_a_peak_pct":  ret_a_peak,
                        "dias_hasta_peak": dias_peak,
                        "days_held":       age,
                        "en_zona_valle":   en_zona_valle,
                        "confidence":      conf,
                    },
                )

        # ---- TIME DECAY — después de curva (puede ser anulado si slope=sube) ----
        if age >= MAX_HOLD_DAYS:
            return self._decision("CLOSE", ticker, "time_decay", fundamental)

        # ---- CONFIDENCE DECAY — después de curva ----
        if conf < MIN_CONFIDENCE_HOLD:
            return self._decision("CLOSE", ticker, "confidence_decay", fundamental)

        # ---- UPDATE PEAK ----
        if price > peak:
            pos["peak_price"] = price

        # ---- HOLD — agrega contexto de curva si disponible ----
        hold_extra = {
            "ret_pct":    round(ret * 100, 2),
            "days":       age,
            "confidence": conf,
        }
        if curva_futura:
            hold_extra["slope_futura"]    = curva_futura.get("slope_futura")
            hold_extra["en_zona_valle"]   = curva_futura.get("en_zona_valle", False)
            hold_extra["ret_a_peak_pct"]  = curva_futura.get("ret_desde_hoy_a_peak_pct")
            hold_extra["dias_hasta_peak"] = curva_futura.get("dias_hasta_peak")

        return self._decision("HOLD", ticker, "healthy", fundamental, hold_extra)

    # --------------------------------------------------
    # ROTATION (GROWTH ONLY)
    # --------------------------------------------------
    def evaluate_rotation(
        self,
        open_positions:  List[Dict[str, Any]],
        new_candidate:   Dict[str, Any],
        latest_signals:  Dict[str, Dict[str, Any]],
        tracker_signals: Optional[Dict[str, Dict]] = None,  # [CF1] señales del tracker
    ) -> Optional[Dict[str, Any]]:

        new_ticker = str(new_candidate.get("ticker", "")).upper()
        new_signal = latest_signals.get(new_ticker)

        if not new_signal:
            return None

        fund = self._get_fundamental_context(new_ticker)

        if fund.get("usable") and fund.get("state") == "OVERVALUED":
            return None

        tracker_signals = tracker_signals or {}
        close_ticker    = None

        for pos in open_positions:
            t   = pos.get("ticker", "").upper()
            sig = latest_signals.get(t)
            ts  = tracker_signals.get(t)         # [CF1] señal tracker para este ticker
            d   = self.evaluate_position(pos, sig, tracker_signal=ts)
            if d["action"] == "HOLD":
                close_ticker = pos["ticker"]
                break

        if not close_ticker:
            return None

        return {
            "action":       "ROTATE",
            "close_ticker": close_ticker,
            "open_ticker":  new_ticker,
            "timestamp":    datetime.now(self.tz).isoformat(),
            "meta": {
                "fundamental_new": fund,
            },
        }

    # --------------------------------------------------
    # NEW POSITION SIZING
    # --------------------------------------------------
    def calculate_new_position(
        self,
        ticker:      str,
        entry_price: float,
        stop_price:  Optional[float] = None,
    ) -> Dict[str, Any]:

        risk_amount = self.fixed_capital * MAX_RISK_PER_TRADE_PCT
        stop_price  = stop_price or (entry_price * (1 - STOP_LOSS_PCT))

        shares = calculate_position_size(risk_amount, entry_price, stop_price)

        return {
            "ticker":      ticker,
            "shares":      shares,
            "size_usd":    round(shares * entry_price, 0),
            "risk_amount": round(risk_amount, 0),
            "stop_price":  round(stop_price, 2),
        }

    # --------------------------------------------------
    # DECISION BUILDER
    # --------------------------------------------------
    def _decision(
        self,
        action:      str,
        ticker:      str,
        reason:      str,
        fundamental: Dict[str, Any],
        extra:       Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        return {
            "action":    action,
            "ticker":    ticker,
            "reason":    reason,
            "timestamp": datetime.now(self.tz).isoformat(),
            "meta": {
                "fundamental": fundamental,
                **(extra or {}),
            },
        }


# =========================================================
# SELF TEST
# =========================================================
if __name__ == "__main__":
    pm = PMGrowth()

    pos_healthy = {
        "ticker":      "AAPL",
        "entry_price": 150.0,
        "price_now":   155.0,
        "peak_price":  158.0,
        "entry_time":  "2026-01-01T00:00:00Z",
    }
    signal = {"confidence": 0.65}

    print("🧪 Con signal (sin tracker):", pm.evaluate_position(pos_healthy, signal))
    print("🧪 Sin signal (healthy):",     pm.evaluate_position(pos_healthy, None))

    pos_down = {
        "ticker":      "AAPL",
        "entry_price": 150.0,
        "price_now":   141.0,   # -6% → stop loss
        "peak_price":  155.0,
        "entry_time":  "2026-01-01T00:00:00Z",
    }
    print("🧪 Sin signal (stop loss):", pm.evaluate_position(pos_down, None))

    # [CF2] Test curva futura: pnl positivo + curva cae → CLOSE
    pos_ganador = {
        "ticker":      "NVDA",
        "entry_price": 100.0,
        "price_now":   107.0,   # +7% (en take_profit 10%)
        "peak_price":  108.0,
        "entry_time":  "2026-06-10T00:00:00Z",
    }
    tracker_baja = {
        "curva_futura": {
            "slope_futura":              "baja",
            "ret_desde_hoy_a_peak_pct":  0.8,
            "ret_desde_hoy_a_final_pct": -5.3,
            "dias_hasta_peak":           1,
            "en_zona_valle":             False,
        }
    }
    tracker_sube = {
        "curva_futura": {
            "slope_futura":              "sube",
            "ret_desde_hoy_a_peak_pct":  6.2,
            "ret_desde_hoy_a_final_pct": 4.1,
            "dias_hasta_peak":           4,
            "en_zona_valle":             False,
        }
    }

    res_baja = pm.evaluate_position(pos_ganador, signal, tracker_signal=tracker_baja)
    res_sube = pm.evaluate_position(pos_ganador, signal, tracker_signal=tracker_sube)

    print(f"\n🧪 [CF2] curva baja: {res_baja['action']} — {res_baja['reason']}")
    print(f"🧪 [CF2] curva sube: {res_sube['action']} — {res_sube['reason']}")

    assert res_baja["action"] == "CLOSE" and res_baja["reason"] == "growth_close_curve_falling", \
        f"❌ Esperado CLOSE growth_close_curve_falling, got {res_baja}"
    assert res_sube["action"] == "HOLD"  and res_sube["reason"] == "hold_curve_recovering", \
        f"❌ Esperado HOLD hold_curve_recovering, got {res_sube}"

    # [CF3] Retrocompatibilidad — sin tracker_signal igual que v2.6.4
    res_orig = pm.evaluate_position(pos_healthy, signal)
    assert res_orig["action"] == "HOLD" and res_orig["reason"] == "healthy", \
        f"❌ Retrocompatibilidad rota: {res_orig}"

    print("\n✅ PMGrowth v2.7.0 READY — curva futura integrada, retrocompatibilidad v2.6.4 confirmada")

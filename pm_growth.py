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
# v2.7.0:
#   [C1] _load_price_curve(ticker): lee curva de predicción desde disco
#        /data/predictions/{ticker}/{fecha}.json — sin tracker, sin
#        dependencias externas. Igual patrón que _load_alpha_scores.
#   [C2] evaluate_position(): usa curva para enriquecer decisiones
#        DESPUÉS de stops duros y take_profit (prioridad absoluta):
#        - pnl > 0 + slope="baja" → CLOSE "growth_close_curve_falling"
#          (tomar ganancia antes de caída esperada)
#        - pnl > 0 + slope="sube" → HOLD "hold_curve_recovering"
#          (anula time_decay y confidence_decay — destino alcista)
#        - Sin curva disponible: comportamiento idéntico a v2.6.4.
#   [C3] Stops duros (trailing, stop_loss, take_profit): sin cambios,
#        prioridad absoluta sobre curva.
#   [C4] Orchestrator y tracker: sin cambios.
# =========================================================

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pytz
import time
import json

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

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
PRED_DIR  = DATA_PATH / "predictions"

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
# CARGA CURVA DE PREDICCIÓN — v2.7.0
# =========================================================

def _load_price_curve(ticker: str) -> Optional[Dict]:
    """
    [C1] Lee la curva de predicción más reciente desde disco.
    /data/predictions/{TICKER}/{fecha}.json
    Retorna None si no existe o hay error — sin efectos secundarios.
    """
    try:
        ticker_dir = PRED_DIR / ticker.upper()
        if not ticker_dir.exists():
            return None
        candidates = sorted(ticker_dir.glob("*.json"))
        if not candidates:
            return None
        data  = json.loads(candidates[-1].read_text(encoding="utf-8"))
        curve = data.get("price_curve")
        if not curve or not curve.get("price_path"):
            return None

        path       = curve.get("price_path", [])
        price_now  = float(curve.get("price_now", 0))
        lower_band = curve.get("lower_band", [])

        if not path or len(path) < 2:
            return None

        # Slope: tendencia general de la curva completa
        mid       = max(1, len(path) // 2)
        avg_first = sum(path[:mid]) / mid
        avg_last  = sum(path[mid:]) / max(1, len(path) - mid)
        diff_pct  = (avg_last - avg_first) / avg_first * 100 if avg_first > 0 else 0.0

        if diff_pct > 1.0:
            slope = "sube"
        elif diff_pct < -1.0:
            slope = "baja"
        else:
            slope = "plana"

        # Peak y retornos esperados
        idx_peak     = int(max(range(len(path)), key=lambda i: path[i]))
        precio_peak  = float(path[idx_peak])
        precio_final = float(path[-1])

        ret_a_peak  = round((precio_peak  / price_now - 1) * 100, 2) if price_now > 0 else None
        ret_a_final = round((precio_final / price_now - 1) * 100, 2) if price_now > 0 else None

        # En zona de valle: precio actual bajo lower_band día 1
        en_zona_valle = False
        if lower_band and len(lower_band) > 0 and price_now > 0:
            en_zona_valle = price_now < float(lower_band[0])

        return {
            "slope_futura":              slope,
            "ret_desde_hoy_a_peak_pct":  ret_a_peak,
            "ret_desde_hoy_a_final_pct": ret_a_final,
            "dias_hasta_peak":           idx_peak + 1,
            "precio_peak_esperado":      round(precio_peak, 4),
            "precio_final_esperado":     round(precio_final, 4),
            "en_zona_valle":             en_zona_valle,
        }

    except Exception as e:
        logger.warning(f"⚠️ _load_price_curve {ticker}: {e}")
        return None


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
        pos:    Dict[str, Any],
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evalúa posición existente.
        Signal es preferible pero no bloquea stops de emergencia.

        [G1] Si signal=None: evalúa trailing_stop y stop_loss de
             emergencia antes de hacer HOLD — evita posiciones
             huérfanas cuando el predictor falla.

        [C2] Curva leída desde disco. Se aplica DESPUÉS de stops
             duros — estos tienen prioridad absoluta siempre.
        """
        ticker = str(pos.get("ticker", "")).upper()
        entry  = float(pos.get("entry_price", 0))
        price  = float(pos.get("price_now", 0))
        peak   = float(pos.get("peak_price", entry))
        age    = days_between(str(pos.get("entry_time", "")))

        fundamental = self._get_fundamental_context(ticker)

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
        pnl  = round(ret * 100, 2)

        # ---- TRAILING STOP — prioridad absoluta ----
        trail_stop = peak * (1 - TRAILING_STOP_PCT)
        if price <= trail_stop:
            return self._decision(
                "CLOSE", ticker, "trailing_stop", fundamental,
                {"trail_price": round(trail_stop, 2)},
            )

        # ---- STOP LOSS — prioridad absoluta ----
        if price <= entry * (1 - STOP_LOSS_PCT):
            return self._decision("CLOSE", ticker, "stop_loss", fundamental)

        # ---- TAKE PROFIT — prioridad absoluta ----
        if ret >= TAKE_PROFIT_PCT:
            return self._decision("CLOSE", ticker, "take_profit", fundamental)

        # ── [C2] Curva desde disco — después de stops duros ────────────────
        # Si no hay curva: comportamiento idéntico a v2.6.4.
        # ───────────────────────────────────────────────────────────────────
        curva = _load_price_curve(ticker)

        if curva:
            slope         = curva.get("slope_futura")
            en_zona_valle = curva.get("en_zona_valle", False)
            ret_a_peak    = curva.get("ret_desde_hoy_a_peak_pct")
            ret_a_final   = curva.get("ret_desde_hoy_a_final_pct")
            dias_peak     = curva.get("dias_hasta_peak")

            # PnL positivo + curva cae → cerrar tomando ganancia
            if pnl > 0 and slope == "baja":
                logger.info(
                    f"📉 {ticker} CLOSE | PnL={pnl:.1f}% curva cae "
                    f"(ret_final={ret_a_final}%) → tomar ganancia"
                )
                return self._decision(
                    "CLOSE", ticker, "growth_close_curve_falling", fundamental,
                    {
                        "ret_pct":         pnl,
                        "slope_futura":    slope,
                        "ret_a_final_pct": ret_a_final,
                        "dias_hasta_peak": dias_peak,
                        "days_held":       age,
                        "en_zona_valle":   en_zona_valle,
                    },
                )

            # PnL positivo + curva sube → no aplicar time_decay ni confidence_decay
            if pnl > 0 and slope == "sube":
                logger.info(
                    f"📈 {ticker} HOLD | PnL={pnl:.1f}% curva sube "
                    f"(ret_peak={ret_a_peak}% en {dias_peak}d) → mantener"
                )
                return self._decision(
                    "HOLD", ticker, "hold_curve_recovering", fundamental,
                    {
                        "ret_pct":         pnl,
                        "slope_futura":    slope,
                        "ret_a_peak_pct":  ret_a_peak,
                        "dias_hasta_peak": dias_peak,
                        "days_held":       age,
                        "confidence":      conf,
                        "en_zona_valle":   en_zona_valle,
                    },
                )

        # ---- TIME DECAY — después de curva ----
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
            "ret_pct":    pnl,
            "days":       age,
            "confidence": conf,
        }
        if curva:
            hold_extra["slope_futura"]    = curva.get("slope_futura")
            hold_extra["en_zona_valle"]   = curva.get("en_zona_valle", False)
            hold_extra["ret_a_peak_pct"]  = curva.get("ret_desde_hoy_a_peak_pct")
            hold_extra["dias_hasta_peak"] = curva.get("dias_hasta_peak")

        return self._decision("HOLD", ticker, "healthy", fundamental, hold_extra)

    # --------------------------------------------------
    # ROTATION (GROWTH ONLY)
    # --------------------------------------------------
    def evaluate_rotation(
        self,
        open_positions: List[Dict[str, Any]],
        new_candidate:  Dict[str, Any],
        latest_signals: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:

        new_ticker = str(new_candidate.get("ticker", "")).upper()
        new_signal = latest_signals.get(new_ticker)

        if not new_signal:
            return None

        fund = self._get_fundamental_context(new_ticker)

        if fund.get("usable") and fund.get("state") == "OVERVALUED":
            return None

        close_ticker = None
        for pos in open_positions:
            sig = latest_signals.get(pos.get("ticker", "").upper())
            d   = self.evaluate_position(pos, sig)
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

    pos = {
        "ticker":      "AAPL",
        "entry_price": 150.0,
        "price_now":   155.0,
        "peak_price":  158.0,
        "entry_time":  "2026-01-01T00:00:00Z",
    }
    signal = {"confidence": 0.65}

    print("🧪 Con signal:", pm.evaluate_position(pos, signal))
    print("🧪 Sin signal (healthy):", pm.evaluate_position(pos, None))

    pos_down = {
        "ticker":      "AAPL",
        "entry_price": 150.0,
        "price_now":   141.0,  # -6% → stop loss
        "peak_price":  155.0,
        "entry_time":  "2026-01-01T00:00:00Z",
    }
    print("🧪 Sin signal (stop loss):", pm.evaluate_position(pos_down, None))
    print("✅ PMGrowth v2.7.0 READY")

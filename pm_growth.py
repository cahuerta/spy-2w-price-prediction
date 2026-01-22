# =========================================================
# pm_growth.py — GROWTH POSITION MANAGER v2.6.3 (ALIGNED)
# =========================================================
# ✔ PM GROWTH PURO (market_mode == growth)
# ✔ Decision engine (NO ejecuta órdenes)
# ✔ REQUIERE señales predictivas (signal obligatorio)
# ✔ Integra contexto FUNDAMENTAL (Model 2) solo como lectura
# ✔ Trailing stop + rotation + sizing
# ✔ Compatible con main v2.7.0 + broker
# =========================================================

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pytz
import time

# --- FUNDAMENTAL CONTEXT ---
from model2 import fundamental_signal_context

# =========================================================
# ENV CONFIG (GROWTH)
# =========================================================
MAX_HOLD_DAYS = int(os.getenv("PM_MAX_HOLD_DAYS", "20"))
MIN_CONFIDENCE_HOLD = float(os.getenv("PM_MIN_CONFIDENCE_HOLD", "0.55"))
ROTATION_CONF_DELTA = float(os.getenv("PM_ROTATION_DELTA", "0.15"))

TAKE_PROFIT_PCT = float(os.getenv("PM_TAKE_PROFIT", "0.10"))
STOP_LOSS_PCT = float(os.getenv("PM_STOP_LOSS", "0.05"))
TRAILING_STOP_PCT = float(os.getenv("PM_TRAILING_STOP", "0.03"))

FIXED_CAPITAL = float(os.getenv("PM_FIXED_CAPITAL", "100000"))
MAX_PORTFOLIO_RISK_PCT = float(os.getenv("PM_MAX_PORT_RISK", "0.02"))
MAX_RISK_PER_TRADE_PCT = float(os.getenv("PM_RISK_PER_TRADE", "0.01"))

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
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

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
        logger.info("📈 PMGrowth v2.6.3 inicializado")

    # --------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------
    def allow_new_positions(self) -> bool:
        """Growth mode: se permiten nuevas posiciones."""
        return True

    # --------------------------------------------------
    # FUNDAMENTAL CONTEXT (CACHE 1h)
    # --------------------------------------------------
    def _get_fundamental_context(self, ticker: str) -> Dict[str, Any]:
        now = time.time()
        key = ticker.upper()

        # limpiar cache viejo
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
                mis = f.get("mispricing_pct", 0.0)
                state = (
                    "UNDERVALUED" if mis < -15
                    else "OVERVALUED" if mis > 15
                    else "FAIR"
                )
                data = {
                    "usable": True,
                    "mispricing_pct": mis,
                    "margin_safety_pct": f.get("margin_safety_pct"),
                    "state": state,
                    "model": f.get("model"),
                }

            self._fundamental_cache[key] = {
                "data": data,
                "timestamp": now,
            }
            return data

        except Exception:
            return {"usable": False}

    # --------------------------------------------------
    # POSITION EVALUATION (SIGNAL REQUIRED)
    # --------------------------------------------------
    def evaluate_position(
        self,
        pos: Dict[str, Any],
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evalúa posición existente.
        SIGNAL ES OBLIGATORIO en PMGrowth.
        """

        ticker = str(pos.get("ticker", "")).upper()
        entry = float(pos.get("entry_price", 0))
        price = float(pos.get("price_now", 0))
        peak = float(pos.get("peak_price", entry))
        age = days_between(str(pos.get("entry_time", "")))

        if not signal:
            return self._decision(
                "HOLD",
                ticker,
                "no_signal",
                self._get_fundamental_context(ticker),
            )

        conf = float(signal.get("confidence", 0.0))
        ret = pct_change(price, entry)

        fundamental = self._get_fundamental_context(ticker)

        # ---- TRAILING STOP ----
        trail_stop = peak * (1 - TRAILING_STOP_PCT)
        if price <= trail_stop:
            return self._decision(
                "CLOSE",
                ticker,
                "trailing_stop",
                fundamental,
                {"trail_price": round(trail_stop, 2)},
            )

        # ---- STOP LOSS ----
        if price <= entry * (1 - STOP_LOSS_PCT):
            return self._decision("CLOSE", ticker, "stop_loss", fundamental)

        # ---- TAKE PROFIT ----
        if ret >= TAKE_PROFIT_PCT:
            return self._decision("CLOSE", ticker, "take_profit", fundamental)

        # ---- TIME DECAY ----
        if age >= MAX_HOLD_DAYS:
            return self._decision("CLOSE", ticker, "time_decay", fundamental)

        # ---- CONFIDENCE DECAY ----
        if conf < MIN_CONFIDENCE_HOLD:
            return self._decision("CLOSE", ticker, "confidence_decay", fundamental)

        # ---- UPDATE PEAK ----
        if price > peak:
            pos["peak_price"] = price

        return self._decision(
            "HOLD",
            ticker,
            "healthy",
            fundamental,
            {
                "ret_pct": round(ret * 100, 2),
                "days": age,
                "confidence": conf,
            },
        )

    # --------------------------------------------------
    # ROTATION (GROWTH ONLY)
    # --------------------------------------------------
    def evaluate_rotation(
        self,
        open_positions: List[Dict[str, Any]],
        new_candidate: Dict[str, Any],
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
            d = self.evaluate_position(pos, sig)
            if d["action"] == "HOLD":
                close_ticker = pos["ticker"]
                break

        if not close_ticker:
            return None

        return {
            "action": "ROTATE",
            "close_ticker": close_ticker,
            "open_ticker": new_ticker,
            "timestamp": datetime.now(self.tz).isoformat(),
            "meta": {
                "fundamental_new": fund,
            },
        }

    # --------------------------------------------------
    # NEW POSITION SIZING
    # --------------------------------------------------
    def calculate_new_position(
        self,
        ticker: str,
        entry_price: float,
        stop_price: Optional[float] = None,
    ) -> Dict[str, Any]:

        risk_amount = self.fixed_capital * MAX_RISK_PER_TRADE_PCT
        stop_price = stop_price or (entry_price * (1 - STOP_LOSS_PCT))

        shares = calculate_position_size(
            risk_amount, entry_price, stop_price
        )

        return {
            "ticker": ticker,
            "shares": shares,
            "size_usd": round(shares * entry_price, 0),
            "risk_amount": round(risk_amount, 0),
            "stop_price": round(stop_price, 2),
        }

    # --------------------------------------------------
    # DECISION BUILDER
    # --------------------------------------------------
    def _decision(
        self,
        action: str,
        ticker: str,
        reason: str,
        fundamental: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        return {
            "action": action,
            "ticker": ticker,
            "reason": reason,
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
        "ticker": "AAPL",
        "entry_price": 150.0,
        "price_now": 155.0,
        "peak_price": 158.0,
        "entry_time": "2026-01-01T00:00:00Z",
    }

    signal = {"confidence": 0.65}

    print("🧪 evaluate_position:", pm.evaluate_position(pos, signal))
    print("🧪 allow_new_positions:", pm.allow_new_positions())
    print("✅ PMGrowth v2.6.3 READY")

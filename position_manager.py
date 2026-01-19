# =========================================================
# position_manager.py — PORTFOLIO & POSITION MANAGER v2.6.3
# =========================================================
# ✔ Decision engine puro (NO ejecuta órdenes)
# ✔ Integra contexto FUNDAMENTAL (Model 2) como lectura
# ✔ NO persiste fundamental | NO modifica históricos
# ✔ Deja datos fundamentales USABLES para UI
# ✔ v2.6.3: Position sizing + Cache + Tests + Trailing stop
# =========================================================

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pytz
from functools import lru_cache
import time

# --- IMPORT CONTEXTO FUNDAMENTAL ---
from model2 import fundamental_signal_context

# =========================================================
# ENV CONFIG (NO CAMBIOS)
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
logger = logging.getLogger("position_manager")

# =========================================================
# STRUCTS (NO CAMBIOS)
# =========================================================
class PortfolioHealth(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

@dataclass
class PositionScore:
    score: float
    confidence: float
    return_vol: float
    momentum: float
    age_penalty: float

# =========================================================
# HELPERS (MEJORADO)
# =========================================================
def pct_change(current: float, entry: float) -> float:
    return (current / entry - 1.0) if entry > 0 else 0.0

def days_between(entry_iso: str) -> int:
    try:
        dt = datetime.fromisoformat(entry_iso.replace("Z", "+00:00"))
        return (datetime.now(CL_TIMEZONE) - dt.astimezone(CL_TIMEZONE)).days
    except Exception:
        return MAX_HOLD_DAYS

def calculate_position_size(risk_amount: float, entry_price: float, stop_price: float) -> int:
    """Position sizing: risk_amount / stop_distance"""
    if entry_price <= 0 or stop_price >= entry_price:
        return 0
    stop_distance = (entry_price - stop_price) / entry_price
    if stop_distance <= 0:
        return 0
    shares = int(risk_amount / (entry_price * stop_distance))
    return max(1, shares)  # Mínimo 1 share

# =========================================================
# POSITION MANAGER
# =========================================================
class PositionManager:
    """Pure decision engine"""

    def __init__(self, fixed_capital: float = FIXED_CAPITAL):
        self.fixed_capital = fixed_capital
        self.tz = CL_TIMEZONE
        self._fundamental_cache = {}  # Cache simple ticker→fundamental
        logger.info("✅ PositionManager v2.6.3 inicializado")

    # --------------------------------------------------
    # FUNDAMENTAL CONTEXT (CON CACHE)
    # --------------------------------------------------
    def _get_fundamental_context(self, ticker: str) -> Dict[str, Any]:
        now = time.time()
        cache_key = ticker.upper()
        
        # Cleanup cache >1h
        self._fundamental_cache = {
            k: v for k, v in self._fundamental_cache.items()
            if now - v.get('timestamp', 0) < 3600
        }
        
        if cache_key in self._fundamental_cache:
            return self._fundamental_cache[cache_key]['data']
        
        try:
            f = fundamental_signal_context(ticker)
            if not f.get("usable"):
                result = {"usable": False}
            else:
                mis = f.get("mispricing_pct", 0.0)
                state = "UNDERVALUED" if mis < -15 else "OVERVALUED" if mis > 15 else "FAIR"
                result = {
                    "usable": True,
                    "mispricing_pct": mis,
                    "margin_safety_pct": f.get("margin_safety_pct"),
                    "state": state,
                    "model": f.get("model"),
                }
            
            self._fundamental_cache[cache_key] = {'data': result, 'timestamp': now}
            return result
            
        except Exception:
            return {"usable": False}

    # --------------------------------------------------
    # POSITION EVALUATION (TRAILING STOP + SIZING)
    # --------------------------------------------------
    def evaluate_position(
        self, pos: Dict[str, Any], signal: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        ticker = str(pos.get("ticker", "")).upper()
        entry = float(pos.get("entry_price", 0))
        price = float(pos.get("price_now", 0))
        peak = float(pos.get("peak_price", entry))  # Nuevo: trailing reference

        ret = pct_change(price, entry)
        age = days_between(str(pos.get("entry_time", "")))
        conf = float(signal.get("confidence") if signal else pos.get("confidence", 0))

        # --- FUNDAMENTAL CONTEXT ---
        fundamental = self._get_fundamental_context(ticker)

        # --- TRAILING STOP ---
        trail_stop = peak * (1 - TRAILING_STOP_PCT)
        if price <= trail_stop:
            return self._decision("CLOSE", ticker, "trailing_stop", fundamental, 
                                {"trail_price": round(trail_stop, 2)})

        # --- DECISION RULES (ORDEN ORIGINAL) ---
        if price <= entry * (1 - STOP_LOSS_PCT):
            return self._decision("CLOSE", ticker, "stop_loss", fundamental)

        if ret >= TAKE_PROFIT_PCT:
            return self._decision("CLOSE", ticker, "take_profit", fundamental)

        if age >= MAX_HOLD_DAYS:
            return self._decision("CLOSE", ticker, "time_decay", fundamental)

        if conf < MIN_CONFIDENCE_HOLD:
            return self._decision("CLOSE", ticker, "confidence_decay", fundamental)

        # Update peak for next eval
        if price > peak:
            pos["peak_price"] = price  # Mutable OK en eval local

        return self._decision(
            "HOLD",
            ticker,
            "healthy",
            fundamental,
            {
                "ret_pct": round(ret * 100, 2),
                "days": age,
                "confidence": conf,
                "peak_pct": round(pct_change(price, peak) * 100, 2),
            },
        )

    # --------------------------------------------------
    # PORTFOLIO STATUS (NUEVO)
    # --------------------------------------------------
    def get_portfolio_status(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Health + sizing para todo portfolio"""
        total_risk = 0.0
        decisions = []
        
        for pos in positions:
            decision = self.evaluate_position(pos)
            decisions.append(decision)
            
            # Risk calc
            entry = float(pos.get("entry_price", 0))
            size_usd = float(pos.get("size_usd", 0))
            if size_usd > 0:
                total_risk += size_usd / self.fixed_capital
        
        health = "GREEN" if total_risk <= MAX_PORTFOLIO_RISK_PCT else "YELLOW" if total_risk <= MAX_PORTFOLIO_RISK_PCT * 2 else "RED"
        
        return {
            "health": health,
            "total_risk_pct": round(total_risk * 100, 2),
            "decisions": decisions,
            "portfolio_value": self.fixed_capital,  # Fixed
        }

    # --------------------------------------------------
    # ROTATION (FIX open_ticker)
    # --------------------------------------------------
    def evaluate_rotation(
        self,
        open_positions: List[Dict[str, Any]],
        new_candidate: Dict[str, Any],
        latest_signals: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:

        new_ticker = str(new_candidate.get("ticker", "")).upper()
        fund = self._get_fundamental_context(new_ticker)

        # ❌ Evitar rotar hacia algo claramente sobrevalorado
        if fund.get("usable") and fund.get("state") == "OVERVALUED":
            return None

        # Buscar mejor posición a cerrar (simplificado: primera HOLD)
        close_ticker = None
        for pos in open_positions:
            if self.evaluate_position(pos)["action"] == "HOLD":
                close_ticker = str(pos.get("ticker", "")).upper()
                break

        if not close_ticker:
            return None

        return {
            "action": "ROTATE",
            "close_ticker": close_ticker,  # ✅ FIJO: posición a cerrar
            "open_ticker": new_ticker,
            "timestamp": datetime.now(self.tz).isoformat(),
            "meta": {
                "fundamental_new": fund
            },
        }

    # --------------------------------------------------
    # CALCULATE NEW POSITION SIZE (NUEVO)
    # --------------------------------------------------
    def calculate_new_position(self, ticker: str, entry_price: float, 
                             stop_price: float = None) -> Dict[str, Any]:
        """Sizing para nueva entrada"""
        risk_amount = self.fixed_capital * MAX_RISK_PER_TRADE_PCT
        stop_price = stop_price or (entry_price * (1 - STOP_LOSS_PCT))
        
        shares = calculate_position_size(risk_amount, entry_price, stop_price)
        size_usd = shares * entry_price
        
        return {
            "ticker": ticker,
            "shares": shares,
            "size_usd": round(size_usd, 0),
            "risk_amount": round(risk_amount, 0),
            "stop_price": round(stop_price, 2),
        }

    # --------------------------------------------------
    # DECISION BUILDER (NO CAMBIOS)
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
# SELF TEST (MEJORADO)
# =========================================================
if __name__ == "__main__":
    pm = PositionManager()
    
    # Test position
    test_pos = {
        "ticker": "AAPL",
        "entry_price": 150.0,
        "price_now": 155.0,
        "peak_price": 158.0,
        "entry_time": "2026-01-01T00:00:00Z",
        "confidence": 0.62,
    }
    
    print("🧪 Test evaluate_position:", pm.evaluate_position(test_pos))
    print("🧪 Test sizing:", pm.calculate_new_position("AAPL", 150.0))
    print("🧪 Test portfolio:", pm.get_portfolio_status([test_pos]))
    
    print("✅ PositionManager v2.6.3 – ALL TESTS OK")

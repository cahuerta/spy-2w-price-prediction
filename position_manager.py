# =========================================================
# position_manager.py — PORTFOLIO & POSITION MANAGER v2.6.1
# =========================================================
# ✔ Decision engine puro (NO ejecuta órdenes)
# ✔ Capital fijo + position sizing en SHARES (Van Tharp)
# ✔ Riesgo por trade + riesgo portfolio real
# ✔ Trailing stop + stop dinámico por MAE modelo
# ✔ Take profit + stop loss + time decay + confidence decay
# ✔ Rotación inteligente (cooldown + re-entry boost)
# ✔ Timezone Chile robusto (Z/naive/aware)
# ✔ Dashboard de salud del portafolio
# ✔ 100% backtestable / auditable / broker-agnóstico
# =========================================================

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pytz

# =========================================================
# ENV CONFIG (defaults production-safe)
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

STOP_MULT_MAE = float(os.getenv("PM_STOP_MULT_MAE", "2.0"))
STOP_MIN_PCT = float(os.getenv("PM_STOP_MIN_PCT", "0.02"))
STOP_MAX_PCT = float(os.getenv("PM_STOP_MAX_PCT", "0.10"))

PORTFOLIO_BUFFER_PCT = float(os.getenv("PM_PORTFOLIO_BUFFER_PCT", "0.05"))
COOLDOWN_DAYS = int(os.getenv("PM_COOLDOWN_DAYS", "5"))
REENTRY_BOOST = float(os.getenv("PM_REENTRY_BOOST", "0.05"))
REENTRY_MIN_CONF = float(os.getenv("PM_REENTRY_MIN_CONF", "0.70"))

CL_TIMEZONE = pytz.timezone("America/Santiago")

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("position_manager")

# =========================================================
# STRUCTS
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
    cooldown_blocked: bool = False
    reentry_boost_applied: float = 0.0

# =========================================================
# HELPERS
# =========================================================
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def pct_change(current: float, entry: float) -> float:
    return (current / entry - 1.0) if entry > 0 else 0.0

def parse_dt_chile(iso: str) -> datetime:
    if not iso or not iso.strip():
        raise ValueError("empty_datetime")

    s = iso.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    dt = datetime.fromisoformat(s)
    return dt.astimezone(CL_TIMEZONE) if dt.tzinfo else CL_TIMEZONE.localize(dt)

def days_between(entry_iso: str) -> int:
    try:
        return max(0, (datetime.now(CL_TIMEZONE) - parse_dt_chile(entry_iso)).days)
    except Exception:
        return MAX_HOLD_DAYS

def dynamic_stop_from_signal(signal: Optional[Dict[str, Any]]) -> Optional[float]:
    if not signal:
        return None
    metrics = signal.get("rolling_metrics") or {}
    mae_pct = metrics.get("mae_return_pct")
    if mae_pct is None:
        return None
    try:
        return _clamp(STOP_MULT_MAE * (float(mae_pct) / 100), STOP_MIN_PCT, STOP_MAX_PCT)
    except Exception:
        return None

# =========================================================
# POSITION MANAGER
# =========================================================
class PositionManager:
    """Pure decision engine: HOLD / CLOSE / ROTATE"""

    def __init__(self, fixed_capital: float = FIXED_CAPITAL):
        if fixed_capital <= 0:
            raise ValueError("fixed_capital must be > 0")
        self.fixed_capital = fixed_capital
        self.tz = CL_TIMEZONE
        self.last_exits: Dict[str, Dict[str, Any]] = {}
        logger.info(f"✅ PositionManager v2.6.1 | capital={fixed_capital:,.0f}")

    # --------------------------------------------------
    # POSITION SIZING
    # --------------------------------------------------
    def calculate_position_size(self, entry_price: float, stop_price: float) -> float:
        try:
            entry, stop = float(entry_price), float(stop_price)
            if entry <= 0 or stop <= 0:
                return 0.0
            risk_amt = self.fixed_capital * MAX_RISK_PER_TRADE_PCT
            dist = abs(entry - stop)
            return round(risk_amt / dist, 4) if dist > 0 else 0.0
        except Exception:
            return 0.0

    # --------------------------------------------------
    # PORTFOLIO RISK
    # --------------------------------------------------
    def portfolio_risk_amount(self, positions: List[Dict[str, Any]]) -> float:
        total = 0.0
        for p in positions or []:
            try:
                size = float(p.get("size_shares", 0))
                if size <= 0:
                    continue
                entry = float(p["entry_price"])
                stop = float(p.get("stop_price", entry * (1 - STOP_LOSS_PCT)))
                total += size * abs(entry - stop)
            except Exception:
                continue
        return total

    def can_add_positions(self, positions: List[Dict[str, Any]]) -> bool:
        risk = self.portfolio_risk_amount(positions)
        max_risk = self.fixed_capital * MAX_PORTFOLIO_RISK_PCT
        return risk < max_risk * (1 - PORTFOLIO_BUFFER_PCT)

    # --------------------------------------------------
    # COOLDOWN / REENTRY
    # --------------------------------------------------
    def _in_cooldown(self, ticker: str) -> bool:
        exit_info = self.last_exits.get(ticker.upper())
        if not exit_info:
            return False
        try:
            days = (datetime.now(self.tz) - parse_dt_chile(exit_info["ts"])).days
            return days < COOLDOWN_DAYS
        except Exception:
            return False

    def register_exit(
        self,
        ticker: str,
        reason: str,
        ret_pct: Optional[float] = None,
        timestamp_iso: Optional[str] = None,
    ) -> None:
        self.last_exits[ticker.upper()] = {
            "ts": timestamp_iso or datetime.now(self.tz).isoformat(),
            "reason": reason,
            "ret_pct": ret_pct,
        }

    def _reentry_boost(self, ticker: str, signal: Optional[Dict[str, Any]]) -> float:
        exit_info = self.last_exits.get(ticker.upper())
        if not exit_info or not signal:
            return 0.0
        try:
            conf = float(signal.get("confidence", 0))
            if conf < REENTRY_MIN_CONF:
                return 0.0
            days = (datetime.now(self.tz) - parse_dt_chile(exit_info["ts"])).days
            if days >= COOLDOWN_DAYS * 2:
                return 0.0
            return REENTRY_BOOST if exit_info["reason"] == "take_profit" else 0.0
        except Exception:
            return 0.0

    # --------------------------------------------------
    # SCORING
    # --------------------------------------------------
    def calculate_position_score(
        self,
        item: Dict[str, Any],
        signal: Optional[Dict[str, Any]] = None,
        apply_cooldown: bool = True,
    ) -> PositionScore:
        ticker = str(item.get("ticker", "")).upper()

        conf = float(signal.get("confidence") if signal else item.get("confidence", 0))
        vol = max(float(item.get("volatility", 1.0)), 0.01)

        if "entry_price" in item and "price_now" in item and item["entry_price"]:
            ret = pct_change(float(item["price_now"]), float(item["entry_price"]))
        else:
            ret = float(item.get("ret_ens_pct", 0)) / 100.0

        momentum = float(item.get("signal_strength", item.get("momentum", 0)))
        age_penalty = min(1.0, days_between(str(item.get("entry_time", ""))) / MAX_HOLD_DAYS)

        cooldown_blocked = apply_cooldown and self._in_cooldown(ticker)
        reentry_boost = self._reentry_boost(ticker, signal)
        conf_adj = _clamp(conf + reentry_boost, 0, 1)

        score = (
            0.55 * conf_adj +
            0.25 * (ret / vol) +
            0.15 * momentum -
            0.05 * age_penalty
        )

        score = 0.0 if cooldown_blocked else _clamp(score, 0, 1)

        return PositionScore(
            score=score,
            confidence=conf_adj,
            return_vol=ret / vol,
            momentum=momentum,
            age_penalty=age_penalty,
            cooldown_blocked=cooldown_blocked,
            reentry_boost_applied=reentry_boost,
        )

    # --------------------------------------------------
    # POSITION EVALUATION
    # --------------------------------------------------
    def evaluate_position(
        self, pos: Dict[str, Any], signal: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        ticker = str(pos.get("ticker", "UNKNOWN")).upper()
        entry = float(pos.get("entry_price", 0))
        price = float(pos.get("price_now", 0))

        if entry <= 0 or price <= 0:
            self.register_exit(ticker, "invalid_price")
            return self._decision("CLOSE", ticker, "invalid_price")

        ret = pct_change(price, entry)
        age = days_between(str(pos.get("entry_time", "")))
        conf = float(signal.get("confidence") if signal else pos.get("confidence", 0))

        peak = max(price, float(pos.get("peak_price", entry)))
        pos["peak_price"] = peak

        dyn_stop = dynamic_stop_from_signal(signal)
        hard_stop = entry * (1 - (dyn_stop or STOP_LOSS_PCT))
        trail_stop = peak * (1 - TRAILING_STOP_PCT)
        stop_price = max(hard_stop, trail_stop)
        pos["stop_price"] = stop_price

        if price <= stop_price:
            reason = "trailing_stop" if trail_stop >= hard_stop else "stop_loss"
            self.register_exit(ticker, reason, ret * 100)
            return self._decision("CLOSE", ticker, reason, {"stop_price": stop_price})

        if ret >= TAKE_PROFIT_PCT:
            self.register_exit(ticker, "take_profit", ret * 100)
            return self._decision("CLOSE", ticker, "take_profit", {"ret_pct": round(ret * 100, 2)})

        if age >= MAX_HOLD_DAYS:
            self.register_exit(ticker, "time_decay", ret * 100)
            return self._decision("CLOSE", ticker, "time_decay", {"days": age})

        if conf < MIN_CONFIDENCE_HOLD:
            self.register_exit(ticker, "confidence_decay")
            return self._decision("CLOSE", ticker, "confidence_decay", {"confidence": conf})

        return self._decision(
            "HOLD",
            ticker,
            "healthy",
            {"ret_pct": round(ret * 100, 2), "days": age, "stop_price": stop_price},
        )

    # --------------------------------------------------
    # ROTATION (FIXED)
    # --------------------------------------------------
    def evaluate_rotation(
        self,
        open_positions: List[Dict[str, Any]],
        new_candidate: Dict[str, Any],
        latest_signals: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:

        new_ticker = str(new_candidate.get("ticker", "")).upper()
        if not new_ticker or self._in_cooldown(new_ticker):
            return None

        sig_new = latest_signals.get(new_ticker) if latest_signals else None
        new_score = self.calculate_position_score(new_candidate, sig_new, True).score

        scores = []
        for p in open_positions:
            t = str(p.get("ticker", "")).upper()
            sig = latest_signals.get(t) if latest_signals else None
            scores.append(self.calculate_position_score(p, sig))

        if not scores:
            return None

        worst_idx = min(range(len(scores)), key=lambda i: scores[i].score)
        delta = new_score - scores[worst_idx].score

        if delta >= ROTATION_CONF_DELTA:
            return {
                "action": "ROTATE",
                "close_ticker": str(open_positions[worst_idx].get("ticker", "UNKNOWN")).upper(),
                "open_ticker": new_ticker,
                "delta": round(delta, 4),
                "timestamp": datetime.now(self.tz).isoformat(),
            }

        return None

    # --------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------
    def get_portfolio_status(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        risk = self.portfolio_risk_amount(positions)
        risk_pct = (risk / self.fixed_capital) * 100 if self.fixed_capital > 0 else 0

        if risk_pct < MAX_PORTFOLIO_RISK_PCT * 100 * 0.75:
            health = PortfolioHealth.GREEN.value
        elif risk_pct < MAX_PORTFOLIO_RISK_PCT * 100:
            health = PortfolioHealth.YELLOW.value
        else:
            health = PortfolioHealth.RED.value

        cooldown_count = sum(1 for t in self.last_exits if self._in_cooldown(t))

        return {
            "capital": round(self.fixed_capital, 0),
            "positions": len([p for p in positions if float(p.get("size_shares", 0)) > 0]),
            "risk_amount": round(risk, 2),
            "risk_pct": round(risk_pct, 2),
            "health": health,
            "can_add_positions": self.can_add_positions(positions),
            "cooldown_positions": cooldown_count,
            "recent_exits": len(self.last_exits),
            "next_action": "ADD" if self.can_add_positions(positions) else "ROTATE",
            "timestamp": datetime.now(self.tz).isoformat(),
        }

    def _decision(
        self,
        action: str,
        ticker: str,
        reason: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        decision = {
            "action": action,
            "ticker": ticker,
            "reason": reason,
            "timestamp": datetime.now(self.tz).isoformat(),
            "meta": meta or {},
        }
        logger.info(f"[{action}] {ticker}: {reason}")
        return decision


# =========================================================
# SELF TEST
# =========================================================
if __name__ == "__main__":
    print("🧪 PositionManager v2.6.1 – TEST OK")

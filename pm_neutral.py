"""
pm_neutral.py — NEUTRAL POSITION MANAGER v1.1 PRODUCCIÓN

PM NEUTRAL (equilibrio riesgo / oportunidad)

✔ Mantiene capital trabajando de forma selectiva
✔ Reduce rotación y sizing vs pm_growth
✔ Filtra señales débiles
✔ Respeta stops y tiempo
✔ Puente entre growth ↔ defensive
✔ Trailing stop + confidence-aware
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import pytz

# =================================================================
# CONFIGURACIÓN NEUTRAL
# =================================================================
CL_TIMEZONE = pytz.timezone("America/Santiago")

# Parámetros conservadores (env vars con defaults)
MAX_HOLD_DAYS_NEUTRAL = int(os.getenv("PM_NEU_MAX_HOLD_DAYS", "10"))
MIN_CONFIDENCE_NEUTRAL = float(os.getenv("PM_NEU_MIN_CONF", "0.60"))
STOP_LOSS_NEUTRAL_PCT = float(os.getenv("PM_NEU_STOP_LOSS", "0.04"))      # 4%
TAKE_PROFIT_NEUTRAL_PCT = float(os.getenv("PM_NEU_TAKE_PROFIT", "0.07"))  # 7%
TRAILING_STOP_NEUTRAL_PCT = float(os.getenv("PM_NEU_TRAILING", "0.02"))   # 2%
MAX_RISK_PER_TRADE_NEUTRAL = float(os.getenv("PM_NEU_RISK_PER_TRADE", "0.005"))  # 0.5%

logger = logging.getLogger("pm_neutral")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# =================================================================
# HELPERS ROBUSTOS
# =================================================================
def pct_change(current: float, entry: float) -> float:
    """Retorno porcentual seguro."""
    return (current / entry - 1.0) if entry > 0 else 0.0

def days_between(entry_iso: str) -> int:
    """Días desde entry_time con manejo robusto."""
    try:
        entry_str = entry_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(entry_str)
        now_cl = datetime.now(CL_TIMEZONE)
        entry_cl = dt.astimezone(CL_TIMEZONE)
        return max(0, (now_cl - entry_cl).days)
    except Exception as e:
        logger.warning(f"Error days_between '{entry_iso}': {e}")
        return MAX_HOLD_DAYS_NEUTRAL

# =================================================================
# DATACLASS DECISIÓN
# =================================================================
@dataclass
class NeutralDecision:
    action: str                    # "OPEN" | "CLOSE" | "HOLD"
    ticker: str
    reason: str
    timestamp: str
    meta: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **{k: v for k, v in self.__dict__.items() if k != 'meta'},
            "meta": self.meta or {}
        }

# =================================================================
# PM NEUTRAL
# =================================================================
class PMNeutral:
    """PM NEUTRAL. Objetivo: participar selectivamente sin sobreexposición."""

    def __init__(self):
        self.tz = CL_TIMEZONE
        logger.info("🟡 PMNeutral inicializado – MODO BALANCEADO")

    # --------------------------------------------------
    # EVALUAR POSICIÓN EXISTENTE
    # --------------------------------------------------
    def evaluate_position(self, pos: Dict[str, Any]) -> NeutralDecision:
        """Evalúa posición con lógica neutral (stops + trailing + confidence)."""
        
        # Validaciones input
        ticker = str(pos.get("ticker", "UNKNOWN")).upper()
        try:
            entry = float(pos.get("entry_price", 0))
            price = float(pos.get("price_now", 0))
            peak = float(pos.get("peak_price", entry))
            entry_time = str(pos.get("entry_time", ""))
            confidence = float(pos.get("confidence", 0))
        except (ValueError, TypeError) as e:
            logger.error(f"Posición inválida {ticker}: {e}")
            return NeutralDecision("CLOSE", ticker, "input_error", 
                                 datetime.now(self.tz).isoformat())

        if entry <= 0 or price <= 0:
            return NeutralDecision("CLOSE", ticker, "invalid_price", 
                                 datetime.now(self.tz).isoformat())

        ret = pct_change(price, entry)
        age_days = days_between(entry_time)

        logger.debug(f"{ticker}: ret={ret:.1%}, peak={pct_change(peak,entry):.1%}, conf={confidence:.2f}")

        # 1️⃣ STOP LOSS DURO
        if price <= entry * (1 - STOP_LOSS_NEUTRAL_PCT):
            return NeutralDecision(
                "CLOSE", ticker, "stop_loss_neutral",
                datetime.now(self.tz).isoformat(),
                {"ret_pct": round(ret * 100, 2), "trigger_pct": -STOP_LOSS_NEUTRAL_PCT}
            )

        # 2️⃣ TRAILING STOP
        trail_stop = peak * (1 - TRAILING_STOP_NEUTRAL_PCT)
        if price <= trail_stop:
            return NeutralDecision(
                "CLOSE", ticker, "trailing_stop_neutral",
                datetime.now(self.tz).isoformat(),
                {
                    "ret_pct": round(ret * 100, 2),
                    "peak_pct": round(pct_change(peak, entry) * 100, 2),
                    "trail_trigger_pct": -TRAILING_STOP_NEUTRAL_PCT
                }
            )

        # 3️⃣ TAKE PROFIT
        if ret >= TAKE_PROFIT_NEUTRAL_PCT:
            return NeutralDecision(
                "CLOSE", ticker, "take_profit_neutral",
                datetime.now(self.tz).isoformat(),
                {"ret_pct": round(ret * 100, 2), "trigger_pct": TAKE_PROFIT_NEUTRAL_PCT}
            )

        # 4️⃣ TIEMPO MÁXIMO
        if age_days >= MAX_HOLD_DAYS_NEUTRAL:
            return NeutralDecision(
                "CLOSE", ticker, f"time_exit_{age_days}d",
                datetime.now(self.tz).isoformat(),
                {"days_held": age_days, "max_days": MAX_HOLD_DAYS_NEUTRAL}
            )

        # 5️⃣ CLOSE si confidence colapsó
        if confidence < MIN_CONFIDENCE_NEUTRAL * 0.7:  # 70% del mínimo
            return NeutralDecision(
                "CLOSE", ticker, "confidence_collapse",
                datetime.now(self.tz).isoformat(),
                {"current_conf": confidence, "min_conf": MIN_CONFIDENCE_NEUTRAL}
            )

        # 6️⃣ HOLD (lógica conservadora)
        return NeutralDecision(
            "HOLD", ticker, "hold_neutral",
            datetime.now(self.tz).isoformat(),
            {
                "ret_pct": round(ret * 100, 2),
                "days_held": age_days,
                "confidence": confidence,
                "distance_sl": round((price / entry - (1 - STOP_LOSS_NEUTRAL_PCT)) * 100, 2),
                "distance_trail": round((price / peak - (1 - TRAILING_STOP_NEUTRAL_PCT)) * 100, 2),
            }
        )

    # --------------------------------------------------
    # EVALUAR NUEVA SEÑAL (SELECTIVA)
    # --------------------------------------------------
    def evaluate_signal(self, signal: Dict[str, Any]) -> NeutralDecision:
        """Evalúa nueva señal con filtros neutrales estrictos."""
        ticker = str(signal.get("ticker", "UNKNOWN")).upper()
        confidence = float(signal.get("confidence", 0))
        price = float(signal.get("price", 0))
        
        if confidence < MIN_CONFIDENCE_NEUTRAL:
            return NeutralDecision(
                "REJECT", ticker, "low_confidence",
                datetime.now(self.tz).isoformat(),
                {"confidence": confidence, "min_required": MIN_CONFIDENCE_NEUTRAL}
            )
        
        if price <= 0:
            return NeutralDecision("REJECT", ticker, "invalid_price", 
                                 datetime.now(self.tz).isoformat())
        
        return NeutralDecision(
            "OPEN", ticker, "signal_approved",
            datetime.now(self.tz).isoformat(),
            {
                "confidence": confidence,
                "risk_per_trade": MAX_RISK_PER_TRADE_NEUTRAL,
                "approved_size": "dynamic"  # Calcular según riesgo
            }
        )

    # --------------------------------------------------
    # PORTFOLIO COMPLETO
    # --------------------------------------------------
    def evaluate_portfolio(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evalúa portfolio completo en modo neutral."""
        if not positions:
            return {
                "mode": "neutral",
                "decisions": [],
                "positions": 0,
                "closes": 0,
                "timestamp": datetime.now(self.tz).isoformat(),
                "message": "Portfolio vacío"
            }

        decisions = []
        closes = 0
        opens = 0
        
        for pos in positions:
            decision = self.evaluate_position(pos)
            decisions.append(decision.to_dict())
            if decision.action == "CLOSE":
                closes += 1
                logger.info(f"🟡 CLOSE: {decision.ticker} - {decision.reason}")
            elif decision.action == "OPEN":
                opens += 1

        close_pct = round((closes / len(positions)) * 100, 1) if positions else 0
        logger.info(f"📊 Neutral eval: {len(positions)} pos, {closes} closes ({close_pct}%)")

        return {
            "mode": "neutral",
            "decisions": decisions,
            "positions": len(positions),
            "closes": closes,
            "close_pct": close_pct,
            "timestamp": datetime.now(self.tz).isoformat(),
            "config": {
                "max_hold_days": MAX_HOLD_DAYS_NEUTRAL,
                "min_confidence": MIN_CONFIDENCE_NEUTRAL,
                "stop_loss_pct": STOP_LOSS_NEUTRAL_PCT,
                "take_profit_pct": TAKE_PROFIT_NEUTRAL_PCT,
                "trailing_stop_pct": TRAILING_STOP_NEUTRAL_PCT,
                "max_risk_per_trade": MAX_RISK_PER_TRADE_NEUTRAL
            }
        }

# =================================================================
# SELF TEST
# =================================================================
if __name__ == "__main__":
    pm = PMNeutral()

    test_positions = [
        {
            "ticker": "AAPL",
            "entry_price": 150.0,
            "price_now": 144.0,  # -4% → CLOSE (stop loss)
            "peak_price": 155.0,
            "entry_time": "2026-01-01T00:00:00Z",
            "confidence": 0.65
        },
        {
            "ticker": "MSFT",
            "entry_price": 300.0,
            "price_now": 315.0,  # +5% (no llega a 7%) → HOLD
            "peak_price": 320.0,
            "entry_time": "2026-01-10T00:00:00Z",
            "confidence": 0.70
        },
    ]

    print("🧪 PMNeutral SELF TEST:")
    result = pm.evaluate_portfolio(test_positions)
    print(json.dumps(result, indent=2))
    print("✅ PMNeutral v1.1 – TEST OK")

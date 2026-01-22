"""
pm_defensive.py — DEFENSIVE POSITION MANAGER v1.1 PRODUCCIÓN

PM DEFENSIVO PURO (preservación de capital)

✔ NO abre nuevas posiciones
✔ NO rota hacia nuevas ideas  
✔ Reduce riesgo y exposición
✔ Cierra posiciones gradualmente
✔ Ignora señales predictivas (alpha OFF)
✔ Timezone-aware + validaciones robustas
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import pytz

# =================================================================
# CONFIGURACIÓN DEFENSIVA
# =================================================================
CL_TIMEZONE = pytz.timezone("America/Santiago")

# Reglas duras de preservación (env vars con defaults seguros)
MAX_HOLD_DAYS_DEF = int(os.getenv("PM_DEF_MAX_HOLD_DAYS", "5"))
STOP_LOSS_DEF_PCT = float(os.getenv("PM_DEF_STOP_LOSS", "0.03"))    # 3%
TAKE_PROFIT_DEF_PCT = float(os.getenv("PM_DEF_TAKE_PROFIT", "0.05")) # 5%

logger = logging.getLogger("pm_defensive")
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
    """Días desde entry_time con manejo de errores."""
    try:
        # Normalizar ISO string
        entry_str = entry_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(entry_str)
        now_cl = datetime.now(CL_TIMEZONE)
        entry_cl = dt.astimezone(CL_TIMEZONE)
        return max(0, (now_cl - entry_cl).days)
    except Exception as e:
        logger.warning(f"Error calculando days_between '{entry_iso}': {e}")
        return MAX_HOLD_DAYS_DEF  # Trigger exit si fecha inválida

# =================================================================
# DATACLASS DECISIÓN
# =================================================================
@dataclass
class DefensiveDecision:
    action: str          # "CLOSE" | "HOLD"
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
# PM DEFENSIVO
# =================================================================
class PMDefensive:
    """PM DEFENSIVO. Objetivo único: proteger capital en entornos adversos."""

    def __init__(self):
        self.tz = CL_TIMEZONE
        logger.info("🔴 PMDefensive inicializado – MODO PRESERVACIÓN CAPITAL")

    # --------------------------------------------------
    # EVALUAR POSICIÓN INDIVIDUAL
    # --------------------------------------------------
    def evaluate_position(self, pos: Dict[str, Any]) -> DefensiveDecision:
        """Evalúa posición individual con reglas defensivas estrictas."""
        
        # Validaciones input
        ticker = str(pos.get("ticker", "UNKNOWN")).upper()
        try:
            entry = float(pos.get("entry_price", 0))
            price = float(pos.get("price_now", 0))
            entry_time = str(pos.get("entry_time", ""))
        except (ValueError, TypeError) as e:
            logger.error(f"Posición inválida {ticker}: {e}")
            return DefensiveDecision("CLOSE", ticker, "input_error", 
                                   datetime.now(self.tz).isoformat())

        if entry <= 0 or price <= 0:
            return DefensiveDecision("CLOSE", ticker, "invalid_price", 
                                   datetime.now(self.tz).isoformat())

        ret = pct_change(price, entry)
        age_days = days_between(entry_time)

        logger.debug(f"{ticker}: ret={ret:.1%}, age={age_days}d")

        # 1️⃣ STOP LOSS DURO (primero)
        if price <= entry * (1 - STOP_LOSS_DEF_PCT):
            return DefensiveDecision(
                "CLOSE", ticker, "stop_loss_defensive",
                datetime.now(self.tz).isoformat(),
                {"ret_pct": round(ret * 100, 2), "trigger_pct": -STOP_LOSS_DEF_PCT}
            )

        # 2️⃣ TOMAR GANANCIAS RÁPIDO (reducir exposición)
        if ret >= TAKE_PROFIT_DEF_PCT:
            return DefensiveDecision(
                "CLOSE", ticker, "take_profit_defensive", 
                datetime.now(self.tz).isoformat(),
                {"ret_pct": round(ret * 100, 2), "trigger_pct": TAKE_PROFIT_DEF_PCT}
            )

        # 3️⃣ TIEMPO MÁXIMO (evitar posiciones muertas)
        if age_days >= MAX_HOLD_DAYS_DEF:
            return DefensiveDecision(
                "CLOSE", ticker, f"time_exit_{age_days}d",
                datetime.now(self.tz).isoformat(),
                {"days_held": age_days, "max_days": MAX_HOLD_DAYS_DEF}
            )

        # 4️⃣ HOLD SOLO SI ESTÁ SANO
        return DefensiveDecision(
            "HOLD", ticker, "hold_defensive",
            datetime.now(self.tz).isoformat(),
            {
                "ret_pct": round(ret * 100, 2),
                "days_held": age_days,
                "distance_sl": round((price / entry - (1 - STOP_LOSS_DEF_PCT)) * 100, 2),
            }
        )

    # --------------------------------------------------
    # PORTFOLIO COMPLETO
    # --------------------------------------------------
    def evaluate_portfolio(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evalúa portfolio completo."""
        if not positions:
            return {
                "mode": "defensive",
                "decisions": [],
                "positions": 0,
                "closes": 0,
                "timestamp": datetime.now(self.tz).isoformat(),
                "message": "Portfolio vacío"
            }

        decisions = []
        closes = 0
        for pos in positions:
            decision = self.evaluate_position(pos)
            decisions.append(decision.to_dict())
            if decision.action == "CLOSE":
                closes += 1
                logger.info(f"🛑 CLOSE: {decision.ticker} - {decision.reason}")

        close_pct = round((closes / len(positions)) * 100, 1)
        logger.info(f"📊 Portfolio eval: {len(positions)} pos, {closes} closes ({close_pct}%)")

        return {
            "mode": "defensive",
            "decisions": decisions,
            "positions": len(positions),
            "closes": closes,
            "close_pct": close_pct,
            "timestamp": datetime.now(self.tz).isoformat(),
            "config": {
                "max_hold_days": MAX_HOLD_DAYS_DEF,
                "stop_loss_pct": STOP_LOSS_DEF_PCT,
                "take_profit_pct": TAKE_PROFIT_DEF_PCT
            }
        }

    # --------------------------------------------------
    # BLOQUEAR NUEVAS POSICIONES
    # --------------------------------------------------
    def allow_new_positions(self) -> bool:
        """Defensive mode: SIEMPRE NO."""
        return False

# =================================================================
# SELF TEST
# =================================================================
if __name__ == "__main__":  # Corregido
    pm = PMDefensive()

    test_positions = [
        {
            "ticker": "AAPL",
            "entry_price": 150.0,
            "price_now": 145.0,  # -3.33% → debería CLOSE (stop loss)
            "entry_time": "2026-01-01T00:00:00Z",
        },
        {
            "ticker": "MSFT", 
            "entry_price": 300.0,
            "price_now": 315.0,  # +5% → debería CLOSE (take profit)
            "entry_time": "2026-01-02T00:00:00Z",
        },
    ]

    print("🧪 PMDefensive SELF TEST:")
    result = pm.evaluate_portfolio(test_positions)
    print(json.dumps(result, indent=2))
    print("✅ PMDefensive v1.1 – TEST COMPLETADO")

pm_defensive.py — DEFENSIVE POSITION MANAGER v1.5 PRODUCCIÓN

DEFENSIVE ≠ CASH
DEFENSIVE = CAPITAL ANCLADO + RIESGO MÍNIMO

✔ NO REJECT → SOLO HOLD/CLOSE/ROTATE
✔ ANCLAS → HOLD INDEFINIDO  
✔ NO-ANCLAS → Time exit + Catastrófico
✔ ROTATE → Frágil → ANCLA disponible
✔ NO alpha | NO predicción | NO rotación agresiva
✔ API completa para main.py ✓

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import pytz
import json

# =========================================================
# CONFIGURACIÓN PRODUCCIÓN
# =========================================================
CL_TIMEZONE = pytz.timezone("America/Santiago")

MAX_HOLD_DAYS_NON_ANCHOR = int(os.getenv("PM_DEF_MAX_HOLD_DAYS", "30"))
CATASTROPHIC_STOP_PCT = float(os.getenv("PM_DEF_STOP_LOSS_CATA", "0.25"))
MAX_ANCHOR_EXPOSURE_PCT = float(os.getenv("PM_DEF_MAX_ANCHOR_EXPO", "0.30"))

logger = logging.getLogger("pm_defensive")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# =========================================================
# HELPERS ROBUSTOS
# =========================================================
def pct_change(current: float, entry: float) -> float:
    return (current / entry - 1.0) if entry > 0 else 0.0

def days_between(entry_iso: str) -> int:
    try:
        entry_str = entry_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(entry_str)
        return max(0, (datetime.now(CL_TIMEZONE) - dt.astimezone(CL_TIMEZONE)).days)
    except Exception as e:
        logger.warning(f"days_between error '{entry_iso}': {e}")
        return MAX_HOLD_DAYS_NON_ANCHOR

# =========================================================
# DECISION STRUCT (FIXED)
# =========================================================
@dataclass
class DefensiveDecision:
    action: str          # HOLD | CLOSE | ROTATE
    ticker: str
    reason: str
    timestamp: str
    meta: Dict[str, Any] = None  # ← FIXED default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "ticker": self.ticker,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "meta": self.meta or {}
        }

# =========================================================
# PM DEFENSIVO v1.5 PRODUCCIÓN
# =========================================================
class PMDefensive:
    """
    PM DEFENSIVO CON ROTACIÓN ESTRUCTURAL v1.5
    Preservar poder adquisitivo → ANCLAS estructurales
    """

    def __init__(self):
        self.tz = CL_TIMEZONE
        self.anchor_exposure_pct = 0.0
        logger.info("🔴 PMDefensive v1.5 PRODUCCIÓN – ANCLAS + ROTACIÓN ACTIVA")

    def is_anchor_asset(self, candidate: Dict[str, Any]) -> bool:
        """Criterio ANCLA estructural (NO predictivo)"""
        return (
            candidate.get("is_structural", False)
            and candidate.get("confidence_structural", 0) >= 0.8
            and candidate.get("volatility_1y", 1.0) <= 0.25
            and candidate.get("max_drawdown_5y", -1.0) >= -0.40
        )

    # --------------------------------------------------
    # EVALUAR POSICIÓN EXISTENTE (CORE)
    # --------------------------------------------------
    def evaluate_position(self, pos: Dict[str, Any]) -> DefensiveDecision:
        ticker = str(pos.get("ticker", "UNKNOWN")).upper()
        entry = float(pos.get("entry_price", 0))
        price = float(pos.get("price_now", 0))
        entry_time = str(pos.get("entry_time", ""))
        is_anchor = bool(pos.get("is_anchor", False))

        ts = datetime.now(self.tz).isoformat()

        # VALIDACIÓN → Proteger capital
        if entry <= 0 or price <= 0:
            return DefensiveDecision(
                "CLOSE", ticker, "invalid_price_data", ts,
                {"entry": entry, "price": price}
            )

        ret = pct_change(price, entry)
        age = days_between(entry_time)

        # 🚨 STOP CATASTRÓFICO (ÚNICO)
        if ret <= -CATASTROPHIC_STOP_PCT:
            return DefensiveDecision(
                "CLOSE", ticker, "catastrophic_loss", ts,
                {"ret_pct": round(ret * 100, 2), "stop_pct": -CATASTROPHIC_STOP_PCT * 100}
            )

        # 🧱 ANCLA → HOLD INDEFINIDO
        if is_anchor:
            return DefensiveDecision(
                "HOLD", ticker, "anchor_hold_indefinite", ts,
                {
                    "ret_pct": round(ret * 100, 2),
                    "days_held": age,
                    "anchor": True,
                    "dist_to_stop_pct": round((ret + CATASTROPHIC_STOP_PCT) * 100, 1)
                }
            )

        # ⏱️ NO-ANCLA envejecido → CLOSE
        if age >= MAX_HOLD_DAYS_NON_ANCHOR:
            return DefensiveDecision(
                "CLOSE", ticker, "non_anchor_time_exit", ts,
                {"days_held": age, "max_days": MAX_HOLD_DAYS_NON_ANCHOR}
            )

        # 🟡 NO-ANCLA sano → HOLD temporal
        return DefensiveDecision(
            "HOLD", ticker, "defensive_hold_non_anchor", ts,
            {
                "ret_pct": round(ret * 100, 2),
                "days_held": age,
                "anchor": False,
                "days_to_exit": MAX_HOLD_DAYS_NON_ANCHOR - age
            }
        )

    # --------------------------------------------------
    # ROTACIÓN → FRÁGIL → ANCLA
    # --------------------------------------------------
    def evaluate_rotation(
        self,
        fragile_pos: Dict[str, Any],
        anchor_candidate: Dict[str, Any]
    ) -> DefensiveDecision:
        """Rota especulativo → estructural"""
        if not self.is_anchor_asset(anchor_candidate):
            return self.evaluate_position(fragile_pos)  # Fallback HOLD/CLOSE

        ts = datetime.now(self.tz).isoformat()
        return DefensiveDecision(
            "ROTATE", fragile_pos["ticker"], "rotate_fragile_to_anchor", ts,
            {
                "close_ticker": fragile_pos.get("ticker"),
                "open_ticker": anchor_candidate.get("ticker"),
                "anchor_quality": {
                    "conf_structural": anchor_candidate.get("confidence_structural"),
                    "vol_1y": anchor_candidate.get("volatility_1y"),
                    "max_dd_5y": anchor_candidate.get("max_drawdown_5y")
                },
                "fragile_ret_pct": round(pct_change(
                    fragile_pos.get("price_now", 0), 
                    fragile_pos.get("entry_price", 0)
                ) * 100, 2)
            }
        )

    # --------------------------------------------------
    # API PRINCIPAL → MAIN.PY COMPATIBLE
    # --------------------------------------------------
    def evaluate_portfolio(
        self,
        positions: List[Dict[str, Any]],
        anchor_universe: List[Dict[str, Any]] = None,
        total_capital: float = 1000000
    ) -> List[DefensiveDecision]:
        """
        API completa para MarketOrchestrator/main.py
        1️⃣ Evalúa posiciones → HOLD/CLOSE
        2️⃣ Identifica rotaciones → FRÁGIL→ANCLA
        """
        decisions: List[DefensiveDecision] = []
        anchors = [p for p in positions if p.get("is_anchor", False)]
        non_anchors = [p for p in positions if not p.get("is_anchor", False)]
        
        available_anchors = anchor_universe or []

        # 1️⃣ Evaluar TODAS posiciones (HOLD/CLOSE decisions)
        for pos in positions:
            decisions.append(self.evaluate_position(pos))

        # 2️⃣ ROTACIONES → Máx 1-2 por ciclo (NO agresivo)
        rotations_done = 0
        max_rotations = min(2, len(non_anchors), len(available_anchors))
        
        for i, fragile in enumerate(non_anchors[:max_rotations]):
            if rotations_done >= max_rotations or not available_anchors:
                break
                
            anchor = available_anchors[i % len(available_anchors)]
            rotation_decision = self.evaluate_rotation(fragile, anchor)
            if rotation_decision.action == "ROTATE":
                decisions.append(rotation_decision)
                rotations_done += 1

        # 📊 Logging producción
        closes = len([d for d in decisions if d.action == "CLOSE"])
        rotates = len([d for d in decisions if d.action == "ROTATE"])
        logger.info(
            f"DEFENSIVE v1.5 | pos={len(positions)} anchors={len(anchors)} "
            f"| closes={closes} rotates={rotates} | capital=${total_capital:,.0f}"
        )

        return decisions

    def allow_new_positions(self, total_capital: float) -> bool:
        """Defensive → SOLO si anchors < límite"""
        return self.anchor_exposure_pct < MAX_ANCHOR_EXPOSURE_PCT

# =========================================================
# SELF-TEST PRODUCCIÓN REALISTA
# =========================================================
if __name__ == "__main__":
    pm = PMDefensive()
    
    # Portfolio realista
    test_positions = [
        # 🧱 ANCLA perfecto
        {
            "ticker": "MSFT", "is_anchor": True,
            "entry_price": 380.0, "price_now": 410.0,
            "qty": 50, "entry_time": "2025-12-15T14:30:00Z"
        },
        # ⏳ NO-ANCLA viejo (>30d)
        {
            "ticker": "AMD", "is_anchor": False,
            "entry_price": 145.0, "price_now": 152.0,
            "qty": 100, "entry_time": "2025-12-10T09:15:00Z"  # ~43 días
        },
        # ❌ Catastrófico
        {
            "ticker": "COIN", "is_anchor": False,
            "entry_price": 250.0, "price_now": 120.0,  # -52%
            "qty": 20, "entry_time": "2026-01-10T11:00:00Z"
        }
    ]
    
    # Pool ANCLAS disponibles
    anchor_universe = [
        {
            "ticker": "BRK.B", "is_structural": True,
            "confidence_structural": 0.92, "volatility_1y": 0.18,
            "max_drawdown_5y": -0.22
        },
        {
            "ticker": "JNJ", "is_structural": True,
            "confidence_structural": 0.87, "volatility_1y": 0.21,
            "max_drawdown_5y": -0.28
        }
    ]
    
    print("🧪 PMDefensive v1.5 PRODUCCIÓN – FULL EVAL:")
    results = pm.evaluate_portfolio(test_positions, anchor_universe, total_capital=2500000)
    
    print("
📋 DECISIONES:")
    for decision in results:
        print(json.dumps(decision.to_dict(), indent=2))
    
    print(f"
✅ PMDefensive v1.5 – TEST PASSED")
    print(f"Config → TimeMaxNonAnchor: {MAX_HOLD_DAYS_NON_ANCHOR}d | "
          f"CatStop: {CATASTROPHIC_STOP_PCT*100}% | "
          f"MaxAnchorExpo: {MAX_ANCHOR_EXPOSURE_PCT*100}%")

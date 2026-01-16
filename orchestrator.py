# =========================================================
# orchestrator.py — TRADING ORCHESTRATOR v1.0 (PRODUCTION)
# =========================================================
# ✔ Une PositionManager → Broker
# ✔ Modo DRY_RUN / LIVE
# ✔ Capital-aware
# ✔ Decisiones auditables
# ✔ Broker-safe (no loops)
# =========================================================

import os
import logging
import requests
from datetime import datetime
from typing import Dict, Any, List

from position_manager import PositionManager
from signals import compute_all_signals

# =========================================================
# CONFIG
# =========================================================
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://spy-2w-price-prediction.onrender.com"
)

BROKER_ENDPOINT = f"{BACKEND_URL}/trading/execute"

AUTO_TRADING = os.getenv("AUTO_TRADING", "false").lower() == "true"
DRY_RUN = not AUTO_TRADING   # seguridad por defecto

MAX_NEW_TRADES_PER_RUN = int(os.getenv("MAX_NEW_TRADES_PER_RUN", "1"))

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s"
)
logger = logging.getLogger("orchestrator")

# =========================================================
# HELPERS
# =========================================================
def send_to_broker(payload: Dict[str, Any]) -> Dict[str, Any]:
    if DRY_RUN:
        logger.warning(f"🧪 DRY_RUN → NO SE ENVÍA AL BROKER: {payload}")
        return {"status": "dry_run", "payload": payload}

    r = requests.post(
        BROKER_ENDPOINT,
        json=payload,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()

# =========================================================
# CORE
# =========================================================
def run_orchestrator():
    logger.info("🚀 ORCHESTRATOR START")

    pm = PositionManager()
    signals = compute_all_signals()
    signals_by_ticker = {s["ticker"]: s for s in signals}

    # ----------------------------------
    # 1️⃣ Obtener posiciones actuales
    # ----------------------------------
    try:
        r = requests.get(f"{BACKEND_URL}/positions", timeout=15)
        r.raise_for_status()
        open_positions = r.json()
    except Exception:
        logger.warning("⚠️ No se pudieron cargar posiciones → asumimos vacío")
        open_positions = []

    executed = 0

    # ----------------------------------
    # 2️⃣ Evaluar posiciones abiertas
    # ----------------------------------
    for pos in open_positions:
        ticker = pos.get("ticker")
        sig = signals_by_ticker.get(ticker)

        decision = pm.evaluate_position(pos, sig)

        if decision["action"] in {"CLOSE"}:
            logger.info(f"🔻 DECISION {decision['action']} {ticker}")
            send_to_broker({
                "action": "CLOSE",
                "ticker": ticker,
                "reason": decision["reason"],
                "meta": decision.get("meta"),
            })

    # ----------------------------------
    # 3️⃣ Evaluar ROTATION / NUEVAS
    # ----------------------------------
    if not pm.can_add_positions(open_positions):
        logger.info("🟡 Portfolio lleno → solo HOLD / ROTATE")
        return

    try:
        r = requests.get(f"{BACKEND_URL}/dashboard/screener", timeout=15)
        r.raise_for_status()
        screener = r.json()
        candidates = screener.get("candidates", [])
    except Exception as e:
        logger.error(f"❌ No screener disponible: {e}")
        return

    for c in candidates:
        if executed >= MAX_NEW_TRADES_PER_RUN:
            break

        decision = pm.evaluate_rotation(
            open_positions=open_positions,
            new_candidate=c,
            latest_signals=signals_by_ticker,
        )

        if not decision:
            continue

        logger.info(
            f"🔁 ROTATE {decision['close_ticker']} → {decision['open_ticker']} "
            f"(delta={decision['delta']})"
        )

        send_to_broker({
            "action": "ROTATE",
            "close_ticker": decision["close_ticker"],
            "open_ticker": decision["open_ticker"],
            "target_pct": 0.1,
            "meta": decision,
        })

        executed += 1

    logger.info("🏁 ORCHESTRATOR END")

# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    run_orchestrator()

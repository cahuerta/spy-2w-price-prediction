# =========================================================
# portfolio_store.py — PORTFOLIO STATE MANAGER v1.7
# =========================================================
#
# FIX v1.6:
#   [F1] load_positions() ya NO sobreescribe el store si el broker
#        devuelve una lista vacía.
#
# FIX v1.7:
#   [F2] load_positions() mergea entry_date desde positions_meta.json
#        Alpaca no retorna fecha de entrada — se guarda por separado
#        y se inyecta aquí para que el resto del sistema lo reciba
#        sin cambios en sus interfaces.
#
# =========================================================

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pytz
import os
import shutil
import httpx
from dataclasses import dataclass
from broker import get_engine
from positions_meta import merge_into_positions


# =========================================================
# CONFIG
# =========================================================

DATA_PATH        = Path(os.getenv("DATA_PATH", "/data"))
POSITIONS_FILE   = DATA_PATH / "positions.json"
POSITIONS_BACKUP = DATA_PATH / "positions_backup.json"

API_BASE = os.getenv(
    "API_BASE_URL",
    "https://spy-2w-price-prediction.onrender.com"
)

CL_TIMEZONE = pytz.timezone("America/Santiago")

logger = logging.getLogger("portfolio_store")
logging.basicConfig(level=logging.INFO)

_store_lock = threading.Lock()

# =========================================================
# STRUCT
# =========================================================

@dataclass
class PortfolioMetrics:
    positions_count: int
    anchors_count: int
    total_value: float
    unrealized_pnl_pct: float
    anchor_exposure_ratio: float
    anchor_exposure_pct: float
    timestamp: str

# =========================================================
# UTILS
# =========================================================

def _now():
    return datetime.now(CL_TIMEZONE).isoformat()

def _ensure_store():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    if not POSITIONS_FILE.exists():
        POSITIONS_FILE.write_text("[]", encoding="utf-8")

def _backup():
    if POSITIONS_FILE.exists():
        shutil.copy2(POSITIONS_FILE, POSITIONS_BACKUP)

def _save_raw(data):
    with _store_lock:
        _backup()
        if isinstance(data, dict):
            data = [{"ticker": t, **v} for t, v in data.items()]
        POSITIONS_FILE.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8"
        )

# =========================================================
# FETCH DESDE /trading/positions
# =========================================================

def _fetch_positions_api():
    url = f"{API_BASE}/trading/positions"
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(url)
        if r.status_code != 200:
            raise RuntimeError("positions API error")
        data = r.json()
        if isinstance(data, dict):
            data = [{"ticker": t, **v} for t, v in data.items()]
        logger.info(f"🔄 Sync API positions → {len(data)}")
        return data
    except Exception as e:
        logger.warning(f"⚠️ API positions fallback: {e}")
        return None

# =========================================================
# LOAD POSITIONS
# =========================================================

def load_positions() -> List[Dict[str, Any]]:
    """
    Fuente de verdad: broker (Alpaca).
    Si el broker devuelve posiciones reales → guardar y retornar.
    Si el broker devuelve vacío → NO sobreescribir, leer desde disco.
    Si disco también vacío → retornar [].

    [F2] Siempre mergea entry_date desde positions_meta.json
    """
    _ensure_store()

    try:
        engine = get_engine()
        data   = engine.get_positions()

        if isinstance(data, dict):
            data = [{"ticker": t, **v} for t, v in data.items()]

        if data:
            logger.info(f"✅ Broker positions: {len(data)}")
            _save_raw(data)
            # [F2] Mergear metadatos antes de retornar
            return merge_into_positions(data)

        logger.info("⚠️ Broker devolvió 0 posiciones → leyendo desde disco")

    except Exception as e:
        logger.warning(f"⚠️ Broker get_positions falló: {e} → leyendo desde disco")

    try:
        data = json.loads(POSITIONS_FILE.read_text())
        if isinstance(data, dict):
            data = [{"ticker": t, **v} for t, v in data.items()]
        logger.info(f"📂 Disco positions: {len(data)}")
        # [F2] Mergear metadatos también en fallback de disco
        return merge_into_positions(data)

    except Exception as e:
        logger.error(f"❌ Error leyendo positions.json: {e}")
        return []

# =========================================================
# SAVE POSITIONS
# =========================================================

def save_positions(positions: List[Dict[str, Any]]):
    _save_raw(positions)

# =========================================================
# CLEAR POSITIONS
# =========================================================

def clear_positions():
    """
    Llamar explícitamente cuando se confirma que el broker
    no tiene posiciones abiertas (portfolio realmente vacío).
    """
    logger.info("🧹 Clearing positions store (broker confirmed empty)")
    _save_raw([])

# =========================================================
# PRICE UPDATE
# =========================================================

def update_prices(price_map: Dict[str, float]):
    positions = load_positions()
    updated   = 0
    for p in positions:
        t = p["ticker"]
        if t in price_map:
            price = float(price_map[t])
            if price > 0:
                p["price_now"] = price
                updated += 1
    if updated:
        _save_raw(positions)
    return updated

# =========================================================
# METRICS
# =========================================================

def portfolio_metrics() -> PortfolioMetrics:
    positions   = load_positions()
    total_value = 0
    unrealized  = 0
    entry_value = 0
    anchors     = 0

    for p in positions:
        price        = p.get("price_now", p["entry_price"])
        value        = p["qty"] * price
        total_value += value
        entry_value += p["qty"] * p["entry_price"]
        unrealized  += (price - p["entry_price"]) * p["qty"]
        if p.get("is_anchor"):
            anchors += 1

    ratio = anchors / len(positions) if positions else 0

    return PortfolioMetrics(
        positions_count=len(positions),
        anchors_count=anchors,
        total_value=round(total_value, 2),
        unrealized_pnl_pct=round((unrealized / entry_value * 100) if entry_value else 0, 2),
        anchor_exposure_ratio=round(ratio, 4),
        anchor_exposure_pct=round(ratio * 100, 1),
        timestamp=_now()
    )
    

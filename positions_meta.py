# =========================================================
# positions_meta.py — POSITION METADATA STORE v1.0
# =========================================================
# Guarda metadatos que Alpaca no retorna:
#   - entry_date: fecha de entrada (para tracking vs curva)
#   - prediction_date: fecha de la predicción que originó la orden
#
# Archivo: /data/positions_meta.json
# Formato: { "AAPL": { "entry_date": "2026-04-18", ... } }
#
# load_positions() en portfolio_store mergea estos datos
# automáticamente — el resto del sistema los recibe sin cambios.
# =========================================================

import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import os

logger    = logging.getLogger("positions_meta")
DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
META_FILE = DATA_PATH / "positions_meta.json"
_lock     = threading.Lock()


def _load_raw() -> Dict[str, Any]:
    if not META_FILE.exists():
        return {}
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"⚠️ positions_meta load error: {e}")
        return {}


def _save_raw(data: Dict[str, Any]) -> None:
    with _lock:
        META_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = META_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(META_FILE)


def set_entry_date(ticker: str, entry_date: Optional[str] = None) -> None:
    """
    Registra la fecha de entrada de una posición.
    Llamar desde trading_orchestrator cuando el fill es exitoso.
    """
    ticker = ticker.upper()
    meta   = _load_raw()
    meta[ticker] = {
        "entry_date":      entry_date or datetime.utcnow().date().isoformat(),
        "registered_at":   datetime.utcnow().isoformat(),
    }
    _save_raw(meta)
    logger.info(f"📅 entry_date registrado | {ticker} → {meta[ticker]['entry_date']}")


def get_entry_date(ticker: str) -> Optional[str]:
    """Retorna entry_date para un ticker, o None si no existe."""
    return _load_raw().get(ticker.upper(), {}).get("entry_date")


def remove_entry(ticker: str) -> None:
    """Limpiar cuando se cierra una posición."""
    meta = _load_raw()
    if ticker.upper() in meta:
        del meta[ticker.upper()]
        _save_raw(meta)
        logger.info(f"🗑 positions_meta limpiado: {ticker}")


def get_all() -> Dict[str, Any]:
    return _load_raw()


def merge_into_positions(positions: list) -> list:
    """
    Agrega entry_date a cada posición desde el meta store.
    Llamado en load_positions() — no modifica posiciones sin meta.
    """
    meta = _load_raw()
    for pos in positions:
        ticker = pos.get("ticker", "").upper()
        if ticker in meta:
            pos["entry_date"] = meta[ticker].get("entry_date")
    return positions

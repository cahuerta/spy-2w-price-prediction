# =========================================================
# screener_data_layer.py
# Screener Profesional — ESTABLE Y ROBUSTO
# Universo = SP500 (Yahoo) → fallback local JSON
# Datos = Alpaca → Yahoo fallback
# =========================================================

from typing import Dict, Any, Optional, List
import os
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
import time
from threading import Lock
import pandas as pd
import yfinance as yf
import json
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
LOOKBACK_DAYS = int(os.getenv("SCREENER_LOOKBACK", "90"))
MIN_REQUIRED_DAYS = 15  # ⬅ más tolerante
MAX_CONCURRENT = 3      # ⬅ Yahoo no tolera alto paralelismo
MAX_UNIVERSE = 300

BASE_DIR = Path(__file__).resolve().parent
LOCAL_SP500_FILE = BASE_DIR / "sp500.json"

# =========================================================
# LOGGING
# =========================================================
logger = logging.getLogger("screener_data_layer")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

# =========================================================
# RATE LIMITING
# =========================================================
_rate_lock = Lock()
_last_fetch = 0.0
_min_interval = 0.2   # ⬅ Más realista para Yahoo

def _rate_limit():
    global _last_fetch
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_fetch
        if elapsed < _min_interval:
            time.sleep(_min_interval - elapsed)
        _last_fetch = time.time()

# =========================================================
# UNIVERSO
# =========================================================
def _get_sp500_yahoo() -> List[str]:
    try:
        table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        symbols = table["Symbol"].tolist()
        symbols = [s.replace(".", "-") for s in symbols]
        logger.info(f"SP500 Yahoo loaded: {len(symbols)}")
        return symbols
    except Exception as e:
        logger.warning(f"SP500 Yahoo failed: {e}")
        return []

def _get_sp500_local() -> List[str]:
    if not LOCAL_SP500_FILE.exists():
        logger.warning("Local SP500 JSON not found")
        return []

    try:
        with open(LOCAL_SP500_FILE, "r") as f:
            symbols = json.load(f)
        logger.info(f"SP500 local loaded: {len(symbols)}")
        return symbols
    except Exception:
        return []

def _discover_universe() -> List[str]:
    logger.info("🔍 Discovering universe...")

    symbols = _get_sp500_yahoo()

    if not symbols:
        symbols = _get_sp500_local()

    symbols = sorted(set(symbols))[:MAX_UNIVERSE]

    logger.info(f"📊 Universe final size: {len(symbols)}")

    return symbols

# =========================================================
# SANITIZE
# =========================================================
def _sanitize_arrays(closes, volumes):
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    mask = (closes > 0) & (volumes > 0)

    closes = closes[mask][-LOOKBACK_DAYS:]
    volumes = volumes[mask][-LOOKBACK_DAYS:]

    return closes, volumes

def _validate_arrays(closes, volumes):
    if closes is None or volumes is None:
        return False
    if len(closes) < MIN_REQUIRED_DAYS:
        return False
    if len(closes) != len(volumes):
        return False
    if np.isnan(closes).any():
        return False
    return True

# =========================================================
# FETCH DATA (SOLO YAHOO ESTABLE)
# =========================================================
def _fetch_from_yahoo(symbol):

    _rate_limit()

    try:
        df = yf.download(
            symbol,
            period=f"{LOOKBACK_DAYS}d",
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df is None or df.empty:
            logger.debug(f"Yahoo empty: {symbol}")
            return None

        return df["Close"].values, df["Volume"].values

    except Exception as e:
        logger.debug(f"Yahoo error {symbol}: {e}")
        return None

# =========================================================
# PUBLIC API
# =========================================================
def fetch_symbol_data(symbol: str) -> Optional[Dict[str, Any]]:

    data = _fetch_from_yahoo(symbol)

    if data is None:
        return None

    closes, volumes = _sanitize_arrays(*data)

    if not _validate_arrays(closes, volumes):
        return None

    return {
        "symbol": symbol,
        "closes": closes,
        "volumes": volumes,
        "source": "yahoo",
        "days": len(closes),
    }

def fetch_multiple_symbols(
    symbols: Optional[List[str]] = None,
    max_concurrent: Optional[int] = None
):

    if symbols is None:
        symbols = _discover_universe()

    if not symbols:
        logger.warning("⚠️ No symbols discovered")
        return []

    workers = max_concurrent or MAX_CONCURRENT

    logger.info(f"Fetching data for {len(symbols)} symbols")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fetch_symbol_data, symbols))

    valid = [r for r in results if r is not None]

    logger.info(f"Valid symbols after fetch: {len(valid)}")

    return valid

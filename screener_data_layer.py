# =========================================================
# screener_data_layer.py
# Screener Profesional — ROBUSTO
# Universo = SP500 (Yahoo) → fallback local JSON
# Datos = Alpaca → Yahoo fallback
# Máximo 300 símbolos
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
MIN_REQUIRED_DAYS = 20
MAX_CONCURRENT = int(os.getenv("SCREENER_CONCURRENT", "10"))
MAX_UNIVERSE = 300

DATA_PATH = Path(os.getenv("DATA_PATH", "."))  # repo root
LOCAL_SP500_FILE = DATA_PATH / "sp500.json"

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
_min_interval = 0.05

def _rate_limit():
    global _last_fetch
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_fetch
        if elapsed < _min_interval:
            time.sleep(_min_interval - elapsed)
        _last_fetch = time.time()

# =========================================================
# ALPACA (OPCIONAL)
# =========================================================
ALPACA_AVAILABLE = True
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except Exception:
    ALPACA_AVAILABLE = False

_data_client = None
_client_lock = Lock()

def _init_alpaca():
    global _data_client
    if not ALPACA_AVAILABLE:
        return None

    if _data_client:
        return _data_client

    with _client_lock:
        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")

        if not key or not secret:
            return None

        try:
            _data_client = StockHistoricalDataClient(key, secret)
            logger.info("✅ Alpaca client initialized")
        except Exception:
            return None

    return _data_client

# =========================================================
# UNIVERSO SP500
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
# SANITIZE + VALIDATE
# =========================================================
def _sanitize_arrays(closes, volumes):
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    closes = np.nan_to_num(closes)
    volumes = np.nan_to_num(volumes)

    mask = (closes > 0) & (volumes > 0)
    closes = closes[mask][-LOOKBACK_DAYS:]
    volumes = volumes[mask][-LOOKBACK_DAYS:]

    return closes, volumes

def _validate_arrays(closes, volumes):
    if len(closes) < MIN_REQUIRED_DAYS:
        return False
    if len(closes) != len(volumes):
        return False
    if np.isnan(closes).all():
        return False
    if np.min(closes) < 0.01:
        return False
    return True

# =========================================================
# FETCH DATA
# =========================================================
def _fetch_from_alpaca(symbol):
    client = _init_alpaca()
    if not client:
        return None

    _rate_limit()

    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            limit=LOOKBACK_DAYS,
        )
        bars = client.get_stock_bars(request).data.get(symbol)
        if not bars:
            return None

        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]
        return closes, volumes
    except Exception:
        return None

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

        if df.empty:
            return None

        return df["Close"].values, df["Volume"].values
    except Exception:
        return None

# =========================================================
# PUBLIC API
# =========================================================
def fetch_symbol_data(symbol: str) -> Optional[Dict[str, Any]]:

    data = _fetch_from_alpaca(symbol)
    source = "alpaca"

    if data is None:
        data = _fetch_from_yahoo(symbol)
        source = "yahoo"

    if data is None:
        return None

    closes, volumes = _sanitize_arrays(*data)

    if not _validate_arrays(closes, volumes):
        return None

    return {
        "symbol": symbol,
        "closes": closes,
        "volumes": volumes,
        "source": source,
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

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fetch_symbol_data, symbols))

    return results

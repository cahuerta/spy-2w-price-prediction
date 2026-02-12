# =========================================================
# screener_data_layer.py
# Data Layer para Screener — CRON MODE ✅
# Sin cache automático
# Alpaca → Yahoo fallback
# Coordinado por screener.py
# =========================================================

from typing import Dict, Any, Optional, Tuple, List
import os
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
import time
from threading import Lock

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
# CONFIG
# =========================================================
LOOKBACK_DAYS = int(os.getenv("SCREENER_LOOKBACK", "90"))
MIN_REQUIRED_DAYS = 20
MAX_CONCURRENT = int(os.getenv("SCREENER_CONCURRENT", "10"))

# =========================================================
# RATE LIMITING (GLOBAL SAFE)
# =========================================================
_rate_lock = Lock()
_last_fetch = 0.0
_min_interval = 0.1  # 100ms entre requests

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
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except Exception:
    ALPACA_AVAILABLE = False

_ALPACA_CLIENT = None
_alpaca_lock = Lock()

def _get_alpaca_client():
    global _ALPACA_CLIENT

    if not ALPACA_AVAILABLE:
        return None

    if _ALPACA_CLIENT is not None:
        return _ALPACA_CLIENT

    with _alpaca_lock:
        if _ALPACA_CLIENT is not None:
            return _ALPACA_CLIENT

        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")

        if not key or not secret:
            return None

        try:
            _ALPACA_CLIENT = StockHistoricalDataClient(key, secret)
            logger.info("✅ Alpaca client inicializado")
            return _ALPACA_CLIENT
        except Exception as e:
            logger.error(f"❌ Alpaca init failed: {e}")
            return None

# =========================================================
# YAHOO FALLBACK
# =========================================================
import yfinance as yf

# =========================================================
# SANITIZACIÓN Y VALIDACIÓN
# =========================================================
def _sanitize_arrays(closes: np.ndarray, volumes: np.ndarray):
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    closes = np.nan_to_num(closes)
    volumes = np.nan_to_num(volumes)

    mask = (closes > 0) & (volumes > 0)
    closes = closes[mask]
    volumes = volumes[mask]

    closes = closes[-LOOKBACK_DAYS:]
    volumes = volumes[-LOOKBACK_DAYS:]

    return closes, volumes

def _validate_arrays(closes: np.ndarray, volumes: np.ndarray):
    if closes is None or volumes is None:
        return False
    if len(closes) < MIN_REQUIRED_DAYS:
        return False
    if len(closes) != len(volumes):
        return False
    if np.isnan(closes).all():
        return False
    if np.min(closes) < 0.01:
        return False
    if np.max(closes) > 1_000_000:
        return False
    return True

# =========================================================
# FETCH ALPACA
# =========================================================
def _fetch_from_alpaca(symbol: str):
    _rate_limit()
    client = _get_alpaca_client()
    if client is None:
        return None

    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            limit=LOOKBACK_DAYS,
        )

        bars = client.get_stock_bars(request).data.get(symbol)

        if not bars or len(bars) < MIN_REQUIRED_DAYS:
            return None

        closes = np.array([b.close for b in bars], dtype=float)
        volumes = np.array([b.volume for b in bars], dtype=float)

        return closes, volumes

    except Exception:
        return None

# =========================================================
# FETCH YAHOO
# =========================================================
def _fetch_from_yahoo(symbol: str):
    _rate_limit()

    try:
        ticker = symbol
        known_suffixes = ['.US', '.TO', '.CA', '.SN', '.DE']
        if not any(s in symbol.upper() for s in known_suffixes):
            ticker = f"{symbol}.US"

        df = yf.download(
            ticker,
            period=f"{LOOKBACK_DAYS}d",
            progress=False,
            auto_adjust=True,
            prepost=False,
            threads=False
        )

        if df is None or df.empty or len(df) < MIN_REQUIRED_DAYS:
            if ticker.endswith(".US") and ticker != symbol:
                df = yf.download(
                    symbol,
                    period=f"{LOOKBACK_DAYS}d",
                    progress=False,
                    auto_adjust=True,
                    prepost=False,
                    threads=False
                )

        if df is None or df.empty or len(df) < MIN_REQUIRED_DAYS:
            return None

        df = df[["Close", "Volume"]].dropna()

        closes = df["Close"].values
        volumes = df["Volume"].values

        return closes, volumes

    except Exception:
        return None

# =========================================================
# API PRINCIPAL
# =========================================================
def fetch_symbol_data(symbol: str) -> Optional[Dict[str, Any]]:

    # 1️⃣ Alpaca
    data = _fetch_from_alpaca(symbol)
    source = "alpaca"

    # 2️⃣ Yahoo fallback
    if data is None:
        data = _fetch_from_yahoo(symbol)
        source = "yahoo"

    if data is None:
        logger.warning(f"❌ No data: {symbol}")
        return None

    closes, volumes = _sanitize_arrays(*data)

    if not _validate_arrays(closes, volumes):
        logger.warning(f"❌ Invalid data: {symbol}")
        return None

    return {
        "symbol": symbol,
        "closes": closes,
        "volumes": volumes,
        "source": source,
        "days": len(closes),
    }

# =========================================================
# BATCH
# =========================================================
def fetch_multiple_symbols(
    symbols: List[str],
    max_concurrent: Optional[int] = None
) -> List[Optional[Dict[str, Any]]]:

    workers = max_concurrent or MAX_CONCURRENT

    if workers == 1:
        return [fetch_symbol_data(s) for s in symbols]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fetch_symbol_data, symbols))

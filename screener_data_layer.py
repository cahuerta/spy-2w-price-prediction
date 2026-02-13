# =========================================================
# screener_data_layer.py
# Data Layer AUTOSUFICIENTE — Screener Profesional vFinal
# Universo = SP500 + Nasdaq100 + Most Actives
# Máximo 300 símbolos
# Alpaca → Yahoo fallback
# =========================================================

from typing import Dict, Any, Optional, List, Set
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
MAX_UNIVERSE = 300  # 🔒 HARD LIMIT

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
# ALPACA IMPORTS
# =========================================================
ALPACA_AVAILABLE = True

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.screener import ScreenerClient
    SCREENER_AVAILABLE = True
except Exception:
    ALPACA_AVAILABLE = False
    SCREENER_AVAILABLE = False

# =========================================================
# CLIENT SINGLETON
# =========================================================
_data_client = None
_screener_client = None
_client_lock = Lock()

def _init_clients():
    global _data_client, _screener_client

    if not ALPACA_AVAILABLE:
        return None, None

    if _data_client and _screener_client:
        return _data_client, _screener_client

    with _client_lock:
        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")

        if not key or not secret:
            logger.warning("⚠️ Alpaca credentials missing")
            return None, None

        try:
            _data_client = StockHistoricalDataClient(key, secret)
            if SCREENER_AVAILABLE:
                _screener_client = ScreenerClient(key, secret)
            logger.info("✅ Alpaca clients initialized")
        except Exception as e:
            logger.error(f"❌ Alpaca init failed: {e}")
            return None, None

    return _data_client, _screener_client

# =========================================================
# SYMBOL FILTER
# =========================================================
def _is_clean_symbol(symbol: str) -> bool:
    s = symbol.upper()
    if "." in s or "-" in s or "/" in s:
        return False
    if len(s) > 5:
        return False
    return True

# =========================================================
# SP500 + NASDAQ100 vía Yahoo
# =========================================================
import yfinance as yf

def _get_sp500() -> Set[str]:
    try:
        table = yf.Ticker("^GSPC").constituents
        return {s for s in table if _is_clean_symbol(s)}
    except Exception:
        return set()

def _get_nasdaq100() -> Set[str]:
    try:
        table = yf.Ticker("^NDX").constituents
        return {s for s in table if _is_clean_symbol(s)}
    except Exception:
        return set()

# =========================================================
# MOST ACTIVES (ALPACA)
# =========================================================
def _get_most_actives(limit=150) -> Set[str]:

    _, screener = _init_clients()
    if not screener:
        return set()

    try:
        actives = screener.get_most_actives()
        symbols = set()

        for a in actives:
            s = a["symbol"]
            if _is_clean_symbol(s):
                symbols.add(s)
            if len(symbols) >= limit:
                break

        return symbols

    except Exception:
        return set()

# =========================================================
# DISCOVER UNIVERSE
# =========================================================
def _discover_universe() -> List[str]:

    logger.info("🔍 Discovering universe...")

    sp500 = _get_sp500()
    nasdaq = _get_nasdaq100()
    actives = _get_most_actives()

    universe = list(sp500 | nasdaq | actives)

    universe = sorted(universe)[:MAX_UNIVERSE]

    logger.info(f"📊 Universe final size: {len(universe)}")

    return universe

# =========================================================
# SANITIZE + VALIDATE
# =========================================================
def _sanitize_arrays(closes: np.ndarray, volumes: np.ndarray):
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    closes = np.nan_to_num(closes)
    volumes = np.nan_to_num(volumes)

    mask = (closes > 0) & (volumes > 0)
    closes = closes[mask][-LOOKBACK_DAYS:]
    volumes = volumes[mask][-LOOKBACK_DAYS:]

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
    return True

# =========================================================
# FETCH ALPACA
# =========================================================
def _fetch_from_alpaca(symbol: str):

    data, _ = _init_clients()
    if not data:
        return None

    _rate_limit()

    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            limit=LOOKBACK_DAYS,
        )

        bars = data.get_stock_bars(request).data.get(symbol)

        if not bars or len(bars) < MIN_REQUIRED_DAYS:
            return None

        closes = np.array([b.close for b in bars], dtype=float)
        volumes = np.array([b.volume for b in bars], dtype=float)

        return closes, volumes

    except Exception:
        return None

# =========================================================
# FETCH YAHOO (FALLBACK)
# =========================================================
def _fetch_from_yahoo(symbol: str):

    _rate_limit()

    try:
        df = yf.download(
            symbol,
            period=f"{LOOKBACK_DAYS}d",
            progress=False,
            auto_adjust=True,
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

# =========================================================
# BATCH
# =========================================================
def fetch_multiple_symbols(
    symbols: Optional[List[str]] = None,
    max_concurrent: Optional[int] = None
) -> List[Optional[Dict[str, Any]]]:

    if symbols is None:
        symbols = _discover_universe()

    if not symbols:
        logger.warning("⚠️ No symbols discovered")
        return []

    workers = max_concurrent or MAX_CONCURRENT

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fetch_symbol_data, symbols))

    return results

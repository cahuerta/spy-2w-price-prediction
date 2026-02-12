# =========================================================
# screener_data_layer.py
# Data Layer AUTOSUFICIENTE para Screener
# Descubre universo + descarga datos
# Alpaca → Yahoo fallback
# =========================================================

from typing import Dict, Any, Optional, List
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
MAX_UNIVERSE = int(os.getenv("SCREENER_MAX_UNIVERSE", "300"))

# =========================================================
# RATE LIMITING
# =========================================================
_rate_lock = Lock()
_last_fetch = 0.0
_min_interval = 0.1

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
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    try:
        from alpaca.data.screener import ScreenerClient
        SCREENER_AVAILABLE = True
    except Exception:
        SCREENER_AVAILABLE = False

except Exception:
    ALPACA_AVAILABLE = False
    SCREENER_AVAILABLE = False


# =========================================================
# CLIENTS SINGLETON
# =========================================================
_trading_client = None
_data_client = None
_client_lock = Lock()

def _init_alpaca_clients():
    global _trading_client, _data_client

    if not ALPACA_AVAILABLE:
        return None, None

    if _trading_client and _data_client:
        return _trading_client, _data_client

    with _client_lock:
        if _trading_client and _data_client:
            return _trading_client, _data_client

        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"

        if not key or not secret:
            logger.warning("⚠️ Alpaca credentials missing")
            return None, None

        try:
            _trading_client = TradingClient(key, secret, paper=paper)
            _data_client = StockHistoricalDataClient(key, secret)
            logger.info("✅ Alpaca clients initialized")
        except Exception as e:
            logger.error(f"❌ Alpaca init failed: {e}")
            return None, None

    return _trading_client, _data_client


# =========================================================
# DISCOVER UNIVERSE
# =========================================================
def _discover_universe(limit: int = MAX_UNIVERSE) -> List[str]:

    trading, data = _init_alpaca_clients()

    if not trading:
        logger.warning("⚠️ No Alpaca → universe empty")
        return []

    # 1️⃣ Intentar Most Actives
    if SCREENER_AVAILABLE:
        try:
            screener = ScreenerClient(
                os.getenv("ALPACA_API_KEY"),
                os.getenv("ALPACA_SECRET_KEY")
            )
            actives = screener.get_most_actives()
            symbols = [a["symbol"] for a in actives[:limit]]
            logger.info(f"📊 Universe: {len(symbols)} most actives")
            return symbols
        except Exception as e:
            logger.warning(f"Most actives failed: {e}")

    # 2️⃣ Fallback assets
    try:
        assets = trading.get_all_assets()

        symbols = [
            a.symbol
            for a in assets
            if a.tradable
            and a.asset_class == AssetClass.US_EQUITY
            and a.exchange in ("NYSE", "NASDAQ")
        ][:limit]

        logger.info(f"📊 Universe fallback: {len(symbols)} assets")
        return symbols

    except Exception as e:
        logger.error(f"Universe fallback failed: {e}")
        return []


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
# FETCH ALPACA DATA
# =========================================================
def _fetch_from_alpaca(symbol: str):

    trading, data = _init_alpaca_clients()
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
# FETCH YAHOO
# =========================================================
import yfinance as yf

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
# BATCH — AUTOSUFICIENTE
# =========================================================
def fetch_multiple_symbols(
    symbols: Optional[List[str]] = None,
    max_concurrent: Optional[int] = None
) -> List[Optional[Dict[str, Any]]]:

    # 🔥 Si no pasan símbolos → descubre universo
    if symbols is None:
        symbols = _discover_universe()

    if not symbols:
        logger.warning("⚠️ No symbols discovered")
        return []

    workers = max_concurrent or MAX_CONCURRENT

    if workers == 1:
        return [fetch_symbol_data(s) for s in symbols]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fetch_symbol_data, symbols))

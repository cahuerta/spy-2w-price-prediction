# =========================================================
# market_data_cache_v2.py - PIPELINE INTEGRATED
# =========================================================
# 🔥 CENTRAL DATA LAYER
# ✔ Lee tickers.json automáticamente
# ✔ Batch paralelo
# ✔ Atomic writes seguros
# ✔ Model / Alpha solo leen cache
# ✔ Pipeline solo llama 1 función
# =========================================================

import os
import json
import logging
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
CACHE_DIR = DATA_PATH / "market_cache"
TICKERS_FILE = DATA_PATH / "tickers.json"

MAX_WORKERS = min(8, os.cpu_count() or 4)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CACHE] %(message)s")
logger = logging.getLogger("market_cache")


# =========================================================
# ATOMIC WRITERS
# =========================================================

def save_json_atomic(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)

    backup = path.with_suffix(".json.bak")
    if path.exists():
        if backup.exists():
            backup.unlink()
        path.replace(backup)

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def save_parquet_atomic(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    backup = path.with_suffix(".parquet.bak")
    if path.exists():
        if backup.exists():
            backup.unlink()
        path.replace(backup)

    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, compression="snappy", index=True)
    tmp.replace(path)


# =========================================================
# CORE REFRESH
# =========================================================

def refresh_market_cache(ticker: str, period: str = "5y") -> bool:
    try:
        logger.info(f"🔄 Refreshing {ticker}")
        tk = yf.Ticker(ticker)

        hist = tk.history(period=period, interval="1d", auto_adjust=True)

        if hist is None or hist.empty:
            logger.error(f"❌ No price data: {ticker}")
            return False

        hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna()

        if len(hist) < 100:
            logger.error(f"❌ Insufficient rows: {ticker}")
            return False

        hist.index = pd.to_datetime(hist.index, utc=True)
        hist.index.name = "Date"

        price_file = CACHE_DIR / f"{ticker}_prices.parquet"
        save_parquet_atomic(hist, price_file)

        raw_info = tk.info or {}
        dividends = tk.dividends.tail(12)

        meta_payload = {
            "ticker": ticker,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "rows_count": len(hist),
            "date_range": {
                "start": hist.index[0].isoformat(),
                "end": hist.index[-1].isoformat(),
            },
            "info": {
                "name": raw_info.get("shortName") or raw_info.get("longName"),
                "sector": raw_info.get("sector"),
                "industry": raw_info.get("industry"),
                "marketCap": raw_info.get("marketCap"),
                "trailingPE": raw_info.get("trailingPE"),
                "forwardPE": raw_info.get("forwardPE"),
                "dividendYield": raw_info.get("dividendYield"),
                "floatShares": raw_info.get("floatShares"),
                "beta": raw_info.get("beta"),
                "averageVolume": raw_info.get("averageVolume"),
                "payoutRatio": raw_info.get("payoutRatio"),
                "dividendRate": raw_info.get("dividendRate"),
                "forwardAnnualDividendRate": raw_info.get("forwardAnnualDividendRate"),
            },
            "dividends_last_year": {
                str(k.date()): float(v) for k, v in dividends.items()
            } if dividends is not None else {}
        }

        meta_file = CACHE_DIR / f"{ticker}_meta.json"
        save_json_atomic(meta_file, meta_payload)

        logger.info(f"✅ {ticker}: {len(hist)} rows cached")
        return True

    except Exception as e:
        logger.error(f"💥 {ticker}: {str(e)[:120]}")
        return False


# =========================================================
# BATCH FROM tickers.json (🔥 NUEVO)
# =========================================================

def load_universe_from_file() -> List[str]:
    if not TICKERS_FILE.exists():
        raise FileNotFoundError("tickers.json not found")

    data = json.loads(TICKERS_FILE.read_text())

    if isinstance(data, dict) and "tickers" in data:
        tickers = data["tickers"]
    elif isinstance(data, list):
        tickers = data
    else:
        raise ValueError("Invalid tickers.json format")

    return sorted(set(tickers))


def refresh_from_universe_file(max_workers: int = None) -> Dict[str, bool]:
    tickers = load_universe_from_file()
    max_workers = max_workers or MAX_WORKERS

    logger.info(f"🔥 Universe refresh | {len(tickers)} tickers | {max_workers} workers")

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(refresh_market_cache, ticker): ticker
            for ticker in tickers
        }

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                results[ticker] = False
                logger.error(f"❌ {ticker} batch error: {e}")

    success_rate = sum(results.values()) / len(results) if results else 0
    logger.info(f"✅ Universe refresh complete | {success_rate:.1%} success")

    return results


# =========================================================
# READERS
# =========================================================

def load_prices_df(ticker: str) -> pd.DataFrame:
    price_file = CACHE_DIR / f"{ticker}_prices.parquet"
    if not price_file.exists():
        raise FileNotFoundError(f"Cache missing: {ticker}")

    df = pd.read_parquet(price_file)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def load_meta_cache(ticker: str) -> Dict[str, Any]:
    meta_file = CACHE_DIR / f"{ticker}_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Meta missing: {ticker}")
    return json.loads(meta_file.read_text())


def get_last_price_from_cache(ticker: str) -> float:
    df = load_prices_df(ticker)
    return float(df["Close"].iloc[-1])


# =========================================================
# TEST LOCAL
# =========================================================

if __name__ == "__main__":
    refresh_from_universe_file()

    cached = load_universe_from_file()
    print(f"Tickers loaded: {len(cached)}")

    df = load_prices_df(cached[0])
    print(f"{cached[0]} rows: {len(df)} | Last: {df['Close'].iloc[-1]:.2f}")

# =========================================================
# market_data_cache.py
# =========================================================
# 🔥 CENTRAL DATA LAYER
# ✔ Descarga Yahoo UNA sola vez
# ✔ Guarda snapshot completo por ticker
# ✔ Sobrescribe mismo archivo diariamente
# ✔ Atomic write seguro
# ✔ Compatible con model / alpha / fundamental
# =========================================================

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import yfinance as yf
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
CACHE_DIR = DATA_PATH / "market_cache"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CACHE] %(message)s")
logger = logging.getLogger("market_cache")


# =========================================================
# HELPERS
# =========================================================

def _atomic_save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _serialize_prices(df: pd.DataFrame) -> list:
    """
    Convierte OHLCV a lista ligera serializable
    """
    df = df.reset_index()
    df["Date"] = df["Date"].astype(str)

    return df[["Date", "Open", "High", "Low", "Close", "Volume"]] \
        .fillna(0) \
        .to_dict(orient="records")


def _serialize_dividends(series: pd.Series) -> list:
    if series is None or series.empty:
        return []
    return [
        {"date": str(idx.date()), "dividend": float(val)}
        for idx, val in series.items()
    ]


# =========================================================
# CORE DOWNLOAD
# =========================================================

def refresh_market_cache(ticker: str, period: str = "max") -> dict:
    """
    Descarga datos completos y sobrescribe archivo cache
    """

    logger.info(f"🔄 Refresh cache {ticker}")

    tk = yf.Ticker(ticker)

    # ---- Price History ----
    hist = tk.history(period=period, interval="1d", auto_adjust=True)

    if hist is None or hist.empty:
        raise RuntimeError(f"No price data for {ticker}")

    hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # ---- Info ----
    info = tk.info or {}

    # ---- Dividends ----
    dividends = tk.dividends

    payload = {
        "ticker": ticker,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "prices": _serialize_prices(hist),
        "info": info,
        "dividends": _serialize_dividends(dividends),
    }

    file_path = CACHE_DIR / f"{ticker}.json"
    _atomic_save(file_path, payload)

    logger.info(f"✅ Cache actualizado {ticker}")

    return payload


# =========================================================
# LOAD (SIN Yahoo)
# =========================================================

def load_market_cache(ticker: str) -> dict:
    """
    Lee archivo cache.
    NO llama Yahoo.
    """
    file_path = CACHE_DIR / f"{ticker}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"No cache for {ticker}")

    return json.loads(file_path.read_text())


# =========================================================
# GET DATAFRAME PARA MODEL
# =========================================================

def get_prices_df_from_cache(ticker: str) -> pd.DataFrame:
    """
    Devuelve DataFrame listo para model.py
    """
    data = load_market_cache(ticker)

    df = pd.DataFrame(data["prices"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    return df


# =========================================================
# GET FUNDAMENTAL SNAPSHOT
# =========================================================

def get_fundamental_from_cache(ticker: str) -> dict:
    data = load_market_cache(ticker)
    return {
        "info": data.get("info", {}),
        "dividends": data.get("dividends", [])
    }


# =========================================================
# USO MANUAL
# =========================================================

if __name__ == "__main__":
    for t in ["SPY", "AAPL", "TSLA"]:
        refresh_market_cache(t)

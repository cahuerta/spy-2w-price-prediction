# =========================================================
# providers/eodhd_provider.py
# Proveedor EOD Historical Data (EODHD)
# ✅ Contrato idéntico: get_price_history(ticker, period, interval) -> pd.DataFrame
# ✅ Soporta mercados USA, España (.MC), Alemania (.XETRA / .F)
# ✅ Plan free: usa API key "demo" (20 req/día, solo AAPL)
# ✅ Plan básico (~$20/mes): API key real, acceso completo
# =========================================================

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger("eodhd_provider")

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
EODHD_API_KEY = os.getenv("EODHD_API_KEY", "demo")
EODHD_BASE_URL = "https://eodhd.com/api/eod"

# Mapeo de sufijos yfinance → sufijos EODHD
# yfinance usa .MC, .DE / EODHD usa .MC, .XETRA
SUFFIX_MAP = {
    ".DE": ".XETRA",   # Alemania: SAP.DE → SAP.XETRA
    ".MC": ".MC",       # España: SAN.MC → SAN.MC (igual)
    ".F":  ".F",        # Frankfurt alternativo (igual)
}

# Cuántos días pedir según period string
PERIOD_DAYS = {
    "1mo":  30,
    "3mo":  90,
    "6mo":  180,
    "1y":   365,
    "2y":   730,
    "5y":   1825,
    "max":  7300,  # ~20 años
}


# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------
def _translate_ticker(ticker: str) -> str:
    """
    Convierte ticker de formato yfinance a formato EODHD.
    Ejemplos:
      SAP.DE   → SAP.XETRA
      SAN.MC   → SAN.MC
      AAPL     → AAPL.US
      SPY      → SPY.US
    """
    ticker = ticker.upper().strip()

    for yf_suffix, eodhd_suffix in SUFFIX_MAP.items():
        if ticker.endswith(yf_suffix):
            base = ticker[: -len(yf_suffix)]
            return f"{base}{eodhd_suffix}"

    # Sin sufijo → mercado USA
    if "." not in ticker:
        return f"{ticker}.US"

    return ticker  # ya tiene sufijo desconocido, pasar tal cual


def _period_to_dates(period: str):
    """Devuelve (from_date, to_date) como strings YYYY-MM-DD."""
    days = PERIOD_DAYS.get(period, 365)
    to_dt = datetime.utcnow()
    from_dt = to_dt - timedelta(days=days)
    return from_dt.strftime("%Y-%m-%d"), to_dt.strftime("%Y-%m-%d")


# ----------------------------------------------------------
# FETCH PRINCIPAL
# ----------------------------------------------------------
def get_price_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Descarga historial OHLCV desde EODHD.

    Retorna DataFrame con columnas:
        Open, High, Low, Close, Volume
    Index: DatetimeIndex (fecha)

    Retorna DataFrame vacío si falla.
    """

    # EODHD solo soporta datos diarios en plan básico
    if interval not in ("1d", "1wk"):
        logger.warning(
            f"[EODHD] interval '{interval}' no soportado, usando '1d'"
        )
        interval = "1d"

    eodhd_ticker = _translate_ticker(ticker)
    from_date, to_date = _period_to_dates(period)

    params = {
        "api_token": EODHD_API_KEY,
        "from":      from_date,
        "to":        to_date,
        "period":    "d" if interval == "1d" else "w",
        "fmt":       "json",
    }

    url = f"{EODHD_BASE_URL}/{eodhd_ticker}"

    try:
        logger.info(f"[EODHD] GET {eodhd_ticker} ({from_date} → {to_date})")
        resp = requests.get(url, params=params, timeout=15)

        if resp.status_code == 429:
            logger.warning(f"[EODHD] Rate limit alcanzado para {eodhd_ticker}")
            return pd.DataFrame()

        if resp.status_code != 200:
            logger.warning(
                f"[EODHD] HTTP {resp.status_code} para {eodhd_ticker}: {resp.text[:200]}"
            )
            return pd.DataFrame()

        data = resp.json()

        if not data or not isinstance(data, list):
            logger.warning(f"[EODHD] Respuesta vacía para {eodhd_ticker}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Renombrar columnas al estándar del proyecto
        df.rename(
            columns={
                "date":          "Date",
                "open":          "Open",
                "high":          "High",
                "low":           "Low",
                "close":         "Close",
                "adjusted_close": "Adj Close",
                "volume":        "Volume",
            },
            inplace=True,
        )

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # Asegurar columnas numéricas
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["Close"], inplace=True)

        logger.info(
            f"[EODHD] ✅ {eodhd_ticker}: {len(df)} filas "
            f"({df.index[0].date()} → {df.index[-1].date()})"
        )

        return df

    except requests.exceptions.Timeout:
        logger.warning(f"[EODHD] Timeout para {eodhd_ticker}")
        return pd.DataFrame()

    except Exception as e:
        logger.warning(f"[EODHD] Error inesperado para {eodhd_ticker}: {e}")
        return pd.DataFrame()

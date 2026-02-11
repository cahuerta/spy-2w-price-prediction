# =========================================================
# providers/yahoo_provider.py
# DATA PROVIDER — YAHOO FINANCE
# =========================================================
# ✔ Fuente actual
# ✔ Usa yfinance
# ✔ Devuelve DataFrame limpio
# =========================================================

import logging
import yfinance as yf
import pandas as pd

logger = logging.getLogger("data_provider.yahoo")


def get_price_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Descarga datos históricos desde Yahoo Finance.
    Retorna DataFrame con índice datetime.
    """

    logger.info(f"[YAHOO] Descargando datos para {ticker}")

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )

        if df is None or df.empty:
            raise RuntimeError(f"Yahoo no devolvió datos para {ticker}")

        df = df.dropna()
        df.index = pd.to_datetime(df.index)

        return df

    except Exception as e:
        logger.error(f"[YAHOO] Error en {ticker}: {e}")
        raise

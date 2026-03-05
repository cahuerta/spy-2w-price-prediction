# =========================================================
# data_provider.py — FACHADA ÚNICA DE DATOS (Mantiene Contrato)
# =========================================================

import os
import logging
import pandas as pd

from providers.cache_provider import get_price_history as cache_fetch
from providers.yahoo_provider import get_price_history as yahoo_fetch
from providers.twelve_provider import get_price_history as twelve_fetch
from providers import chile_provider

from market_data_cache import load_meta_cache

logger = logging.getLogger("data_provider")

DEFAULT_PROVIDER = os.getenv("DATA_PROVIDER", "cache").lower()


def get_fundamental_meta(ticker: str):
    return load_meta_cache(ticker)


# =========================================================
# MAIN ENTRYPOINT (CONTRATO INTACTO)
# =========================================================
def get_price_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fachada con lógica de Fallback robusta.

    Orden de prioridad:
    1. Cache (rápido)
    2. Chile provider (si ticker .SN/.SCL)
    3. Yahoo Finance
    4. Twelve Data
    """

    ticker_upper = ticker.upper()
    df = pd.DataFrame()

    # -----------------------------------------------------
    # PASO 1 — CACHE (prioridad por velocidad)
    # -----------------------------------------------------
    try:
        logger.info(f"[DATA_PROVIDER] CACHE → {ticker_upper}")
        df = cache_fetch(ticker, period, interval)

        if df is not None and not df.empty:
            df = df.dropna()
            if not df.empty:
                return df

    except Exception as e:
        logger.warning(f"[DATA_PROVIDER] Cache error {ticker_upper}: {e}")

    # -----------------------------------------------------
    # PASO 2 — CHILE PROVIDER
    # -----------------------------------------------------
    if ticker_upper.endswith((".SN", ".SCL")):
        try:
            logger.info(f"[DATA_PROVIDER] CHILE_PROVIDER → {ticker_upper}")

            df = chile_provider.get_price_history(
                ticker,
                period,
                interval
            )

            if df is not None and not df.empty:
                df = df.dropna()
                if not df.empty:
                    return df

        except Exception as e:
            logger.warning(f"[DATA_PROVIDER] Chile provider error {ticker_upper}: {e}")

    # -----------------------------------------------------
    # PASO 3 — YAHOO
    # -----------------------------------------------------
    try:
        logger.info(f"[DATA_PROVIDER] YAHOO → {ticker_upper}")

        df = yahoo_fetch(
            ticker,
            period,
            interval
        )

        if df is not None and not df.empty:
            df = df.dropna()
            if not df.empty:
                return df

    except Exception as e:
        logger.warning(f"[DATA_PROVIDER] Yahoo error {ticker_upper}: {e}")

    # -----------------------------------------------------
    # PASO 4 — TWELVE DATA (último recurso)
    # -----------------------------------------------------
    try:
        logger.info(f"[DATA_PROVIDER] TWELVE → {ticker_upper}")

        df = twelve_fetch(
            ticker,
            period,
            interval
        )

        if df is not None and not df.empty:
            df = df.dropna()
            if not df.empty:
                return df

    except Exception as e:
        logger.warning(f"[DATA_PROVIDER] Twelve error {ticker_upper}: {e}")

    # -----------------------------------------------------
    # SI TODO FALLA
    # -----------------------------------------------------
    logger.critical(
        f"[DATA_PROVIDER] No se pudieron obtener datos para {ticker_upper}"
    )

    return pd.DataFrame()

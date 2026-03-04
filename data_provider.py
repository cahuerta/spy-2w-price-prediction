# =========================================================
# data_provider.py — FACHADA ÚNICA DE DATOS
# =========================================================
# ✔ Model.py SOLO importa desde aquí
# ✔ Permite cambiar proveedor sin tocar modelo
# ✔ Soporta múltiples fuentes
# ✔ CACHE FIRST (producción segura)
# =========================================================

import os
import logging
import pandas as pd

from providers.cache_provider import get_price_history as cache_fetch
from providers.yahoo_provider import get_price_history as yahoo_fetch
from providers.twelve_provider import get_price_history as twelve_fetch
from market_data_cache import load_meta_cache
logger = logging.getLogger("data_provider")

# =========================================================
# CONFIG
# =========================================================
# 🔥 DEFAULT = CACHE
DEFAULT_PROVIDER = os.getenv("DATA_PROVIDER", "cache").lower()

def get_fundamental_meta(ticker: str):
    return load_meta_cache(ticker)
    
# =========================================================
# MAIN ENTRYPOINT
# =========================================================
def get_price_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Punto único de acceso a datos históricos.
    """

    provider = DEFAULT_PROVIDER

    logger.info(f"[DATA_PROVIDER] {provider.upper()} → {ticker}")

    if provider == "cache":
        return cache_fetch(ticker, period, interval)

    elif provider == "yahoo":
        return yahoo_fetch(ticker, period, interval)

    elif provider == "twelve":
        return twelve_fetch(ticker, period, interval)

    else:
        raise RuntimeError(f"Proveedor no soportado: {provider}")

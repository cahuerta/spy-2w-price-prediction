# =========================================================
# data_provider.py — MODIFICADO PARA CHILE
# =========================================================

import os
import logging
import pandas as pd

from providers.cache_provider import get_price_history as cache_fetch
from providers.yahoo_provider import get_price_history as yahoo_fetch
from providers.twelve_provider import get_price_history as twelve_fetch
# 🔥 Importamos el nuevo proveedor
from providers.chile_provider import get_price_history as chile_fetch 

from market_data_cache import load_meta_cache

logger = logging.getLogger("data_provider")

DEFAULT_PROVIDER = os.getenv("DATA_PROVIDER", "cache").lower()

def is_chilean_ticker(ticker: str) -> bool:
    """Detecta si el instrumento pertenece a la Bolsa de Santiago."""
    return ticker.upper().endswith((".SN", ".SCL"))

def get_price_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    
    # 1. Ruteo Forzado para Chile (Yahoo falla mucho aquí)
    if is_chilean_ticker(ticker):
        logger.info(f"[DATA_PROVIDER] CHILE-ROUTE → {ticker}")
        # Si usas cache, podrías pasar por cache_fetch primero, 
        # pero para asegurar data fresca de la Bolsa:
        return chile_fetch(ticker, period)

    # 2. Ruteo Estándar (USA / Otros)
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
        

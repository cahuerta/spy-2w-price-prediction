# =========================================================
# providers/twelve_provider.py
# DATA PROVIDER — TWELVE DATA (STUB ACTUAL)
# =========================================================
# ✔ Estructura preparada
# ✔ Actualmente usa Yahoo internamente
# ✔ Luego solo reemplazamos la lógica interna
# =========================================================

import logging
import pandas as pd

# Por ahora reutilizamos yahoo
from providers.yahoo_provider import get_price_history as yahoo_fetch

logger = logging.getLogger("data_provider.twelve")


def get_price_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Stub provider.
    Actualmente usa Yahoo.
    En el futuro será reemplazado por TwelveData API real.
    """

    logger.info(f"[TWELVE-STUB] Usando Yahoo como backend para {ticker}")

    return yahoo_fetch(ticker, period=period, interval=interval)

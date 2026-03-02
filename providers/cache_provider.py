# =========================================================
# providers/cache_provider.py
# =========================================================

from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

from market_data_cache_v2 import (
    load_prices_df,
    refresh_market_cache,
    CACHE_DIR
)

MAX_CACHE_AGE_HOURS = 18


def _is_cache_stale(ticker: str) -> bool:
    meta_file = CACHE_DIR / f"{ticker}_meta.json"

    if not meta_file.exists():
        return True

    try:
        meta = json.loads(meta_file.read_text())
        updated_at = datetime.fromisoformat(meta["updated_at"])
        age_hours = (datetime.now(timezone.utc) - updated_at).total_seconds() / 3600
        return age_hours > MAX_CACHE_AGE_HOURS
    except Exception:
        return True


def get_price_history(
    ticker: str,
    period: str = "5y",
    interval: str = "1d"
):
    """
    Cache-first provider.
    Si no existe o está vencido → refresca automáticamente.
    """

    if _is_cache_stale(ticker):
        refresh_market_cache(ticker, period=period)

    return load_prices_df(ticker)

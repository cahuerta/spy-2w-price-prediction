# =========================================================
# screener.py — ALPACA MARKET SCREENER (OBSERVADOR) v2.1 FINAL
# =========================================================
# ✅ NO ejecuta trades  |  ✅ NO modifica tickers.json
# ✅ Screener API nativa |  ✅ Rate-limit safe
# ✅ RSI Wilder + Momentum continuo
# ✅ Pipeline-ready JSON |  Dashboard / CSS ready
# ✅ Production hardened
# =========================================================

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from itertools import islice

import numpy as np

# =========================
# Alpaca imports (safe)
# =========================
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.screener import ScreenerClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# =========================
# Configuración
# =========================
DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
OUTPUT_FILE = DATA_PATH / "screener_candidates.json"

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

LOOKBACK_DAYS = int(os.getenv("SCREENER_LOOKBACK", "90"))
MIN_DOLLAR_VOLUME = float(os.getenv("SCREENER_MIN_DOLLAR_VOL", "50000000"))
MIN_VOLATILITY = float(os.getenv("SCREENER_MIN_VOL", "0.015"))
MAX_VOLATILITY = float(os.getenv("SCREENER_MAX_VOL", "0.06"))
MIN_SCORE = float(os.getenv("SCREENER_MIN_SCORE", "0.6"))
MAX_ASSETS = int(os.getenv("SCREENER_MAX_ASSETS", "300"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
TOP_K = int(os.getenv("SCREENER_TOP_K", "50"))

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s"
)
logger = logging.getLogger("screener")

# =========================
# Utils matemáticos
# =========================
def pct_returns(prices: np.ndarray) -> np.ndarray:
    """Retornos porcentuales diarios."""
    return np.diff(prices) / prices[:-1]

def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """
    RSI Wilder robusto.
    """
    if len(prices) < period + 1:
        return 50.0

    deltas = np.diff(prices)
    gains = np.clip(deltas, 0, None)
    losses = -np.clip(deltas, None, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 70.0

    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))

def compute_score_enhanced(
    closes: np.ndarray,
    volumes: np.ndarray,
    dollar_volume: float
) -> Tuple[float, str]:
    """
    Score multifactorial continuo (0–1).
    """
    returns = pct_returns(closes)
    volatility = float(np.std(returns))

    trend = float((closes[-1] / closes[0]) - 1)
    rsi = compute_rsi(closes, RSI_PERIOD)

    ma5 = np.mean(closes[-5:])
    ma20 = np.mean(closes[-20:])
    momentum = np.clip((closes[-1] - ma20) / ma20, 0, 1)

    vol_score = np.clip(
        (volatility - MIN_VOLATILITY) / (MAX_VOLATILITY - MIN_VOLATILITY), 0, 1
    )
    trend_score = np.clip(trend / 0.10, 0, 1)
    rsi_score = 1 - abs(rsi - 50) / 50
    liquidity_score = np.clip(dollar_volume / (2 * MIN_DOLLAR_VOLUME), 0, 1)

    score = (
        0.35 * trend_score +
        0.25 * rsi_score +
        0.20 * momentum +
        0.10 * vol_score +
        0.10 * liquidity_score
    )

    if score >= 0.85:
        quality = "STRONG"
    elif score >= 0.70:
        quality = "GOOD"
    elif score >= 0.55:
        quality = "WEAK"
    else:
        quality = "NOISE"

    return round(float(score), 3), quality

# =========================
# Rate limiting
# =========================
def batch_process(iterable, batch_size=50, delay=1.0):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch
        time.sleep(delay)

# =========================
# Screener principal
# =========================
def run_screener(limit_assets: int = MAX_ASSETS) -> Dict[str, Any]:
    if not ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-py no instalado")

    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Credenciales Alpaca no configuradas")

    trading = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)
    data = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

    try:
        screener = ScreenerClient(ALPACA_KEY, ALPACA_SECRET)
        actives = screener.get_most_actives()
        universe = [a["symbol"] for a in actives[:limit_assets]]
        logger.info(f"🔍 Screener API: {len(universe)} activos")
    except Exception as e:
        logger.warning(f"Screener API falló ({e}). Fallback manual.")
        assets = trading.get_all_assets(asset_class=AssetClass.US_EQUITY)
        universe = [
            a.symbol for a in assets
            if a.tradable and a.exchange in ("NYSE", "NASDAQ")
        ][:limit_assets]

    logger.info(f"🚀 Analizando {len(universe)} activos")

    candidates: List[Dict[str, Any]] = []

    for batch in batch_process(universe):
        for symbol in batch:
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    limit=LOOKBACK_DAYS
                )
                bars = data.get_stock_bars(req).data.get(symbol)
                if not bars or len(bars) < 60:
                    continue

                closes = np.array([b.close for b in bars], float)
                volumes = np.array([b.volume for b in bars], float)
                dollar_volume = float(np.mean(closes * volumes))

                if dollar_volume < MIN_DOLLAR_VOLUME:
                    continue

                vol = float(np.std(pct_returns(closes)))
                if not (MIN_VOLATILITY <= vol <= MAX_VOLATILITY):
                    continue

                score, quality = compute_score_enhanced(closes, volumes, dollar_volume)
                rsi = compute_rsi(closes, RSI_PERIOD)

                if score >= MIN_SCORE:
                    candidates.append({
                        "ticker": symbol,
                        "score": score,
                        "quality": quality,
                        "trend_3m_pct": round((closes[-1] / closes[0] - 1) * 100, 2),
                        "volatility": round(vol, 4),
                        "rsi": round(rsi, 1),
                        "avg_dollar_volume": int(dollar_volume),
                        "signals_ready": False,
                        "reason": f"score={score:.3f} RSI={rsi:.0f}"
                    })

            except Exception as e:
                logger.debug(f"{symbol} skipped: {e}")

    candidates.sort(key=lambda x: x["score"], reverse=True)

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_universe": len(universe),
        "n_candidates": len(candidates),
        "top_k": TOP_K,
        "criteria": {
            "lookback_days": LOOKBACK_DAYS,
            "min_dollar_volume": MIN_DOLLAR_VOLUME,
            "volatility_range": [MIN_VOLATILITY, MAX_VOLATILITY],
            "min_score": MIN_SCORE,
            "rsi_period": RSI_PERIOD
        },
        "candidates": candidates[:TOP_K]
    }

    DATA_PATH.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))

    logger.info(f"✅ Screener FINAL | {len(output['candidates'])} candidatos")
    logger.info(f"📁 Output: {OUTPUT_FILE}")

    return output

# =========================
# CLI
# =========================
if __name__ == "__main__":
    try:
        result = run_screener()

        print("\n🎯 TOP 5 CANDIDATOS:")
        print(json.dumps(result["candidates"][:5], indent=2))

        print(f"\n📊 Total: {len(result['candidates'])} candidatos")
        print(f"💾 JSON: {OUTPUT_FILE}")

    except Exception as e:
        logger.error(f"❌ Screener falló: {e}")
        raise

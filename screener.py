# =========================================================
# screener.py — ALPACA MARKET SCREENER (OBSERVADOR) v2.3 FUNDAMENTAL
# =========================================================
# ✅ NO ejecuta trades  |  ✅ NO modifica tickers.json
# ✅ Diseñado para ser llamado por ENDPOINT / CRON
# ✅ RSI Wilder + Momentum continuo (MA20 + MA50)
# ✅ + CONTEXTO FUNDAMENTAL (model2_improved) SOLO como ETIQUETA
# ✅ Pipeline-ready JSON | Production hardened
# =========================================================

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from itertools import islice

import numpy as np

# =========================================================
# Fundamental (safe import)
# =========================================================
try:
    from model2_improved import fundamental_signal_context
    FUNDAMENTAL_AVAILABLE = True
except Exception:
    FUNDAMENTAL_AVAILABLE = False

    def fundamental_signal_context(ticker: str) -> Dict[str, Any]:
        return {"usable": False}

# =========================================================
# Alpaca imports (safe)
# =========================================================
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

# =========================================================
# Configuración
# =========================================================
DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
OUTPUT_FILE = DATA_PATH / "screener_candidates.json"

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

LOOKBACK_DAYS = int(os.getenv("SCREENER_LOOKBACK", "90"))
MIN_DOLLAR_VOLUME = float(os.getenv("SCREENER_MIN_DOLLAR_VOL", "50_000_000"))
MIN_VOLATILITY = float(os.getenv("SCREENER_MIN_VOL", "0.015"))
MAX_VOLATILITY = float(os.getenv("SCREENER_MAX_VOL", "0.06"))
MIN_SCORE = float(os.getenv("SCREENER_MIN_SCORE", "0.6"))
MAX_ASSETS = int(os.getenv("SCREENER_MAX_ASSETS", "300"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
TOP_K = int(os.getenv("SCREENER_TOP_K", "50"))

# Fundamental tags (solo etiqueta)
FUNDAMENTAL_DEEP_VALUE = float(os.getenv("FUNDAMENTAL_DEEP_VALUE", "-30"))
FUNDAMENTAL_OVERHEATED = float(os.getenv("FUNDAMENTAL_OVERHEATED", "30"))

# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s"
)
logger = logging.getLogger("screener")

# =========================================================
# Utils matemáticos
# =========================================================
def pct_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(prices) / prices[:-1]


def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
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
    returns = pct_returns(closes)
    volatility = float(np.std(returns))

    trend = float((closes[-1] / closes[0]) - 1)

    ma20 = np.mean(closes[-20:])
    ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else np.mean(closes)

    momentum_short = np.clip((closes[-1] - ma20) / ma20, 0, 1)
    momentum_long = np.clip((closes[-1] - ma50) / ma50, 0, 1)
    momentum = 0.6 * momentum_short + 0.4 * momentum_long

    rsi = compute_rsi(closes, RSI_PERIOD)

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

# =========================================================
# Rate limiting helper
# =========================================================
def batch_process(iterable, batch_size: int = 50, delay: float = 1.0):
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch
        time.sleep(delay)

# =========================================================
# Fundamental tagging (NO score)
# =========================================================
def fundamental_tag(symbol: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not FUNDAMENTAL_AVAILABLE:
        return None, None

    f = fundamental_signal_context(symbol) or {}
    if not f.get("usable"):
        return None, None

    mp = f.get("mispricing_pct")
    ms = f.get("margin_safety_pct")

    flag = "⚪ FAIR"
    if mp is not None and mp <= FUNDAMENTAL_DEEP_VALUE:
        flag = "🟢 DEEP_VALUE"
    elif mp is not None and mp >= FUNDAMENTAL_OVERHEATED:
        flag = "🔴 OVERHEATED"
    elif ms is not None and ms >= 25:
        flag = "🟡 VALUE"

    return flag, {
        "mispricing_pct": f.get("mispricing_pct"),
        "margin_safety_pct": f.get("margin_safety_pct"),
        "fundamental_reco": f.get("fundamental_reco"),
        "model": f.get("model"),
    }

# =========================================================
# SCREENER PRINCIPAL (CALLABLE)
# =========================================================
def run_screener(limit_assets: int = MAX_ASSETS) -> Dict[str, Any]:
    if not ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-py no instalado")

    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Credenciales Alpaca no configuradas")

    trading = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)
    data = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

    # Universo
    try:
        screener = ScreenerClient(ALPACA_KEY, ALPACA_SECRET)
        actives = screener.get_most_actives()
        universe = [a["symbol"] for a in actives[:limit_assets]]
    except Exception:
        assets = trading.get_all_assets(asset_class=AssetClass.US_EQUITY)
        universe = [
            a.symbol for a in assets
            if a.tradable and a.exchange in ("NYSE", "NASDAQ")
        ][:limit_assets]

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
                if score < MIN_SCORE:
                    continue

                f_flag, f_ctx = fundamental_tag(symbol)

                candidates.append({
                    "ticker": symbol,
                    "score": score,
                    "quality": quality,
                    "trend_3m_pct": round((closes[-1] / closes[0] - 1) * 100, 2),
                    "volatility": round(vol, 4),
                    "rsi": round(compute_rsi(closes), 1),
                    "avg_dollar_volume": int(dollar_volume),
                    "fundamental_flag": f_flag,
                    "fundamental": f_ctx,
                    "signals_ready": False,
                })

            except Exception:
                continue

    candidates.sort(key=lambda x: x["score"], reverse=True)

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_universe": len(universe),
        "n_candidates": len(candidates),
        "top_k": TOP_K,
        "fundamental_available": FUNDAMENTAL_AVAILABLE,
        "candidates": candidates[:TOP_K],
    }

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")

    logger.info(f"✅ Screener v2.3 FUNDAMENTAL | {len(output['candidates'])} candidatos")
    return output

# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    r = run_screener()
    print(json.dumps(r["candidates"][:5], indent=2))

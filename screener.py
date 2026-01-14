# =========================================================
# screener.py — IBKR MARKET SCREENER (REAL) v3.1 COMPATIBLE
# =========================================================
# ✅ NO ejecuta trades
# ✅ NO modifica tickers.json
# ✅ Descubre oportunidades reales (IBKR Market Scanner)
# ✅ Quant first, IA second (contextual)
# ✅ RSI Wilder + Sharpe Ratio + Beta (SPY)
# ✅ Wrapper sync seguro (solo CLI)
# ✅ JSON de salida 100% compatible
# =========================================================

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from ib_insync import IB, Stock, ScannerSubscription

# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
)
logger = logging.getLogger("screener")

# =========================================================
# IA Screener Context (SAFE / OPTIONAL)
# =========================================================
try:
    from ia import enrich_screener_candidates_batch
    IA_AVAILABLE = True
    logger.info("🧠 IA module loaded")
except Exception as e:
    IA_AVAILABLE = False
    logger.warning(f"⚠️ IA module not available: {e}")

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
# Configuración
# =========================================================
DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
OUTPUT_FILE = DATA_PATH / "screener_candidates.json"

LOOKBACK_DAYS = int(os.getenv("SCREENER_LOOKBACK", "90"))
MIN_DOLLAR_VOLUME = float(os.getenv("SCREENER_MIN_DOLLAR_VOL", "50_000_000"))
MIN_VOLATILITY = float(os.getenv("SCREENER_MIN_VOL", "0.015"))
MAX_VOLATILITY = float(os.getenv("SCREENER_MAX_VOL", "0.06"))
MIN_SCORE = float(os.getenv("SCREENER_MIN_SCORE", "0.6"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
TOP_K = int(os.getenv("SCREENER_TOP_K", "50"))

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "7497"))   # 7497 paper / 7496 live
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "19"))

# =========================================================
# Utils matemáticos (SIN CAMBIOS)
# =========================================================
def pct_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return np.array([], dtype=float)
    return np.diff(prices) / prices[:-1]


def compute_rsi_wilder(prices: np.ndarray, period: int = 14) -> float:
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 1:
        return 50.0

    deltas = np.diff(prices)
    gains = pd.Series(np.clip(deltas, 0, None))
    losses = pd.Series(np.clip(-deltas, 0, None))

    avg_gain = gains.ewm(span=period, adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(span=period, adjust=False).mean().iloc[-1]

    if avg_loss == 0:
        return 70.0

    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def compute_score_enhanced_v3(
    closes: np.ndarray,
    volumes: np.ndarray,
    dollar_volume: float,
    benchmark_returns: Optional[np.ndarray] = None,
) -> Tuple[float, str, float]:

    returns = pct_returns(closes)
    if len(returns) < 10:
        return 0.0, "❌ NOISE", 0.0

    volatility = float(np.std(returns))
    trend = float((closes[-1] / closes[0]) - 1)

    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
    ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else np.mean(closes)

    momentum_short = max((closes[-1] - ma20) / ma20, 0) if ma20 > 0 else 0
    momentum_long = max((closes[-1] - ma50) / ma50, 0) if ma50 > 0 else 0
    momentum = 0.6 * momentum_short + 0.4 * momentum_long

    rsi = compute_rsi_wilder(closes, RSI_PERIOD)

    sharpe = np.mean(returns) / volatility * np.sqrt(252) if volatility > 0 else 0
    sharpe_score = np.clip(sharpe / 1.5, 0, 1)

    beta = 0.0
    beta_score = 0.0
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        n = min(len(returns), len(benchmark_returns))
        cov = np.cov(returns[-n:], benchmark_returns[-n:])[0, 1]
        var = np.var(benchmark_returns[-n:])
        if var > 0:
            beta = cov / var
            beta_score = np.clip(1 - abs(beta - 1), 0, 1)

    vol_score = np.clip((volatility - MIN_VOLATILITY) / (MAX_VOLATILITY - MIN_VOLATILITY), 0, 1)
    trend_score = np.clip(trend / 0.10, 0, 1)
    rsi_score = 1 - abs(rsi - 50) / 50
    liquidity_score = np.clip(dollar_volume / (2 * MIN_DOLLAR_VOLUME), 0, 1)

    score = (
        0.25 * trend_score +
        0.20 * rsi_score +
        0.20 * momentum +
        0.15 * sharpe_score +
        0.10 * vol_score +
        0.10 * liquidity_score +
        0.05 * beta_score
    )

    score = float(np.clip(score, 0, 1))

    if score >= 0.85:
        quality = "🚀 STRONG"
    elif score >= 0.70:
        quality = "✅ GOOD"
    elif score >= 0.55:
        quality = "⚠️ WEAK"
    else:
        quality = "❌ NOISE"

    return round(score, 3), quality, round(beta, 3)

# =========================================================
# IBKR Market Scanner (DESCUBRIMIENTO REAL)
# =========================================================
async def scan_market_ibkr(ib: IB, limit: int = 100) -> List[str]:
    sub = ScannerSubscription(
        instrument="STK",
        locationCode="STK.US.MAJOR",
        scanCode="TOP_PERC_GAIN",
    )

    results = await ib.reqScannerSubscription(sub)
    symbols = [
        r.contractDetails.contract.symbol
        for r in results[:limit]
        if r.contractDetails and r.contractDetails.contract
    ]
    return list(dict.fromkeys(symbols))

# =========================================================
# Benchmark SPY (IBKR)
# =========================================================
async def load_benchmark_returns_ibkr(ib: IB) -> Optional[np.ndarray]:
    contract = Stock("SPY", "SMART", "USD")
    await ib.qualifyContracts(contract)

    bars = await ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=f"{LOOKBACK_DAYS} D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
    )

    if not bars or len(bars) < 60:
        return None

    closes = np.array([b.close for b in bars], float)
    return pct_returns(closes)

# =========================================================
# Fetch + evaluación por símbolo (IBKR)
# =========================================================
async def fetch_symbol_data_ibkr(
    ib: IB,
    symbol: str,
    benchmark_returns: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:

    contract = Stock(symbol, "SMART", "USD")
    await ib.qualifyContracts(contract)

    bars = await ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=f"{LOOKBACK_DAYS} D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
    )

    if not bars or len(bars) < 60:
        return None

    closes = np.array([b.close for b in bars], float)
    volumes = np.array([b.volume for b in bars], float)

    dollar_volume = float(np.mean(closes * volumes))
    if dollar_volume < MIN_DOLLAR_VOLUME:
        return None

    returns = pct_returns(closes)
    vol = float(np.std(returns))
    if not (MIN_VOLATILITY <= vol <= MAX_VOLATILITY):
        return None

    score, quality, beta = compute_score_enhanced_v3(
        closes, volumes, dollar_volume, benchmark_returns
    )

    if score < MIN_SCORE:
        return None

    sharpe = float(np.mean(returns) / vol * np.sqrt(252)) if vol > 0 else 0
    rsi = compute_rsi_wilder(closes, RSI_PERIOD)

    f_flag, f_ctx = (None, None)
    if FUNDAMENTAL_AVAILABLE:
        f_flag, f_ctx = fundamental_signal_context(symbol), None

    return {
        "ticker": symbol,
        "score": score,
        "quality": quality,
        "trend_3m_pct": round((closes[-1] / closes[0] - 1) * 100, 2),
        "volatility": round(vol, 4),
        "rsi_wilder": round(rsi, 1),
        "sharpe_ratio": round(sharpe, 2),
        "beta_spy": beta,
        "avg_dollar_volume": int(dollar_volume),
        "fundamental_flag": f_flag,
        "fundamental": f_ctx,
    }

# =========================================================
# MAIN ASYNC SCREENER
# =========================================================
async def run_screener_async() -> Dict[str, Any]:
    ib = IB()
    await ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)

    symbols = await scan_market_ibkr(ib)
    benchmark_returns = await load_benchmark_returns_ibkr(ib)

    candidates: List[Dict[str, Any]] = []
    for symbol in symbols:
        r = await fetch_symbol_data_ibkr(ib, symbol, benchmark_returns)
        if r:
            candidates.append(r)

    ib.disconnect()

    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:TOP_K]

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "version": "3.1",
        "n_universe": len(symbols),
        "n_candidates": len(candidates),
        "ia_available": IA_AVAILABLE,
        "fundamental_available": FUNDAMENTAL_AVAILABLE,
        "params": {
            "lookback_days": LOOKBACK_DAYS,
            "min_dollar_volume": MIN_DOLLAR_VOLUME,
            "min_score": MIN_SCORE,
            "top_k": TOP_K,
        },
        "candidates": candidates,
    }

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")

    return output

# =========================================================
# SYNC WRAPPER (CLI ONLY)
# =========================================================
def run_screener() -> Dict[str, Any]:
    return asyncio.run(run_screener_async())

# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    result = run_screener()
    print(json.dumps(result["candidates"][:5], indent=2))
    print(f"\n📊 Full results saved: {OUTPUT_FILE}")

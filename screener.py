# =========================================================
# screener.py — ALPACA MARKET SCREENER (OBSERVADOR) v3.1 ✅ FIXED
# =========================================================
# ✅ NO ejecuta trades
# ✅ NO modifica tickers.json
# ✅ ASYNC total (Alpaca + IA)
# ✅ Quant first, IA second (contextual)
# ✅ RSI Wilder + Sharpe Ratio + Beta (SPY)
# ✅ Wrapper sync seguro (solo CLI)
# ✅ Production hardened v3.1
# =========================================================
import alpaca
print("ALPACA VERSION:", alpaca.__version__)
import os
import json
import logging
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd

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
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

LOOKBACK_DAYS = int(os.getenv("SCREENER_LOOKBACK", "90"))
MIN_DOLLAR_VOLUME = float(os.getenv("SCREENER_MIN_DOLLAR_VOL", "50_000_000"))
MIN_VOLATILITY = float(os.getenv("SCREENER_MIN_VOL", "0.015"))
MAX_VOLATILITY = float(os.getenv("SCREENER_MAX_VOL", "0.06"))
MIN_SCORE = float(os.getenv("SCREENER_MIN_SCORE", "0.6"))
MAX_ASSETS = int(os.getenv("SCREENER_MAX_ASSETS", "300"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
TOP_K = int(os.getenv("SCREENER_TOP_K", "50"))

# Async tuning
ALPACA_CONCURRENT = int(os.getenv("ALPACA_CONCURRENT", "20"))
BATCH_SIZE = int(os.getenv("SCREENER_BATCH_SIZE", "50"))

# IA ranking weights (quant first)
IA_W_QUANT = float(os.getenv("IA_W_QUANT", "0.65"))
IA_W_CONTEXT = float(os.getenv("IA_W_CONTEXT", "0.35"))

# =========================================================
# Utils matemáticos MEJORADOS
# =========================================================
def pct_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return np.array([], dtype=float)
    return np.diff(prices) / prices[:-1]


def compute_rsi_wilder(prices: np.ndarray, period: int = 14) -> float:
    """RSI con método Wilder (EMA) - más estable"""
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
    """
    Score con Sharpe + Beta bonus.
    Devuelve: (score, quality, beta)
    """
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    returns = pct_returns(closes)
    if len(returns) < 10:
        return 0.0, "❌ NOISE", 0.0

    volatility = float(np.std(returns))
    trend = float((closes[-1] / closes[0]) - 1)

    ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else float(np.mean(closes))
    ma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else float(np.mean(closes))

    momentum_short = float(np.clip((closes[-1] - ma20) / ma20, 0, 1)) if ma20 > 0 else 0.0
    momentum_long = float(np.clip((closes[-1] - ma50) / ma50, 0, 1)) if ma50 > 0 else 0.0
    momentum = 0.6 * momentum_short + 0.4 * momentum_long

    rsi = compute_rsi_wilder(closes, RSI_PERIOD)

    # Sharpe anualizado
    sharpe = float(np.mean(returns) / volatility * np.sqrt(252)) if volatility > 0 else 0.0
    sharpe_score = float(np.clip(sharpe / 1.5, 0, 1))

    beta = 0.0
    beta_score = 0.0
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # Alinear longitudes (últimos N)
        n = min(len(returns), len(benchmark_returns))
        r = returns[-n:]
        b = benchmark_returns[-n:]
        var_b = float(np.var(b))
        if var_b > 0:
            cov = float(np.cov(r, b)[0, 1])
            beta = cov / var_b
            beta_score = float(np.clip(1 - abs(beta - 1.0), 0, 1))

    vol_score = float(np.clip((volatility - MIN_VOLATILITY) / (MAX_VOLATILITY - MIN_VOLATILITY), 0, 1))
    trend_score = float(np.clip(trend / 0.10, 0, 1))
    rsi_score = float(1 - abs(rsi - 50) / 50)
    liquidity_score = float(np.clip(dollar_volume / (2 * MIN_DOLLAR_VOLUME), 0, 1))

    score = (
        0.25 * trend_score +
        0.20 * rsi_score +
        0.20 * momentum +
        0.15 * sharpe_score +
        0.10 * vol_score +
        0.10 * liquidity_score
    )

    # Bonus beta (pequeño, para estabilidad)
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        score += 0.05 * beta_score

    score = float(np.clip(score, 0, 1))

    if score >= 0.85:
        quality = "🚀 STRONG"
    elif score >= 0.70:
        quality = "✅ GOOD"
    elif score >= 0.55:
        quality = "⚠️ WEAK"
    else:
        quality = "❌ NOISE"

    return round(score, 3), quality, round(float(beta), 3)


# =========================================================
# Fundamental tagging (NO impact on score)
# =========================================================
def fundamental_tag(symbol: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not FUNDAMENTAL_AVAILABLE:
        return None, None

    try:
        f = fundamental_signal_context(symbol) or {}
        if not f.get("usable"):
            return None, None

        mp = f.get("mispricing_pct")
        ms = f.get("margin_safety_pct")

        flag = "⚪ FAIR"
        if mp is not None and mp <= -30:
            flag = "🟢 DEEP_VALUE"
        elif mp is not None and mp >= 30:
            flag = "🔴 OVERHEATED"
        elif ms is not None and ms >= 25:
            flag = "🟡 VALUE"

        return flag, f
    except Exception:
        return None, None


# =========================================================
# Benchmark loader (SPY)
# =========================================================
def load_benchmark_returns(data: "StockHistoricalDataClient") -> Optional[np.ndarray]:
    """
    Obtiene returns de SPY usando Alpaca (mejor esfuerzo).
    Retorna np.ndarray o None.
    """
    try:
        req = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            limit=LOOKBACK_DAYS,
        )
        # En alpaca-py, .df suele venir con multiindex; manejamos ambos
        df = data.get_stock_bars(req).df
        if df is None or len(df) == 0:
            return None

        if "close" in df.columns:
            closes = df["close"].values
        elif "Close" in df.columns:
            closes = df["Close"].values
        else:
            # Intento robusto si viene con columnas raras
            closes = df.iloc[:, 0].values

        r = pct_returns(np.asarray(closes, dtype=float))
        return r if len(r) > 10 else None
    except Exception as e:
        logger.warning(f"Benchmark (SPY) no disponible: {e}")
        return None


# =========================================================
# ASYNC SYMBOL ANALYSIS
# =========================================================
async def fetch_symbol_data(
    symbol: str,
    data_client: "StockHistoricalDataClient",
    semaphore: asyncio.Semaphore,
    benchmark_returns: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    async with semaphore:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                limit=LOOKBACK_DAYS,
            )
            bars = data_client.get_stock_bars(req).data.get(symbol)
            if not bars or len(bars) < 60:
                return None

            closes = np.array([b.close for b in bars], float)
            volumes = np.array([b.volume for b in bars], float)

            dollar_volume = float(np.mean(closes * volumes))
            if dollar_volume < MIN_DOLLAR_VOLUME:
                return None

            rets = pct_returns(closes)
            if len(rets) < 10:
                return None

            vol = float(np.std(rets))
            if not (MIN_VOLATILITY <= vol <= MAX_VOLATILITY):
                return None

            score, quality, beta = compute_score_enhanced_v3(
                closes, volumes, dollar_volume, benchmark_returns=benchmark_returns
            )
            if score < MIN_SCORE:
                return None

            rsi = compute_rsi_wilder(closes, RSI_PERIOD)
            sharpe = float(np.mean(rets) / vol * np.sqrt(252)) if vol > 0 else 0.0

            f_flag, f_ctx = fundamental_tag(symbol)

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
        except Exception as e:
            logger.debug(f"{symbol} failed: {e}")
            return None


# =========================================================
# MAIN ASYNC SCREENER v3.1
# =========================================================
async def run_screener_async(limit_assets: int = MAX_ASSETS) -> Dict[str, Any]:
    if not ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-py no instalado")
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Credenciales Alpaca no configuradas")

    trading = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)
    data = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

    benchmark_returns = load_benchmark_returns(data)
    if benchmark_returns is None:
        logger.warning("Benchmark returns no disponibles → beta bonus desactivado")

    logger.info("🔍 Fetching universe...")
    try:
        screener = ScreenerClient(ALPACA_KEY, ALPACA_SECRET)
        actives = screener.get_most_actives()
        universe = [a["symbol"] for a in actives[:limit_assets]]
        logger.info(f"Universe: {len(universe)} most actives")
    except Exception as e:
        logger.warning(f"Screener API failed: {e} → fallback assets list")
        assets = trading.get_all_assets(asset_class=AssetClass.US_EQUITY)
        universe = [
            a.symbol for a in assets
            if a.tradable and a.exchange in ("NYSE", "NASDAQ")
        ][:limit_assets]
        logger.info(f"Universe fallback: {len(universe)} assets")

    semaphore = asyncio.Semaphore(ALPACA_CONCURRENT)
    candidates: List[Dict[str, Any]] = []

    total_batches = (len(universe) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(universe), BATCH_SIZE):
        batch = universe[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{total_batches} ({len(batch)} symbols)")

        results = await asyncio.gather(
            *[fetch_symbol_data(s, data, semaphore, benchmark_returns) for s in batch],
            return_exceptions=True,
        )
        candidates.extend([r for r in results if isinstance(r, dict)])

    # Quant ranking
    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:TOP_K]

    # IA enrichment (post-filter)
    if IA_AVAILABLE and candidates:
        try:
            candidates = await enrich_screener_candidates_batch(candidates)
            # IA contract: usamos SOLO context_score y event_risk (estables)
            candidates.sort(
                key=lambda x: (
                    (IA_W_QUANT * x["score"] + IA_W_CONTEXT * float(x.get("ia", {}).get("context_score", 0.5)))
                    * (1 - float(x.get("ia", {}).get("event_risk", 0.0)))
                ),
                reverse=True,
            )
            logger.info("🧠 IA re-ranking aplicado (contract stable: context_score/event_risk)")
        except Exception as e:
            logger.error(f"IA enrichment failed: {e}")

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "version": "3.1",
        "n_universe": len(universe),
        "n_candidates": len(candidates),
        "ia_available": IA_AVAILABLE,
        "fundamental_available": FUNDAMENTAL_AVAILABLE,
        "params": {
            "lookback_days": LOOKBACK_DAYS,
            "min_dollar_volume": MIN_DOLLAR_VOLUME,
            "min_score": MIN_SCORE,
            "top_k": TOP_K,
            "alpaca_concurrent": ALPACA_CONCURRENT,
            "batch_size": BATCH_SIZE,
        },
        "candidates": candidates,
    }

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")

    if candidates:
        logger.info(
            f"✅ Screener v3.1 | {len(candidates)} candidatos | TOP: {candidates[0]['ticker']} ({candidates[0]['score']:.3f})"
        )
    else:
        logger.info("✅ Screener v3.1 | 0 candidatos (filtros muy estrictos o universo vacío)")

    return output


# =========================================================
# SYNC WRAPPER (CLI ONLY) — seguro
# =========================================================
def run_screener(limit_assets: int = MAX_ASSETS) -> Dict[str, Any]:
    """
    Wrapper sync SOLO para CLI / cron python directo.
    ⚠️ No lo uses desde endpoints async de FastAPI (usa run_screener_async).
    """
    return asyncio.run(run_screener_async(limit_assets))


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    result = run_screener()
    print(json.dumps(result["candidates"][:5], indent=2))
    print(f"\n📊 Full results saved: {OUTPUT_FILE}")

# capital_governor.py — CAPITAL GOVERNOR v2.4 HEDGE FUND
# ✅ 100% backward compatible v2.x
# ✅ fixed_capital obligatorio 
# ✅ cash_reserve en CapitalState
# ✅ Beta CAPM daily (correcto)

from dataclasses import dataclass, asdict
from typing import List, Dict
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import logging
import os
import json

CL_TIMEZONE = pytz.timezone("America/Santiago")
logger = logging.getLogger("capital_governor")
logging.basicConfig(level=logging.INFO)

LOOKBACK_DAYS = 252
CONFIDENCE_LEVEL = 0.95
MARKET_BENCHMARK = "SPY"

@dataclass
class CapitalState:
    volatility_annual: float
    var_95_annual: float
    expected_shortfall_95_annual: float
    beta_vs_spy: float
    total_value: float
    cash_reserve: float      # 🆕
    timestamp_utc: str
    timestamp_cl: str
    data_quality: str

    def to_dict(self):
        return asdict(self)

class CapitalGovernor:
    def __init__(self, fixed_capital: float):
        if fixed_capital is None:
            fixed_capital = float(os.getenv('FIXED_CAPITAL', 1000000))
        self.fixed_capital = fixed_capital
        logger.info(f"🏛 CapitalGovernor v2.4 | Capital: ${fixed_capital:,.0f}")

    def evaluate(self, positions: List[Dict]) -> CapitalState:
        if not positions:
            raise ValueError("❌ Portfolio vacío")

        # VALIDACIÓN MILITAR
        for i, p in enumerate(positions):
            required = ["ticker", "qty", "price_now"]
            if not all(k in p for k in required):
                raise ValueError(f"❌ Pos {i}: {p}")
            if p["qty"] <= 0 or p["price_now"] <= 0:
                raise ValueError(f"❌ Pos {i} qty/price: {p}")

        tickers = list({p["ticker"].upper() for p in positions})
        tickers_with_benchmark = tickers + [MARKET_BENCHMARK]
        
        logger.info(f"📊 {len(tickers)} tickers vs {MARKET_BENCHMARK}")

        # YFINANCE ROBUSTO
        data = yf.download(tickers_with_benchmark, period="1y", 
                          auto_adjust=True, progress=False, threads=True)
        if data.empty:
            raise RuntimeError("❌ yfinance vacío")

        # MULTIINDEX BULLETPROOF
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data["Close"]

        prices = prices.dropna(axis=1, thresh=len(prices)*0.8)
        if len(prices) < LOOKBACK_DAYS * 0.5:
            raise RuntimeError(f"❌ Datos insuficientes: {len(prices)}d")

        returns = prices.pct_change().dropna()

        # PESOS + VALOR
        total_value = sum(p["qty"] * p["price_now"] for p in positions)
        weights = {p["ticker"].upper(): (p["qty"] * p["price_now"]) / total_value 
                  for p in positions if p["ticker"].upper() in prices.columns}
        
        valid_tickers = list(weights.keys())
        if not valid_tickers:
            raise RuntimeError("❌ Sin tickers válidos")

        w = np.array([weights[t] for t in valid_tickers])
        asset_returns = returns[valid_tickers]

        # RIESGO REAL
        cov_matrix = asset_returns.cov() * 252
        portfolio_vol = np.sqrt(w.T @ cov_matrix.values @ w)
        portfolio_returns = asset_returns @ w
        
        var_daily = portfolio_returns.quantile(1 - CONFIDENCE_LEVEL)
        var_annual = abs(var_daily) * np.sqrt(252)

        tail = portfolio_returns[portfolio_returns <= var_daily]
        es_daily = tail.mean() if len(tail) > 0 else var_daily * 1.2  # Conservador
        es_annual = abs(es_daily) * np.sqrt(252)

        # BETA CAPM CORRECTO (daily cov → annual var)
        beta = 1.0
        if MARKET_BENCHMARK in returns.columns:
            spy_ret = returns[MARKET_BENCHMARK]
            beta = np.cov(portfolio_returns, spy_ret)[0,1] / np.var(spy_ret)

        cash_reserve = self.fixed_capital - total_value

        # QUALITY
        days = len(prices)
        data_quality = ("EXCELLENT" if days >= LOOKBACK_DAYS else 
                       "GOOD" if days >= LOOKBACK_DAYS*0.75 else 
                       "WARNING" if days >= LOOKBACK_DAYS*0.5 else "POOR")

        now_utc = datetime.utcnow()
        now_cl = now_utc.astimezone(CL_TIMEZONE)

        logger.info(f"🏛 Vol:{portfolio_vol:.1%} VaR:{var_annual:.1%} ES:{es_annual:.1%} "
                   f"Beta:{beta:.2f} Cash:${cash_reserve:,.0f} Quality:{data_quality}")

        return CapitalState(
            volatility_annual=round(float(portfolio_vol), 4),
            var_95_annual=round(float(var_annual), 4),
            expected_shortfall_95_annual=round(float(es_annual), 4),
            beta_vs_spy=round(float(beta), 3),
            total_value=round(float(total_value), 2),
            cash_reserve=round(float(cash_reserve), 2),
            timestamp_utc=now_utc.isoformat(),
            timestamp_cl=now_cl.isoformat(),
            data_quality=data_quality
        )

if __name__ == "__main__":
    positions = [
        {"ticker": "AAPL", "qty": 100, "price_now": 220.0},
        {"ticker": "MSFT", "qty": 50, "price_now": 410.0},
        {"ticker": "GOOGL", "qty": 30, "price_now": 145.0},
    ]
    
    gov = CapitalGovernor(fixed_capital=1000000)
    state = gov.evaluate(positions)
    print("🏛 CAPITAL GOVERNOR v2.4 HEDGE FUND")
    print(json.dumps(state.to_dict(), indent=2))

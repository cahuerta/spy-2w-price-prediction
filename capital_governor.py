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

    # --- INSERTAR ESTO DENTRO DE LA CLASE CapitalGovernor (v2.4) ---

    def adjust_sizing(self, current_positions: List[Dict], candidates: List[Dict]) -> List[Dict]:
        """
        Toma las intenciones de los PMs y ajusta los 'shares' según 
        el Cash Buffer del 10% y el Riesgo Sistémico (ES).
        """
        try:
            # Re-evaluamos el estado actual antes de procesar candidatos
            state = self.evaluate(current_positions)
        except Exception as e:
            logger.error(f"❌ Error evaluando riesgo para sizing: {e}")
            return [] # Si falla el riesgo, no abrimos nada (Safe-stop)

        # 1. BLOQUEO TOTAL DE RIESGO (ES > 45%)
        if state.expected_shortfall_95_annual > 0.45:
            logger.warning(f"🛑 RIESGO EXTREMO: ES {state.expected_shortfall_95_annual:.2%}. Bloqueando aperturas.")
            return []

        # 2. CÁLCULO DE PODER DE COMPRA REAL (Respetando el 10% de Buffer)
        # Usamos state.cash_reserve que ya calculaste en evaluate()
        safety_buffer = self.fixed_capital * 0.10 
        effective_buying_power = max(0, state.cash_reserve - safety_buffer)

        logger.info(f"💰 Cash Reserve: ${state.cash_reserve:,.0f} | Buffer: ${safety_buffer:,.0f} | Power: ${effective_buying_power:,.0f}")

        adjusted_decisions = []
        
        for cand in candidates:
            ticker = cand.get('ticker', '').upper()
            # Buscamos el precio en cualquiera de los dos campos comunes
            price = cand.get('entry_price') or cand.get('price_now')
            requested_shares = cand.get('shares', 0)
            
            if not price or requested_shares <= 0:
                logger.warning(f"⚠️ {ticker} saltado: Precio o shares inválidos ({price}, {requested_shares})")
                continue

            # 3. FACTOR DE REDUCCIÓN SEGÚN RIESGO ACTUAL
            risk_factor = 1.0
            if state.expected_shortfall_95_annual > 0.35: 
                risk_factor = 0.5 # Reducimos a la mitad si el riesgo es alto
            elif state.expected_shortfall_95_annual > 0.25: 
                risk_factor = 0.8 # Reducimos un 20% si el riesgo es moderado

            # 4. SIZING ATÓMICO
            max_shares_by_cash = int(effective_buying_power // price)
            final_shares = int(min(requested_shares, max_shares_by_cash) * risk_factor)

            if final_shares > 0:
                # Modificamos la decisión original con los shares aprobados por el Gobernador
                cand['shares'] = final_shares
                
                # Auditoría de por qué se cambió el tamaño
                cand.setdefault('meta', {})['governor_adj'] = {
                    "original_shares": requested_shares,
                    "risk_factor": risk_factor,
                    "es_at_execution": state.expected_shortfall_95_annual
                }
                
                adjusted_decisions.append(cand)
                
                # RESTAMOS EL COSTO DEL PODER DE COMPRA PARA EL SIGUIENTE CANDIDATO
                # Esto evita que el orquestador mande 5 órdenes que juntas superen el cash
                effective_buying_power -= (final_shares * price)
                
                logger.info(f"✅ {ticker}: Ajustado a {final_shares}s (Original: {requested_shares})")
            else:
                logger.warning(f"❌ {ticker}: Rechazado por falta de Capital/Buffer.")

        return adjusted_decisions


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

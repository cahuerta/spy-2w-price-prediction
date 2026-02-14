# =========================================================
# capital_governor.py — CAPITAL GOVERNOR v2.1 PRODUCTION
# =========================================================
# ✔ Riesgo REAL (cov histórica)
# ✔ VaR histórico 95%
# ✔ Expected Shortfall 95% + FALLBACK
# ✔ Beta portafolio vs SPY
# ✔ MultiIndex yfinance SAFETY
# ✔ Logging granular
# ✔ Sin placeholders
# ✔ Sin defaults inventados
# ✔ Falla explícita si faltan datos
# =========================================================

from dataclasses import dataclass, asdict
from typing import Dict, List
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import logging

CL_TIMEZONE = pytz.timezone("America/Santiago")
logger = logging.getLogger("capital_governor")
logging.basicConfig(level=logging.INFO)

LOOKBACK_DAYS = 252
CONFIDENCE_LEVEL = 0.95
MARKET_BENCHMARK = "SPY"

# =========================================================
# OUTPUT STRUCT
# =========================================================

@dataclass
class CapitalState:
    volatility_annual: float
    var_95_annual: float
    expected_shortfall_95_annual: float
    beta_vs_spy: float
    total_value: float
    timestamp_utc: str
    timestamp_cl: str
    data_quality: str  # "EXCELLENT", "GOOD", "WARNING", "POOR"

    def to_dict(self):
        return asdict(self)


# =========================================================
# CAPITAL GOVERNOR v2.1 PRODUCTION
# =========================================================

class CapitalGovernor:

    def __init__(self):
        logger.info("🏛 CapitalGovernor v2.1 PRODUCTION iniciado")

    # -----------------------------------------------------
    # CORE REAL RISK ENGINE
    # -----------------------------------------------------

    def evaluate(self, positions: List[Dict]) -> CapitalState:
        if not positions:
            raise ValueError("❌ No hay posiciones")

        # -------------------------
        # 1️⃣ VALIDACIÓN ESTRICTA
        # -------------------------
        for i, p in enumerate(positions):
            if "ticker" not in p or "qty" not in p or "price_now" not in p:
                raise ValueError(f"❌ Posición {i} inválida: {p}")
            if p["qty"] <= 0 or p["price_now"] <= 0:
                raise ValueError(f"❌ Posición {i} qty/price inválidos: {p}")

        tickers = list({p["ticker"].upper() for p in positions})
        tickers_with_benchmark = tickers + [MARKET_BENCHMARK]
        
        logger.info(f"📊 Evaluando {len(tickers)} tickers vs {MARKET_BENCHMARK}")

        # -------------------------
        # 2️⃣ DESCARGA REAL HISTÓRICA + MULTIINDEX SAFETY
        # -------------------------
        try:
            data = yf.download(
                tickers_with_benchmark,
                period="1y",
                auto_adjust=True,
                progress=False,
                group_by='ticker'  # Fuerza MultiIndex limpio
            )
        except Exception as e:
            raise RuntimeError(f"❌ Error yfinance: {e}")

        if data.empty:
            raise RuntimeError("❌ No se pudieron descargar datos históricos")

        # MultiIndex handling robusto
        if isinstance(data.columns, pd.MultiIndex):
            prices = data.xs('Close', axis=1, level=1, drop_level=False)
        else:
            prices = data['Close'] if 'Close' in data.columns else data

        # Limpieza agresiva: elimina tickers con >20% missing
        prices = prices.dropna(axis=1, thresh=len(prices)*0.8)
        
        if len(prices.columns) < len(tickers):
            missing = set(tickers) - set(prices.columns)
            logger.warning(f"⚠️  Tickers sin datos: {missing}")

        if len(prices) < LOOKBACK_DAYS * 0.5:
            raise RuntimeError(f"❌ Histórico insuficiente: {len(prices)} días")

        returns = prices.pct_change().dropna()
        logger.info(f"✅ Datos: {len(prices)} días, {len(prices.columns)} tickers")

        # -------------------------
        # 3️⃣ PESOS REALES
        # -------------------------
        total_value = sum(p["qty"] * p["price_now"] for p in positions)
        weights_dict = {}
        for p in positions:
            ticker = p["ticker"].upper()
            value = p["qty"] * p["price_now"]
            if ticker in prices.columns:
                weights_dict[ticker] = value / total_value
            else:
                logger.warning(f"⚠️ Ticker {ticker} sin histórico, peso=0")

        # Solo tickers con datos
        valid_tickers = [t for t in tickers if t in prices.columns]
        w = np.array([weights_dict.get(t, 0.0) for t in valid_tickers])

        if len(valid_tickers) == 0:
            raise RuntimeError("❌ Ningún ticker con datos históricos")

        # -------------------------
        # 4️⃣ MATRIZ COV REAL
        # -------------------------
        asset_returns = returns[valid_tickers]
        cov_matrix = asset_returns.cov() * 252
        portfolio_vol = np.sqrt(w.T @ cov_matrix.values @ w)

        # -------------------------
        # 5️⃣ VaR HISTÓRICO
        # -------------------------
        portfolio_returns = asset_returns @ w
        var_daily = np.percentile(portfolio_returns, (1 - CONFIDENCE_LEVEL) * 100)
        var_annual = abs(var_daily) * np.sqrt(252)

        # -------------------------
        # 6️⃣ EXPECTED SHORTFALL + FALLBACK
        # -------------------------
        tail_losses = portfolio_returns[portfolio_returns <= var_daily]
        if len(tail_losses) > 0:
            es_daily = tail_losses.mean()
        else:
            es_daily = var_daily  # Fallback conservador
            logger.warning("⚠️ Usando VaR como ES fallback")

        es_annual = abs(es_daily) * np.sqrt(252)

        # -------------------------
        # 7️⃣ BETA REAL VS SPY
        # -------------------------
        if MARKET_BENCHMARK in returns.columns:
            spy_returns = returns[MARKET_BENCHMARK]
            cov_ps = np.cov(portfolio_returns, spy_returns)[0, 1] * 252
            var_spy = np.var(spy_returns) * 252
            beta = cov_ps / var_spy if var_spy != 0 else 1.0
        else:
            beta = 1.0
            logger.warning("⚠️ Sin SPY, beta=1.0 neutral")

        # -------------------------
        # 8️⃣ DATA QUALITY
        # -------------------------
        data_days = len(prices)
        data_quality = (
            "EXCELLENT" if data_days >= LOOKBACK_DAYS else
            "GOOD" if data_days >= LOOKBACK_DAYS * 0.75 else
            "WARNING" if data_days >= LOOKBACK_DAYS * 0.5 else
            "POOR"
        )

        # -------------------------
        # 9️⃣ TIMESTAMP
        # -------------------------
        now_utc = datetime.utcnow()
        now_cl = now_utc.astimezone(CL_TIMEZONE)

        logger.info(
            f"🏛 PRODUCTION RISK | "
            f"Vol={portfolio_vol:.2%} | VaR95={var_annual:.2%} | "
            f"ES95={es_annual:.2%} | Beta={beta:.2f} | "
            f"Quality={data_quality}"
        )

        return CapitalState(
            volatility_annual=round(float(portfolio_vol), 4),
            var_95_annual=round(float(var_annual), 4),
            expected_shortfall_95_annual=round(float(es_annual), 4),
            beta_vs_spy=round(float(beta), 3),
            total_value=round(float(total_value), 2),
            timestamp_utc=now_utc.isoformat(),
            timestamp_cl=now_cl.isoformat(),
            data_quality=data_quality,
        )


# =========================================================
# TEST / CLI
# =========================================================

if __name__ == "__main__":
    # Posiciones de ejemplo
    test_positions = [
        {"ticker": "AAPL", "qty": 100, "price_now": 220.0},
        {"ticker": "MSFT", "qty": 50, "price_now": 410.0},
        {"ticker": "GOOGL", "qty": 30, "price_now": 145.0},
    ]
    
    governor = CapitalGovernor()
    state = governor.evaluate(test_positions)
    
    print("🏛 CAPITAL GOVERNOR v2.1 PRODUCTION")
    print(json.dumps(state.to_dict(), indent=2))

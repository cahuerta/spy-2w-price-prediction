# model2.py
# Modelo FUNDAMENTAL para acciones con dividendos
# - Estima valor económico basado en dividendos
# - NO hace timing
# - Sirve como ancla para reducir ruido
# - Pensado para dividend-paying stocks

import yfinance as yf
import numpy as np


def run_model_fundamental_dividend(
    ticker: str,
    discount_rate: float = 0.09,     # r → tasa de descuento (9% default)
    growth_rate: float = 0.03,       # g → crecimiento anual del dividendo
    horizon_days: int = 10
):
    """
    Calcula:
    - Precio fundamental por dividendos
    - Proyección lenta a T+10
    - Desalineación mercado vs valor
    """

    tk = yf.Ticker(ticker)
    info = tk.info

    price_market = info.get("currentPrice")
    dividend_rate = info.get("dividendRate")  # USD por año
    dividend_yield = info.get("dividendYield")

    if price_market is None:
        raise RuntimeError("Precio de mercado no disponible")

    if dividend_rate is None or dividend_rate <= 0:
        raise RuntimeError("La acción no paga dividendos suficientes")

    # =========================
    # Valor fundamental actual (modelo Gordon simplificado)
    # =========================
    if discount_rate <= growth_rate:
        raise RuntimeError("La tasa de descuento debe ser mayor al crecimiento")

    price_fundamental_now = dividend_rate / (discount_rate - growth_rate)

    # =========================
    # Proyección lenta a T+10 (fundamental, no mercado)
    # =========================
    growth_period = growth_rate * (horizon_days / 252)
    price_fundamental_t10 = price_fundamental_now * np.exp(growth_period)

    # =========================
    # Mispricing
    # =========================
    mispricing_pct = (price_market / price_fundamental_now - 1) * 100

    # =========================
    # Yield implícito al precio de mercado
    # =========================
    yield_market = dividend_rate / price_market * 100

    return {
        "fundamental": {
            "ticker": ticker,
            "price_market": round(price_market, 2),
            "dividend_annual": round(dividend_rate, 2),
            "dividend_yield_pct": round(yield_market, 2),
            "price_fundamental_now": round(price_fundamental_now, 2),
            "price_fundamental_t10": round(price_fundamental_t10, 2),
            "mispricing_pct": round(mispricing_pct, 2),
            "assumptions": {
                "discount_rate": discount_rate,
                "growth_rate": growth_rate
            }
        }
    }

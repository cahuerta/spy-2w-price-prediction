# model2_improved.py
# ======================================================
# MODELO FUNDAMENTAL MEJORADO (DIVIDEND STOCKS)
# ------------------------------------------------------
# ✔ Gordon Growth + Two-Stage DDM
# ✔ CAPM dinámico (beta)
# ✔ Validaciones robustas
# ✔ SOLO informativo (NO timing, NO señales)
# ✔ Diseñado como ANCLA fundamental
# ======================================================

import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from scipy.optimize import newton


# ======================================================
# Supuestos configurables
# ======================================================
@dataclass
class ModelAssumptions:
    discount_rate: float = 0.09        # fallback
    growth_rate: float = 0.03          # crecimiento estable
    growth_high: float = 0.06           # crecimiento fase inicial
    years_high_growth: int = 5
    horizon_days: int = 252             # 1Y trading days


# ======================================================
# Helpers financieros
# ======================================================
def get_risk_adjusted_discount_rate(info: Dict) -> float:
    """CAPM simplificado y robusto"""
    beta = info.get("beta")
    beta = float(beta) if beta is not None else 1.0

    risk_free = 0.035      # ~US 10Y
    equity_premium = 0.055 # ERP conservador

    return risk_free + beta * equity_premium


def validate_dividend_stock(
    info: Dict,
    dividends: pd.Series,
    min_years: int = 5
) -> bool:
    """Chequeos mínimos de calidad"""
    payout = info.get("payoutRatio")
    if payout is not None:
        if payout < 0.25 or payout > 0.85:
            return False

    if dividends is None or len(dividends) < min_years * 4:
        return False

    recent_growth = dividends.pct_change().tail(8).mean()
    if recent_growth is not None and recent_growth < -0.10:
        return False

    return True


def gordon_growth_model(dividend: float, r: float, g: float) -> float:
    if r <= g:
        raise ValueError("r must be > g")
    return dividend * (1 + g) / (r - g)


def two_stage_ddm(
    dividend: float,
    r: float,
    g_high: float,
    g_stable: float,
    years_high: int
) -> float:
    pv_high = sum(
        dividend * (1 + g_high) ** t / (1 + r) ** t
        for t in range(1, years_high + 1)
    )

    terminal = (
        dividend
        * (1 + g_high) ** years_high
        * (1 + g_stable)
        / (r - g_stable)
    )

    pv_terminal = terminal / (1 + r) ** years_high
    return pv_high + pv_terminal


def implied_discount_rate(
    price_market: float,
    dividend: float,
    g: float
) -> float:
    def eq(r):
        return dividend * (1 + g) / (r - g) - price_market

    return float(newton(eq, 0.08, maxiter=50))


# ======================================================
# MODELO PRINCIPAL
# ======================================================
def run_model_fundamental_dividend_improved(
    ticker: str,
    use_two_stage: bool = True,
    assumptions: Optional[ModelAssumptions] = None
) -> Dict:
    """
    MODELO FUNDAMENTAL ROBUSTO
    ------------------------------------
    Retorna:
    - Valor intrínseco
    - Margen de seguridad
    - Mispricing %
    - TIR implícita
    - Recomendación FUNDAMENTAL
    """

    assumptions = assumptions or ModelAssumptions()

    tk = yf.Ticker(ticker)
    info = tk.info or {}
    dividends = tk.dividends

    # =========================
    # Datos de mercado
    # =========================
    price_market = info.get("currentPrice")
    dividend_rate = info.get("dividendRate", 0.0)
    forward_dividend = info.get(
        "forwardAnnualDividendRate",
        dividend_rate
    )

    if price_market is None or price_market <= 0:
        raise RuntimeError("Market price unavailable")

    if dividend_rate is None or dividend_rate <= 0:
        raise RuntimeError("No meaningful dividends")

    # =========================
    # CAPM dinámico
    # =========================
    r_capm = get_risk_adjusted_discount_rate(info)
    assumptions.discount_rate = r_capm

    # =========================
    # Validaciones
    # =========================
    valid_dividend_stock = validate_dividend_stock(info, dividends)

    # =========================
    # Valor fundamental
    # =========================
    if use_two_stage and valid_dividend_stock:
        intrinsic_price = two_stage_ddm(
            dividend=forward_dividend,
            r=r_capm,
            g_high=assumptions.growth_high,
            g_stable=assumptions.growth_rate,
            years_high=assumptions.years_high_growth,
        )
        model_used = "Two-Stage DDM"
    else:
        intrinsic_price = gordon_growth_model(
            dividend_rate,
            r_capm,
            assumptions.growth_rate,
        )
        model_used = "Gordon Growth"

    # =========================
    # Métricas
    # =========================
    yield_market = dividend_rate / price_market * 100
    mispricing_pct = (price_market / intrinsic_price - 1) * 100
    margin_safety_pct = max(0.0, 1 - price_market / intrinsic_price) * 100

    try:
        tir_implicita = implied_discount_rate(
            price_market,
            dividend_rate,
            assumptions.growth_rate,
        )
    except Exception:
        tir_implicita = r_capm

    price_1y_projection = intrinsic_price * (1 + assumptions.growth_rate)

    # =========================
    # Recomendación FUNDAMENTAL
    # =========================
    if abs(mispricing_pct) < 15:
        recommendation = "HOLD"
    elif mispricing_pct < -15:
        recommendation = "BUY"
    else:
        recommendation = "SELL"

    # =========================
    # OUTPUT FINAL
    # =========================
    return {
        "ticker": ticker,
        "model_used": model_used,
        "valid_dividend_stock": valid_dividend_stock,
        "market_data": {
            "price_market": round(price_market, 2),
            "dividend_annual": round(dividend_rate, 3),
            "dividend_forward": round(forward_dividend, 3),
            "yield_market_pct": round(yield_market, 2),
            "payout_ratio_pct": round((info.get("payoutRatio") or 0) * 100, 1),
            "beta": round(info.get("beta") or 1.0, 2),
        },
        "fundamental_value": {
            "price_intrinsic": round(intrinsic_price, 2),
            "price_1y_projection": round(price_1y_projection, 2),
            "mispricing_pct": round(mispricing_pct, 2),
            "margin_safety_pct": round(margin_safety_pct, 1),
            "implied_market_return_pct": round(tir_implicita * 100, 1),
        },
        "assumptions": {
            "discount_rate_capm": round(r_capm, 3),
            "growth_short_term": assumptions.growth_high if use_two_stage else assumptions.growth_rate,
            "growth_long_term": assumptions.growth_rate,
            "two_stage_used": use_two_stage,
        },
        "recommendation": recommendation,
    }


# ======================================================
# USO LOCAL (TEST)
# ======================================================
if __name__ == "__main__":
    tickers = ["KO", "JNJ", "PG", "MCD"]
    rows = []

    for t in tickers:
        try:
            r = run_model_fundamental_dividend_improved(t)
            rows.append({
                "Ticker": r["ticker"],
                "Precio": r["market_data"]["price_market"],
                "Intrínseco": r["fundamental_value"]["price_intrinsic"],
                "Margen %": r["fundamental_value"]["margin_safety_pct"],
                "Reco": r["recommendation"],
            })
        except Exception as e:
            rows.append({"Ticker": t, "Error": str(e)})

    df = pd.DataFrame(rows)
    print("\n📊 RESUMEN FUNDAMENTAL\n")
    print(df.to_string(index=False))

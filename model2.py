# ======================================================
# MODELO FUNDAMENTAL MEJORADO (DIVIDEND STOCKS)
# ------------------------------------------------------
# ✔ Gordon Growth + Two-Stage DDM
# ✔ CAPM dinámico (beta)
# ✔ Validaciones robustas
# ✔ SOLO informativo (NO timing, NO señales)
# ✔ Diseñado como ANCLA fundamental
# ✔ COMPATIBLE con signals.py (safe + silent)
# ======================================================


import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from scipy.optimize import newton
from data_provider import get_fundamental_meta
from market_data_cache import get_last_price_from_cache

# ======================================================
# Supuestos configurables
# ======================================================
@dataclass
class ModelAssumptions:
    discount_rate: float = 0.09
    growth_rate: float = 0.03
    growth_high: float = 0.06
    years_high_growth: int = 5
    horizon_days: int = 252


# ======================================================
# Helpers financieros
# ======================================================
def get_risk_adjusted_discount_rate(info: Dict) -> float:
    beta = info.get("beta")
    beta = float(beta) if beta is not None else 1.0
    risk_free = 0.035
    equity_premium = 0.055
    return risk_free + beta * equity_premium


def validate_dividend_stock(
    info: Dict,
    dividends: pd.Series,
    min_years: int = 5
) -> bool:
    payout = info.get("payoutRatio")
    if payout is not None and (payout < 0.25 or payout > 0.85):
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


def implied_discount_rate_safe(
    price_market: float,
    dividend: float,
    g: float,
    fallback: float
) -> float:
    try:
        def eq(r):
            return dividend * (1 + g) / (r - g) - price_market
        return float(newton(eq, 0.08, maxiter=50))
    except Exception:
        return fallback


# ======================================================
# MODELO PRINCIPAL (SAFE + SILENT)
# ======================================================
def run_model_fundamental_dividend_improved(
    ticker: str,
    use_two_stage: bool = True,
    assumptions: Optional[ModelAssumptions] = None,
    *,
    silent: bool = True,
    safe: bool = True
) -> Dict:
    """
    MODELO FUNDAMENTAL ROBUSTO
    - silent=True  → sin prints (signals-safe)
    - safe=True    → NO lanza excepciones fatales
    """

    assumptions = assumptions or ModelAssumptions()

    try:
        meta = get_fundamental_meta(ticker)

        info = meta.get("info", {})
        dividends_dict = meta.get("dividends_last_year", {})
        dividends = pd.Series(dividends_dict)

        price_market = get_last_price_from_cache(ticker)
        dividend_rate = info.get("dividendRate", 0.0)
        forward_dividend = info.get(
            "forwardAnnualDividendRate",
            dividend_rate
        )

        if price_market is None or price_market <= 0:
            raise RuntimeError("price_unavailable")

        if dividend_rate is None or dividend_rate <= 0:
            raise RuntimeError("no_dividends")

        r_capm = get_risk_adjusted_discount_rate(info)
        assumptions.discount_rate = r_capm

        valid_dividend_stock = validate_dividend_stock(info, dividends)

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

        mispricing_pct = (price_market / intrinsic_price - 1) * 100
        margin_safety_pct = max(0.0, 1 - price_market / intrinsic_price) * 100

        tir_implicita = implied_discount_rate_safe(
            price_market,
            dividend_rate,
            assumptions.growth_rate,
            fallback=r_capm
        )

        if abs(mispricing_pct) < 15:
            recommendation = "HOLD"
        elif mispricing_pct < -15:
            recommendation = "BUY"
        else:
            recommendation = "SELL"

        return {
            "usable": True,
            "ticker": ticker,
            "model_used": model_used,
            "valid_dividend_stock": valid_dividend_stock,
            "mispricing_pct": round(mispricing_pct, 2),
            "margin_safety_pct": round(margin_safety_pct, 1),
            "implied_market_return_pct": round(tir_implicita * 100, 1),
            "recommendation": recommendation,
        }

    except Exception as e:
        if safe:
            return {
                "usable": False,
                "ticker": ticker,
                "reason": str(e),
            }
        raise


# ======================================================
# CONTEXTO LIGERO PARA signals.py  ⭐ CLAVE ⭐
# ======================================================
def fundamental_signal_context(ticker: str) -> Dict:
    """
    Wrapper mínimo y rápido para signals.py
    """
    r = run_model_fundamental_dividend_improved(
        ticker,
        silent=True,
        safe=True
    )

    if not r.get("usable"):
        return {"usable": False}

    return {
        "usable": True,
        "mispricing_pct": r["mispricing_pct"],
        "margin_safety_pct": r["margin_safety_pct"],
        "fundamental_reco": r["recommendation"],
        "model": r["model_used"],
    }


# ======================================================
# USO LOCAL (TEST)
# ======================================================
if __name__ == "__main__":
    for t in ["KO", "JNJ", "PG", "MCD"]:
        print(fundamental_signal_context(t))

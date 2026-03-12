# capital_governor.py — CAPITAL GOVERNOR v2.5
# ✅ 100% backward compatible v2.4
# ✅ Normaliza posiciones del broker (market_value/qty → price_now)
# ✅ adjust_sizing con fallback cuando portfolio vacío
# ✅ fixed_capital obligatorio
# ✅ cash_reserve en CapitalState
# ✅ Beta CAPM daily (correcto)
#
# FIX v2.5:
#   [F1] _normalize_positions(): acepta market_value+avg_entry_price del broker
#        y deriva price_now = market_value / qty
#   [F2] adjust_sizing(): fallback cuando portfolio vacío (no crashea)
#   [F3] Validación militar mejorada: mensaje de error más claro

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
    cash_reserve: float
    timestamp_utc: str
    timestamp_cl: str
    data_quality: str

    def to_dict(self):
        return asdict(self)


class CapitalGovernor:
    def __init__(self, fixed_capital: float):
        if fixed_capital is None:
            fixed_capital = float(os.getenv("FIXED_CAPITAL", 1_000_000))
        self.fixed_capital = fixed_capital
        logger.info(f"🏛 CapitalGovernor v2.5 | Capital: ${fixed_capital:,.0f}")

    # =========================================================
    # [F1] NORMALIZADOR — broker → formato interno
    # =========================================================
    @staticmethod
    def _normalize_positions(positions: List[Dict]) -> List[Dict]:
        """
        Acepta posiciones del broker (market_value + avg_entry_price)
        o el formato interno (price_now) y devuelve siempre formato interno.

        Broker entrega:
            {"ticker": "APA", "qty": 100.0, "market_value": 3392.0,
             "avg_entry_price": 29.29, "unrealized_pl": 463.0}

        Formato interno requerido:
            {"ticker": "APA", "qty": 100.0, "price_now": 33.92}
        """
        normalized = []
        for p in positions:
            pos = dict(p)  # no mutar el original

            if "price_now" not in pos or pos["price_now"] is None:
                qty = float(pos.get("qty") or 0)
                market_value = pos.get("market_value")

                if market_value is not None and qty > 0:
                    # Precio actual real = valor de mercado / cantidad
                    pos["price_now"] = float(market_value) / qty
                else:
                    # Fallback: usar precio de entrada si no hay market_value
                    avg_entry = pos.get("avg_entry_price")
                    if avg_entry is not None:
                        pos["price_now"] = float(avg_entry)
                    else:
                        logger.warning(
                            f"⚠️ {pos.get('ticker')} sin price_now ni market_value — saltado"
                        )
                        continue

            normalized.append(pos)
        return normalized

    # =========================================================
    # EVALUATE — calcula riesgo del portfolio
    # =========================================================
    def evaluate(self, positions: List[Dict]) -> CapitalState:
        if not positions:
            raise ValueError("❌ Portfolio vacío")

        # [F1] Normalizar antes de validar
        positions = self._normalize_positions(positions)

        # VALIDACIÓN MILITAR — ahora price_now siempre existe
        for i, p in enumerate(positions):
            required = ["ticker", "qty", "price_now"]
            if not all(k in p for k in required):
                raise ValueError(
                    f"❌ Pos {i} campos faltantes — tiene: {list(p.keys())} — requiere: {required}"
                )
            if p["qty"] <= 0 or p["price_now"] <= 0:
                raise ValueError(
                    f"❌ Pos {i} qty/price inválidos: qty={p['qty']} price_now={p['price_now']}"
                )

        tickers = list({p["ticker"].upper() for p in positions})
        tickers_with_benchmark = tickers + [MARKET_BENCHMARK]

        logger.info(f"📊 {len(tickers)} tickers vs {MARKET_BENCHMARK}")

        # YFINANCE ROBUSTO
        data = yf.download(
            tickers_with_benchmark, period="1y",
            auto_adjust=True, progress=False, threads=True
        )
        if data.empty:
            raise RuntimeError("❌ yfinance vacío")

        # MULTIINDEX BULLETPROOF
        prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
        prices = prices.dropna(axis=1, thresh=len(prices) * 0.8)

        if len(prices) < LOOKBACK_DAYS * 0.5:
            raise RuntimeError(f"❌ Datos insuficientes: {len(prices)}d")

        returns = prices.pct_change().dropna()

        # PESOS + VALOR
        total_value = sum(p["qty"] * p["price_now"] for p in positions)
        weights = {
            p["ticker"].upper(): (p["qty"] * p["price_now"]) / total_value
            for p in positions
            if p["ticker"].upper() in prices.columns
        }

        valid_tickers = list(weights.keys())
        if not valid_tickers:
            raise RuntimeError("❌ Sin tickers válidos en yfinance")

        w = np.array([weights[t] for t in valid_tickers])
        asset_returns = returns[valid_tickers]

        # RIESGO REAL
        cov_matrix = asset_returns.cov() * 252
        portfolio_vol = np.sqrt(w.T @ cov_matrix.values @ w)
        portfolio_returns = asset_returns @ w

        var_daily = portfolio_returns.quantile(1 - CONFIDENCE_LEVEL)
        var_annual = abs(var_daily) * np.sqrt(252)

        tail = portfolio_returns[portfolio_returns <= var_daily]
        es_daily = tail.mean() if len(tail) > 0 else var_daily * 1.2
        es_annual = abs(es_daily) * np.sqrt(252)

        # BETA CAPM
        beta = 1.0
        if MARKET_BENCHMARK in returns.columns:
            spy_ret = returns[MARKET_BENCHMARK]
            beta = np.cov(portfolio_returns, spy_ret)[0, 1] / np.var(spy_ret)

        cash_reserve = self.fixed_capital - total_value

        # QUALITY
        days = len(prices)
        data_quality = (
            "EXCELLENT" if days >= LOOKBACK_DAYS else
            "GOOD"      if days >= LOOKBACK_DAYS * 0.75 else
            "WARNING"   if days >= LOOKBACK_DAYS * 0.5 else
            "POOR"
        )

        now_utc = datetime.utcnow()
        now_cl = now_utc.astimezone(CL_TIMEZONE)

        logger.info(
            f"🏛 Vol:{portfolio_vol:.1%} VaR:{var_annual:.1%} ES:{es_annual:.1%} "
            f"Beta:{beta:.2f} Cash:${cash_reserve:,.0f} Quality:{data_quality}"
        )

        return CapitalState(
            volatility_annual=round(float(portfolio_vol), 4),
            var_95_annual=round(float(var_annual), 4),
            expected_shortfall_95_annual=round(float(es_annual), 4),
            beta_vs_spy=round(float(beta), 3),
            total_value=round(float(total_value), 2),
            cash_reserve=round(float(cash_reserve), 2),
            timestamp_utc=now_utc.isoformat(),
            timestamp_cl=now_cl.isoformat(),
            data_quality=data_quality,
        )

    # =========================================================
    # ADJUST SIZING — governor de capital
    # =========================================================
    def adjust_sizing(self, current_positions: List[Dict], candidates: List[Dict]) -> List[Dict]:
        """
        Ajusta shares según Cash Buffer (10%) y Riesgo Sistémico (ES).
        [F2] Si el portfolio está vacío usa capital completo con sizing conservador.
        """
        # [F2] FALLBACK: portfolio vacío → no hay riesgo que calcular
        if not current_positions:
            logger.info("📭 Portfolio vacío → sizing conservador sin cálculo de riesgo")
            safety_buffer = self.fixed_capital * 0.10
            effective_buying_power = self.fixed_capital - safety_buffer
            return self._size_candidates(candidates, effective_buying_power, es=0.0)

        try:
            # [F1] Normalizar antes de evaluar
            normalized = self._normalize_positions(current_positions)
            state = self.evaluate(normalized)
        except Exception as e:
            logger.error(f"❌ Error evaluando riesgo para sizing: {e}")
            return []  # Safe-stop: si falla el riesgo, no abrimos nada

        # BLOQUEO TOTAL DE RIESGO (ES > 45%)
        if state.expected_shortfall_95_annual > 0.45:
            logger.warning(
                f"🛑 RIESGO EXTREMO: ES {state.expected_shortfall_95_annual:.2%}. "
                f"Bloqueando aperturas."
            )
            return []

        safety_buffer = self.fixed_capital * 0.10
        effective_buying_power = max(0, state.cash_reserve - safety_buffer)

        logger.info(
            f"💰 Cash Reserve: ${state.cash_reserve:,.0f} | "
            f"Buffer: ${safety_buffer:,.0f} | "
            f"Power: ${effective_buying_power:,.0f}"
        )

        return self._size_candidates(
            candidates, effective_buying_power,
            es=state.expected_shortfall_95_annual
        )

    def _size_candidates(
        self,
        candidates: List[Dict],
        buying_power: float,
        es: float,
    ) -> List[Dict]:
        """Lógica de sizing atómico — extraída para reutilizar en fallback."""
        # Factor de reducción según riesgo actual
        if es > 0.35:
            risk_factor = 0.5
        elif es > 0.25:
            risk_factor = 0.8
        else:
            risk_factor = 1.0

        adjusted = []
        remaining_power = buying_power

        for cand in candidates:
            ticker = cand.get("ticker", "").upper()
            price = cand.get("entry_price") or cand.get("price_now")
            requested_shares = cand.get("shares", 0)

            if not price or requested_shares <= 0:
                logger.warning(
                    f"⚠️ {ticker} saltado: precio o shares inválidos "
                    f"(price={price}, shares={requested_shares})"
                )
                continue

            max_shares_by_cash = int(remaining_power // price)
            final_shares = int(min(requested_shares, max_shares_by_cash) * risk_factor)

            if final_shares > 0:
                cand = dict(cand)  # no mutar el original
                cand["shares"] = final_shares
                cand.setdefault("meta", {})["governor_adj"] = {
                    "original_shares": requested_shares,
                    "risk_factor": risk_factor,
                    "es_at_execution": round(es, 4),
                }
                adjusted.append(cand)
                remaining_power -= final_shares * price
                logger.info(f"✅ {ticker}: {final_shares}s (original: {requested_shares})")
            else:
                logger.warning(f"❌ {ticker}: Rechazado por falta de capital/buffer.")

        return adjusted


# =========================================================
# MAIN — test local
# =========================================================
if __name__ == "__main__":
    positions = [
        {"ticker": "AAPL", "qty": 100, "price_now": 220.0},
        {"ticker": "MSFT", "qty": 50,  "price_now": 410.0},
        {"ticker": "GOOGL","qty": 30,  "price_now": 145.0},
    ]

    gov = CapitalGovernor(fixed_capital=1_000_000)
    state = gov.evaluate(positions)
    print("🏛 CAPITAL GOVERNOR v2.5 HEDGE FUND")
    print(json.dumps(state.to_dict(), indent=2))

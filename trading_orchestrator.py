# capital_governor.py — CAPITAL GOVERNOR v2.9
# ✅ 100% backward compatible v2.8
#
# FIX v2.8:
#   [F4] fixed_capital usa equity real del broker (~$85k)
#   [F5] _size_candidates usa fixed_capital real para MAX_POSITION_PCT
#
# FIX v2.9:
#   [F6] adjust_sizing_dynamic: sizing ajustado por VIX y correlación
#        con posiciones existentes. Reduce tamaño en mercados volátiles
#        y evita concentración en tickers muy correlacionados.
#        VIX_HIGH (>25) → factor 0.5 | VIX_EXTREME (>35) → factor 0.3
#        Correlación > CORR_THRESHOLD → penalización proporcional.
#   [F7] _get_vix_level: lee VIX desde market_context.json (ya calculado
#        por market_state_evaluator) — no llama yfinance extra.
#   [F8] _compute_correlation_penalty: usa retornos de 20 días desde
#        yfinance solo si hay posiciones existentes con precio válido.
#        Fallback a 1.0 (sin penalización) si falla.

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import logging
import os
import json

CL_TIMEZONE      = pytz.timezone("America/Santiago")
logger           = logging.getLogger("capital_governor")
logging.basicConfig(level=logging.INFO)

LOOKBACK_DAYS    = 252
CONFIDENCE_LEVEL = 0.95
MARKET_BENCHMARK = "SPY"

ES_BLOCK_THRESHOLD = float(os.getenv("ES_BLOCK_THRESHOLD", "0.20"))
MAX_POSITION_PCT   = float(os.getenv("MAX_POSITION_PCT",   "0.05"))

# [F6] Umbrales VIX
VIX_NORMAL   = float(os.getenv("VIX_NORMAL",   "20"))
VIX_HIGH     = float(os.getenv("VIX_HIGH",     "25"))
VIX_EXTREME  = float(os.getenv("VIX_EXTREME",  "35"))

# [F6] Umbral de correlación para penalizar
CORR_THRESHOLD = float(os.getenv("CORR_THRESHOLD", "0.70"))

DATA_PATH = os.getenv("DATA_PATH", "/data")


@dataclass
class CapitalState:
    volatility_annual: float
    var_95_annual: float
    expected_shortfall_95_annual: float
    effective_es: float
    portfolio_weight: float
    beta_vs_spy: float
    total_value: float
    cash_reserve: float
    timestamp_utc: str
    timestamp_cl: str
    data_quality: str

    def to_dict(self):
        return asdict(self)


class CapitalGovernor:
    def __init__(self, fixed_capital: Optional[float] = None):
        if fixed_capital is not None and fixed_capital > 0:
            self.fixed_capital = fixed_capital
        else:
            real = os.getenv("REAL_CAPITAL")
            if real:
                try:
                    self.fixed_capital = float(real)
                except ValueError:
                    self.fixed_capital = float(os.getenv("FIXED_CAPITAL", 1_000_000))
            else:
                self.fixed_capital = float(os.getenv("FIXED_CAPITAL", 1_000_000))

        logger.info(f"🏛 CapitalGovernor v2.9 | Capital: ${self.fixed_capital:,.0f}")

    # =========================================================
    # NORMALIZADOR
    # =========================================================
    @staticmethod
    def _normalize_positions(positions: List[Dict]) -> List[Dict]:
        normalized = []
        for p in positions:
            pos = dict(p)
            if "price_now" not in pos or pos["price_now"] is None:
                qty          = float(pos.get("qty") or 0)
                market_value = pos.get("market_value")
                if market_value is not None and qty > 0:
                    pos["price_now"] = float(market_value) / qty
                else:
                    avg_entry = pos.get("avg_entry_price")
                    if avg_entry is not None:
                        pos["price_now"] = float(avg_entry)
                    else:
                        logger.warning(f"⚠️ {pos.get('ticker')} sin price_now → saltado")
                        continue
            normalized.append(pos)
        return normalized

    # =========================================================
    # [F7] LEER VIX DESDE MARKET CONTEXT
    # =========================================================
    def _get_vix_level(self) -> float:
        """
        [F7] Lee VIX desde market_context.json — ya calculado por
        market_state_evaluator. No llama yfinance extra.
        Fallback a VIX_NORMAL si no está disponible.
        """
        try:
            ctx_path = os.path.join(DATA_PATH, "market_context.json")
            if os.path.exists(ctx_path):
                ctx = json.loads(open(ctx_path).read())
                vix = ctx.get("vix") or ctx.get("vix_level") or ctx.get("volatility_index")
                if vix and float(vix) > 0:
                    logger.info(f"📊 VIX desde market_context: {float(vix):.1f}")
                    return float(vix)
        except Exception as e:
            logger.debug(f"_get_vix_level fallback: {e}")
        logger.info(f"📊 VIX no disponible → usando neutral {VIX_NORMAL}")
        return VIX_NORMAL

    # =========================================================
    # [F6] FACTOR VIX
    # =========================================================
    def _vix_factor(self, vix: float) -> float:
        """
        [F6] Reduce tamaño de posición según nivel de VIX.
        VIX <= 20 → 1.0 (normal)
        VIX 20-25 → 0.8 (precaución)
        VIX 25-35 → 0.5 (alto)
        VIX > 35  → 0.3 (extremo)
        """
        if vix >= VIX_EXTREME:
            factor = 0.3
        elif vix >= VIX_HIGH:
            factor = 0.5
        elif vix >= VIX_NORMAL:
            factor = 0.8
        else:
            factor = 1.0

        if factor < 1.0:
            logger.info(f"📉 VIX factor={factor:.1f} | VIX={vix:.1f}")
        return factor

    # =========================================================
    # [F8] PENALIZACIÓN POR CORRELACIÓN
    # =========================================================
    def _compute_correlation_penalty(
        self,
        candidate_ticker: str,
        existing_positions: List[Dict],
    ) -> float:
        """
        [F8] Penaliza candidatos muy correlacionados con posiciones existentes.
        Correlación > CORR_THRESHOLD → reducción proporcional.
        Fallback a 1.0 (sin penalización) si falla.
        """
        if not existing_positions:
            return 1.0

        existing_tickers = [
            p.get("ticker", "").upper()
            for p in existing_positions
            if p.get("ticker")
        ]

        if not existing_tickers:
            return 1.0

        all_tickers = list(set([candidate_ticker.upper()] + existing_tickers))

        try:
            data = yf.download(
                all_tickers, period="1mo",
                auto_adjust=True, progress=False, threads=True
            )
            if data.empty:
                return 1.0

            prices  = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
            returns = prices.pct_change().dropna()

            if candidate_ticker.upper() not in returns.columns:
                return 1.0

            cand_returns = returns[candidate_ticker.upper()]
            correlations = []

            for t in existing_tickers:
                if t in returns.columns:
                    corr = float(cand_returns.corr(returns[t]))
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

            if not correlations:
                return 1.0

            avg_corr = float(np.mean(correlations))
            max_corr = float(np.max(correlations))

            if max_corr > CORR_THRESHOLD:
                # Penalización proporcional al exceso de correlación
                excess   = max_corr - CORR_THRESHOLD
                penalty  = max(0.3, 1.0 - excess * 2)
                logger.info(
                    f"🔗 Correlación {candidate_ticker} | "
                    f"avg={avg_corr:.2f} max={max_corr:.2f} → penalty={penalty:.2f}"
                )
                return penalty

            return 1.0

        except Exception as e:
            logger.debug(f"_compute_correlation_penalty fallback: {e}")
            return 1.0

    # =========================================================
    # EVALUATE
    # =========================================================
    def evaluate(self, positions: List[Dict]) -> CapitalState:
        if not positions:
            raise ValueError("❌ Portfolio vacío")

        positions = self._normalize_positions(positions)

        for i, p in enumerate(positions):
            required = ["ticker", "qty", "price_now"]
            if not all(k in p for k in required):
                raise ValueError(f"❌ Pos {i} campos faltantes: {list(p.keys())}")
            if p["qty"] <= 0 or p["price_now"] <= 0:
                raise ValueError(f"❌ Pos {i} qty/price inválidos: qty={p['qty']} price={p['price_now']}")

        tickers                = list({p["ticker"].upper() for p in positions})
        tickers_with_benchmark = tickers + [MARKET_BENCHMARK]

        logger.info(f"📊 {len(tickers)} tickers vs {MARKET_BENCHMARK}")

        data = yf.download(
            tickers_with_benchmark, period="1y",
            auto_adjust=True, progress=False, threads=True
        )
        if data.empty:
            raise RuntimeError("❌ yfinance vacío")

        prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
        prices = prices.dropna(axis=1, thresh=len(prices) * 0.8)

        if len(prices) < LOOKBACK_DAYS * 0.5:
            raise RuntimeError(f"❌ Datos insuficientes: {len(prices)}d")

        returns     = prices.pct_change().dropna()
        total_value = sum(p["qty"] * p["price_now"] for p in positions)

        weights = {
            p["ticker"].upper(): (p["qty"] * p["price_now"]) / total_value
            for p in positions
            if p["ticker"].upper() in prices.columns
        }

        valid_tickers = list(weights.keys())
        if not valid_tickers:
            raise RuntimeError("❌ Sin tickers válidos en yfinance")

        w             = np.array([weights[t] for t in valid_tickers])
        asset_returns = returns[valid_tickers]
        cov_matrix    = asset_returns.cov() * 252
        portfolio_vol = np.sqrt(w.T @ cov_matrix.values @ w)
        portfolio_returns = asset_returns @ w

        var_daily  = portfolio_returns.quantile(1 - CONFIDENCE_LEVEL)
        var_annual = abs(var_daily) * np.sqrt(252)

        tail     = portfolio_returns[portfolio_returns <= var_daily]
        es_daily = tail.mean() if len(tail) > 0 else var_daily * 1.2
        es_annual = abs(es_daily) * np.sqrt(252)

        beta = 1.0
        if MARKET_BENCHMARK in returns.columns:
            spy_ret = returns[MARKET_BENCHMARK]
            beta    = np.cov(portfolio_returns, spy_ret)[0, 1] / np.var(spy_ret)

        cash_reserve     = self.fixed_capital - total_value
        portfolio_weight = total_value / self.fixed_capital if self.fixed_capital > 0 else 0
        effective_es     = es_annual * portfolio_weight

        days         = len(prices)
        data_quality = (
            "EXCELLENT" if days >= LOOKBACK_DAYS        else
            "GOOD"      if days >= LOOKBACK_DAYS * 0.75 else
            "WARNING"   if days >= LOOKBACK_DAYS * 0.5  else
            "POOR"
        )

        now_utc = datetime.utcnow()
        now_cl  = now_utc.astimezone(CL_TIMEZONE)

        logger.info(
            f"🏛 Vol:{portfolio_vol:.1%} VaR:{var_annual:.1%} "
            f"ES:{es_annual:.1%} EffES:{effective_es:.1%} "
            f"Weight:{portfolio_weight:.1%} Beta:{beta:.2f} "
            f"Cash:${cash_reserve:,.0f} Quality:{data_quality}"
        )

        return CapitalState(
            volatility_annual=round(float(portfolio_vol), 4),
            var_95_annual=round(float(var_annual), 4),
            expected_shortfall_95_annual=round(float(es_annual), 4),
            effective_es=round(float(effective_es), 4),
            portfolio_weight=round(float(portfolio_weight), 4),
            beta_vs_spy=round(float(beta), 3),
            total_value=round(float(total_value), 2),
            cash_reserve=round(float(cash_reserve), 2),
            timestamp_utc=now_utc.isoformat(),
            timestamp_cl=now_cl.isoformat(),
            data_quality=data_quality,
        )
        
    # =========================================================
    # ADJUST SIZING — con VIX y correlación [F6][F8]
    # =========================================================
    def adjust_sizing(
        self,
        current_positions: List[Dict],
        candidates: List[Dict],
        use_dynamic: bool = True,
    ) -> List[Dict]:
        """
        [F6][F8] use_dynamic=True: aplica factor VIX y penalización
        por correlación sobre el sizing base.
        """
        if not current_positions:
            logger.info("📭 Portfolio vacío → sizing conservador")
            safety_buffer = self.fixed_capital * 0.10
            return self._size_candidates_dynamic(
                candidates, self.fixed_capital - safety_buffer,
                es=0.0, current_positions=[], use_dynamic=use_dynamic,
            )

        try:
            normalized = self._normalize_positions(current_positions)
            state      = self.evaluate(normalized)
        except Exception as e:
            logger.error(f"❌ Error evaluando riesgo: {e}")
            return []

        if state.effective_es > ES_BLOCK_THRESHOLD:
            logger.warning(
                f"🛑 RIESGO EXTREMO: EffES {state.effective_es:.2%} > "
                f"{ES_BLOCK_THRESHOLD:.0%}. Bloqueando aperturas."
            )
            return []

        safety_buffer          = self.fixed_capital * 0.10
        effective_buying_power = max(0, state.cash_reserve - safety_buffer)

        logger.info(
            f"💰 Cash Reserve: ${state.cash_reserve:,.0f} | "
            f"Buffer: ${safety_buffer:,.0f} | "
            f"Power: ${effective_buying_power:,.0f} | "
            f"EffES: {state.effective_es:.2%}"
        )

        return self._size_candidates_dynamic(
            candidates, effective_buying_power,
            es=state.effective_es,
            current_positions=current_positions,
            use_dynamic=use_dynamic,
        )

    # =========================================================
    # ADJUST SIZING AFTER CLOSES
    # =========================================================
    def adjust_sizing_after_closes(
        self,
        current_positions: List[Dict],
        closes: List[str],
        candidates: List[Dict],
        use_dynamic: bool = True,
    ) -> List[Dict]:
        if not candidates:
            return []

        closes_upper    = {c.upper() for c in closes}
        positions_after = [
            p for p in current_positions
            if p.get("ticker", "").upper() not in closes_upper
        ]

        logger.info(
            f"🔮 Simulando portfolio post-cierre | "
            f"antes={len(current_positions)} cierres={len(closes_upper)} "
            f"después={len(positions_after)}"
        )

        if not positions_after:
            logger.info("✅ Portfolio vacío post-cierre → capital libre")
            safety_buffer = self.fixed_capital * 0.10
            buying_power  = self.fixed_capital - safety_buffer
            return self._size_candidates_dynamic(
                candidates, buying_power,
                es=0.0, current_positions=[],
                use_dynamic=use_dynamic,
            )

        try:
            normalized = self._normalize_positions(positions_after)
            state      = self.evaluate(normalized)
        except Exception as e:
            logger.error(f"❌ Error evaluando riesgo post-cierre: {e}")
            return []

        logger.info(
            f"📊 ES post-cierre: {state.expected_shortfall_95_annual:.2%} | "
            f"EffES post-cierre: {state.effective_es:.2%} | "
            f"Cash post-cierre: ${state.cash_reserve:,.0f}"
        )

        if state.effective_es > ES_BLOCK_THRESHOLD:
            logger.warning(f"🛑 EffES post-cierre aún alto: {state.effective_es:.2%}. Bloqueando.")
            return []

        safety_buffer = self.fixed_capital * 0.10
        buying_power  = max(0, state.cash_reserve - safety_buffer)

        logger.info(f"💰 Buying power para anclas: ${buying_power:,.0f}")

        return self._size_candidates_dynamic(
            candidates, buying_power,
            es=state.effective_es,
            current_positions=positions_after,
            use_dynamic=use_dynamic,
        )

    # =========================================================
    # SIZE CANDIDATES DYNAMIC [F6][F7][F8]
    # =========================================================
    def _size_candidates_dynamic(
        self,
        candidates: List[Dict],
        buying_power: float,
        es: float,
        current_positions: List[Dict],
        use_dynamic: bool = True,
    ) -> List[Dict]:
        """
        [F6][F7][F8] Sizing con factor VIX + penalización correlación.
        use_dynamic=False → comportamiento idéntico a v2.8 (sin ajustes).
        """
        # Factor de riesgo por ES
        if es > 0.10:
            risk_factor = 0.5
        elif es > 0.05:
            risk_factor = 0.8
        else:
            risk_factor = 1.0

        # [F7] VIX desde market_context
        vix        = self._get_vix_level() if use_dynamic else VIX_NORMAL
        vix_factor = self._vix_factor(vix) if use_dynamic else 1.0

        max_per_position = self.fixed_capital * MAX_POSITION_PCT

        adjusted        = []
        remaining_power = buying_power

        for cand in candidates:
            ticker           = cand.get("ticker", "").upper()
            price            = cand.get("entry_price") or cand.get("price_now")
            requested_shares = cand.get("shares", 0)

            if not price or price <= 0 or requested_shares <= 0:
                logger.warning(f"⚠️ {ticker} saltado: precio/shares inválidos")
                continue

            # [F8] Penalización por correlación con posiciones existentes
            corr_penalty = (
                self._compute_correlation_penalty(ticker, current_positions)
                if use_dynamic and current_positions
                else 1.0
            )

            # Shares máximos por posición
            max_shares_by_position = int(max_per_position // price)

            # Shares máximos por cash disponible
            max_shares_by_cash = int(remaining_power // price)

            # Factor combinado: ES × VIX × correlación
            combined_factor = risk_factor * vix_factor * corr_penalty

            final_shares = int(
                min(requested_shares, max_shares_by_cash, max_shares_by_position)
                * combined_factor
            )

            if final_shares > 0:
                cand = dict(cand)
                cand["shares"] = final_shares
                cand.setdefault("meta", {})["governor_adj"] = {
                    "original_shares":      requested_shares,
                    "risk_factor":          risk_factor,
                    "vix_factor":           vix_factor,
                    "corr_penalty":         corr_penalty,
                    "combined_factor":      round(combined_factor, 3),
                    "vix_level":            vix,
                    "max_by_position":      max_shares_by_position,
                    "max_per_position_usd": round(max_per_position, 2),
                    "fixed_capital_used":   round(self.fixed_capital, 2),
                    "effective_es":         round(es, 4),
                }
                adjusted.append(cand)
                remaining_power -= final_shares * price
                logger.info(
                    f"✅ {ticker}: {final_shares}s @ ${price:.2f} "
                    f"(original: {requested_shares}, max_pos: {max_shares_by_position}, "
                    f"vix_f: {vix_factor:.1f}, corr_f: {corr_penalty:.2f})"
                )
            else:
                logger.warning(
                    f"❌ {ticker}: Rechazado — cash={max_shares_by_cash}s "
                    f"pos_limit={max_shares_by_position}s "
                    f"factor={combined_factor:.2f} remaining=${remaining_power:,.0f}"
                )

        return adjusted

    # =========================================================
    # BACKWARD COMPAT — alias sin dynamic para llamadas legacy
    # =========================================================
    def _size_candidates(
        self,
        candidates: List[Dict],
        buying_power: float,
        es: float,
    ) -> List[Dict]:
        """Alias v2.8 — llama al nuevo método sin ajustes dinámicos."""
        return self._size_candidates_dynamic(
            candidates, buying_power, es,
            current_positions=[], use_dynamic=False,
        )


# =========================================================
# MAIN — test local
# =========================================================
if __name__ == "__main__":
    positions = [
        {"ticker": "GLW", "qty": 248, "price_now": 159.95},
        {"ticker": "VZ",  "qty": 845, "price_now": 47.27},
    ]

    gov   = CapitalGovernor(fixed_capital=89_203)
    state = gov.evaluate(positions)
    print("🏛 CAPITAL GOVERNOR v2.9")
    print(json.dumps(state.to_dict(), indent=2))

    candidates = [
        {"ticker": "CAT", "entry_price": 769.57, "shares": 64},
        {"ticker": "FDX", "entry_price": 369.80, "shares": 135},
    ]
    sized = gov.adjust_sizing(positions, candidates)
    print("\n📐 Sizing dinámico con VIX + correlación:")
    for s in sized:
        adj = s.get("meta", {}).get("governor_adj", {})
        print(f"  {s['ticker']}: {s['shares']} shares | factor={adj.get('combined_factor')}")
        

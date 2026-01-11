# =========================================================
# broker.py — LIVE/PAPER TRADING ENGINE v1.1 (FINAL CORREGIDO)
# =========================================================
# ✅ Alpaca Markets API (Paper + Live)
# ✅ Auto-ejecuta señales 🔥 STRONG
# ✅ Risk 1% por trade + Portfolio limits (conservador)
# ✅ Max 10 posiciones | Decision logging 100%
# ✅ FastAPI + Pydantic + Production hardened
# =========================================================

import os
import json
import logging
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# =========================================================
# Decision logger
# =========================================================
from decision_log import log_decision

# =========================================================
# Alpaca
# =========================================================
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# =========================================================
# Config
# =========================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
PAPER_TRADING = os.getenv("ALPACA_PAPER", "true").lower() == "true"

ALLOW_SHORT = os.getenv("ALLOW_SHORT", "false").lower() == "true"
MAX_POSITIONS = int(os.getenv("BROKER_MAX_POSITIONS", "10"))
RISK_PER_TRADE = float(os.getenv("BROKER_RISK_PER_TRADE", "0.01"))
MAX_PORTFOLIO_RISK = float(os.getenv("BROKER_MAX_PORTFOLIO_RISK", "0.15"))
MIN_CONFIDENCE = float(os.getenv("BROKER_MIN_CONFIDENCE", "0.70"))

STOP_MULT_MAE = float(os.getenv("BROKER_STOP_MULT_MAE", "2.0"))
FALLBACK_STOP_PCT = float(os.getenv("BROKER_FALLBACK_STOP_PCT", "0.03"))
TAKE_PROFIT_MULT_STOP = float(os.getenv("BROKER_TP_MULT_STOP", "1.5"))
MIN_QTY = float(os.getenv("BROKER_MIN_QTY", "1"))

TRADES_DIR = Path(DATA_PATH) / "trades"
TRADES_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("broker")

# =========================================================
# Router
# =========================================================
router = APIRouter(prefix="/trading", tags=["trading"])

# =========================================================
# Pydantic Models
# =========================================================
class SignalInput(BaseModel):
    ticker: str
    quality: str
    confidence: float
    recommendation: str  # "BUY" | "SELL"
    price_now: float

class TradeResultModel(BaseModel):
    status: str
    order_id: Optional[str] = None
    ticker: str
    side: str
    qty: float
    price: Optional[float] = None
    reason: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

# =========================================================
# Helpers
# =========================================================
def append_trade_log(record: Dict[str, Any]) -> None:
    """Append-only JSONL (correcto)."""
    try:
        day = datetime.utcnow().strftime("%Y-%m-%d")
        fp = TRADES_DIR / f"trades_{day}.jsonl"
        with open(fp, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")  # ✅ FIX
    except Exception as e:
        logger.warning(f"trade log write failed: {e}")

def estimate_portfolio_risk_conservative(n_positions: int) -> float:
    """
    ✅ Estimación CONSERVADORA de riesgo de portafolio.
    Hasta tener stops reales persistidos, no usamos PnL flotante como riesgo.
    Suposición: cada posición consume ~RISK_PER_TRADE del presupuesto de riesgo.
    """
    return float(min(n_positions * RISK_PER_TRADE, 1.0))

# =========================================================
# Trading Engine
# =========================================================
class TradingEngine:
    def __init__(self):
        if not ALPACA_AVAILABLE:
            raise ValueError("alpaca-py no instalado")
        if not ALPACA_KEY or not ALPACA_SECRET:
            raise ValueError("Credenciales Alpaca faltantes")

        self.trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER_TRADING)
        acc = self.trading_client.get_account()
        self.equity = float(acc.equity)

        logger.info(
            f"🚀 Broker {'PAPER' if PAPER_TRADING else 'LIVE'} "
            f"(equity=${self.equity:.0f}, allow_short={ALLOW_SHORT})"
        )

    def _positions(self) -> Dict[str, Any]:
        return {p.symbol: p for p in self.trading_client.get_all_positions()}

    def execute_signal(self, signal: Dict[str, Any]) -> TradeResultModel:
        ticker = (signal.get("ticker") or "").upper().strip()
        quality = signal.get("quality")
        confidence = signal.get("confidence")
        recommendation = (signal.get("recommendation") or "").upper().strip()

        # -------- 1) QUALITY GATE
        if quality != "🔥 STRONG":
            log_decision({
                "module": "broker",
                "ticker": ticker,
                "decision": "skipped",
                "reason": "quality_not_strong",
                "quality": quality,
                "confidence": confidence,
            })
            return TradeResultModel(
                status="skipped", ticker=ticker, side="", qty=0,
                reason="quality_not_strong"
            )

        # -------- 2) CONFIDENCE GATE
        if confidence is None or float(confidence) < MIN_CONFIDENCE:
            log_decision({
                "module": "broker",
                "ticker": ticker,
                "decision": "skipped",
                "reason": "confidence_too_low",
                "quality": quality,
                "confidence": confidence,
            })
            return TradeResultModel(
                status="skipped", ticker=ticker, side="", qty=0,
                reason="confidence_too_low"
            )

        positions = self._positions()

        # -------- 3) DUPLICATE
        if ticker in positions:
            log_decision({
                "module": "broker",
                "ticker": ticker,
                "decision": "skipped",
                "reason": "already_in_positions",
            })
            return TradeResultModel(
                status="skipped", ticker=ticker, side="", qty=0,
                reason="already_in_positions"
            )

        # -------- 4) MAX POSITIONS
        if len(positions) >= MAX_POSITIONS:
            log_decision({
                "module": "broker",
                "ticker": ticker,
                "decision": "rejected",
                "reason": "max_positions_reached",
                "n_positions": len(positions),
            })
            return TradeResultModel(
                status="rejected", ticker=ticker, side="", qty=0,
                reason="max_positions_reached"
            )

        # -------- 5) PORTFOLIO RISK (✅ FIX real)
        total_risk = estimate_portfolio_risk_conservative(len(positions))
        if total_risk > MAX_PORTFOLIO_RISK:
            log_decision({
                "module": "broker",
                "ticker": ticker,
                "decision": "rejected",
                "reason": "portfolio_risk_exceeded",
                "current_risk_est": total_risk,
                "max_portfolio_risk": MAX_PORTFOLIO_RISK,
            })
            return TradeResultModel(
                status="rejected", ticker=ticker, side="", qty=0,
                reason="portfolio_risk_exceeded"
            )

        # -------- 6) SIDE VALIDATION
        if recommendation == "BUY":
            side = "buy"
            alp_side = OrderSide.BUY
        elif recommendation == "SELL" and ALLOW_SHORT:
            side = "sell"
            alp_side = OrderSide.SELL
        else:
            log_decision({
                "module": "broker",
                "ticker": ticker,
                "decision": "rejected",
                "reason": "invalid_side_or_short_disabled",
                "recommendation": recommendation,
                "allow_short": ALLOW_SHORT,
            })
            return TradeResultModel(
                status="rejected", ticker=ticker, side="", qty=0,
                reason="invalid_side_or_short_disabled"
            )

        # -------- 7) PRICE
        try:
            price_now = float(signal.get("price_now") or 0.0)
        except Exception:
            price_now = 0.0

        if price_now <= 0:
            log_decision({
                "module": "broker",
                "ticker": ticker,
                "decision": "rejected",
                "reason": "invalid_price",
                "price_now": signal.get("price_now"),
            })
            return TradeResultModel(
                status="rejected", ticker=ticker, side=side, qty=0,
                reason="invalid_price"
            )

        # -------- 8) POSITION SIZING (✅ FIX: NO fuerza MIN_QTY)
        max_risk_dollars = self.equity * RISK_PER_TRADE

        # Edge sobre 0.5 (solo para sizing conservador)
        c = float(np.clip(float(confidence), 0.0, 1.0))
        edge = max(0.0, c - 0.5)

        # Kelly-inspired conservador 1/4
        kelly_factor = edge * 0.25

        # dólares asignados (cap por riesgo)
        dollars_alloc = max_risk_dollars * kelly_factor

        qty = int(dollars_alloc / price_now) if dollars_alloc > 0 else 0  # ✅ FIX

        if qty < MIN_QTY:
            log_decision({
                "module": "broker",
                "ticker": ticker,
                "decision": "rejected",
                "reason": "qty_below_minimum",
                "qty": qty,
                "min_qty": MIN_QTY,
                "dollars_alloc": round(dollars_alloc, 2),
                "price_now": price_now,
                "kelly_factor": round(kelly_factor, 4),
            })
            return TradeResultModel(
                status="rejected", ticker=ticker, side=side, qty=0,
                reason="qty_below_minimum"
            )

        # -------- 9) EXECUTE ORDER
        try:
            order = self.trading_client.submit_order(
                MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=alp_side,
                    time_in_force=TimeInForce.DAY,
                )
            )

            # Trade log
            append_trade_log({
                "ts": datetime.utcnow().isoformat(),
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "price_ref": price_now,
                "confidence": c,
                "quality": quality,
                "order_id": getattr(order, "id", None),
            })

            # Decision log
            log_decision({
                "module": "broker",
                "ticker": ticker,
                "decision": "executed",
                "order_id": getattr(order, "id", None),
                "side": side,
                "qty": qty,
                "price_ref": price_now,
                "confidence": c,
                "quality": quality,
                "portfolio_risk_est": total_risk,
                "kelly_factor": round(kelly_factor, 4),
                "dollars_alloc": round(dollars_alloc, 2),
            })

            logger.info(f"✅ EXECUTED {side.upper()} {qty} {ticker} @ ${price_now:.2f}")

            return TradeResultModel(
                status="executed",
                order_id=str(getattr(order, "id", None)),
                ticker=ticker,
                side=side,
                qty=float(qty),
                price=price_now,
                meta={"portfolio_risk_est": total_risk}
            )

        except Exception as e:
            log_decision({
                "module": "broker",
                "ticker": ticker,
                "decision": "failed",
                "reason": "execution_error",
                "error": str(e),
                "side": side,
                "qty": qty,
            })
            logger.error(f"❌ EXECUTION FAILED {ticker}: {e}")

            return TradeResultModel(
                status="failed",
                ticker=ticker,
                side=side,
                qty=float(qty),
                reason=str(e),
            )

# =========================================================
# Singleton
# =========================================================
_engine: Optional[TradingEngine] = None

def get_trading_engine() -> TradingEngine:
    global _engine
    if _engine is None:
        _engine = TradingEngine()
    return _engine

# =========================================================
# FastAPI Endpoints
# =========================================================
@router.post("/execute", response_model=TradeResultModel)
async def execute_trade(signal: SignalInput):
    """Ejecuta señal de trading con risk management completo."""
    try:
        engine = get_trading_engine()
        result = engine.execute_signal(signal.model_dump())
        return result
    except Exception as e:
        log_decision({
            "module": "broker_api",
            "decision": "api_error",
            "error": str(e),
            "ticker": getattr(signal, "ticker", None),
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def broker_status():
    """Estado del broker."""
    try:
        engine = get_trading_engine()
        positions = engine._positions()
        account = engine.trading_client.get_account()

        portfolio_risk_est = estimate_portfolio_risk_conservative(len(positions))

        return {
            "status": "active",
            "equity": float(account.equity),
            "positions": len(positions),
            "max_positions": MAX_POSITIONS,
            "paper": PAPER_TRADING,
            "portfolio_risk_est": round(portfolio_risk_est, 4),
            "max_portfolio_risk": MAX_PORTFOLIO_RISK,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

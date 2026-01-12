# =========================================================
# broker.py — EXECUTION ENGINE v2.1 (PM-DRIVEN + DYNAMIC SIZING)
# =========================================================
# ✔ Position sizing automático (% equity)
# ✔ Account health checks
# ✔ Order status polling
# ✔ Rate limiting
# ✔ Enhanced validation
# =========================================================

import os
import json
import logging
import sys
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from functools import wraps

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
import asyncio

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
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import LatestQuoteRequest
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

TRADES_DIR = Path(DATA_PATH) / "trades"
TRADES_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("broker")

# =========================================================
# Router
# =========================================================
router = APIRouter(prefix="/trading", tags=["trading"])

# =========================================================
# Pydantic Models (ENHANCED)
# =========================================================
class DecisionInput(BaseModel):
    action: str  # OPEN | CLOSE | ROTATE
    ticker: Optional[str] = None
    close_ticker: Optional[str] = None
    open_ticker: Optional[str] = None
    target_pct: Optional[float] = None  # % del equity (reemplaza shares)
    meta: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None

    @validator('action')
    def valid_action(cls, v):
        if v not in ['OPEN', 'CLOSE', 'ROTATE']:
            raise ValueError('action debe ser OPEN, CLOSE o ROTATE')
        return v

class TradeResultModel(BaseModel):
    status: str
    ticker: Optional[str] = None
    side: Optional[str] = None
    qty: Optional[float] = None
    order_id: Optional[str] = None
    reason: Optional[str] = None
    equity_used_pct: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None

# =========================================================
# Rate Limiting
# =========================================================
def rate_limit(calls_per_min: int = 30):
    last_calls = []
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            now = time.time()
            last_calls[:] = [t for t in last_calls if now - t < 60]
            if len(last_calls) >= calls_per_min:
                raise HTTPException(429, f"Rate limit: {calls_per_min}/min")
            last_calls.append(now)
            return await fn(*args, **kwargs)
        return wrapper
    return decorator

# =========================================================
# Helpers
# =========================================================
def append_trade_log(record: Dict[str, Any]) -> None:
    try:
        day = datetime.utcnow().strftime("%Y-%m-%d")
        fp = TRADES_DIR / f"trades_{day}.jsonl"
        with open(fp, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "
")
    except Exception as e:
        logger.warning(f"trade log write failed: {e}")

# =========================================================
# Trading Engine (v2.1)
# =========================================================
class TradingEngine:
    def __init__(self):
        if not ALPACA_AVAILABLE:
            raise ValueError("alpaca-py no instalado")
        if not ALPACA_KEY or not ALPACA_SECRET:
            raise ValueError("Credenciales Alpaca faltantes")

        self.client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER_TRADING)
        self.data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        
        # Health check
        acc = self.client.get_account()
        if acc.trading_blocked:
            raise ValueError("🚫 Trading blocked en cuenta")
        if float(acc.buying_power) < 10:
            logger.warning(f"⚠️ Buying power bajo: ${acc.buying_power}")
        
        self.equity = float(acc.equity)
        logger.info(
            f"🚀 Broker {'PAPER' if PAPER_TRADING else 'LIVE'} "
            f"(equity=${self.equity:.0f}, buying_power=${acc.buying_power})"
        )

    # -------------------------
    # Position sizing DINÁMICO
    # -------------------------
    def calculate_qty(self, ticker: str, target_pct: float) -> float:
        """Calcula shares basado en % del equity"""
        if target_pct <= 0 or target_pct > 0.5:  # Max 50%
            raise ValueError(f"target_pct inválido: {target_pct}")
        
        try:
            # Precio actual
            quote = self.data_client.get_stock_latest_quote([ticker]).latest_quote[ticker]
            price = (quote.ask + quote.bid) / 2
            
            # Shares = (equity * %) / precio
            acc = self.client.get_account()
            target_value = float(acc.equity) * target_pct
            qty = target_value / price
            
            # Redondeo inteligente (min 1 share)
            qty = max(1, int(qty * 100) / 100)  # 2 decimales
            
            logger.info(f"📊 {ticker}: {target_pct:.1%} equity → "
                       f"${target_value:.0f} → {qty} shares @ ${price:.2f}")
            
            return qty
        except Exception as e:
            logger.error(f"❌ Position sizing failed {ticker}: {e}")
            raise

    # -------------------------
    # Execution primitives (con polling)
    # -------------------------
    async def open_market(self, ticker: str, qty: float):
        order = self.client.submit_order(
            MarketOrderRequest(
                symbol=ticker,
                qty=int(qty),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        
        # Polling hasta filled o timeout
        for _ in range(10):  # 10s max
            await asyncio.sleep(1)
            order = self.client.get_order(order.id)
            if order.status == OrderStatus.FILLED:
                return order
            if order.status in [OrderStatus.REJECTED, OrderStatus.CANCELLED]:
                raise ValueError(f"Order {order.id} {order.status}")
        
        raise ValueError(f"Order {order.id} timeout: {order.status}")

    def close_market(self, ticker: str):
        return self.client.close_position(ticker)

    # -------------------------
    # Decision executor
    # -------------------------
    async def execute_decision(self, decision: Dict[str, Any]) -> TradeResultModel:
        action = decision.get("action")
        meta = decision.get("meta") or {}
        target_pct = decision.get("target_pct") or meta.get("target_pct")

        try:
            # =====================
            # OPEN
            # =====================
            if action == "OPEN":
                ticker = decision.get("ticker")
                if not ticker:
                    return TradeResultModel(status="rejected", reason="missing_ticker")

                qty = self.calculate_qty(ticker, target_pct or 0.1)
                order = await self.open_market(ticker, qty)

                self._log_execution("OPEN", ticker, qty, decision, order)
                return TradeResultModel(
                    status="executed",
                    ticker=ticker,
                    side="buy",
                    qty=qty,
                    order_id=str(order.id),
                    equity_used_pct=target_pct,
                )

            # =====================
            # CLOSE
            # =====================
            if action == "CLOSE":
                ticker = decision.get("ticker")
                if not ticker:
                    return TradeResultModel(status="rejected", reason="missing_ticker")

                self.close_market(ticker)
                self._log_execution("CLOSE", ticker, None, decision, None)
                return TradeResultModel(status="executed", ticker=ticker, side="sell")

            # =====================
            # ROTATE
            # =====================
            if action == "ROTATE":
                close_ticker = decision.get("close_ticker")
                open_ticker = decision.get("open_ticker")
                
                if not close_ticker or not open_ticker:
                    return TradeResultModel(status="rejected", reason="missing_tickers")

                self.close_market(close_ticker)
                qty = self.calculate_qty(open_ticker, target_pct or 0.1)
                order = await self.open_market(open_ticker, qty)

                self._log_execution("ROTATE", f"{close_ticker}→{open_ticker}", qty, decision, order)
                return TradeResultModel(
                    status="executed",
                    ticker=open_ticker,
                    side="buy",
                    qty=qty,
                    order_id=str(order.id),
                    equity_used_pct=target_pct,
                )

            return TradeResultModel(status="ignored", reason=f"unknown_action:{action}")

        except Exception as e:
            logger.error(f"❌ EXECUTION FAILED: {e}")
            log_decision({
                "module": "broker",
                "decision": "failed",
                "error": str(e),
                "payload": decision,
            })
            return TradeResultModel(status="failed", reason=str(e))

    # -------------------------
    # Logging
    # -------------------------
    def _log_execution(
        self,
        action: str,
        ticker: str,
        qty: Optional[float],
        decision: Dict[str, Any],
        order: Any,
    ):
        record = {
            "ts": datetime.utcnow().isoformat(),
            "action": action,
            "ticker": ticker,
            "qty": qty,
            "order_id": getattr(order, "id", None),
            "decision": decision,
        }

        append_trade_log(record)
        log_decision({"module": "broker", "decision": "executed", **record})
        logger.info(f"✅ {action} {ticker} qty={qty}")

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
@rate_limit(30)
async def execute_trade(decision: DecisionInput):
    """Ejecuta decisión con position sizing dinámico"""
    try:
        engine = get_trading_engine()
        return await engine.execute_decision(decision.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def broker_status():
    """Estado completo del broker"""
    try:
        engine = get_trading_engine()
        acc = engine.client.get_account()
        positions = engine.client.get_all_positions()

        return {
            "status": "active",
            "equity": float(acc.equity),
            "buying_power": float(acc.buying_power),
            "positions": len(positions),
            "trading_blocked": acc.trading_blocked,
            "paper": PAPER_TRADING,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

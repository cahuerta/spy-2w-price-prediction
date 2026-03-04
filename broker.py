# =========================================================
# broker.py — V3.0 PURE EXECUTOR (NO SIZING)
# =========================================================

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from fastapi import Header, APIRouter, HTTPException
from pydantic import BaseModel

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

logger = logging.getLogger("broker")
router = APIRouter(prefix="/trading", tags=["trading"])

# =========================================================
# MODELS
# =========================================================
class PureDecisionInput(BaseModel):
    action: str
    ticker: str
    shares: Optional[int] = 0  # <--- OBLIGATORIO PARA OPEN/ROTATE
    reason: Optional[str] = None
    meta: Optional[Dict] = None

# =========================================================
# ENGINE
# =========================================================
class TradingEngine:
    def __init__(self):
        self.key = os.getenv("ALPACA_API_KEY")
        self.secret = os.getenv("ALPACA_SECRET_KEY")
        self.paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        
        if not self.key or not self.secret:
            raise ValueError("Faltan credenciales de Alpaca")

        self.client = TradingClient(self.key, self.secret, paper=self.paper)
        logger.info(f"🔌 Broker Pure Executor conectado ({'PAPER' if self.paper else 'LIVE'})")

    def is_executable(self, ticker: str) -> bool:
        return ticker is not None and not ticker.upper().endswith(".SN")
        
    async def get_account(self):
        """
        Devuelve la cuenta Alpaca para el dashboard
        """
        return self.client.get_account()
    async def execute_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        action = decision.get("action")
        ticker = decision.get("ticker", "").upper()
        shares = int(decision.get("shares", 0))

        if not self.is_executable(ticker):
            return {"status": "skipped", "reason": "non_executable_market"}

        try:
            if action == "OPEN":
                if shares <= 0:
                    return {"status": "rejected", "reason": "zero_shares_governor"}
                
                return await self._place_market_order(ticker, shares, OrderSide.BUY)

            elif action == "CLOSE":
                # Close no necesita shares, cierra todo lo que hay
                res = self.client.close_position(ticker)
                logger.info(f"⚰️ CLOSED position for {ticker}")
                return {"status": "executed", "order_id": getattr(res, "id", "sync_close")}

            elif action == "ROTATE":
                # Rotate en el broker son dos pasos atómicos
                close_ticker = decision.get("close_ticker", "").upper()
                if close_ticker:
                    self.client.close_position(close_ticker)
                
                if shares > 0:
                    return await self._place_market_order(ticker, shares, OrderSide.BUY)
                return {"status": "partially_executed", "reason": "close_done_open_failed_shares"}

        except Exception as e:
            logger.error(f"❌ EXECUTION ERROR for {ticker}: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _place_market_order(self, ticker: str, qty: int, side: OrderSide):
        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        order = self.client.submit_order(req)
        
        # Espera simple de confirmación
        for _ in range(5):
            await asyncio.sleep(1)
            order = self.client.get_order(order.id)
            if order.status == OrderStatus.FILLED:
                logger.info(f"✅ {side} {qty} {ticker} FILLED at ${order.filled_avg_price}")
                return {"status": "executed", "order_id": str(order.id), "price": order.filled_avg_price}
        
        return {"status": "pending", "order_id": str(order.id)}

# =========================================================
# SINGLETON & ENDPOINT
# =========================================================
_engine = None

def get_engine():
    global _engine
    if _engine is None: _engine = TradingEngine()
    return _engine

@router.post("/execute")
async def execute(decision: PureDecisionInput, x_api_key: str = Header(None)):
    if x_api_key != os.getenv("BROKER_EXECUTION_KEY"):
        raise HTTPException(401)
    return await get_engine().execute_decision(decision.dict())

# =========================================================
# broker.py — V3.2 PURE EXECUTOR (NO SIZING)
# =========================================================
#
# FIX v3.1:
#   [B1] cancel_all_orders() — cancela todas las órdenes abiertas
#   [B2] cancel_orders_for_ticker() — cancela órdenes de un ticker
#   [B3] get_positions() incluye current_price
#
# FIX v3.2:
#   [B4] _place_market_order() ahora asigna client_order_id con
#        prefijo "sys_" para identificar órdenes del sistema
#        vs órdenes manuales en análisis posteriores.
#        Formato: sys_{TICKER}_{timestamp_ms}
# =========================================================

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from fastapi import Header, APIRouter, HTTPException
from pydantic import BaseModel

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

logger = logging.getLogger("broker")
router = APIRouter(prefix="/trading", tags=["trading"])

# Prefijo que identifica órdenes del sistema automático
SYSTEM_ORDER_PREFIX = "sys_"


# =========================================================
# MODELS
# =========================================================
class PureDecisionInput(BaseModel):
    action: str
    ticker: str
    shares: Optional[int] = 0
    reason: Optional[str] = None
    meta: Optional[Dict] = None


# =========================================================
# ENGINE
# =========================================================
class TradingEngine:
    def __init__(self):
        self.key    = os.getenv("ALPACA_API_KEY")
        self.secret = os.getenv("ALPACA_SECRET_KEY")
        self.paper  = os.getenv("ALPACA_PAPER", "true").lower() == "true"

        if not self.key or not self.secret:
            raise ValueError("Faltan credenciales de Alpaca")

        self.client = TradingClient(self.key, self.secret, paper=self.paper)
        logger.info(f"🔌 Broker V3.2 conectado ({'PAPER' if self.paper else 'LIVE'})")

    def is_executable(self, ticker: str) -> bool:
        return ticker is not None and not ticker.upper().endswith(".SN")

    async def get_account(self):
        return self.client.get_account()

    def get_positions(self) -> Dict:
        try:
            alpaca_positions = self.client.get_all_positions()
        except Exception as e:
            logger.error(f"❌ Error obteniendo posiciones: {e}")
            return {}

        result = {}
        for p in alpaca_positions:
            result[p.symbol] = {
                "qty":             float(p.qty),
                "market_value":    float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_pl":   float(p.unrealized_pl),
                # [B3] current_price para enriquecer posiciones en orchestrator
                "current_price":   float(p.current_price) if hasattr(p, "current_price") else None,
                "price_now":       float(p.current_price) if hasattr(p, "current_price") else float(p.avg_entry_price),
            }

        return result

    # =========================================================
    # [B1] CANCEL ALL ORDERS
    # =========================================================
    async def cancel_all_orders(self):
        """Cancela todas las órdenes abiertas (DAY, GTC, etc)."""
        try:
            cancel_statuses = self.client.cancel_orders()
            cancelled = len(cancel_statuses) if cancel_statuses else 0
            logger.info(f"🗑 cancel_all_orders: {cancelled} órdenes canceladas")
            return cancelled
        except Exception as e:
            logger.warning(f"⚠️ cancel_all_orders falló: {e}")
            return 0

    # =========================================================
    # [B2] CANCEL ORDERS FOR TICKER
    # =========================================================
    async def cancel_orders_for_ticker(self, ticker: str):
        """Cancela órdenes abiertas para un ticker específico."""
        ticker = ticker.upper()
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[ticker],
            )
            open_orders = self.client.get_orders(filter=request)

            if not open_orders:
                logger.info(f"✅ {ticker}: sin órdenes abiertas")
                return 0

            cancelled = 0
            for order in open_orders:
                try:
                    self.client.cancel_order_by_id(order.id)
                    cancelled += 1
                    logger.info(f"🗑 {ticker}: orden {order.id} cancelada")
                except Exception as e:
                    logger.warning(f"⚠️ {ticker}: no se pudo cancelar orden {order.id}: {e}")

            return cancelled

        except Exception as e:
            logger.warning(f"⚠️ cancel_orders_for_ticker({ticker}) falló: {e}")
            return 0

    # =========================================================
    # EXECUTE DECISION
    # =========================================================
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
                res = self.client.close_position(ticker)
                logger.info(f"⚰️ CLOSED position for {ticker}")
                return {"status": "executed", "order_id": getattr(res, "id", "sync_close")}

            elif action == "ROTATE":
                close_ticker = decision.get("close_ticker", "").upper()
                if close_ticker:
                    self.client.close_position(close_ticker)

                if shares > 0:
                    return await self._place_market_order(ticker, shares, OrderSide.BUY)
                return {"status": "partially_executed", "reason": "close_done_open_failed_shares"}

        except Exception as e:
            logger.error(f"❌ EXECUTION ERROR for {ticker}: {str(e)}")
            return {"status": "error", "message": str(e)}

    # =========================================================
    # [B4] PLACE MARKET ORDER — con client_order_id del sistema
    # =========================================================
    async def _place_market_order(self, ticker: str, qty: int, side: OrderSide):
        # Genera ID único con prefijo "sys_" para identificar órdenes automáticas
        ts_ms     = int(datetime.utcnow().timestamp() * 1000)
        client_id = f"{SYSTEM_ORDER_PREFIX}{ticker.lower()}_{ts_ms}"

        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_id,   # [B4] marca de origen sistema
        )

        order = self.client.submit_order(req)
        logger.info(f"📤 Orden enviada {ticker} client_id={client_id}")

        for _ in range(5):
            await asyncio.sleep(1)
            order = self.client.get_order_by_id(order.id)
            if order.status == OrderStatus.FILLED:
                logger.info(f"✅ {side} {qty} {ticker} FILLED at ${order.filled_avg_price}")
                return {
                    "status":           "executed",
                    "order_id":         str(order.id),
                    "client_order_id":  client_id,
                    "price":            float(order.filled_avg_price) if order.filled_avg_price else None,
                }

        return {
            "status":          "pending",
            "order_id":        str(order.id),
            "client_order_id": client_id,
        }


# =========================================================
# SINGLETON & ENDPOINTS
# =========================================================
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = TradingEngine()
    return _engine


@router.post("/execute")
async def execute(decision: PureDecisionInput, x_api_key: str = Header(None)):
    if x_api_key != os.getenv("BROKER_EXECUTION_KEY"):
        raise HTTPException(401)
    return await get_engine().execute_decision(decision.dict())


@router.get("/status")
async def trading_status():
    engine  = get_engine()
    account = engine.client.get_account()
    positions = engine.client.get_all_positions()
    return {
        "status":          account.status,
        "equity":          float(account.equity),
        "buying_power":    float(account.buying_power),
        "paper":           engine.paper,
        "positions":       len(positions),
        "trading_blocked": account.trading_blocked,
    }


@router.get("/positions")
async def trading_positions():
    engine = get_engine()
    try:
        alpaca_positions = engine.client.get_all_positions()
    except Exception as e:
        logger.error(f"❌ Error obteniendo posiciones: {e}")
        return {}

    result = {}
    for p in alpaca_positions:
        result[p.symbol] = {
            "qty":             float(p.qty),
            "market_value":    float(p.market_value),
            "avg_entry_price": float(p.avg_entry_price),
            "unrealized_pl":   float(p.unrealized_pl),
            "current_price":   float(p.current_price) if hasattr(p, "current_price") else None,
        }

    return result


# =========================================================
# CANCEL ENDPOINTS
# =========================================================

@router.delete("/orders")
async def cancel_all(x_api_key: str = Header(None)):
    """Cancela todas las órdenes abiertas."""
    if x_api_key != os.getenv("BROKER_EXECUTION_KEY"):
        raise HTTPException(401)
    engine    = get_engine()
    cancelled = await engine.cancel_all_orders()
    return {"cancelled": cancelled}


@router.delete("/orders/{ticker}")
async def cancel_ticker_orders(ticker: str, x_api_key: str = Header(None)):
    """Cancela órdenes abiertas para un ticker específico."""
    if x_api_key != os.getenv("BROKER_EXECUTION_KEY"):
        raise HTTPException(401)
    engine    = get_engine()
    cancelled = await engine.cancel_orders_for_ticker(ticker)
    return {"ticker": ticker.upper(), "cancelled": cancelled}

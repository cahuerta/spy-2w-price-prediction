# =========================================================
# broker.py — EXECUTION ENGINE v2.1 (PM-DRIVEN + DYNAMIC SIZING)
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
from fastapi import Header, APIRouter, HTTPException
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
BROKER_EXECUTION_KEY = os.getenv("BROKER_EXECUTION_KEY")

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
# Helpers
# =========================================================
def is_executable_usa(ticker: str) -> bool:
    return ticker is not None and not ticker.upper().endswith(".SN")


def append_trade_log(record: Dict[str, Any]) -> None:
    try:
        day = datetime.utcnow().strftime("%Y-%m-%d")
        fp = TRADES_DIR / f"trades_{day}.jsonl"
        with open(fp, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"trade log write failed: {e}")

# =========================================================
# Models
# =========================================================
class DecisionInput(BaseModel):
    action: str
    ticker: Optional[str] = None
    close_ticker: Optional[str] = None
    open_ticker: Optional[str] = None
    target_pct: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None

    @validator("action")
    def valid_action(cls, v):
        if v not in ["OPEN", "CLOSE", "ROTATE"]:
            raise ValueError("action debe ser OPEN, CLOSE o ROTATE")
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
# Rate Limit
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
# Trading Engine
# =========================================================
class TradingEngine:

    def __init__(self):

        if not ALPACA_AVAILABLE:
            raise ValueError("alpaca-py no instalado")

        if not ALPACA_KEY or not ALPACA_SECRET:
            raise ValueError("Credenciales Alpaca faltantes")

        self.client = TradingClient(
            ALPACA_KEY,
            ALPACA_SECRET,
            paper=PAPER_TRADING
        )

        self.data_client = StockHistoricalDataClient(
            ALPACA_KEY,
            ALPACA_SECRET
        )

        acc = self.client.get_account()

        if acc.trading_blocked:
            raise ValueError("🚫 Trading blocked en cuenta")

        self.equity = float(acc.equity)

        logger.info(
            f"🚀 Broker {'PAPER' if PAPER_TRADING else 'LIVE'} "
            f"(equity=${self.equity:.0f})"
        )

    async def get_account(self):
        return self.client.get_account()
        def sync_positions_from_broker(self):
    """
    Descarga posiciones reales desde Alpaca
    y sobrescribe /data/positions.json
    """

    from portfolio_store import save_positions

    alpaca_positions = self.client.get_all_positions()

    new_positions = []

    for p in alpaca_positions:
        new_positions.append({
            "id": f"{p.symbol}-{datetime.utcnow().isoformat()}",
            "ticker": p.symbol,
            "qty": float(p.qty),
            "entry_price": float(p.avg_entry_price),
            "price_now": float(p.current_price),
            "peak_price": float(p.current_price),
            "days_at_peak": 1,
            "entry_time": datetime.utcnow().isoformat(),
            "is_anchor": False,
            "market_mode_entry": None,
            "meta": {
                "source": "alpaca_sync"
            }
        })

    save_positions(new_positions)

    logger.info(f"🔄 Synced {len(new_positions)} positions from Alpaca")

    def calculate_qty(self, ticker: str, target_pct: float) -> float:

        if not is_executable_usa(ticker):
            raise RuntimeError(f"EXECUTION BLOCKED (CHILE): {ticker}")

        if target_pct <= 0 or target_pct > 0.5:
            raise ValueError(f"target_pct inválido: {target_pct}")

        quote = self.data_client.get_stock_latest_quote([ticker]).latest_quote[ticker]
        price = (quote.ask + quote.bid) / 2

        acc = self.client.get_account()
        target_value = float(acc.equity) * target_pct
        qty = max(1, int((target_value / price) * 100) / 100)

        return qty

    async def open_market(self, ticker: str, qty: float):

        if not is_executable_usa(ticker):
            raise RuntimeError(f"EXECUTION BLOCKED (CHILE): {ticker}")

        order = self.client.submit_order(
            MarketOrderRequest(
                symbol=ticker,
                qty=int(qty),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )

        for _ in range(10):
            await asyncio.sleep(1)
            order = self.client.get_order(order.id)

            if order.status == OrderStatus.FILLED:
                return order

            if order.status in [OrderStatus.REJECTED, OrderStatus.CANCELLED]:
                raise RuntimeError(f"Order {order.id} {order.status}")

        raise RuntimeError(f"Order timeout: {order.id}")

    def close_market(self, ticker: str):

        if not is_executable_usa(ticker):
            raise RuntimeError(f"EXECUTION BLOCKED (CHILE): {ticker}")

        return self.client.close_position(ticker)

    async def execute_decision(self, decision: Dict[str, Any]) -> TradeResultModel:

        action = decision.get("action")
        target_pct = decision.get("target_pct") or (decision.get("meta") or {}).get("target_pct")

        try:

            if action == "OPEN":
                ticker = decision.get("ticker")
                if not ticker:
                    return TradeResultModel(status="rejected", reason="missing_ticker")

                if not is_executable_usa(ticker):
                    return TradeResultModel(status="skipped", ticker=ticker, reason="CHILE_NO_EXEC")

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

            if action == "CLOSE":
                ticker = decision.get("ticker")
                if not ticker:
                    return TradeResultModel(status="rejected", reason="missing_ticker")

                if not is_executable_usa(ticker):
                    return TradeResultModel(status="skipped", ticker=ticker, reason="CHILE_NO_EXEC")

                self.close_market(ticker)
                self._log_execution("CLOSE", ticker, None, decision, None)

                return TradeResultModel(status="executed", ticker=ticker, side="sell")

            if action == "ROTATE":
                ct = decision.get("close_ticker")
                ot = decision.get("open_ticker")

                if not ct or not ot:
                    return TradeResultModel(status="rejected", reason="missing_tickers")

                if not is_executable_usa(ct) or not is_executable_usa(ot):
                    return TradeResultModel(status="skipped", ticker=ot, reason="CHILE_NO_EXEC")

                self.close_market(ct)

                qty = self.calculate_qty(ot, target_pct or 0.1)
                order = await self.open_market(ot, qty)

                self._log_execution("ROTATE", f"{ct}->{ot}", qty, decision, order)

                return TradeResultModel(
                    status="executed",
                    ticker=ot,
                    side="buy",
                    qty=qty,
                    order_id=str(order.id),
                    equity_used_pct=target_pct,
                )

            return TradeResultModel(status="ignored", reason=f"unknown_action:{action}")

        except Exception as e:
            logger.error(f"❌ EXECUTION FAILED: {e}")
            log_decision({"module": "broker", "decision": "failed", "error": str(e)})
            return TradeResultModel(status="failed", reason=str(e))

    def _log_execution(self, action, ticker, qty, decision, order):

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
# Endpoints
# =========================================================
@router.post("/execute", response_model=TradeResultModel)
@rate_limit(30)
async def execute_trade(
    decision: DecisionInput,
    x_api_key: str = Header(None)
):
    if not BROKER_EXECUTION_KEY:
        raise HTTPException(status_code=500, detail="Broker key not configured")

    if x_api_key != BROKER_EXECUTION_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    engine = get_trading_engine()
    return await engine.execute_decision(decision.model_dump())


@router.get("/status")
async def broker_status():
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
@router.get("/positions")
async def broker_positions():
    engine = get_trading_engine()
    positions = engine.client.get_all_positions()

    result = {}

    for p in positions:
        result[p.symbol] = {
            "market_value": float(p.market_value),
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "unrealized_pl": float(p.unrealized_pl),
            "side": p.side,
        }

    return result

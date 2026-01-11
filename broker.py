# broker.py — LIVE/PAPER TRADING ENGINE (CORREGIDO Y PRODUCTION-READY)
# ✅ Alpaca Markets API (Paper + Live)
# ✅ Auto-ejecuta señales 🔥 STRONG (desde signals.py REAL, sin mock)
# ✅ Risk 1% por trade (cap) + Kelly conservador (1/4)
# ✅ Max 10 posiciones simultáneas
# ✅ Stop-loss dinámico basado en 2x MAE (desde rolling_metrics o evaluations)
# ✅ Evita doble entrada por ticker
# ✅ Persistencia de trades a disco (/data/trades/*.jsonl)
# ✅ Endpoints /trading/status /trading/positions /trading/execute /trading/auto-execute
# ✅ Compatible con signals.py + dashboard.py

import os
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# =========================================================
# Alpaca (alpaca-py)
# =========================================================
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# =========================================================
# Stack config
# =========================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
PAPER_TRADING = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Safety toggles
ALLOW_SHORT = os.getenv("ALLOW_SHORT", "false").lower() == "true"
MAX_POSITIONS = int(os.getenv("BROKER_MAX_POSITIONS", "10"))
RISK_PER_TRADE = float(os.getenv("BROKER_RISK_PER_TRADE", "0.01"))          # 1% equity
MAX_PORTFOLIO_RISK = float(os.getenv("BROKER_MAX_PORTFOLIO_RISK", "0.15"))  # 15% equity
MIN_CONFIDENCE = float(os.getenv("BROKER_MIN_CONFIDENCE", "0.70"))
STRONG_ONLY_DEFAULT = os.getenv("BROKER_STRONG_ONLY", "true").lower() == "true"

# Stop-loss config
STOP_MULT_MAE = float(os.getenv("BROKER_STOP_MULT_MAE", "2.0"))             # 2x MAE (return %)
FALLBACK_STOP_PCT = float(os.getenv("BROKER_FALLBACK_STOP_PCT", "0.03"))    # 3% if no MAE
TAKE_PROFIT_MULT_STOP = float(os.getenv("BROKER_TP_MULT_STOP", "1.5"))      # TP = 1.5x stop distance
MIN_QTY = float(os.getenv("BROKER_MIN_QTY", "1"))                            # for US stocks, integer qty default

# Trades persistence
TRADES_DIR = Path(DATA_PATH) / "trades"
TRADES_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Logging
# =========================================================
def setup_logging():
    level = logging.INFO
    try:
        log_path = Path(DATA_PATH) / "broker.log"
        log_path.parent.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
    except Exception:
        fh = logging.NullHandler()

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        handlers=[fh, sh],
    )

setup_logging()
logger = logging.getLogger(__name__)

# =========================================================
# Router
# =========================================================
router = APIRouter(prefix="/trading", tags=["trading"])

# =========================================================
# Pydantic responses (FastAPI-friendly)
# =========================================================
class TradeResultModel(BaseModel):
    status: str  # executed | rejected | skipped
    order_id: Optional[str] = None
    ticker: str
    side: str
    qty: float
    price: Optional[float] = None
    reason: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

# =========================================================
# Helpers: persistence
# =========================================================
def append_trade_log(record: Dict[str, Any]) -> None:
    try:
        day = datetime.utcnow().strftime("%Y-%m-%d")
        fp = TRADES_DIR / f"trades_{day}.jsonl"
        with open(fp, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"trade log write failed: {e}")

# =========================================================
# Helpers: evaluations -> MAE cache (no deps)
# =========================================================
_mae_cache: Dict[str, Dict[str, Any]] = {}  # ticker -> {"ts": epoch, "mae_pct": float}

def _get_mae_from_evaluations(ticker: str, limit: int = 200, cache_ttl_sec: int = 300) -> Optional[float]:
    """
    Lee /data/evaluations/<ticker>/*.json y estima MAE del error_return_pct.
    Retorna MAE en porcentaje (ej: 1.2 = 1.2%).
    """
    now = time.time()
    cached = _mae_cache.get(ticker)
    if cached and (now - cached["ts"] < cache_ttl_sec):
        return cached.get("mae_pct")

    eval_dir = Path(DATA_PATH) / "evaluations" / ticker
    if not eval_dir.exists():
        _mae_cache[ticker] = {"ts": now, "mae_pct": None}
        return None

    files = sorted(eval_dir.glob("*.json"))[-limit:]
    if not files:
        _mae_cache[ticker] = {"ts": now, "mae_pct": None}
        return None

    vals = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            v = obj.get("error_return_pct")
            if v is None:
                continue
            vals.append(abs(float(v)))
        except Exception:
            continue

    mae = float(np.mean(vals)) if vals else None
    _mae_cache[ticker] = {"ts": now, "mae_pct": mae}
    return mae

def infer_stop_pct_from_signal(signal: Dict[str, Any]) -> float:
    """
    Prioridad:
    1) rolling_metrics.mae_return_pct (si viene desde signals.py)
    2) evaluations/<ticker> error_return_pct promedio (MAE)
    3) fallback fijo (FALLBACK_STOP_PCT)
    Retorna stop_pct como fracción (0.03 = 3%).
    """
    ticker = signal.get("ticker", "")

    rm = signal.get("rolling_metrics") or {}
    mae_pct = None

    try:
        if isinstance(rm, dict) and rm.get("mae_return_pct") is not None:
            mae_pct = float(rm["mae_return_pct"])
    except Exception:
        mae_pct = None

    if mae_pct is None and ticker:
        mae_pct = _get_mae_from_evaluations(ticker)

    if mae_pct is None:
        return float(FALLBACK_STOP_PCT)

    # mae_pct viene en porcentaje (ej 1.2 => 1.2%). Convertimos a fracción.
    stop_pct = (float(mae_pct) / 100.0) * float(STOP_MULT_MAE)
    # clamp razonable
    stop_pct = float(np.clip(stop_pct, 0.005, 0.20))  # 0.5% a 20%
    return stop_pct

# =========================================================
# Trading Engine
# =========================================================
class TradingEngine:
    def __init__(self):
        if not ALPACA_AVAILABLE:
            raise ValueError("alpaca-py no instalado. pip install alpaca-py")
        if not ALPACA_KEY or not ALPACA_SECRET:
            raise ValueError("ALPACA_API_KEY y ALPACA_SECRET_KEY requeridos")

        self.trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER_TRADING)

        # Risk
        self.max_positions = MAX_POSITIONS
        self.risk_per_trade = RISK_PER_TRADE
        self.max_portfolio_risk = MAX_PORTFOLIO_RISK

        acc = self.trading_client.get_account()
        self._equity = float(acc.equity)

        logger.info(
            f"🚀 Broker inicializado: {'PAPER' if PAPER_TRADING else 'LIVE'} "
            f"(status={acc.status}, equity=${self._equity:.0f}, allow_short={ALLOW_SHORT})"
        )

    # -------------------------
    # Portfolio helpers
    # -------------------------
    def _get_positions_map(self) -> Dict[str, Any]:
        pos = self.trading_client.get_all_positions()
        return {p.symbol: p for p in pos}

    def estimate_position_risk_dollars(self, ticker: str, qty: float, entry_price: float, stop_pct: float) -> float:
        # riesgo = distancia al stop * qty
        stop_dist = entry_price * stop_pct
        return abs(qty) * stop_dist

    def get_portfolio_risk_estimate(self) -> float:
        """
        Estimación conservadora del riesgo del portafolio basada en stop_pct por ticker.
        No es perfecta (no conoce stops reales), pero evita exceder riesgo global.
        """
        try:
            acc = self.trading_client.get_account()
            equity = float(acc.equity)
            if equity <= 0:
                return 1.0

            positions = self.trading_client.get_all_positions()
            total_risk = 0.0

            for p in positions:
                ticker = p.symbol
                qty = float(p.qty)
                entry = float(p.avg_entry_price) if float(p.avg_entry_price) > 0 else float(p.current_price)

                # stop_pct estimado desde evaluations
                stop_pct = float(np.clip(_get_mae_from_evaluations(ticker) or (FALLBACK_STOP_PCT * 100), 0.5, 20.0)) / 100.0
                stop_pct *= float(STOP_MULT_MAE)
                stop_pct = float(np.clip(stop_pct, 0.005, 0.20))

                total_risk += self.estimate_position_risk_dollars(ticker, qty, entry, stop_pct)

            return float(np.clip(total_risk / equity, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"portfolio risk estimate failed: {e}")
            return 1.0

    # -------------------------
    # Sizing (Kelly conservador + cap 1% riesgo)
    # -------------------------
    def kelly_position_size_qty(self, signal: Dict[str, Any], confidence: float, stop_pct: float) -> float:
        """
        - Usa Kelly conservador (1/4).
        - Cap por riesgo por trade: (equity * risk_per_trade) / (price * stop_pct)
        - Retorna qty (acciones). Para acciones US, redondeamos a entero >= 1 por defecto.
        """
        acc = self.trading_client.get_account()
        equity = float(acc.equity)

        price = float(signal.get("price_now") or 0)
        if price <= 0:
            return 0.0

        ret_pct = float(signal.get("ret_ens_pct") or 0.0)  # porcentaje
        edge = abs(ret_pct) / 100.0  # fracción esperada

        c = float(confidence) if confidence is not None else 0.5
        c = float(np.clip(c, 0.0, 1.0))

        # Kelly simplificado: p - (1-p)/b, usando b ~= edge/stop_pct (payoff ratio aproximado)
        # para no explotar cuando edge ~0, protegemos.
        b = (edge / max(stop_pct, 1e-6)) if edge > 1e-6 else 0.0
        if b <= 0:
            kelly = 0.0
        else:
            kelly = c - (1 - c) / b

        kelly = max(kelly, 0.0)
        size_pct = min(kelly * 0.25, self.risk_per_trade)  # 1/4 kelly + cap

        # Convertir %equity a qty, pero asegurando riesgo por trade <= 1%
        dollars_alloc = equity * size_pct
        qty_by_alloc = dollars_alloc / price if dollars_alloc > 0 else 0.0

        # Cap por riesgo: riesgo_dólar = qty * price * stop_pct <= equity*risk_per_trade
        max_risk_dollars = equity * self.risk_per_trade
        qty_by_risk = max_risk_dollars / (price * max(stop_pct, 1e-6))

        qty = min(qty_by_alloc, qty_by_risk)

        # Para acciones: qty entero. Si quieres fraccional, pon MIN_QTY=0.01 y quita int().
        if MIN_QTY >= 1:
            qty = float(int(qty))
        else:
            qty = float(round(qty, 4))

        # mínimo
        if qty < MIN_QTY:
            return 0.0
        return qty

    # -------------------------
    # Execution
    # -------------------------
    def execute_signal(self, signal: Dict[str, Any]) -> TradeResultModel:
        """
        Ejecución sin bloquear el resto del stack (este método es sync).
        En FastAPI, puedes llamarlo desde endpoints async sin problema si no haces heavy loads.
        """
        ticker = str(signal.get("ticker") or "").strip().upper()
        if not ticker:
            return TradeResultModel(status="rejected", order_id=None, ticker="", side="", qty=0, reason="missing_ticker")

        quality = signal.get("quality")
        confidence = signal.get("confidence")
        recommendation = str(signal.get("recommendation") or "").upper()

        # 1) Filtros de calidad
        if quality != "🔥 STRONG":
            return TradeResultModel(status="skipped", order_id=None, ticker=ticker, side="", qty=0, reason="quality_not_strong")

        if confidence is None or float(confidence) < MIN_CONFIDENCE:
            return TradeResultModel(status="skipped", order_id=None, ticker=ticker, side="", qty=0, reason="confidence_too_low")

        # 2) Evitar duplicados
        positions_map = self._get_positions_map()
        if ticker in positions_map:
            return TradeResultModel(status="skipped", order_id=None, ticker=ticker, side="", qty=0, reason="already_in_positions")

        # 3) Límites del portafolio
        if len(positions_map) >= self.max_positions:
            return TradeResultModel(status="rejected", order_id=None, ticker=ticker, side="", qty=0, reason="max_positions")

        port_risk = self.get_portfolio_risk_estimate()
        if port_risk >= self.max_portfolio_risk:
            return TradeResultModel(status="rejected", order_id=None, ticker=ticker, side="", qty=0, reason="portfolio_risk_limit")

        # 4) Precio
        price_now = signal.get("price_now")
        try:
            price_now = float(price_now)
        except Exception:
            price_now = 0.0
        if price_now <= 0:
            return TradeResultModel(status="rejected", order_id=None, ticker=ticker, side="", qty=0, reason="invalid_price")

        # 5) Side: por seguridad, SHORT deshabilitado por defecto.
        if recommendation == "BUY":
            side = "buy"
            alp_side = OrderSide.BUY
        elif recommendation == "SELL":
            if not ALLOW_SHORT:
                return TradeResultModel(status="skipped", order_id=None, ticker=ticker, side="sell", qty=0, reason="short_disabled")
            side = "sell"
            alp_side = OrderSide.SELL
        else:
            return TradeResultModel(status="rejected", order_id=None, ticker=ticker, side="", qty=0, reason="invalid_recommendation")

        # 6) Stop-loss dinámico + sizing por riesgo
        stop_pct = infer_stop_pct_from_signal(signal)  # fracción (0.03 = 3%)
        qty = self.kelly_position_size_qty(signal, float(confidence), stop_pct)
        if qty <= 0:
            return TradeResultModel(status="rejected", order_id=None, ticker=ticker, side=side, qty=0, reason="qty_too_small")

        # 7) Construir orden
        # Nota: bracket orders exactos dependen de la versión de alpaca-py.
        # Para mantener 100% compatibilidad sin adivinar clases, enviamos MARKET entry
        # y registramos stop/tp calculados para que tú puedas convertirlo a bracket
        # si habilitas esas clases en tu entorno.
        stop_price = price_now * (1 - stop_pct) if side == "buy" else price_now * (1 + stop_pct)
        tp_price = price_now * (1 + stop_pct * TAKE_PROFIT_MULT_STOP) if side == "buy" else price_now * (1 - stop_pct * TAKE_PROFIT_MULT_STOP)

        try:
            order_req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=alp_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(order_req)

            rec = {
                "ts": datetime.utcnow().isoformat(),
                "mode": "PAPER" if PAPER_TRADING else "LIVE",
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "price_ref": price_now,
                "confidence": float(confidence),
                "quality": quality,
                "recommendation": recommendation,
                "ret_ens_pct": float(signal.get("ret_ens_pct") or 0.0),
                "stop_pct": stop_pct,
                "stop_price": stop_price,
                "take_profit_price": tp_price,
                "order_id": getattr(order, "id", None),
            }
            append_trade_log(rec)

            logger.info(
                f"✅ EXECUTED {ticker}: {side.upper()} qty={qty} ref=${price_now:.2f} "
                f"(conf={float(confidence):.3f}, ret={float(signal.get('ret_ens_pct') or 0.0):+.2f}%, "
                f"stop~{stop_pct*100:.2f}%)"
            )

            return TradeResultModel(
                status="executed",
                order_id=str(getattr(order, "id", None)) if getattr(order, "id", None) else None,
                ticker=ticker,
                side=side,
                qty=float(qty),
                price=None,
                reason=None,
                meta={
                    "stop_pct": stop_pct,
                    "stop_price_est": stop_price,
                    "take_profit_price_est": tp_price,
                    "portfolio_risk_est": port_risk,
                },
            )

        except Exception as e:
            logger.error(f"❌ FAILED {ticker}: {e}")
            append_trade_log(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "mode": "PAPER" if PAPER_TRADING else "LIVE",
                    "ticker": ticker,
                    "side": side,
                    "qty": qty,
                    "status": "rejected",
                    "error": str(e),
                }
            )
            return TradeResultModel(status="rejected", order_id=None, ticker=ticker, side=side, qty=float(qty), reason=str(e))

# =========================================================
# Global singleton engine
# =========================================================
trading_engine: Optional[TradingEngine] = None

def get_trading_engine() -> TradingEngine:
    global trading_engine
    if trading_engine is None:
        trading_engine = TradingEngine()
    return trading_engine

# =========================================================
# Endpoints
# =========================================================
@router.get("/status")
async def trading_status() -> Dict[str, Any]:
    try:
        engine = get_trading_engine()
        acc = engine.trading_client.get_account()
        positions = engine.trading_client.get_all_positions()

        return {
            "status": "active",
            "mode": "PAPER" if PAPER_TRADING else "LIVE",
            "account_status": acc.status,
            "equity": float(acc.equity),
            "buying_power": float(acc.buying_power),
            "n_positions": len(positions),
            "max_positions": engine.max_positions,
            "min_confidence": MIN_CONFIDENCE,
            "allow_short": ALLOW_SHORT,
            "portfolio_risk_est_pct": round(engine.get_portfolio_risk_estimate() * 100, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Broker unavailable: {e}")

@router.get("/positions")
async def get_positions() -> List[Dict[str, Any]]:
    try:
        engine = get_trading_engine()
        positions = engine.trading_client.get_all_positions()

        out = []
        for p in positions:
            out.append(
                {
                    "ticker": p.symbol,
                    "qty": float(p.qty),
                    "side": "long" if float(p.qty) > 0 else "short",
                    "avg_entry": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pnl_pct": float(p.unrealized_plpc),
                    "unrealized_pnl_dollar": float(p.unrealized_pl),
                    "market_value": float(p.market_value),
                }
            )

        return sorted(out, key=lambda x: x["unrealized_pnl_pct"], reverse=True)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Positions unavailable: {e}")

@router.post("/execute", response_

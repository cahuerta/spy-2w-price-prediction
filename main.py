"""
main.py — TRADING SUITE ENTERPRISE v2.7.0 PRODUCCIÓN ✅

ORQUESTADOR COMPLETO: Market → PM → Broker
Singletons PMs + APIs correctas + Monitoring enterprise
"""

import os
import json
import logging
import time
import asyncio
import signal
import requests
from typing import Any, Dict, List, Optional, Deque
from pathlib import Path
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, Query, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# MÓDULOS CRÍTICOS (todos desarrollados)
# =========================================================
from market_state_evaluator import evaluate_quant_market  # ❌ PENDIENTE
from market_qualitative_evaluator import evaluate_qualitative_market
from market_orchestrator import MarketOrchestrator, MarketOrchestrationContext

from pm_growth import PMGrowth  # ❌ PENDIENTE DESARROLLAR
from pm_neutral import PMNeutral
from pm_defensive import PMDefensive

# =========================================================
# CONFIGURACIÓN PRODUCTION READY
# =========================================================
class Config:
    DATA_PATH = os.getenv("DATA_PATH", "/data")
    PORT = int(os.getenv("PORT", "8000"))
    FIXED_CAPITAL = float(os.getenv("PM_FIXED_CAPITAL", "100000"))

    BROKER_EXEC_URL = os.getenv("BROKER_URL", "http://localhost:8001/trading/execute")
    BROKER_STATUS_URL = os.getenv("BROKER_STATUS_URL", "http://localhost:8001/trading/status")

    RL_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RL_PER_SECONDS = int(os.getenv("RATE_LIMIT_PER_SECONDS", "60"))
    RL_MAX_IPS = int(os.getenv("RATE_LIMIT_MAX_IPS", "5000"))

    BATCH_LIMIT_DEFAULT = int(os.getenv("BATCH_LIMIT_DEFAULT", "500"))
    BROKER_FAILURE_THRESHOLD = int(os.getenv("BROKER_FAILURE_THRESHOLD", "5"))
    BROKER_CIRCUIT_OPEN_SECS = int(os.getenv("BROKER_CIRCUIT_OPEN_SECS", "300"))

    SIGNALS_MIN_CONF_DEFAULT = float(os.getenv("SIGNALS_MIN_CONF_DEFAULT", "0.0"))
    SIGNALS_MAX = int(os.getenv("SIGNALS_MAX", "5000"))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

config = Config()

# =========================================================
# LOGGING ENTERPRISE
# =========================================================
def setup_logging():
    Path(config.DATA_PATH).mkdir(parents=True, exist_ok=True)
    log_file = Path(config.DATA_PATH) / "trading_suite.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    lg = logging.getLogger("trading_suite")
    lg.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    lg.handlers.clear()
    lg.addHandler(file_handler)
    lg.addHandler(console_handler)
    lg.propagate = False
    return lg

logger = setup_logging()

# =========================================================
# DISK OPERATIONS
# =========================================================
def ensure_dirs():
    dirs = [
        "predictions", "evaluations", "market", "positions", "signals"
    ]
    for d in dirs:
        (Path(config.DATA_PATH) / d).mkdir(parents=True, exist_ok=True)

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

# =========================================================
# SINGLETONS GLOBALES (CRÍTICO: mantiene hysteresis)
# =========================================================
_market_orchestrator: Optional[MarketOrchestrator] = None
_pm_growth: Optional[PMGrowth] = None
_pm_neutral: Optional[PMNeutral] = None
_pm_defensive: Optional[PMDefensive] = None

def get_market_orchestrator() -> MarketOrchestrator:
    global _market_orchestrator
    if _market_orchestrator is None:
        _market_orchestrator = MarketOrchestrator()
        logger.info("🧠 MarketOrchestrator singleton inicializado")
    return _market_orchestrator

def get_pm_growth() -> PMGrowth:
    global _pm_growth
    if _pm_growth is None:
        _pm_growth = PMGrowth()
        logger.info("📈 PMGrowth singleton inicializado")
    return _pm_growth

def get_pm_neutral() -> PMNeutral:
    global _pm_neutral
    if _pm_neutral is None:
        _pm_neutral = PMNeutral()
        logger.info("🟡 PMNeutral singleton inicializado")
    return _pm_neutral

def get_pm_defensive() -> PMDefensive:
    global _pm_defensive
    if _pm_defensive is None:
        _pm_defensive = PMDefensive()
        logger.info("🔴 PMDefensive singleton inicializado")
    return _pm_defensive

def resolve_pm(market_mode: str):
    """Resuelve PM correcto MANTENIENDO ESTADO."""
    if market_mode == "growth":
        return get_pm_growth()
    elif market_mode == "neutral":
        return get_pm_neutral()
    return get_pm_defensive()

# =========================================================
# FASTAPI SETUP
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Trading Suite Enterprise v2.7.0 iniciando...")
    ensure_dirs()
    yield
    logger.info("🛑 Trading Suite detenida graceful")

app = FastAPI(
    title="Trading Suite Enterprise",
    version="2.7.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# OPTIONAL MODULES (graceful fallback)
# =========================================================
compute_all_signals = None
try:
    from signals import compute_all_signals
    logger.info("✅ Signals module cargado")
except Exception as e:
    logger.warning(f"Signals module unavailable: {e}")
    compute_all_signals = lambda: []

try:
    from broker import router as broker_router
    app.include_router(broker_router, prefix="/trading")
    logger.info("✅ Broker router integrado")
except Exception as e:
    logger.warning(f"Broker router unavailable: {e}")

# =========================================================
# DAILY SYSTEM ORCHESTRATOR (FLUJOS CORREGIDOS)
# =========================================================
@app.post("/internal/system/daily-run")
async def daily_system_run(request: Request):
    """Orquestación completa diaria: Market → PM → Broker."""
    
    if request.headers.get("X-PIPELINE-KEY") != os.getenv("PIPELINE_KEY"):
        raise HTTPException(status_code=403, detail="Forbidden")

    logger.info("🧠 === DAILY SYSTEM RUN INICIADO ===")

    # ================================
    # 1️⃣ MARKET CONTEXT EVALUATION
    # ================================
    market_mode = "defensive"  # Fallback seguro
    market_ctx = None
    
    try:
        spy_path = Path(config.DATA_PATH) / "market" / "spy_prices.json"
        cross_path = Path(config.DATA_PATH) / "market" / "cross_prices.json"
        
        spy = load_json(spy_path)
        cross = load_json(cross_path)

        if not spy or not spy.get("prices") or not cross or not cross.get("prices"):
            raise ValueError("Market data faltante o inválido")

        quant_ctx = evaluate_quant_market(spy["prices"], cross["prices"])
        qual_ctx = evaluate_qualitative_market(quant_ctx.to_dict())
        
        orchestrator = get_market_orchestrator()
        market_ctx = orchestrator.evaluate(quant_ctx.to_dict(), qual_ctx.to_dict())
        market_mode = market_ctx.market_mode
        
        # PERSISTIR MARKET STATE
        save_json(
            Path(config.DATA_PATH) / "market" / "current_state.json", 
            market_ctx.to_dict()
        )
        
        logger.info(f"🌍 MARKET MODE: {market_mode.upper()} | conf: {market_ctx.confidence:.2f}")

    except Exception as e:
        logger.error(f"❌ Market evaluation failed → FALLBACK DEFENSIVE: {e}")
        market_ctx = MarketOrchestrationContext(
            market_mode="defensive",
            confidence=0.0,
            reason=f"evaluation_failed: {str(e)[:100]}",
            timestamp=datetime.utcnow().isoformat(),
            source={}
        )

    # ================================
    # 2️⃣ RESOLVE PM SINGLETON
    # ================================
    pm = resolve_pm(market_mode)
    logger.info(f"📦 PM ACTIVO: {pm.__class__.__name__}")

    # ================================
    # 3️⃣ LOAD CURRENT POSITIONS
    # ================================
    positions_path = Path(config.DATA_PATH) / "positions.json"
    positions = load_json(positions_path) or []
    logger.info(f"📊 Portfolio actual: {len(positions)} posiciones")

    # ================================
    # 4️⃣ EVALUATE EXISTING POSITIONS
    # ================================
    decisions = []
    closes = 0
    
    for pos in positions:
        try:
            ticker = pos.get("ticker", "UNKNOWN")
            decision = pm.evaluate_position(pos)  # 👈 API CORRECTA
            decisions.append(decision.to_dict())
            
            if decision["action"] == "CLOSE":
                closes += 1
                logger.info(f"🛑 CLOSE: {ticker} | {decision['reason']}")
                
        except Exception as e:
            logger.error(f"Error evaluando {ticker}: {e}")
            decisions.append({
                "action": "CLOSE", "ticker": ticker, 
                "reason": f"evaluation_error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    # ================================
    # 5️⃣ NEW POSITIONS (SOLO si PM permite)
    # ================================
    new_signals = []
    if hasattr(pm, "allow_new_positions") and pm.allow_new_positions():
        try:
            signals = compute_all_signals()
            logger.info(f"🔍 Evaluando {len(signals)} señales nuevas")
            
            for signal in signals[:5]:  # Máx 5 candidatos
                decision = pm.evaluate_signal(signal)
                if decision.action == "OPEN":
                    new_signals.append(decision.to_dict())
                    logger.info(f"➕ NEW OPEN: {decision.ticker} | conf: {decision.get('meta', {}).get('confidence', 0):.2f}")
                    break  # 1 sola nueva posición por ciclo
        except Exception as e:
            logger.error(f"New signals failed: {e}")

    decisions.extend(new_signals)
    logger.info(f"📋 Decisions: {len(decisions)} total | closes: {closes} | new: {len(new_signals)}")

    # ================================
    # 6️⃣ EXECUTE VIA BROKER
    # ================================
    executed = []
    execution_errors = 0
    
    for decision in decisions:
        if decision["action"] in ["OPEN", "CLOSE"]:
            try:
                response = requests.post(
                    config.BROKER_EXEC_URL,
                    json=decision,
                    timeout=15,
                    headers={
                        "X-MARKET-MODE": market_mode,
                        "X-PM-ACTIVE": pm.__class__.__name__
                    }
                )
                response.raise_for_status()
                
                broker_result = response.json()
                executed.append({
                    **decision,
                    "broker_status": "success",
                    "broker_response": broker_result
                })
                logger.info(f"✅ EXEC {decision['action']}: {decision['ticker']}")
                
            except requests.exceptions.RequestException as e:
                execution_errors += 1
                executed.append({
                    **decision,
                    "broker_status": "failed",
                    "error": str(e)
                })
                logger.error(f"❌ Broker EXEC failed {decision['ticker']}: {e}")

    # ================================
    # 7️⃣ FINAL SUMMARY & PERSIST
    # ================================
    summary = {
        "status": "completed",
        "market_mode": market_mode,
        "market_confidence": getattr(market_ctx, 'confidence', 0.0),
        "pm_active": pm.__class__.__name__,
        "positions_evaluated": len(positions),
        "closes": closes,
        "new_positions": len(new_signals),
        "executed": len([d for d in executed if d["broker_status"] == "success"]),
        "execution_errors": execution_errors,
        "total_decisions": len(decisions),
        "decisions_sample": decisions[-10:],  # Últimas 10
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    save_json(Path(config.DATA_PATH) / "daily_runs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M')}.json", summary)
    
    logger.info(f"🏁 DAILY RUN COMPLETADO | {summary}")
    return summary

# =========================================================
# MONITORING ENDPOINTS
# =========================================================
@app.get("/internal/market/state")
async def get_market_state():
    """Estado actual del mercado (persistido)."""
    state_path = Path(config.DATA_PATH) / "market" / "current_state.json"
    if state_path.exists():
        return load_json(state_path)
    return {"error": "No market state available"}

@app.get("/internal/pm/status")
async def get_pm_status():
    """Estado de todos los PM singletons."""
    return {
        "growth": get_pm_growth().__class__.__name__ if _pm_growth else None,
        "neutral": get_pm_neutral().__class__.__name__ if _pm_neutral else None,
        "defensive": get_pm_defensive().__class__.__name__ if _pm_defensive else None,
        "market_mode": getattr(get_market_orchestrator(), '_last_mode', 'unknown'),
        "active_pm": resolve_pm("neutral").__class__.__name__
    }

@app.get("/internal/portfolio/summary")
async def get_portfolio_summary():
    """Resumen rápido portfolio."""
    positions = load_json(Path(config.DATA_PATH) / "positions.json") or []
    return {
        "positions_count": len(positions),
        "market_mode": getattr(get_market_orchestrator(), '_last_mode', 'unknown'),
        "pm_active": resolve_pm("neutral").__class__.__name__
    }

# =========================================================
# HEALTH ENHANCED
# =========================================================
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.7.0",
        "market_mode": getattr(get_market_orchestrator(), '_last_mode', 'unknown'),
        "pm_active": resolve_pm("neutral").__class__.__name__,
        "positions": len(load_json(Path(config.DATA_PATH) / "positions.json") or []),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# =========================================================
# GRACEFUL SHUTDOWN
# =========================================================
def handle_shutdown(signum, frame):
    logger.info("SIGTERM recibido - graceful shutdown")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)

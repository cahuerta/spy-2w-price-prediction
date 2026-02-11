# =========================================================
# main.py — TRADING SUITE ENTERPRISE v2.9.0 PRODUCCIÓN
# =========================================================
# ✔ Runtime de trading (NO batch)
# ✔ NO decide mercado
# ✔ NO ejecuta modelos
# ✔ NO ejecuta pipeline
# ✔ RECIBE resultados del pipeline
# ✔ GRABA a disco (fuente única)
# ✔ Ejecuta SOLO TradingOrchestrator
# =========================================================

import os
import json
import logging
import signal
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any
from signals_router import router as signals_router
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# ROUTERS
# =========================================================
from dashboard import router as dashboard_router

# =========================================================
# CORE
# =========================================================
from market_orchestrator import MarketOrchestrationContext
from trading_orchestrator import TradingOrchestrator

# =========================================================
# PIPELINE ROUTER
# =========================================================
from pipeline_router import router as pipeline_router

# =========================================================
# CONFIG
# =========================================================
class Config:
    DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
    PORT = int(os.getenv("PORT", "8000"))
    PIPELINE_KEY = os.getenv("PIPELINE_KEY")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

config = Config()

DATA_PATH = config.DATA_PATH
MARKET_CTX_FILE = DATA_PATH / "market_context.json"
SCREENER_FILE = DATA_PATH / "screener_candidates.json"
PIPELINE_AUDIT_FILE = DATA_PATH / "last_pipeline.json"

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trading_suite")

# =========================================================
# HELPERS
# =========================================================
def save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def load_market_context() -> MarketOrchestrationContext:
    if not MARKET_CTX_FILE.exists():
        raise RuntimeError("market_context.json no existe")

    data = json.loads(MARKET_CTX_FILE.read_text())
    return MarketOrchestrationContext(**data)

# =========================================================
# FASTAPI APP
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Trading Suite v2.9.0 START")
    yield
    logger.info("Trading Suite v2.9.0 STOP")

app = FastAPI(
    title="Trading Suite Enterprise",
    version="2.9.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)
app.include_router(pipeline_router)

# =========================================================
# PIPELINE COMMIT (RECIBE Y GRABA)
# =========================================================
@app.post("/internal/pipeline/commit")
async def pipeline_commit(payload: Dict[str, Any], request: Request):
    """
    Recibe output de pipeline_daily.main()
    y lo persiste en disco.
    """
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    logger.info("💾 Committing pipeline payload")

    try:
        # --- guardar screener ---
        if "screener" in payload:
            save_json(SCREENER_FILE, payload["screener"])
            logger.info("📄 screener_candidates.json guardado")

        # --- guardar market context ---
        if "market_ctx" in payload:
            save_json(MARKET_CTX_FILE, payload["market_ctx"])
            logger.info("📄 market_context.json guardado")

        # --- auditoría completa ---
        save_json(PIPELINE_AUDIT_FILE, payload)

    except Exception as e:
        logger.error("❌ Commit failed")
        raise HTTPException(500, str(e))

    return {
        "status": "ok",
        "message": "pipeline committed",
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# TRADING ENDPOINT
# =========================================================
@app.post("/internal/trading/run")
async def trading_run(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    logger.info("🔔 Trading run triggered")

    market_ctx = load_market_context()
    orchestrator = TradingOrchestrator()
    result = await orchestrator.run(market_ctx.to_dict())

    return {
        "status": "ok",
        "market_mode": market_ctx.market_mode,
        "result": result,
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# HEALTH
# =========================================================
app.include_router(signals_router)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "spy-2w-price-prediction",
        "env": "production"
    }
# =========================================================
# SHUTDOWN
# =========================================================
def handle_shutdown(signum, frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.PORT,
        log_level="error",
                   )

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
# ✔ 🔥 MERGE DE TICKERS EN STARTUP (NUNCA BORRA)
# =========================================================

import os
import json
import logging
import signal
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from signals_router import router as signals_router
from dashboard import router as dashboard_router
from pipeline_router import router as pipeline_router

from market_orchestrator import MarketOrchestrationContext
from trading_orchestrator import TradingOrchestrator
from debug_data_router import router as debug_router
from admin_tickers_router import router as admin_router
from broker import router as trading_router

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

TICKERS_FILE = DATA_PATH / "tickers.json"
REPO_TICKERS_FILE = Path(__file__).resolve().parent / "tickers.json"
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
# HELPERS DISCO
# =========================================================
def save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())

# =========================================================
# 🔥 MERGE DE TICKERS (STARTUP)
# =========================================================
def merge_tickers_on_startup():
    """
    Une tickers base + tickers dinámicos.
    NUNCA borra. SOLO agrega.
    """
    logger.info("🔧 Merging tickers on startup")

    # --- tickers existentes en disco (fuente real) ---
    disk_tickers: List[str] = load_json(TICKERS_FILE, [])

    # --- tickers base desde el repo ---
    if REPO_TICKERS_FILE.exists():
        try:
            repo_raw = json.loads(REPO_TICKERS_FILE.read_text())
            if isinstance(repo_raw, list):
                base_tickers = [str(t).strip() for t in repo_raw if isinstance(t, str)]
            elif isinstance(repo_raw, dict) and "tickers" in repo_raw:
                base_tickers = [str(t).strip() for t in repo_raw["tickers"] if isinstance(t, str)]
            else:
                base_tickers = []
        except Exception:
            base_tickers = []
    else:
        base_tickers = []

    merged = sorted(set(disk_tickers) | set(base_tickers))

    if merged != disk_tickers:
        save_json(TICKERS_FILE, merged)
        logger.info(
            f"📈 tickers.json actualizado | total={len(merged)} "
            f"(antes={len(disk_tickers)})"
        )
    else:
        logger.info("✔ tickers.json ya estaba actualizado")

# =========================================================
# MARKET CONTEXT
# =========================================================
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
    logger.info("🚀 Trading Suite v2.9.0 START")
    merge_tickers_on_startup()
    yield
    logger.info("🛑 Trading Suite v2.9.0 STOP")

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

# =========================================================
# ROUTERS
# =========================================================
app.include_router(dashboard_router)
app.include_router(pipeline_router)
app.include_router(signals_router)
app.include_router(debug_router)
app.include_router(admin_router)
app.include_router(trading_router)

# =========================================================
# PIPELINE COMMIT
# =========================================================
@app.post("/internal/pipeline/commit")
async def pipeline_commit(payload: Dict[str, Any], request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    logger.info("💾 Committing pipeline payload")

    try:
        if "screener" in payload:
            save_json(SCREENER_FILE, payload["screener"])

        if "market_ctx" in payload:
            save_json(MARKET_CTX_FILE, payload["market_ctx"])

        save_json(PIPELINE_AUDIT_FILE, payload)

    except Exception as e:
        logger.error("❌ Commit failed")
        raise HTTPException(500, str(e))

    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# =========================================================
# TRADING
# =========================================================
@app.post("/internal/trading/run")
async def trading_run(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

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
# INTERNAL INSPECTION (para frontend / debug)
# =========================================================

@app.get("/internal/pipeline-last")
async def internal_pipeline_last(request: Request):
    # Protegido con PIPELINE_KEY
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    # last_pipeline.json
    return load_json(PIPELINE_AUDIT_FILE, {})


@app.get("/internal/trades-today")
async def internal_trades_today(request: Request):
    # Protegido con PIPELINE_KEY
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    today = datetime.utcnow().strftime("%Y-%m-%d")
    fp = DATA_PATH / "trades" / f"trades_{today}.jsonl"

    if not fp.exists():
        return []

    out = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            # si hay una línea corrupta, la ignoramos
            continue
    return out


@app.get("/internal/positions")
async def internal_positions(request: Request):
    # Protegido con PIPELINE_KEY
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    from portfolio_store import load_positions
    return load_positions()
# =========================================================
# HEALTH
# =========================================================
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
@app.get("/internal/pipeline/last")
async def get_last_pipeline():
    if not PIPELINE_AUDIT_FILE.exists():
        return {"status": "no_pipeline_executed"}

    return load_json(PIPELINE_AUDIT_FILE, {})
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

# =====================================================
# main.py — TRADING SUITE ENTERPRISE v2.6.0 (PRODUCTION READY ✅)
# =====================================================

import os
import json
import logging
import time
import asyncio
import signal
from typing import Any, Dict, List, Optional, Deque
from pathlib import Path
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, Query, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# Config - PRODUCTION READY
# =========================================================
class Config:
    DATA_PATH = os.getenv("DATA_PATH", "/data")
    PORT = int(os.getenv("PORT", "8000"))
    FIXED_CAPITAL = float(os.getenv("PM_FIXED_CAPITAL", "100000"))

    BROKER_EXEC_URL = os.getenv("BROKER_URL", "http://localhost:8000/trading/execute")
    BROKER_STATUS_URL = os.getenv("BROKER_STATUS_URL", "http://localhost:8000/trading/status")

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
# Logging - DUAL FILE/CONSOLE
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
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
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
# Disk Operations - CACHED & RESILIENT
# =========================================================
def ensure_dirs():
    (Path(config.DATA_PATH) / "predictions").mkdir(parents=True, exist_ok=True)
    (Path(config.DATA_PATH) / "evaluations").mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Directorios preparados en {config.DATA_PATH}")

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"No se pudo cargar {path}: {e}")
        return None

def safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

@lru_cache(maxsize=1024)
def list_prediction_tickers_cached() -> List[str]:
    root = Path(config.DATA_PATH) / "predictions"
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())

def latest_prediction_for_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    p = Path(config.DATA_PATH) / "predictions" / ticker
    if not p.exists():
        return None
    files = sorted(p.glob("*.json"))
    return load_json(files[-1]) if files else None

# =========================================================
# GLOBAL STATE - THREAD-SAFE POSITION MANAGER
# =========================================================
_pm_cache: Optional["PositionManager"] = None
_pm_lock = asyncio.Lock()

async def get_position_manager() -> "PositionManager":
    global _pm_cache
    if _pm_cache is None:
        async with _pm_lock:
            if _pm_cache is None:
                try:
                    from position_manager import PositionManager
                    _pm_cache = PositionManager()
                    logger.info("💼 PositionManager inicializado")
                except Exception as e:
                    logger.error(f"❌ PositionManager fallo: {e}")
                    raise HTTPException(503, "PositionManager no disponible")
    return _pm_cache

# =========================================================
# BROKER CIRCUIT BREAKER - PRODUCTION GRADE
# =========================================================
class BrokerCircuit:
    def __init__(self):
        self.failures = 0
        self.open_until = 0
        self._lock = asyncio.Lock()

    async def is_open(self) -> bool:
        async with self._lock:
            return time.time() < self.open_until

    async def record_failure(self):
        async with self._lock:
            self.failures += 1
            if self.failures >= config.BROKER_FAILURE_THRESHOLD:
                self.open_until = time.time() + config.BROKER_CIRCUIT_OPEN_SECS
                logger.error(f"🔌 BROKER CIRCUIT OPEN - Failures: {self.failures}")
                return True
        return False

    async def reset(self):
        async with self._lock:
            self.failures = 0
            self.open_until = 0
            logger.info("🔌 Broker circuit RESET")

broker_circuit = BrokerCircuit()

# =========================================================
# ASYNC RATE LIMITER - SCALABLE
# =========================================================
class AsyncRateLimiter:
    def __init__(self, requests: int, per_seconds: int, max_ips: int):
        self.requests = requests
        self.per_seconds = per_seconds
        self.max_ips = max_ips
        self.buckets: Dict[str, Deque[float]] = {}
        self._lock = asyncio.Lock()

    async def __call__(self, request: Request) -> bool:
        ip = request.client.host if request.client else "unknown"
        now = time.time()

        async with self._lock:
            if ip not in self.buckets:
                if len(self.buckets) >= self.max_ips:
                    self._cleanup(now)
                self.buckets[ip] = deque(maxlen=self.requests * 2)

            q = self.buckets[ip]
            while q and now - q[0] > self.per_seconds:
                q.popleft()

            if len(q) >= self.requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit excedido",
                )

            q.append(now)
            self._cleanup(now)

        return True

    def _cleanup(self, now: float):
        expired = [
            ip for ip, q in self.buckets.items()
            if (not q) or (now - q[-1] > self.per_seconds * 2)
        ]
        for ip in expired:
            self.buckets.pop(ip, None)

rate_limiter = AsyncRateLimiter(
    config.RL_REQUESTS, config.RL_PER_SECONDS, config.RL_MAX_IPS
)

# =========================================================
# RATE LIMIT DEPENDENCY
# =========================================================
async def verify_rate_limit(request: Request):
    await rate_limiter(request)

# =========================================================
# LIFESPAN - GRACEFUL START/STOP
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Trading Suite Enterprise v2.6.0 iniciando...")
    ensure_dirs()
    await broker_circuit.reset()
    yield
    logger.info("🛑 Trading Suite Enterprise deteniéndose...")

# =========================================================
# FASTAPI APP - ENTERPRISE READY
# =========================================================
app = FastAPI(
    title="Trading Suite Enterprise",
    version="2.6.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# OPTIONAL MODULES - RESILIENT LOADING
# =========================================================
modules_status = {
    "model": False,
    "evaluator": False,
    "position_manager": False,
    "broker": False,
}

try:
    from model import run_model
    modules_status["model"] = True
except Exception:
    run_model = None

try:
    from evaluator import evaluate_all
    modules_status["evaluator"] = True
except Exception:
    evaluate_all = None

try:
    from position_manager import PositionManager
    modules_status["position_manager"] = True
except Exception:
    pass

try:
    from broker import router as broker_router
    app.include_router(broker_router, prefix="/trading")
    modules_status["broker"] = True
except Exception:
    pass
try:
    from signals import compute_all_signals
    modules_status["signals"] = True
except Exception as e:
    compute_all_signals = None
    modules_status["signals"] = False
    
# =========================================================
# SIGNALS ENDPOINT (BACKEND ↔ FRONTEND)
# =========================================================
@app.get("/signals")
async def signals_endpoint(
    request: Request,
    min_confidence: float = Query(config.SIGNALS_MIN_CONF_DEFAULT),
    _=Depends(verify_rate_limit),
):
    if compute_all_signals is None:
        raise HTTPException(503, "Signals module not available")

    data = compute_all_signals()

    if min_confidence > 0:
        data = [
            s for s in data
            if (s.get("confidence") or 0) >= min_confidence
        ]

    return {
        "signals": data,
        "count": len(data),
        "min_confidence": min_confidence,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

# =====================================================
# Screener disk path (SINGLE SOURCE OF TRUTH)
# =====================================================
DATA_PATH = Path(os.getenv("DATA_PATH", config.DATA_PATH))
SCREENER_FILE = DATA_PATH / "screener_candidates.json"
# =====================================================
# Screener candidates endpoint (READ ONLY | DEBUG + FRONTEND)
# =====================================================
@app.get("/dashboard/screener")
def get_screener_candidates():
    path = SCREENER_FILE  # /data/screener_candidates.json

    # Caso 1: archivo aún no generado
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "generated_at": None,
            "n_candidates": 0,
            "candidates": [],
            "message": "screener_candidates.json no existe aún"
        }

    # Caso 2: archivo existe → leerlo
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)

        stat = path.stat()

        return {
            "exists": True,
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
            # Payload esperado por frontend
            "generated_at": data.get("generated_at"),
            "version": data.get("version"),
            "n_universe": data.get("n_universe"),
            "n_candidates": data.get("n_candidates"),
            "candidates": data.get("candidates", []),
            # Debug útil (no rompe UI)
            "meta": {
                "ia_available": data.get("ia_available"),
                "fundamental_available": data.get("fundamental_available"),
                "params": data.get("params"),
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error leyendo screener_candidates.json: {e}"
        )

@app.post("/internal/screener/result")
async def receive_screener_result(
    payload: Dict[str, Any],
    request: Request,
):
    if request.headers.get("X-PIPELINE-KEY") != os.getenv("PIPELINE_KEY"):
        raise HTTPException(403, "Forbidden")

    path = Path(config.DATA_PATH) / "screener_candidates.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info("📁 Screener JSON recibido y guardado")

    return {"status": "ok"}

# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.6.0",
        "modules": modules_status,
        "broker_circuit": await broker_circuit.is_open(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# =========================================================
# DASHBOARD ENDPOINTS (FIXED PARAM ORDER)
# =========================================================
@app.get("/dashboard/predictions")
async def dashboard_predictions(
    request: Request,
    ticker: str = Query(...),
    _=Depends(verify_rate_limit)
):
    pred_dir = Path(config.DATA_PATH) / "predictions" / ticker
    if not pred_dir.exists():
        return {"data": [], "ticker": ticker, "count": 0}

    data = []
    for f in sorted(pred_dir.glob("*.json")):
        j = load_json(f)
        if j and "prediction" in j:
            data.append({**j, "filename": f.name, "timestamp": f.stat().st_mtime})

    return {"ticker": ticker, "count": len(data), "latest": data[-1] if data else None, "data": data}

@app.get("/dashboard/evaluations")
async def dashboard_evaluations(
    request: Request,
    ticker: str = Query(...),
    _=Depends(verify_rate_limit)
):
    eval_dir = Path(config.DATA_PATH) / "evaluations" / ticker
    if not eval_dir.exists():
        return {"data": [], "ticker": ticker, "count": 0}

    data = []
    for f in sorted(eval_dir.glob("*.json")):
        j = load_json(f)
        if j:
            data.append({**j, "filename": f.name, "timestamp": f.stat().st_mtime})

    return {"ticker": ticker, "count": len(data), "latest": data[-1] if data else None, "data": data}

@app.get("/dashboard/tickers")
async def dashboard_tickers(
    request: Request,
    _=Depends(verify_rate_limit)
):
    tickers = list_prediction_tickers_cached()
    return {"tickers": tickers, "count": len(tickers), "cache_hits": list_prediction_tickers_cached.cache_info()}

@app.get("/dashboard/latest/{ticker}")
async def dashboard_latest(
    request: Request,
    ticker: str,
    _=Depends(verify_rate_limit)
):
    pred = latest_prediction_for_ticker(ticker)
    if not pred:
        raise HTTPException(404, f"No predictions found for {ticker}")
    return {"ticker": ticker, "latest": pred}
@app.get("/dashboard/predictions/summary")
async def dashboard_predictions_summary(
    request: Request,
    ticker: str = Query(...),
    limit: int = Query(60, ge=1, le=500),
    _=Depends(verify_rate_limit),
):
    pred_dir = Path(config.DATA_PATH) / "predictions" / ticker
    if not pred_dir.exists():
        raise HTTPException(404, f"No predictions for {ticker}")

    files = sorted(pred_dir.glob("*.json"))
    if not files:
        raise HTTPException(404, f"No prediction files for {ticker}")

    data = []
    for fp in files[-limit:]:
        obj = load_json(fp)
        if not obj:
            continue

        p = obj.get("prediction", {})
        if not p:
            continue

        data.append({
            "date_base": p.get("date_base"),
            "price_now": safe_float(p.get("price_now")),
            "price_pred": safe_float(p.get("price_pred")),
            "ret_ens_pct": safe_float(p.get("ret_ens_pct")),
            "recommendation": p.get("recommendation"),
        })

    if not data:
        raise HTTPException(404, f"No usable data for {ticker}")

    return {
        "ticker": ticker,
        "count": len(data),
        "data": data,
    }

# =========================================================
# POSITION MANAGER ENDPOINT
# =========================================================
@app.get("/positions")
async def get_positions(
    request: Request,
    _=Depends(verify_rate_limit)
):
    pm = await get_position_manager()
    return await pm.get_positions()

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=config.DEBUG_MODE,
        log_level="error" if not config.DEBUG_MODE else "debug",
            )

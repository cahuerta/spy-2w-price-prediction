# =====================================================
# main.py — TRADING SUITE ENTERPRISE v2.5.2 (RENDER FIXED ✅)
# =====================================================
# ✅ FIX 1: TrustedHostMiddleware removido (no acepta "*" y rompe startup en Render)
# ✅ FIX 2: lifespan usa broker_circuit.reset() (no existía reset_broker_circuit)
# ✅ Mantiene Universe + Signals + Health + Metrics + Circuit + PM cache
# =====================================================

import os
import json
import logging
import time
import asyncio
import signal
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, Query, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
import httpx

# =========================================================
# Config - MEJORADA CON VALIDACIÓN
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
# Logging - MEJORADO (file + console)
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
# Disk helpers
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

def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

@lru_cache(maxsize=1024)
def list_prediction_tickers_cached() -> List[str]:
    """Cache de tickers con LRU"""
    root = Path(config.DATA_PATH) / "predictions"
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())

def latest_prediction_for_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    """Obtiene la última predicción para un ticker específico"""
    p = Path(config.DATA_PATH) / "predictions" / ticker
    if not p.exists():
        return None
    files = sorted(p.glob("*.json"))
    return load_json(files[-1]) if files else None

# =========================================================
# GLOBAL STATE - THREAD-SAFE
# =========================================================
_pm_cache: Optional["PositionManager"] = None
_pm_lock = asyncio.Lock()

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

    async def reset(self):
        async with self._lock:
            self.failures = 0
            self.open_until = 0
            logger.info("🔌 Broker circuit RESET")

broker_circuit = BrokerCircuit()

# =========================================================
# Rate Limiter (simple)
# =========================================================
class AsyncRateLimiter:
    def __init__(self, requests: int, per_seconds: int, max_ips: int):
        self.requests = requests
        self.per_seconds = per_seconds
        self.max_ips = max_ips
        self.buckets: Dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def __call__(self, request: Request) -> bool:
        ip = request.client.host if request.client else "unknown"
        now = time.time()

        async with self._lock:
            if ip not in self.buckets:
                # soft bound
                if len(self.buckets) >= self.max_ips:
                    self._cleanup(now)
                self.buckets[ip] = deque(maxlen=self.requests * 2)

            q = self.buckets[ip]
            while q and now - q[0] > self.per_seconds:
                q.popleft()

            if len(q) >= self.requests:
                logger.warning(f"Rate limit excedido para IP {ip}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit excedido. Intente en {max(1, int(self.per_seconds - (now - q[0])))}s",
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

rate_limiter = AsyncRateLimiter(config.RL_REQUESTS, config.RL_PER_SECONDS, config.RL_MAX_IPS)

# =========================================================
# LIFESPAN (startup/shutdown) ✅ FIXED
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Trading Suite Enterprise v2.5.2 iniciando...")
    ensure_dirs()
    await broker_circuit.reset()   # ✅ FIX: existía broker_circuit.reset()
    yield
    logger.info("🛑 Trading Suite Enterprise deteniéndose...")

# =========================================================
# FastAPI
# =========================================================
app = FastAPI(
    title="Trading Suite Enterprise",
    version="2.5.2",
    lifespan=lifespan,
)

# ✅ FIX: NO TrustedHostMiddleware aquí (en Render rompe si pones '*')
# Si luego quieres endurecer, agrega host exacto de tu app onrender.com

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Optional modules - fallbacks
# =========================================================
modules_status = {
    "model": False,
    "evaluator": False,
    "position_manager": False,
    "broker": False,
}

def safe_import(attr_name: str, module_path: str):
    try:
        module = __import__(module_path, fromlist=[attr_name])
        fn = getattr(module, attr_name)
        modules_status[module_path] = True  # solo informativo
        logger.info(f"✅ {module_path}.{attr_name} cargado")
        return fn
    except Exception as e:
        logger.warning(f"⚠️  {module_path}.{attr_name} no disponible: {e}")
        return None

# Mantengo tu idea, pero guardo flags correctos
try:
    from model import run_model as run_model
    modules_status["model"] = True
    logger.info("✅ model.py loaded")
except Exception as e:
    run_model = None
    logger.warning(f"⚠️ model.py missing: {e}")

try:
    from evaluator import evaluate_all as evaluate_all
    modules_status["evaluator"] = True
    logger.info("✅ evaluator.py loaded")
except Exception as e:
    evaluate_all = None
    logger.warning(f"⚠️ evaluator.py missing: {e}")

try:
    from position_manager import PositionManager as PositionManager
    modules_status["position_manager"] = True
    logger.info("🔥 position_manager loaded")
except Exception as e:
    PositionManager = None
    logger.warning(f"⚠️ position_manager missing: {e}")

try:
    from broker import router as broker_router
    # OJO: tu broker.py ya viene con prefix="/trading"
    # Si acá pones prefix="/broker", quedaría /broker/trading/...
    # Para NO romper, lo incluyo SIN prefix:
    app.include_router(broker_router)
    modules_status["broker"] = True
    logger.info("✅ broker router included")
except Exception as e:
    logger.warning(f"⚠️ broker router missing: {e}")

# =========================================================
# PositionManager cache
# =========================================================
async def get_position_manager() -> Optional["PositionManager"]:
    global _pm_cache
    if PositionManager is None:
        return None

    async with _pm_lock:
        if _pm_cache is None:
            try:
                _pm_cache = PositionManager(fixed_capital=config.FIXED_CAPITAL)
                logger.info(f"💼 PositionManager inicializado con capital: ${config.FIXED_CAPITAL:,.0f}")
            except Exception as e:
                logger.error(f"Error inicializando PositionManager: {e}")
                return None
        return _pm_cache

# =========================================================
# Universe API
# =========================================================
@app.get("/assets", summary="Lista todos los assets disponibles")
async def assets(
    limit: int = Query(config.BATCH_LIMIT_DEFAULT, ge=1, le=config.SIGNALS_MAX),
    refresh: bool = Query(False),
    _: Any = Depends(rate_limiter),
):
    if refresh:
        list_prediction_tickers_cached.cache_clear()

    all_tickers = list_prediction_tickers_cached()
    tickers = all_tickers[:limit]
    return {
        "assets": [{"ticker": t} for t in tickers],
        "count": len(tickers),
        "total_available": len(all_tickers),
        "source": "predictions",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_hits": list_prediction_tickers_cached.cache_info().hits,
    }

@app.get("/signals", summary="Obtiene señales ordenadas por confianza")
async def signals(
    min_confidence: float = Query(config.SIGNALS_MIN_CONF_DEFAULT, ge=0.0, le=1.0),
    limit: int = Query(2000, ge=1, le=config.SIGNALS_MAX),
    refresh: bool = Query(False),
    _: Any = Depends(rate_limiter),
):
    if refresh:
        list_prediction_tickers_cached.cache_clear()

    tickers = list_prediction_tickers_cached()
    out: List[Dict[str, Any]] = []

    for ticker in tickers:
        pred = latest_prediction_for_ticker(ticker)
        if not pred:
            continue

        p = pred.get("prediction", {}) or {}
        conf = safe_float(p.get("confidence")) or 0.0
        if conf < min_confidence:
            continue

        out.append({
            "ticker": ticker,
            "quality": p.get("quality", p.get("signal_quality", "NO_DATA")),
            "confidence": conf,
            "ret_ens_pct": safe_float(p.get("ret_ens_pct")) or 0.0,
            "date_base": p.get("date_base") or p.get("date"),
        })

    out.sort(key=lambda x: (-x["confidence"], -abs(x["ret_ens_pct"])))
    return {
        "signals": out[:limit],
        "count": len(out[:limit]),
        "filtered_by_confidence": min_confidence,
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# Health / Metrics
# =========================================================
@app.get("/health", summary="Estado del sistema")
async def health(_: Any = Depends(rate_limiter)):
    circuit_open = await broker_circuit.is_open()
    return {
        "status": "ok" if not circuit_open else "degraded",
        "broker_circuit_open": circuit_open,
        "broker_failures": broker_circuit.failures,
        "modules": modules_status,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/metrics", summary="Métricas del sistema")
async def metrics(_: Any = Depends(rate_limiter)):
    pm_positions = 0
    pm = await get_position_manager()
    if pm:
        pm_positions = len(getattr(pm, "positions", []))

    return {
        "pm_positions_open": pm_positions,
        "pm_capital_fixed": config.FIXED_CAPITAL,
        "broker_failures": broker_circuit.failures,
        "broker_circuit_open": await broker_circuit.is_open(),
        "rate_limiter_buckets": len(rate_limiter.buckets),
        "modules_status": modules_status,
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# Broker Circuit Management
# =========================================================
@app.post("/broker/reset-circuit", summary="Resetea el circuit breaker")
async def broker_reset(_: Any = Depends(rate_limiter)):
    await broker_circuit.reset()
    return {"status": "reset", "timestamp": datetime.utcnow().isoformat()}

@app.get("/broker/circuit", summary="Estado del circuit breaker")
async def broker_circuit_status(_: Any = Depends(rate_limiter)):
    return {
        "is_open": await broker_circuit.is_open(),
        "failures": broker_circuit.failures,
        "open_until": broker_circuit.open_until,
        "threshold": config.BROKER_FAILURE_THRESHOLD,
        "reset_after": config.BROKER_CIRCUIT_OPEN_SECS,
    }

# =========================================================
# Graceful Shutdown
# =========================================================
def handle_shutdown(signum: int, frame: Any):
    logger.info(f"Señal {signal.Signals(signum).name} recibida. Cerrando graceful...")
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(shutdown())
    except Exception:
        # en algunos entornos no hay loop aquí
        pass

async def shutdown():
    logger.info("Iniciando shutdown graceful...")
    await broker_circuit.reset()
    logger.info("Trading Suite Enterprise v2.5.2 detenido correctamente")

# =========================================================
# Run (local)
# =========================================================
if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=config.DEBUG_MODE,
        log_level="debug" if config.DEBUG_MODE else "info",
                )

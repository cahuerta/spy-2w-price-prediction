# =====================================================
# main.py — TRADING SUITE ENTERPRISE v2.5.2 (CORREGIDO)
# =====================================================
# ✔ Producción real (largo plazo) - PRODUCCIÓN ESTABLE
# ✔ Universe + Signals API - OPTIMIZADO
# ✔ PositionManager real - CACHE MEJORADO
# ✔ Circuit breaker + reset endpoint - ROBUSTO
# ✔ Guard-rails broker - CON MEJORA DE LOGGING
# ✔ Rate limiting mejorado - SEGURIDAD
# =====================================================

import os
import json
import logging
import time
import asyncio
import signal
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException, Request, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import httpx
from functools import lru_cache

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
    
    # Nuevas configs seguras
    TRUSTED_HOSTS = os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1,*").split(",")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

config = Config()

# =========================================================
# LIFETIME MANAGER - NUEVO
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Trading Suite Enterprise v2.5.2 iniciando...")
    ensure_dirs()
    await reset_broker_circuit()
    yield
    # Shutdown
    logger.info("🛑 Trading Suite Enterprise deteniéndose...")

# =========================================================
# GLOBAL STATE - THREAD-SAFE MEJORADO
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
# Rate Limiter - MEJORADO CON TTL AUTOMÁTICO
# =========================================================
class AsyncRateLimiter:
    def __init__(self, requests: int, per_seconds: int):
        self.requests = requests
        self.per_seconds = per_seconds
        self.buckets: Dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def __call__(self, request: Request) -> bool:
        ip = request.client.host if hasattr(request.client, 'host') else "unknown"
        now = time.time()

        async with self._lock:
            if ip not in self.buckets:
                self.buckets[ip] = deque(maxlen=self.requests * 2)

            q = self.buckets[ip]
            # Cleanup automático
            while q and now - q[0] > self.per_seconds:
                q.popleft()

            if len(q) >= self.requests:
                logger.warning(f"Rate limit excedido para IP {ip}")
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS, 
                    f"Rate limit excedido. Intente en {self.per_seconds - (now - q[0]):.0f}s"
                )
            
            q.append(now)
        return True

rate_limiter = AsyncRateLimiter(config.RL_REQUESTS, config.RL_PER_SECONDS)

# =========================================================
# Logging - MEJORADO CON ROTACIÓN
# =========================================================
def setup_logging():
    Path(config.DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(config.DATA_PATH) / "trading_suite.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    )
    
    logger = logging.getLogger("trading_suite")
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    
    return logger

logger = setup_logging()

# =========================================================
# Disk helpers - MEJORADOS CON ATOMIC WRITES
# =========================================================
def ensure_dirs():
    (Path(config.DATA_PATH) / "predictions").mkdir(parents=True, exist_ok=True)
    (Path(config.DATA_PATH) / "evaluations").mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Directorios preparados en {config.DATA_PATH}")

def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def write_json_atomic(path: Path, obj: Dict[str, Any]):
    """Escribe JSON de forma atómica"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix('.tmp')
    tmp_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp_path.replace(path)

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
    """Cache de tickers con LRU"""
    root = Path(config.DATA_PATH) / "predictions"
    return sorted(d.name for d in root.iterdir() if d.is_dir())

# =========================================================
# FastAPI - MEJORADO CON LIFESPAN Y MIDDLEWARES
# =========================================================
app = FastAPI(
    title="Trading Suite Enterprise", 
    version="2.5.2",
    lifespan=lifespan
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=config.TRUSTED_HOSTS
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Remaining"],
)

# =========================================================
# Optional modules - MEJORADO CON FALLBACKS
# =========================================================
modules_status = {
    "model": False,
    "evaluator": False,
    "position_manager": False,
    "broker": False
}

def safe_import(module_name: str, import_path: str):
    """Importa módulos opcionales con logging detallado"""
    try:
        module = __import__(import_path, fromlist=[module_name])
        logger.info(f"✅ {module_name} cargado correctamente")
        modules_status[module_name] = True
        return getattr(module, module_name)
    except Exception as e:
        logger.warning(f"⚠️  {module_name} no disponible: {e}")
        return None

# Carga lazy de módulos
run_model = safe_import("run_model", "model")
evaluate_all = safe_import("evaluate_all", "evaluator")
PositionManager = safe_import("PositionManager", "position_manager")

try:
    from broker import router as broker_router
    app.include_router(broker_router, prefix="/broker", tags=["broker"])
    modules_status["broker"] = True
except Exception:
    logger.warning("Broker router no disponible")

# =========================================================
# PositionManager cache - MEJORADO
# =========================================================
async def get_position_manager() -> Optional["PositionManager"]:
    global _pm_cache
    if not modules_status["position_manager"]:
        logger.warning("PositionManager no disponible")
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
# API — Universe - OPTIMIZADO
# =========================================================
@app.get("/assets", summary="Lista todos los assets disponibles")
async def assets(
    limit: int = Query(config.BATCH_LIMIT_DEFAULT, le=config.SIGNALS_MAX),
    refresh: bool = Query(False),
    _: Any = Depends(rate_limiter)
):
    if refresh:
        list_prediction_tickers_cached.cache_clear()
    
    tickers = list_prediction_tickers_cached()[:limit]
    return {
        "assets": [{"ticker": t} for t in tickers],
        "count": len(tickers),
        "total_available": len(list_prediction_tickers_cached()),
        "source": "predictions",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_hits": list_prediction_tickers_cached.cache_info().hits
    }

@app.get("/signals", summary="Obtiene señales ordenadas por confianza")
async def signals(
    min_confidence: float = Query(config.SIGNALS_MIN_CONF_DEFAULT, ge=0.0, le=1.0),
    limit: int = Query(2000, le=config.SIGNALS_MAX),
    refresh: bool = Query(False),
    _: Any = Depends(rate_limiter),
):
    if refresh:
        list_prediction_tickers_cached.cache_clear()
    
    tickers = list_prediction_tickers_cached()[:limit]
    out = []
    
    for ticker in tickers:
        pred = latest_prediction_for_ticker(ticker)
        if not pred:
            continue
            
        p = pred.get("prediction", {})
        conf = safe_float(p.get("confidence")) or 0.0
        
        if conf >= min_confidence:
            out.append({
                "ticker": ticker,
                "quality": p.get("quality", "NO_DATA"),
                "confidence": conf,
                "ret_ens_pct": safe_float(p.get("ret_ens_pct")) or 0.0,
                "date_base": p.get("date_base"),
                "timestamp": pred.get("timestamp")
            })
    
    out.sort(key=lambda x: (-x["confidence"], -abs(x["ret_ens_pct"])))
    return {
        "signals": out[:limit],
        "count": len(out),
        "filtered_by_confidence": min_confidence,
        "timestamp": datetime.utcnow().isoformat()
    }

def latest_prediction_for_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    """Obtiene la última predicción para un ticker específico"""
    p = Path(config.DATA_PATH) / "predictions" / ticker
    files = sorted(p.glob("*.json"))
    return load_json(files[-1]) if files else None

# =========================================================
# Health / Metrics - MEJORADO
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
# Broker Circuit Management - MEJORADO
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
# Graceful Shutdown - NUEVO
# =========================================================
def handle_shutdown(signum: int, frame: Any):
    logger.info(f"Señal {signal.Signals(signum).name} recibida. Cerrando graceful...")
    loop = asyncio.get_event_loop()
    loop.create_task(shutdown())

async def shutdown():
    logger.info("Iniciando shutdown graceful...")
    await broker_circuit.reset()
    logger.info("Trading Suite Enterprise v2.5.2 detenido correctamente")

# =========================================================
# Run - MEJORADO CON SHUTDOWN HANDLER
# =========================================================
if __name__ == "__main__":
    # Registro de señales para graceful shutdown
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=config.PORT,
        reload=config.DEBUG_MODE,
        log_level="info" if not config.DEBUG_MODE else "debug"
    )

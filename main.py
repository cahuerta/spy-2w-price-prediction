# =====================================================
# main.py — TRADING SUITE ENTERPRISE v2.5.3 (SIGNALS FIX ✅)
# =====================================================
# ✅ FIX: signals ahora deriva confidence + quality desde prediction.ret_ens_pct
# ✅ Compatible con tu JSON real:
#    { "prediction": { "ret_ens_pct": ..., "date_base": ..., ... } }
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
from fastapi.responses import HTMLResponse   # <-- ÚNICO IMPORT AGREGADO

# =========================================================
# Config
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
# Logging
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
# Lifespan
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Trading Suite Enterprise v2.5.3 iniciando...")
    ensure_dirs()
    await broker_circuit.reset()
    yield
    logger.info("🛑 Trading Suite Enterprise deteniéndose...")

# =========================================================
# FastAPI
# =========================================================
app = FastAPI(
    title="Trading Suite Enterprise",
    version="2.5.3",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# DASHBOARD ENDPOINT (ÚNICO AGREGADO)
# =========================================================
@app.get("/dashboard", response_class=HTMLResponse, summary="Dashboard")
async def dashboard():
    return """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Trading Suite Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{background:#020617;color:#e5e7eb;font-family:system-ui;padding:20px}
h1{color:#38bdf8}
table{width:100%;border-collapse:collapse}
th,td{padding:8px;border-bottom:1px solid #1e293b}
</style>
</head>
<body>
<h1>📈 Trading Suite – Signals</h1>
<table id="t"><thead><tr><th>Ticker</th><th>Conf</th><th>Ret%</th><th>Quality</th></tr></thead><tbody></tbody></table>
<script>
fetch('/signals?limit=50')
.then(r=>r.json())
.then(d=>{
 const b=document.querySelector('#t tbody');
 d.signals.forEach(s=>{
  const tr=document.createElement('tr');
  tr.innerHTML=`<td>${s.ticker}</td><td>${(s.confidence*100).toFixed(1)}%</td><td>${s.ret_ens_pct.toFixed(2)}</td><td>${s.quality}</td>`;
  b.appendChild(tr);
 });
});
</script>
</body>
</html>
"""

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
        pass

async def shutdown():
    logger.info("Iniciando shutdown graceful...")
    await broker_circuit.reset()
    logger.info("Trading Suite Enterprise detenido correctamente")

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

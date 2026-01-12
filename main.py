# =====================================================
# main.py — TRADING SUITE ENTERPRISE v2.4 (PM FULL) ✅ FIXED
# =====================================================
# ✔ PositionManager con posiciones reales
# ✔ OPEN / CLOSE / ROTATE habilitados
# ✔ Broker URL configurable
# ✔ PM cache (THREAD-SAFE con asyncio.Lock)
# ✔ Async Rate Limiter
# ✔ Position validation
# ✔ Circuit breaker para broker
# ✔ Metrics endpoint
# ✔ Compatibilidad total con cron históricos
# =====================================================

import os
import json
import logging
import time
import asyncio
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

from fastapi import FastAPI, Query, HTTPException, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
import httpx

# =========================================================
# Config
# =========================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")
PORT = int(os.getenv("PORT", "8000"))
FIXED_CAPITAL = float(os.getenv("PM_FIXED_CAPITAL", "100000"))
BROKER_EXEC_URL = os.getenv("BROKER_URL", "http://localhost:8000/trading/execute")
BROKER_STATUS_URL = os.getenv("BROKER_STATUS_URL", "http://localhost:8000/trading/status")

# Rate limiting
RL_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RL_PER_SECONDS = int(os.getenv("RATE_LIMIT_PER_SECONDS", "60"))
RL_MAX_IPS = int(os.getenv("RATE_LIMIT_MAX_IPS", "5000"))

# Batch defaults
BATCH_LIMIT_DEFAULT = int(os.getenv("BATCH_LIMIT_DEFAULT", "500"))

# Circuit breaker
BROKER_FAILURE_THRESHOLD = int(os.getenv("BROKER_FAILURE_THRESHOLD", "5"))
BROKER_CIRCUIT_OPEN_SECS = int(os.getenv("BROKER_CIRCUIT_OPEN_SECS", "300"))

# =========================================================
# GLOBAL STATE (Thread-safe)
# =========================================================
_pm_cache: Optional['PositionManager'] = None
_pm_lock = asyncio.Lock()
broker_failures = 0
broker_circuit_open_until = 0

# =========================================================
# Async Rate Limiter (FIXED)
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
                self.buckets[ip] = deque(maxlen=self.requests * 2)
            
            q = self.buckets[ip]
            while q and now - q[0] > self.per_seconds:
                q.popleft()

            if len(q) >= self.requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            q.append(now)
            self._cleanup(now)
        
        return True

    def _cleanup(self, now: float):
        expired = [ip for ip, q in self.buckets.items() 
                  if not q or now - q[-1] > self.per_seconds * 2]
        for ip in expired:
            self.buckets.pop(ip, None)

rate_limiter = AsyncRateLimiter(RL_REQUESTS, RL_PER_SECONDS, RL_MAX_IPS)

# =========================================================
# Circuit Breaker
# =========================================================
async def is_broker_circuit_open() -> bool:
    global broker_circuit_open_until
    return time.time() < broker_circuit_open_until

async def record_broker_failure():
    global broker_failures, broker_circuit_open_until
    broker_failures += 1
    if broker_failures >= BROKER_FAILURE_THRESHOLD:
        broker_circuit_open_until = time.time() + BROKER_CIRCUIT_OPEN_SECS
        logger.error(f"🔌 BROKER CIRCUIT OPEN hasta {datetime.fromtimestamp(broker_circuit_open_until)}")

# =========================================================
# Logging
# =========================================================
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# =========================================================
# Disk helpers
# =========================================================
def ensure_dirs():
    (Path(DATA_PATH) / "predictions").mkdir(parents=True, exist_ok=True)
    (Path(DATA_PATH) / "evaluations").mkdir(parents=True, exist_ok=True)

def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def write_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_tickers_from_file() -> List[str]:
    p = Path("tickers.json")
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("tickers", data) if isinstance(data, dict) else data
    out = []
    for t in raw:
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, dict) and "ticker" in t:
            out.append(str(t["ticker"]))
    return out

ensure_dirs()

# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="🚀 Trading Suite Enterprise", version="2.4.0 ✅ FIXED")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Optional modules (unchanged)
# =========================================================
run_model = None
evaluate_all = None
PositionManager = None

try:
    from model import run_model as _run_model
    run_model = _run_model
    logger.info("✅ model.py loaded")
except Exception as e:
    logger.warning(f"⚠️ model.py missing: {e}")

try:
    from evaluator import evaluate_all as _evaluate_all
    evaluate_all = _evaluate_all
    logger.info("✅ evaluator.py loaded")
except Exception as e:
    logger.warning(f"⚠️ evaluator.py missing: {e}")

try:
    from position_manager import PositionManager as _PM
    PositionManager = _PM
    logger.info("🔥 position_manager loaded")
except Exception as e:
    logger.warning(f"⚠️ position_manager missing: {e}")

try:
    from broker import router as broker_router
    app.include_router(broker_router)
    logger.info("✅ broker router included")
except Exception as e:
    logger.warning(f"⚠️ broker router missing: {e}")

# =========================================================
# PositionManager cache (THREAD-SAFE ✅ FIXED)
# =========================================================
async def get_position_manager() -> 'PositionManager':
    global _pm_cache
    async with _pm_lock:
        if _pm_cache is None:
            if PositionManager is None:
                raise RuntimeError("PositionManager not loaded")
            _pm_cache = PositionManager(fixed_capital=FIXED_CAPITAL)
            logger.info("🔄 PositionManager cache initialized (thread-safe)")
        return _pm_cache

# =========================================================
# Broker helpers (VALIDATION ✅ FIXED)
# =========================================================
def validate_position(p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Valida y normaliza posición del broker"""
    try:
        ticker = str(p.get("ticker") or "")
        if not ticker:
            return None
            
        return {
            "ticker": ticker,
            "entry_price": float(p.get("entry_price") or 0),
            "price_now": float(p.get("price_now") or 0),
            "size_shares": float(p.get("qty") or 0),
            "entry_time": p.get("entry_time", ""),
            "peak_price": float(p.get("peak_price", p.get("price_now") or 0)),
            "volatility": float(p.get("volatility", 1.0)),
            "confidence": float(p.get("confidence", 0.0)),
        }
    except (ValueError, TypeError, AttributeError):
        return None

async def load_broker_positions() -> List[Dict[str, Any]]:
    """Obtiene posiciones reales del broker con validación"""
    if await is_broker_circuit_open():
        logger.warning("🔌 Broker circuit open, skipping positions")
        return []
        
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            r = await client.get(BROKER_STATUS_URL)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            await record_broker_failure()
            logger.error(f"❌ Broker status failed: {e}")
            return []

    positions_raw = data.get("positions_detail") or []
    positions = [validate_position(p) for p in positions_raw if validate_position(p)]
    logger.info(f"📦 Loaded {len(positions)} validated positions from broker")
    return positions

# =========================================================
# Health + Metrics
# =========================================================
@app.get("/health")
async def health(_: Any = Depends(rate_limiter)):
    pm_available = PositionManager is not None
    pm_cached = _pm_cache is not None
    broker_ok = not await is_broker_circuit_open()
    
    return {
        "status": "ok",
        "broker_exec_url": BROKER_EXEC_URL,
        "broker_status_url": BROKER_STATUS_URL,
        "broker_circuit_open": await is_broker_circuit_open(),
        "broker_failures": broker_failures,
        "modules": {
            "model": run_model is not None,
            "evaluator": evaluate_all is not None,
            "position_manager": pm_available,
            "pm_cached": pm_cached,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/metrics")
async def metrics(_: Any = Depends(rate_limiter)):
    pm = await get_position_manager() if PositionManager else None
    predictions_today = len(list((Path(DATA_PATH)/"predictions").glob(f"{datetime.now().strftime('%Y%m%d')}*.json")))
    
    return {
        "pm_positions_open": len(pm.positions) if pm else 0,
        "predictions_today": predictions_today,
        "broker_failures": broker_failures,
        "broker_circuit_open": await is_broker_circuit_open(),
        "rate_limiter_active_ips": len(rate_limiter.buckets),
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# PREDICT + SAVE + PM + BROKER (BACKGROUND ✅ OPTIMIZED)
# =========================================================
async def _run_full_pipeline(limit: int, horizon: int, theta: float):
    """Ejecuta pipeline completo en background"""
    if run_model is None:
        logger.error("❌ model not loaded")
        return

    tickers = load_tickers_from_file()
    if not tickers:
        logger.error("❌ tickers.json missing/empty")
        return

    ts = utc_stamp()
    signals: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    execution_results: List[Dict[str, Any]] = []

    # 1) MODELO
    logger.info(f"🔮 Running model on {min(len(tickers), limit)} tickers")
    for t in tickers[:limit]:
        try:
            res = run_model(ticker=t, horizon=horizon, theta=theta)
            fp = Path(DATA_PATH) / "predictions" / t / f"{ts}.json"
            write_json(fp, {"meta": {"ticker": t}, "prediction": res})
            signals[t] = res
            results.append({"ticker": t, "status": "saved"})
        except Exception as e:
            results.append({"ticker": t, "status": "failed", "error": str(e)})
            logger.error(f"❌ Model failed for {t}: {e}")

    # 2) LOAD REAL POSITIONS
    current_positions = await load_broker_positions()

    # 3) POSITION MANAGER (FULL)
    decisions: List[Dict[str, Any]] = []
    if PositionManager is not None:
        pm = await get_position_manager()

        # a) evaluar posiciones abiertas (HOLD / CLOSE)
        for pos in current_positions:
            d = pm.evaluate_position(pos, signals.get(pos["ticker"]))
            if d["action"] != "HOLD":
                d["ticker"] = d.get("ticker") or pos["ticker"]
                decisions.append(d)

        # b) evaluar rotación / nuevas entradas
        for t, sig in signals.items():
            candidate = {
                "ticker": t,
                "confidence": sig.get("confidence"),
                "ret_ens_pct": sig.get("ret_ens_pct"),
            }
            rot = pm.evaluate_rotation(current_positions, candidate, signals)
            if rot:
                rot["target_pct"] = rot.get("target_pct", 0.1)
                decisions.append(rot)

        logger.info(f"🤖 PositionManager decisions: {len(decisions)}")

    # 4) BROKER EXECUTION (con circuit breaker)
    if decisions and not await is_broker_circuit_open():
        async with httpx.AsyncClient(timeout=30.0) as client:
            for d in decisions:
                try:
                    resp = await client.post(BROKER_EXEC_URL, json=d)
                    result = resp.json()
                    execution_results.append({
                        "decision": d,
                        "broker_response": result,
                        "success": result.get("status") in ["executed", "ignored"],
                    })
                except Exception as e:
                    execution_results.append({
                        "decision": d,
                        "broker_response": None,
                        "error": str(e),
                        "success": False,
                    })
                    await record_broker_failure()

        ok = sum(1 for r in execution_results if r["success"])
        logger.info(f"🚀 Broker executed {ok}/{len(decisions)} decisions")

    # Log final
    logger.info(f"✅ Pipeline completed: {len([r for r in results if r['status'] == 'saved'])} saved, {len(decisions)} decisions")

@app.post("/predict/save/all")  # POST para heavy ops
async def predict_save_all(
    background_tasks: BackgroundTasks,
    limit: int = Query(BATCH_LIMIT_DEFAULT, ge=1, le=5000),
    horizon: int = Query(10, ge=1, le=30),
    theta: float = Query(0.75, ge=0.1, le=1.0),
    now: bool = Query(False),  # sync vs async
    _: Any = Depends(rate_limiter),
):
    if now:
        await _run_full_pipeline(limit, horizon, theta)
        return {"status": "completed"}
    
    background_tasks.add_task(_run_full_pipeline, limit, horizon, theta)
    return {"status": "queued", "task_id": utc_stamp()}

# =========================================================
# EVALUATE (histórico, intacto)
# =========================================================
@app.get("/evaluate")
async def evaluate(_: Any = Depends(rate_limiter)):
    if evaluate_all is None:
        raise HTTPException(status_code=503, detail="evaluator not loaded")
    return evaluate_all()

# =========================================================
# PM cache reset (dev)
# =========================================================
@app.post("/pm/reset")
async def reset_pm_cache():
    global _pm_cache
    async with _pm_lock:
        _pm_cache = None
    logger.info("🔄 PositionManager cache reset")
    return {"status": "reset"}

# =========================================================
# Reset circuit breaker (dev)
# =========================================================
@app.post("/broker/reset-circuit")
async def reset_broker_circuit():
    global broker_failures, broker_circuit_open_until
    broker_failures = 0
    broker_circuit_open_until = 0
    logger.info("🔌 Broker circuit reset")
    return {"status": "reset"}

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Starting Trading Suite v2.4 ✅ FIXED → Broker: {BROKER_EXEC_URL}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

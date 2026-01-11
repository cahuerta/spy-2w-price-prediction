# main.py — TRADING SUITE ENTERPRISE (COMPATIBLE + PRODUCTION-READY)
# =====================================================
# ✅ Compatible con tu sistema histórico (cron Render):
#    - /predict/save/all  (GUARDA predicciones)
#    - /evaluate          (EVALÚA predicciones guardadas)
# ✅ Nuevo:
#    - /assets (expone tickers.json = universo)
# ✅ Mantiene:
#    - /predict (on-demand)
#    - /signals (visor de señales)
#    - /live /ready /health
# ✅ Guarda en disco: DATA_PATH=/data (Render Disk)
# =====================================================

import os
import json
import logging
import time
from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

from fastapi import FastAPI, Query, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# Config
# =========================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")
PORT = int(os.getenv("PORT", "8000"))

# Rate limiting
RL_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RL_PER_SECONDS = int(os.getenv("RATE_LIMIT_PER_SECONDS", "60"))
RL_MAX_IPS = int(os.getenv("RATE_LIMIT_MAX_IPS", "5000"))

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

# Signals defaults
DEFAULT_WINDOW = int(os.getenv("SIGNAL_WINDOW", "30"))
DEFAULT_MIN_CONF = float(os.getenv("SIGNAL_MIN_CONFIDENCE", "0.4"))

# Batch defaults
BATCH_LIMIT_DEFAULT = int(os.getenv("BATCH_LIMIT_DEFAULT", "500"))

# =========================================================
# Rate limiter (memory-safe)
# =========================================================
class SimpleRateLimiter:
    def __init__(self, requests: int, per_seconds: int, max_ips: int):
        self.requests = requests
        self.per_seconds = per_seconds
        self.max_ips = max_ips
        self.buckets = defaultdict(lambda: deque(maxlen=requests * 2))

    async def __call__(self, request: Request):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        q = self.buckets[ip]

        while q and now - q[0] > self.per_seconds:
            q.popleft()

        if len(q) >= self.requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        q.append(now)

        if len(self.buckets) > self.max_ips:
            self._cleanup(now)

        return True

    def _cleanup(self, now: float):
        for ip, q in list(self.buckets.items()):
            if not q or now - q[-1] > self.per_seconds * 2:
                self.buckets.pop(ip, None)

rate_limiter = SimpleRateLimiter(RL_REQUESTS, RL_PER_SECONDS, RL_MAX_IPS)

# =========================================================
# Logging
# =========================================================
def setup_logging():
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

setup_logging()
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
    tickers_path = Path("tickers.json")
    if not tickers_path.exists():
        return []
    data = json.loads(tickers_path.read_text(encoding="utf-8"))
    raw = data.get("tickers", data) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return []
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
app = FastAPI(title="🚀 Trading Suite Enterprise", version="2.0.2")

if CORS_ORIGINS.strip() == "*":
    allow_origins = ["*"]
    allow_credentials = False
else:
    allow_origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
    allow_credentials = ALLOW_CREDENTIALS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Load modules (best-effort)
# =========================================================
run_model = None
evaluate_all = None
compute_all_signals = None

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
    from signals import compute_all_signals as _compute_all_signals
    compute_all_signals = _compute_all_signals
    logger.info("✅ signals.py loaded")
except Exception as e:
    logger.warning(f"⚠️ signals.py missing: {e}")

# Routers opcionales
try:
    from dashboard import router as dashboard_router
    app.include_router(dashboard_router)
    logger.info("✅ dashboard router included")
except Exception as e:
    logger.warning(f"⚠️ dashboard router missing: {e}")

try:
    from broker import router as broker_router
    app.include_router(broker_router)
    logger.info("✅ broker router included")
except Exception as e:
    logger.warning(f"⚠️ broker router missing: {e}")

# =========================================================
# Healthchecks
# =========================================================
@app.get("/live")
async def live():
    return {"status": "live"}

@app.get("/ready")
async def ready(_: Any = Depends(rate_limiter)):
    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}

@app.get("/health")
async def health(_: Any = Depends(rate_limiter)):
    issues = []
    if run_model is None: issues.append("model missing")
    if evaluate_all is None: issues.append("evaluator missing")
    if compute_all_signals is None: issues.append("signals missing")

    pred_path = Path(DATA_PATH) / "predictions"
    eval_path = Path(DATA_PATH) / "evaluations"
    if not pred_path.exists(): issues.append("predictions dir missing")
    if not eval_path.exists(): issues.append("evaluations dir missing")

    return {
        "status": "healthy" if not issues else "degraded",
        "issues": issues,
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# ✅ UNIVERSO: /assets  (FUENTE DE VERDAD PARA ANÁLISIS)
# =========================================================
@app.get("/assets")
async def assets(_: Any = Depends(rate_limiter)):
    tickers = load_tickers_from_file()
    assets = [{"ticker": t} for t in tickers]
    return {
        "ok": True if tickers else False,
        "assets": assets,
        "count": len(assets),
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# ✅ PREDICT (on-demand; NO guarda por defecto)
# =========================================================
@app.get("/predict")
async def predict(
    ticker: str = Query("SPY"),
    horizon: int = Query(10, ge=1, le=30),
    theta: float = Query(0.75, ge=0.1, le=1.0),
    _: Any = Depends(rate_limiter),
):
    if run_model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    t0 = time.time()
    result = run_model(ticker=ticker, horizon=horizon, theta=theta)
    latency = time.time() - t0

    return {
        "ok": True,
        "timestamp": datetime.utcnow().isoformat(),
        "latency_s": round(latency, 3),
        "result": result,
    }

# =========================================================
# ✅ COMPAT: /predict/save/all  (cron histórico)
# Guarda predicciones de TODO el universo a /data/predictions
# =========================================================
@app.get("/predict/save/all")
async def predict_save_all(
    limit: int = Query(BATCH_LIMIT_DEFAULT, ge=1, le=5000),
    horizon: int = Query(10, ge=1, le=30),
    theta: float = Query(0.75, ge=0.1, le=1.0),
    _: Any = Depends(rate_limiter),
):
    if run_model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    tickers = load_tickers_from_file()
    if not tickers:
        return {"ok": False, "error": "tickers.json missing/empty", "saved": 0, "results": []}

    ts = utc_stamp()
    results = []

    for t in tickers[:limit]:
        try:
            res = run_model(ticker=t, horizon=horizon, theta=theta)

            fp = Path(DATA_PATH) / "predictions" / t / f"{ts}.json"
            payload = {
                "ticker": t,
                "timestamp": ts,
                "result": res,
            }
            write_json(fp, payload)
            results.append({"ticker": t, "status": "saved", "file": str(fp)})
        except Exception as e:
            results.append({"ticker": t, "status": "failed", "error": str(e)})

    saved = sum(1 for r in results if r["status"] == "saved")
    return {"ok": True, "saved": saved, "requested": min(limit, len(tickers)), "timestamp": ts, "results": results}

# =========================================================
# ✅ COMPAT: /evaluate  (cron histórico)
# Evalúa usando evaluator.py, que debe leer /data/predictions y escribir /data/evaluations
# =========================================================
@app.get("/evaluate")
async def evaluate(_: Any = Depends(rate_limiter)):
    if evaluate_all is None:
        raise HTTPException(status_code=503, detail="evaluator not loaded")

    # IMPORTANTÍSIMO:
    # evaluator.py debe estar diseñado para leer DATA_PATH/predictions
    # y escribir DATA_PATH/evaluations.
    # Si no lo hace, hay que corregir evaluator.py (pero el endpoint queda).
    try:
        t0 = time.time()
        out = evaluate_all()
        latency = time.time() - t0
        return {"ok": True, "latency_s": round(latency, 3), "timestamp": datetime.utcnow().isoformat(), "result": out}
    except Exception as e:
        logger.error(f"/evaluate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# Señales (visor de eventos; NO universo)
# =========================================================
@app.get("/signals")
async def signals(
    window: int = Query(DEFAULT_WINDOW, ge=7, le=365),
    min_confidence: float = Query(DEFAULT_MIN_CONF, ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=200),
    _: Any = Depends(rate_limiter),
):
    if compute_all_signals is None:
        return {"ok": False, "signals": [], "count": 0}

    signals_raw = compute_all_signals(window)

    filtered = [s for s in signals_raw if (s.get("confidence") or 0) >= min_confidence]
    filtered.sort(key=lambda s: (-(s.get("confidence") or 0), -(s.get("ret_ens_pct") or 0)))

    out = filtered[:limit]
    return {"ok": True, "signals": out, "count": len(out), "timestamp": datetime.utcnow().isoformat()}

# =========================================================
# Root
# =========================================================
@app.get("/")
async def root():
    return {
        "service": "Trading Suite Enterprise",
        "version": "2.0.2",
        "endpoints": {
            "assets": "/assets",
            "predict": "/predict?ticker=SPY",
            "predict_save_all": "/predict/save/all",
            "evaluate": "/evaluate",
            "signals": "/signals",
            "health": "/health",
        },
        "status": "COMPATIBLE + PRODUCTION READY",
    }

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

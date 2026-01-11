# main.py — TRADING SUITE ENTERPRISE (PRODUCTION-READY)
# =====================================================
# ✔ model.py + evaluator.py + signals.py + broker.py + dashboard.py
# ✔ Healthchecks (live/ready/health)
# ✔ Rate limiting memory-safe
# ✔ CORS seguro
# ✔ Signals NO tocado (solo eventos)
# ✔ /assets expone tickers.json (universo)
# ✔ ✅ LEGACY RESTAURADO: /predict/save/all (CRON COMPATIBLE)

import os
import json
import logging
import time
from typing import Dict, Any
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

RL_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RL_PER_SECONDS = int(os.getenv("RATE_LIMIT_PER_SECONDS", "60"))
RL_MAX_IPS = int(os.getenv("RATE_LIMIT_MAX_IPS", "5000"))

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

DEFAULT_WINDOW = int(os.getenv("SIGNAL_WINDOW", "30"))
DEFAULT_MIN_CONF = float(os.getenv("SIGNAL_MIN_CONFIDENCE", "0.4"))

# =========================================================
# Rate limiter
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
# FastAPI
# =========================================================
app = FastAPI(
    title="🚀 Trading Suite Enterprise",
    version="2.0.1",
)

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
# Load modules
# =========================================================
ALL_MODULES_LOADED = False
run_model = evaluate_all = compute_all_signals = None

try:
    from model import run_model
    from evaluator import evaluate_all
    from signals import compute_all_signals
    ALL_MODULES_LOADED = True
    logger.info("✅ Core modules loaded")
except Exception as e:
    logger.warning(f"⚠️ Core modules missing: {e}")

try:
    from dashboard import router as dashboard_router
    app.include_router(dashboard_router)
except Exception as e:
    logger.warning(f"⚠️ dashboard router missing: {e}")

try:
    from broker import router as broker_router
    app.include_router(broker_router)
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
    return {
        "status": "healthy" if ALL_MODULES_LOADED else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# ASSETS / UNIVERSE
# =========================================================
@app.get("/assets")
async def get_assets(_: Any = Depends(rate_limiter)):
    tickers_path = Path("tickers.json")

    if not tickers_path.exists():
        return {"ok": False, "assets": [], "count": 0}

    with open(tickers_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("tickers", data) if isinstance(data, dict) else data
    assets = [{"ticker": t} if isinstance(t, str) else t for t in raw]

    return {
        "ok": True,
        "assets": assets,
        "count": len(assets),
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# 🔁 LEGACY ENDPOINT — CRON COMPATIBLE
# =========================================================
@app.post("/predict/save/all")
async def predict_save_all(_: Any = Depends(rate_limiter)):
    """
    🔁 ENDPOINT LEGACY (NO SE ELIMINA)
    - Usado por CRON de Render
    - Ejecuta run_model para TODO el universo
    - Guarda JSON diarios en /data/predictions/<TICKER>/
    """

    if run_model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    tickers_path = Path("tickers.json")
    if not tickers_path.exists():
        raise HTTPException(status_code=404, detail="tickers.json missing")

    with open(tickers_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tickers = data.get("tickers", data) if isinstance(data, dict) else data

    results = []
    for ticker in tickers:
        try:
            result = run_model(ticker=ticker)

            pred_dir = Path(DATA_PATH) / "predictions" / ticker
            pred_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            fp = pred_dir / f"{ts}.json"
            fp.write_text(json.dumps(result), encoding="utf-8")

            results.append({"ticker": ticker, "status": "saved"})
        except Exception as e:
            results.append({"ticker": ticker, "status": "error", "error": str(e)})

    return {
        "ok": True,
        "saved": len([r for r in results if r["status"] == "saved"]),
        "results": results,
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# Signals (EVENTOS — NO universo)
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

    raw = compute_all_signals(window)

    filtered = [
        s for s in raw
        if (s.get("confidence") or 0) >= min_confidence
    ]

    filtered.sort(
        key=lambda s: (
            -(s.get("confidence") or 0),
            -(s.get("ret_ens_pct") or 0),
        )
    )

    return {
        "ok": True,
        "signals": filtered[:limit],
        "count": len(filtered),
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# Root
# =========================================================
@app.get("/")
async def root():
    return {
        "service": "Trading Suite Enterprise",
        "version": "2.0.1",
        "endpoints": {
            "assets": "/assets",
            "predict_save_all": "/predict/save/all",
            "signals": "/signals",
            "health": "/health",
        },
        "status": "🚀 PRODUCTION READY",
    }

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

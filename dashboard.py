# dashboard.py — VERSIÓN FINAL CORREGIDA (ENTERPRISE / MODELO 3)
# 🔹 LEE SOLO DESDE /data/predictions
# 🔹 signals.py NO es fuente de análisis
# 🔹 Frontend usa predictions como VERDAD
# 🔹 signals = visor secundario (pestaña señales)

import os
import json
import logging
import sys
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from functools import lru_cache

import numpy as np
from fastapi import (
    FastAPI,
    APIRouter,
    Query,
    HTTPException,
    Request,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================================================
# Config
# =========================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")
MIN_CONFIDENCE = float(os.getenv("SIGNAL_MIN_CONFIDENCE", "0.4"))

# =========================================================
# Logging
# =========================================================
def setup_logging():
    level = logging.INFO
    try:
        log_path = Path(DATA_PATH) / "dashboard.log"
        log_path.parent.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_path)
    except Exception:
        fh = logging.NullHandler()

    sh = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        handlers=[fh, sh],
    )

setup_logging()
logger = logging.getLogger(__name__)

# =========================================================
# Rate Limiter
# =========================================================
class SimpleRateLimiter:
    def __init__(self, requests=20, per_seconds=60, max_ips=5000):
        self.requests = requests
        self.per_seconds = per_seconds
        self.max_ips = max_ips
        self.buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=requests * 2))

    async def __call__(self, request: Request):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        q = self.buckets[ip]

        while q and now - q[0] > self.per_seconds:
            q.popleft()

        if len(q) >= self.requests:
            raise HTTPException(429, "Rate limit exceeded")

        q.append(now)
        return True

rate_limiter = SimpleRateLimiter()

# =========================================================
# Router
# =========================================================
router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# =========================================================
# Models
# =========================================================
class TickerStats(BaseModel):
    ticker: str
    n_predictions: int = 0
    n_evaluations: int = 0

# =========================================================
# Helpers
# =========================================================
@lru_cache(maxsize=2)
def list_tickers(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

# =========================================================
# ENDPOINTS
# =========================================================
@router.get("/status")
async def status(_: Any = Depends(rate_limiter)):
    pred = Path(DATA_PATH) / "predictions"
    return {
        "status": "ok",
        "predictions_exists": pred.exists(),
        "n_tickers": len(list_tickers(str(pred))),
        "timestamp": datetime.utcnow().isoformat(),
    }

@router.get("/tickers", response_model=List[TickerStats])
async def tickers(_: Any = Depends(rate_limiter)):
    pred_path = Path(DATA_PATH) / "predictions"
    out = []
    for t in list_tickers(str(pred_path)):
        n = len(list((pred_path / t).glob("*.json")))
        out.append(TickerStats(ticker=t, n_predictions=n))
    return out

# =========================================================
# 🔑 ENDPOINT CLAVE — ESTE ERA EL QUE FALTABA
# =========================================================
@router.get("/predictions/summary")
async def prediction_summary(
    ticker: str,
    limit: int = Query(60, ge=1, le=500),
    _: Any = Depends(rate_limiter),
):
    """
    📈 Fuente ÚNICA del análisis.
    LEE datos históricos ya calculados por model.py
    """
    pred_dir = Path(DATA_PATH) / "predictions" / ticker
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
# App
# =========================================================
app = FastAPI(title="Trading Dashboard API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

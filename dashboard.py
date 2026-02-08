# =========================================================
# dashboard.py — ENTERPRISE DASHBOARD MODULE (PRODUCCIÓN)
# =========================================================
# 🔹 LEE SOLO DESDE /data/predictions
# 🔹 signals.py NO es fuente de análisis
# 🔹 Frontend usa predictions como VERDAD
# 🔹 signals = visor secundario (pestaña señales)
# 🔹 ESTE ARCHIVO NO MONTA FASTAPI (solo router)
# =========================================================

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

from fastapi import (
    APIRouter,
    Query,
    HTTPException,
    Request,
    Depends,
)
from pydantic import BaseModel

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")
MIN_CONFIDENCE = float(os.getenv("SIGNAL_MIN_CONFIDENCE", "0.4"))

# =========================================================
# LOGGING (AISLADO)
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
logger = logging.getLogger("dashboard")

# =========================================================
# RATE LIMITER (LOCAL)
# =========================================================
class SimpleRateLimiter:
    def __init__(self, requests=20, per_seconds=60):
        self.requests = requests
        self.per_seconds = per_seconds
        self.buckets: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=requests * 2)
        )

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
# ROUTER (CONTRATO FIJO)
# =========================================================
router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# =========================================================
# MODELS
# =========================================================
class TickerStats(BaseModel):
    ticker: str
    n_predictions: int = 0

# =========================================================
# HELPERS
# =========================================================
@lru_cache(maxsize=2)
def list_tickers(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

# =========================================================
# ENDPOINTS (CONTRATO FRONTEND INTACTO)
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
    out: List[TickerStats] = []

    for t in list_tickers(str(pred_path)):
        n = len(list((pred_path / t).glob("*.json")))
        out.append(TickerStats(ticker=t, n_predictions=n))

    return out

# =========================================================
# 🔑 ENDPOINT CLAVE — FUENTE ÚNICA DEL FRONTEND
# =========================================================
@router.get("/predictions/summary")
async def prediction_summary(
    ticker: str,
    limit: int = Query(60, ge=1, le=500),
    _: Any = Depends(rate_limiter),
):
    """
    📈 Fuente ÚNICA del análisis.
    Lee datos YA calculados por model.py
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

        p = obj.get("prediction")
        if not isinstance(p, dict):
            continue

        data.append({
            "date_base": p.get("date_base"),
            "price_now": safe_float(p.get("price_now")),
            "price_pred": safe_float(p.get("price_pred")),
            "ret_ens_pct": safe_float(p.get("ret_ens_pct")),
            "recommendation": p.get("recommendation"),
        })

    if not data:
        raise HTTPException(404, "No usable prediction data")

    return {
        "ticker": ticker,
        "count": len(data),
        "data": data,
    }
@router.get("/latest/{ticker}")
async def latest_snapshot(
    ticker: str,
    _: Any = Depends(rate_limiter),
):
    pred_dir = Path(DATA_PATH) / "predictions" / ticker
    if not pred_dir.exists():
        raise HTTPException(404, f"No predictions for {ticker}")

    files = sorted(pred_dir.glob("*.json"))
    if not files:
        raise HTTPException(404, f"No prediction files for {ticker}")

    last = load_json(files[-1])
    if not last:
        raise HTTPException(404, "Invalid prediction file")

    return {
        "ticker": ticker,
        "latest": last
    }


# =========================================================
# SCREENER (READ ONLY)
# =========================================================
@router.get("/screener")
async def screener(_: Any = Depends(rate_limiter)):
    p = Path(DATA_PATH) / "screener" / "screener_latest.json"
    if not p.exists():
        raise HTTPException(404, "Screener not available")

    data = load_json(p)
    if not data:
        raise HTTPException(500, "Invalid screener file")

    return data

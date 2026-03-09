# =========================================================
# dashboard.py — ENTERPRISE DASHBOARD MODULE (PRODUCCIÓN)
# =========================================================
# 🔹 LEE SOLO DESDE /data/predictions
# 🔹 signals.py NO es fuente de análisis
# 🔹 Frontend usa predictions como VERDAD
# 🔹 signals = visor secundario (pestaña señales)
# 🔹 ESTE ARCHIVO NO MONTA FASTAPI (solo router)
# 🔹 RATE LIMIT SOLO EN ENDPOINT PESADO (summary)
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
from trading_orchestrator import TradingOrchestrator
from fastapi import (
    APIRouter,
    Query,
    HTTPException,
    Request,
    Depends,
)

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")

# =========================================================
# LOGGING
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
# RATE LIMITER (SOLO PARA SUMMARY)
# =========================================================
class SimpleRateLimiter:
    def __init__(self, requests=30, per_seconds=60):
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
# ROUTER
# =========================================================
router = APIRouter(prefix="/dashboard", tags=["dashboard"])

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
# STATUS (SIN RATE LIMIT)
# =========================================================
@router.get("/status")
async def status():
    pred = Path(DATA_PATH) / "predictions"
    return {
        "status": "ok",
        "predictions_exists": pred.exists(),
        "n_tickers": len(list_tickers(str(pred))),
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# TICKERS (SIN RATE LIMIT)
# =========================================================
@router.get("/tickers")
async def tickers():
    pred_path = Path(DATA_PATH) / "predictions"
    return {
        "tickers": list_tickers(str(pred_path))
    }

# =========================================================
# SUMMARY (ÚNICO CON RATE LIMIT)
# =========================================================
@router.get("/predictions/summary")
async def prediction_summary(
    ticker: str,
    limit: int = Query(60, ge=1, le=500),
    _: Any = Depends(rate_limiter),
):
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

# =========================================================
# LATEST (SIN RATE LIMIT)
# =========================================================
@router.get("/latest/{ticker}")
async def latest_snapshot(ticker: str):
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
# EVALUATION LATEST (SIN RATE LIMIT)
# LEE DESDE /data/evaluations/<TICKER>/*.json
# =========================================================
@router.get("/evaluation-latest/{ticker}")
async def evaluation_latest(ticker: str):
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise HTTPException(400, "ticker requerido")

    eval_dir = Path(DATA_PATH) / "evaluations" / ticker

    if not eval_dir.exists():
        raise HTTPException(404, f"No evaluations for {ticker}")

    files = sorted(eval_dir.glob("*.json"), key=lambda p: p.name)
    if not files:
        raise HTTPException(404, f"No evaluation files for {ticker}")

    last_fp = files[-1]
    last = load_json(last_fp)
    if not last:
        raise HTTPException(404, "Invalid evaluation file")

    return {
        "ticker": ticker,
        "evaluation": last,
        "file": last_fp.name,
    }

# =========================================================
# SCREENER (SIN RATE LIMIT)
# =========================================================
@router.get("/screener")
async def screener():
    p = Path(DATA_PATH) / "screener_candidates.json"
    if not p.exists():
        raise HTTPException(404, "Screener not available")

    data = load_json(p)
    if not data:
        raise HTTPException(500, "Invalid screener file")

    return data

# =========================================================
# MARKET CONTEXT (SIN RATE LIMIT)
# =========================================================
@router.get("/market-context")
async def market_context():
    p = Path(DATA_PATH) / "market_context.json"

    if not p.exists():
        raise HTTPException(404, "Market context not available")

    data = load_json(p)
    if not data:
        raise HTTPException(500, "Invalid market context file")

    return data

@router.get("/executability-preview")
async def executability_preview():
    from trading_orchestrator import TradingOrchestrator

    orchestrator = TradingOrchestrator()

    market_ctx_path = Path(DATA_PATH) / "market_context.json"
    if not market_ctx_path.exists():
        raise HTTPException(404, "market_context.json not found")

    market_ctx = json.loads(market_ctx_path.read_text())

    return await orchestrator.preview_executability(
        market_ctx=market_ctx
    )

@router.get("/universe")
async def universe():
    from portfolio_store import load_positions

    # 1. Usar el archivo exacto que usa el orquestador
    ALPHA_FILE = Path(DATA_PATH) / "alpha_last.json"
    market_ctx_path = Path(DATA_PATH) / "market_context.json"

    # 2. Cargar Alpha
    alpha_data = load_json(ALPHA_FILE) or {}
    alpha_map = alpha_data.get("results", {})

    # 3. Cargar Posiciones y Contexto
    positions_list = load_positions()

    positions = {
        p.get("ticker", "").upper(): p
        for p in positions_list
        if "ticker" in p
    }
    market_ctx = load_json(market_ctx_path) or {}
    mode = market_ctx.get("market_mode", "neutral")

    # 4. Definir Thresholds (espejo del Orquestador)
    thresholds = {"growth": 0.65, "neutral": 0.75, "defensive": 0.85}
    current_threshold = thresholds.get(mode, 0.75)

    rows = []

    alpha_tickers = set(alpha_map.keys())
    position_tickers = set(positions.keys())

    universe_tickers = alpha_tickers | position_tickers

    for ticker in universe_tickers:

        data = alpha_map.get(ticker, {})
        score = data.get("alpha_score") 

        is_executable = False
        block_reason = None

        if score is None:
            block_reason = "no_alpha"

        else:
            # Kill switch → ejecutable (cerrar)
            if score <= -0.40:
                is_executable = True
                block_reason = "kill_switch"

            # Alpha suficiente
            elif score >= current_threshold:
                is_executable = True

            # Alpha insuficiente
            else:
                block_reason = "alpha_below_threshold"

        # Si el alpha_engine ya generó un motivo explícito
        if data.get("reason"):
            block_reason = data["reason"]

        rows.append({
            "ticker": ticker,
            "alpha": score,
            "confidence": data.get("confidence"),
            "positionValue": positions.get(ticker.upper(), {}).get("market_value", 0),
            "executable": is_executable,
            "block_reason": block_reason,
            "mode_context": mode,  # Para que el front sepa por qué el threshold es ese
        })

    # Ordenar por Alpha
    rows.sort(
        key=lambda x: x["alpha"] if x["alpha"] is not None else -999,
        reverse=True
    )
    return {"rows": rows}

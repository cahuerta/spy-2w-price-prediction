# =========================================================
# alpha_router.py — ALPHA SNAPSHOT API (ROOT)
# =========================================================
# ✔ Lee snapshot liviano
# ✔ Recalcula bajo demanda
# ✔ Solo usa tickers.json
# ✔ Producción ready
# =========================================================

import os
import json
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException

from alpha_engine_v4 import compute_and_persist_alpha


router = APIRouter()

DATA_PATH = os.getenv("DATA_PATH", "/data")
ALPHA_FILE = Path(DATA_PATH) / "alpha_last.json"
TICKERS_FILE = Path(DATA_PATH) / "tickers.json"


# =========================================================
# GET /alpha  (solo lectura snapshot)
# =========================================================
@router.get("/alpha")
def get_alpha_snapshot() -> Dict:

    if not ALPHA_FILE.exists():
        return {
            "status": "no_alpha_calculated"
        }

    return json.loads(ALPHA_FILE.read_text())


# =========================================================
# POST /internal/alpha/recompute
# =========================================================
@router.post("/internal/alpha/recompute")
def recompute_alpha() -> Dict:

    if not TICKERS_FILE.exists():
        raise HTTPException(400, "tickers.json not found")

    universe = json.loads(TICKERS_FILE.read_text())

    if not universe:
        raise HTTPException(400, "tickers.json vacío")

    payload = compute_and_persist_alpha(universe)

    return {
        "status": "recomputed",
        "calculated": payload["calculated"],
        "timestamp": payload["timestamp"],
    }
# =========================================================
# GET /alpha/test/{ticker}
# DEBUG endpoint (no afecta producción)
# =========================================================
from alpha_engine_v4 import compute_alpha_for_ticker

@router.get("/alpha/test/{ticker}")
def test_alpha_ticker(ticker: str) -> Dict:

    result = compute_alpha_for_ticker(ticker.upper())

    if not result:
        return {
            "ticker": ticker.upper(),
            "status": "alpha_returned_none"
        }

    return result
@router.get("/alpha/debug/history/{ticker}")
def debug_history(ticker: str):
    from data_provider import get_price_history
    raw = get_price_history(ticker, period="1y", interval="1d")
    if raw is None:
        return {"status": "raw_none"}
    return {"rows": len(raw)}

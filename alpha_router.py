# =========================================================
# alpha_router.py — ALPHA API ROOT
# =========================================================
# ✔ Calcula SOLO para tickers.json
# ✔ Ubicado en root del proyecto
# ✔ No acepta tickers externos
# ✔ Producción ready
# ✔ Endpoint: /alpha
# =========================================================

import os
import json
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter

from alpha_engine_v4 import compute_batch


# =========================================================
# Router ROOT
# =========================================================
router = APIRouter()

DATA_PATH = os.getenv("DATA_PATH", "/data")


# =========================================================
# Load Universe from tickers.json
# =========================================================
def load_universe() -> List[str]:

    p = Path(DATA_PATH) / "tickers.json"

    if not p.exists():
        return []

    try:
        with open(p, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [str(t).upper() for t in data]

        if isinstance(data, dict):
            return [str(t).upper() for t in data.get("tickers", [])]

    except Exception:
        return []

    return []


# =========================================================
# alpha_router.py — ALPHA SNAPSHOT API
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
# (Protegido si quieres)
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

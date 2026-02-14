# =========================================================
# admin_tickers_router.py — TICKERS ADMIN CONTROL
# =========================================================
# ✅ Limpia formato a LIST[str]
# ✅ Permite wipe total
# ✅ Protegido con X-PIPELINE-KEY
# =========================================================

import json
import os
from pathlib import Path
from typing import List, Any
from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/internal/admin", tags=["admin"])

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
PIPELINE_KEY = os.getenv("PIPELINE_KEY")

TICKERS_FILE = DATA_PATH / "tickers.json"


# =========================================================
# Helpers
# =========================================================

def load_raw(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def extract_tickers(data: Any) -> List[str]:
    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "tickers" in data:
        return data["tickers"]

    raise RuntimeError("Formato inválido en tickers.json")


def save_atomic(path: Path, tickers: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(set(tickers)), indent=2))
    tmp.replace(path)


# =========================================================
# ENDPOINT: NORMALIZE
# =========================================================

@router.post("/tickers/normalize")
async def normalize_tickers(request: Request):

    if request.headers.get("X-PIPELINE-KEY") != PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    raw = load_raw(TICKERS_FILE)

    try:
        tickers = extract_tickers(raw)
    except Exception as e:
        raise HTTPException(400, str(e))

    save_atomic(TICKERS_FILE, tickers)

    return {
        "status": "normalized",
        "total": len(set(tickers))
    }


# =========================================================
# ENDPOINT: WIPE
# =========================================================

@router.post("/tickers/wipe")
async def wipe_tickers(request: Request):

    if request.headers.get("X-PIPELINE-KEY") != PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    save_atomic(TICKERS_FILE, [])

    return {
        "status": "wiped",
        "total": 0
    }

# =========================================================
# admin_tickers_router.py — TICKERS ADMIN CONTROL v3.0
# =========================================================
# ✔ tickers.json SIEMPRE es List[str]
# ✔ Permite WIPE total
# ✔ Permite SANITIZE (limpia metadata)
# ✔ Escritura atómica
# ✔ Protegido con X-PIPELINE-KEY
# ✔ NO rompe contrato del sistema
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
    """
    Acepta:
    - List[str]
    - Dict con clave 'tickers'
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "tickers" in data and isinstance(data["tickers"], list):
        return data["tickers"]

    raise RuntimeError("Formato inválido en tickers.json")


def save_atomic(path: Path, tickers: List[str]):
    """
    GUARDA SIEMPRE COMO LISTA PURA
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    clean = sorted(set(str(t).strip().upper() for t in tickers if t))

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, indent=2))
    tmp.replace(path)


# =========================================================
# ENDPOINT: SANITIZE
# =========================================================
@router.post("/tickers/sanitize")
async def sanitize_tickers(request: Request):

    if request.headers.get("X-PIPELINE-KEY") != PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    raw = load_raw(TICKERS_FILE)

    try:
        tickers = extract_tickers(raw)
    except Exception as e:
        raise HTTPException(400, str(e))

    save_atomic(TICKERS_FILE, tickers)

    return {
        "status": "sanitized",
        "total": len(set(tickers))
    }


# =========================================================
# ENDPOINT: WIPE TOTAL
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


# =========================================================
# ENDPOINT: VIEW
# =========================================================
@router.get("/tickers/view")
async def view_tickers():

    if not TICKERS_FILE.exists():
        return {
            "exists": False,
            "total": 0,
            "tickers": []
        }

    raw = load_raw(TICKERS_FILE)

    try:
        tickers = extract_tickers(raw)
    except Exception:
        raise HTTPException(500, "tickers.json corrupt")

    return {
        "exists": True,
        "total": len(tickers),
        "tickers": tickers
    }

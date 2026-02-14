# =========================================================
# admin_tickers_router.py — TICKERS ADMIN CONTROL v4.0
# =========================================================
# ✔ tickers.json SIEMPRE es List[str]
# ✔ Limpieza REAL (elimina metadata contaminada)
# ✔ Permite WIPE total
# ✔ Validación estricta de ticker
# ✔ Escritura atómica
# ✔ Protegido con X-PIPELINE-KEY
# ✔ Nunca vuelve a mezclar dict keys como tickers
# =========================================================

import json
import os
from pathlib import Path
from typing import List, Any
from fastapi import APIRouter, HTTPException, Request, Header


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


# ---------------------------------------------------------
# VALIDACIÓN ESTRICTA DE TICKER
# ---------------------------------------------------------
def is_valid_ticker(t: str) -> bool:

    if not isinstance(t, str):
        return False

    t = t.strip().upper()

    if len(t) < 1 or len(t) > 15:
        return False

    if " " in t:
        return False

    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ.")
    return all(c in allowed for c in t)


# ---------------------------------------------------------
# ESCRITURA ATÓMICA SEGURA
# ---------------------------------------------------------
def save_atomic(path: Path, tickers: List[str]):

    path.parent.mkdir(parents=True, exist_ok=True)

    clean = sorted(
        set(
            t.strip().upper()
            for t in tickers
            if is_valid_ticker(t)
        )
    )

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, indent=2))
    tmp.replace(path)


# =========================================================
# ENDPOINT: SANITIZE (LIMPIEZA INTELIGENTE)
# =========================================================

@router.post("/tickers/sanitize")
async def sanitize_tickers(
    x_pipeline_key: str = Header(...)
):
    if x_pipeline_key != PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    raw = load_raw(TICKERS_FILE)

    try:
        tickers = extract_tickers(raw)
    except Exception as e:
        raise HTTPException(400, str(e))

    save_atomic(TICKERS_FILE, tickers)

    return {
        "status": "sanitized",
        "total": len(
            [
                t for t in tickers
                if is_valid_ticker(t)
            ]
        )
    }


# =========================================================
# ENDPOINT: WIPE TOTAL
# =========================================================

@router.post("/tickers/wipe")
async def wipe_tickers(
    x_pipeline_key: str = Header(...)
):
    if x_pipeline_key != PIPELINE_KEY:
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

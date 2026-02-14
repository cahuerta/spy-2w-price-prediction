# =========================================================
# debug_data_router.py
# DIAGNÓSTICO PROFUNDO DE /data
# NO afecta pipeline
# SOLO lectura
# =========================================================

from fastapi import APIRouter
from pathlib import Path
import json
from typing import Any

router = APIRouter(prefix="/internal/debug", tags=["debug"])

DATA_DIR = Path("data")


# =========================================================
# 1️⃣ LISTAR TODO EN /data
# =========================================================
@router.get("/files")
def list_data_files():

    if not DATA_DIR.exists():
        return {"error": "data directory not found"}

    files = []

    for p in DATA_DIR.rglob("*"):
        files.append({
            "path": str(p),
            "is_file": p.is_file(),
            "size_bytes": p.stat().st_size if p.is_file() else None
        })

    return {
        "total": len(files),
        "files": files
    }


# =========================================================
# 2️⃣ VER CONTENIDO DE UN JSON
# =========================================================
@router.get("/json")
def read_json(path: str):

    file_path = Path(path)

    if not file_path.exists():
        return {"error": "file not found"}

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        return {
            "path": str(file_path),
            "type": type(data).__name__,
            "keys": list(data.keys()) if isinstance(data, dict) else None,
            "preview": data if isinstance(data, (dict, list)) else str(data)
        }

    except Exception as e:
        return {"error": str(e)}


# =========================================================
# 3️⃣ LISTAR PREDICCIONES
# =========================================================
@router.get("/predictions")
def list_predictions():

    predictions_dir = DATA_DIR / "predictions"

    if not predictions_dir.exists():
        return {"error": "predictions folder not found"}

    files = list(predictions_dir.glob("*.json"))

    return {
        "count": len(files),
        "files": [str(f) for f in files]
    }


# =========================================================
# 4️⃣ VER PREDICCIÓN ESPECÍFICA
# =========================================================
@router.get("/prediction/{ticker}")
def read_prediction(ticker: str):

    file_path = DATA_DIR / "predictions" / f"{ticker}.json"

    if not file_path.exists():
        return {"error": f"prediction not found for {ticker}"}

    with open(file_path, "r") as f:
        data = json.load(f)

    return data

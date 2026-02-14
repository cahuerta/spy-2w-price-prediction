# =========================================================
# debug_data_router.py
# DIAGNÓSTICO REAL DE /data EN RENDER
# =========================================================

from fastapi import APIRouter
from pathlib import Path
import json
import os

router = APIRouter(prefix="/internal/debug", tags=["debug"])

DATA_DIR = Path(os.getenv("DATA_PATH", "/data"))


# =========================================================
# 1️⃣ LISTAR TODO EN /data REAL
# =========================================================
@router.get("/files")
def list_data_files():

    if not DATA_DIR.exists():
        return {
            "error": "data directory not found",
            "checked_path": str(DATA_DIR)
        }

    files = []

    for p in DATA_DIR.rglob("*"):
        files.append({
            "path": str(p),
            "is_file": p.is_file(),
            "size_bytes": p.stat().st_size if p.is_file() else None
        })

    return {
        "checked_path": str(DATA_DIR),
        "total": len(files),
        "files": files
    }


# =========================================================
# 2️⃣ VER JSON ESPECÍFICO
# =========================================================
@router.get("/json")
def read_json(path: str):

    file_path = Path(path)

    if not file_path.exists():
        return {"error": "file not found", "checked_path": str(file_path)}

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        return {
            "path": str(file_path),
            "type": type(data).__name__,
            "keys": list(data.keys()) if isinstance(data, dict) else None,
            "preview": data
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
        return {
            "error": "predictions folder not found",
            "checked_path": str(predictions_dir)
        }

    files = list(predictions_dir.glob("*.json"))

    return {
        "checked_path": str(predictions_dir),
        "count": len(files),
        "files": [str(f) for f in files]
    }

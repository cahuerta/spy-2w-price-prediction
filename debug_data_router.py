# =========================================================
# debug_data_router.py
# DIAGNÓSTICO REAL DE /data EN RENDER (MEJORADO)
# =========================================================
# ✅ /files      -> dump completo (recursivo) como antes
# ✅ /ls         -> listar SOLO una carpeta o un archivo (respeta ?path=)
# ✅ /json       -> ver JSON específico (con guardrails)
# ✅ /predictions/{ticker} -> listar predicciones por ticker
# ✅ /evaluations/{ticker} -> listar evaluaciones por ticker
# ✅ /glob       -> buscar por patrón (dentro de /data)
# ✅ /tail       -> leer últimas líneas de un log
# =========================================================

from fastapi import APIRouter, Query
from pathlib import Path
import json
import os

router = APIRouter(prefix="/internal/debug", tags=["debug"])

DATA_DIR = Path(os.getenv("DATA_PATH", "/data"))

# ---------------------------------------------------------
# Helpers (seguridad: solo permitir rutas dentro de /data)
# ---------------------------------------------------------
def _resolve_safe(path_str: str) -> Path | None:
    """
    Devuelve Path resuelto si está dentro de DATA_DIR.
    Si no, retorna None.
    """
    try:
        target = Path(path_str).resolve()
        base = DATA_DIR.resolve()

        # permitir base exacta o cualquier hijo
        if target == base or base in target.parents:
            return target
        return None
    except Exception:
        return None


def _file_info(p: Path):
    return {
        "path": str(p),
        "is_file": p.is_file(),
        "size_bytes": p.stat().st_size if p.is_file() else None,
        "name": p.name,
    }


# =========================================================
# 1️⃣ LISTAR TODO EN /data REAL (dump recursivo)
# =========================================================
@router.get("/files")
def list_data_files():
    if not DATA_DIR.exists():
        return {"error": "data directory not found", "checked_path": str(DATA_DIR)}

    files = []
    for p in DATA_DIR.rglob("*"):
        try:
            files.append(
                {
                    "path": str(p),
                    "is_file": p.is_file(),
                    "size_bytes": p.stat().st_size if p.is_file() else None,
                }
            )
        except Exception:
            # por si algún archivo cambia mientras iteras
            files.append({"path": str(p), "is_file": None, "size_bytes": None})

    return {"checked_path": str(DATA_DIR), "total": len(files), "files": files}


# =========================================================
# 1.5️⃣ LISTAR SOLO UNA CARPETA O ARCHIVO (respeta ?path=)
# =========================================================
@router.get("/ls")
def ls(path: str = Query("/data")):
    target = _resolve_safe(path)
    if target is None:
        return {"error": "forbidden path", "checked_path": path}

    if not target.exists():
        return {"error": "not found", "checked_path": str(target)}

    if target.is_file():
        return {"checked_path": str(target), "is_file": True, "item": _file_info(target)}

    items = []
    try:
        for p in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            items.append(_file_info(p))
    except Exception as e:
        return {"error": str(e), "checked_path": str(target)}

    return {"checked_path": str(target), "count": len(items), "items": items}


# =========================================================
# 2️⃣ VER JSON ESPECÍFICO (con guardrails)
# =========================================================
@router.get("/json")
def read_json(path: str):
    file_path = _resolve_safe(path)
    if file_path is None:
        return {"error": "forbidden path", "checked_path": path}

    if not file_path.exists():
        return {"error": "file not found", "checked_path": str(file_path)}

    if not file_path.is_file():
        return {"error": "not a file", "checked_path": str(file_path)}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "path": str(file_path),
            "type": type(data).__name__,
            "keys": list(data.keys()) if isinstance(data, dict) else None,
            "preview": data,
        }
    except Exception as e:
        return {"error": str(e), "checked_path": str(file_path)}


# =========================================================
# 3️⃣ LISTAR PREDICCIONES (estructura real: /predictions/<TICKER>/*.json)
# =========================================================
@router.get("/predictions/{ticker}")
def list_predictions_ticker(ticker: str):
    d = DATA_DIR / "predictions" / ticker
    if not d.exists():
        return {"error": "ticker folder not found", "checked_path": str(d)}

    files = sorted([p.name for p in d.glob("*.json")])
    return {"checked_path": str(d), "count": len(files), "files": files}


# =========================================================
# 4️⃣ LISTAR EVALUACIONES (estructura real: /evaluations/<TICKER>/*.json)
# =========================================================
@router.get("/evaluations/{ticker}")
def list_evaluations_ticker(ticker: str):
    d = DATA_DIR / "evaluations" / ticker
    if not d.exists():
        return {"error": "ticker folder not found", "checked_path": str(d)}

    files = sorted([p.name for p in d.glob("*.json")])
    return {"checked_path": str(d), "count": len(files), "files": files}


# =========================================================
# 5️⃣ BUSCAR POR PATRÓN (glob dentro de /data)
# Ej: /internal/debug/glob?pattern=/data/evaluations/JNJ/*.json
# =========================================================
@router.get("/glob")
def glob_search(pattern: str):
    # patrón debe comenzar dentro de /data
    safe_root = str(DATA_DIR.resolve())
    if not pattern.startswith(safe_root):
        return {"error": "pattern must start with DATA_PATH", "checked_pattern": pattern, "data_path": safe_root}

    try:
        matches = sorted([str(p) for p in Path("/").glob(pattern.lstrip("/"))])
        # Filtrar matches que no estén dentro de DATA_DIR (doble seguridad)
        out = []
        base = DATA_DIR.resolve()
        for m in matches:
            rp = Path(m).resolve()
            if rp == base or base in rp.parents:
                out.append(m)
        return {"checked_pattern": pattern, "count": len(out), "matches": out}
    except Exception as e:
        return {"error": str(e), "checked_pattern": pattern}


# =========================================================
# 6️⃣ LEER ÚLTIMAS N LÍNEAS DE UN ARCHIVO (logs)
# Ej: /internal/debug/tail?path=/data/signals.log&n=200
# =========================================================
@router.get("/tail")
def tail(path: str, n: int = 200):
    fp = _resolve_safe(path)
    if fp is None:
        return {"error": "forbidden path", "checked_path": path}

    if not fp.exists():
        return {"error": "file not found", "checked_path": str(fp)}

    if not fp.is_file():
        return {"error": "not a file", "checked_path": str(fp)}

    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-max(1, min(n, 5000)) :]
        return {"path": str(fp), "n": n, "lines": lines}
    except Exception as e:
        return {"error": str(e), "checked_path": str(fp)}

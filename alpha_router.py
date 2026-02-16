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
# GET /alpha
# =========================================================
@router.get("/alpha")
def get_alpha_universe() -> Dict:

    universe = load_universe()

    if not universe:
        return {
            "status": "error",
            "reason": "tickers.json not found or empty",
        }

    results = compute_batch(universe)

    if not results:
        return {
            "status": "error",
            "reason": "no alpha calculable",
            "universe_size": len(universe),
        }

    # Ordenado por alpha desc
    ranked = dict(
        sorted(
            results.items(),
            key=lambda x: x[1]["alpha_score"],
            reverse=True
        )
    )

    return {
        "status": "ok",
        "universe_size": len(universe),
        "calculated": len(ranked),
        "results": ranked,
    }

# dashboard.py
# Endpoints de lectura para dashboard (solo lectura de disco)

import os
import json
from fastapi import APIRouter, Query

DATA_PATH = os.getenv("DATA_PATH", "/data")

router = APIRouter()

# =========================
# Helpers
# =========================
def list_dirs(path):
    if not os.path.exists(path):
        return []
    return sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ])


def load_json_files(path):
    if not os.path.exists(path):
        return []

    files = sorted(
        [f for f in os.listdir(path) if f.endswith(".json")]
    )

    data = []
    for f in files:
        try:
            with open(os.path.join(path, f), "r") as fh:
                data.append(json.load(fh))
        except Exception:
            continue
    return data


# =========================
# Endpoints
# =========================

@router.get("/dashboard/status")
def status():
    return {
        "status": "ok",
        "data_path": DATA_PATH
    }


@router.get("/dashboard/tickers")
def get_tickers():
    pred_path = os.path.join(DATA_PATH, "predictions")
    eval_path = os.path.join(DATA_PATH, "evaluations")

    tickers = set(list_dirs(pred_path)) | set(list_dirs(eval_path))

    return {
        "tickers": sorted(list(tickers))
    }


@router.get("/dashboard/predictions")
def get_predictions(
    ticker: str = Query(...),
):
    path = os.path.join(DATA_PATH, "predictions", ticker)
    data = load_json_files(path)

    return {
        "ticker": ticker,
        "count": len(data),
        "data": data
    }


@router.get("/dashboard/evaluations")
def get_evaluations(
    ticker: str = Query(...),
):
    path = os.path.join(DATA_PATH, "evaluations", ticker)
    data = load_json_files(path)

    return {
        "ticker": ticker,
        "count": len(data),
        "data": data
    }


@router.get("/dashboard/metrics")
def get_metrics(
    ticker: str = Query(...),
):
    evals = load_json_files(
        os.path.join(DATA_PATH, "evaluations", ticker)
    )

    if not evals:
        return {"ticker": ticker, "error": "No evaluations yet"}

    hits = [e["hit"] for e in evals if "hit" in e]
    errors = [abs(e["error_pct"]) for e in evals if "error_pct" in e]

    hit_rate = sum(hits) / len(hits) if hits else None
    mae = sum(errors) / len(errors) if errors else None

    return {
        "ticker": ticker,
        "n_evals": len(evals),
        "hit_rate": hit_rate,
        "mae_pct": mae
    }

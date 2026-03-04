# =========================================================
# performance_router.py — INSTITUTIONAL PERFORMANCE ENGINE
# =========================================================

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter

from broker import get_engine

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
META_FILE = DATA_PATH / "account_meta.json"

router = APIRouter(prefix="/dashboard", tags=["performance"])

# =========================================================
# HELPERS
# =========================================================

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def list_evaluation_files() -> List[Path]:
    root = DATA_PATH / "evaluations"
    if not root.exists():
        return []

    files = []
    for ticker_dir in root.iterdir():
        if ticker_dir.is_dir():
            files.extend(ticker_dir.glob("*.json"))

    return sorted(files)


def compute_model_metrics():

    files = list_evaluation_files()

    if not files:
        return {
            "win_rate_pct": None,
            "avg_prediction_error_pct": None,
            "total_predictions": 0,
            "evaluated_predictions": 0,
            "pending_predictions": 0,
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
        }

    total = 0
    evaluated = 0
    correct = 0
    errors = []
    returns = []

    for f in files:
        data = load_json(f)
        if not data:
            continue

        total += 1

        if data.get("real_return_pct") is None:
            continue

        evaluated += 1

        if data.get("decision_correct") is True:
            correct += 1

        if data.get("error_return_pct") is not None:
            errors.append(float(data["error_return_pct"]))

        if data.get("real_return_pct") is not None:
            returns.append(float(data["real_return_pct"]))

    pending = total - evaluated

    win_rate = round((correct / evaluated) * 100, 2) if evaluated > 0 else None
    avg_error = round(np.mean(errors), 4) if errors else None

    sharpe = None
    max_dd = None

    if returns:
        returns_arr = np.array(returns) / 100.0

        mean_ret = np.mean(returns_arr)
        std_ret = np.std(returns_arr)

        if std_ret > 0:
            sharpe = round((mean_ret / std_ret) * np.sqrt(252), 3)

        equity_curve = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max

        max_dd = round(np.min(drawdowns) * 100, 2)

    return {
        "win_rate_pct": win_rate,
        "avg_prediction_error_pct": avg_error,
        "total_predictions": total,
        "evaluated_predictions": evaluated,
        "pending_predictions": pending,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
    }


# =========================================================
# PERFORMANCE ENDPOINT
# =========================================================

@router.get("/performance")
async def performance():

    # ================= ACCOUNT PERFORMANCE =================
    try:
        engine = get_engine()
        account = await engine.get_account()
        equity = float(account.equity)
    except Exception:
        equity = None

    meta = load_json(META_FILE)

    if not meta and equity:
        meta = {
            "initial_equity": equity,
            "start_date": datetime.utcnow().isoformat(),
            "high_water_mark": equity,
        }
        META_FILE.write_text(json.dumps(meta, indent=2))

    if meta and equity:
        initial_equity = float(meta["initial_equity"])
        high_water_mark = float(meta.get("high_water_mark", equity))

        if equity > high_water_mark:
            high_water_mark = equity
            meta["high_water_mark"] = equity
            META_FILE.write_text(json.dumps(meta, indent=2))

        total_return_pct = round(
            (equity - initial_equity) / initial_equity * 100, 2
        )

        drawdown_pct = round(
            (equity - high_water_mark) / high_water_mark * 100, 2
        )
    else:
        total_return_pct = None
        drawdown_pct = None
        high_water_mark = None

    # ================= MODEL METRICS =================
    model_metrics = compute_model_metrics()

    return {
        "equity": equity,
        "total_return_pct": total_return_pct,
        "drawdown_pct": drawdown_pct,
        "high_water_mark": high_water_mark,
        "since": meta["start_date"] if meta else None,

        # Model quality
        "win_rate_pct": model_metrics["win_rate_pct"],
        "avg_prediction_error_pct": model_metrics["avg_prediction_error_pct"],
        "total_predictions": model_metrics["total_predictions"],
        "evaluated_predictions": model_metrics["evaluated_predictions"],
        "pending_predictions": model_metrics["pending_predictions"],
        "sharpe_ratio": model_metrics["sharpe_ratio"],
        "max_drawdown_pct": model_metrics["max_drawdown_pct"],
    }

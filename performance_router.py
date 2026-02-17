# =========================================================
# performance_router.py — REAL PERFORMANCE (ALPACA)
# =========================================================

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from broker import get_trading_engine  # Ya existe en tu sistema

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
META_FILE = DATA_PATH / "account_meta.json"

router = APIRouter(prefix="/dashboard", tags=["performance"])


# =========================================================
# HELPERS
# =========================================================

def save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


# =========================================================
# PERFORMANCE ENDPOINT
# =========================================================

@router.get("/performance")
async def performance():

    engine = get_trading_engine()
    account = await engine.get_account()

    equity = float(account.equity)

    # Primera vez → guardar capital inicial
    meta = load_json(META_FILE)
    if not meta:
        meta = {
            "initial_equity": equity,
            "start_date": datetime.utcnow().isoformat(),
            "high_water_mark": equity,
        }
        save_json(META_FILE, meta)

    initial_equity = float(meta["initial_equity"])
    high_water_mark = float(meta.get("high_water_mark", equity))

    # Actualizar HWM si corresponde
    if equity > high_water_mark:
        high_water_mark = equity
        meta["high_water_mark"] = equity
        save_json(META_FILE, meta)

    total_return_pct = round(
        (equity - initial_equity) / initial_equity * 100,
        2,
    )

    drawdown_pct = round(
        (equity - high_water_mark) / high_water_mark * 100,
        2,
    )

    return {
        "equity": round(equity, 2),
        "total_return_pct": total_return_pct,
        "drawdown_pct": drawdown_pct,
        "high_water_mark": round(high_water_mark, 2),
        "since": meta["start_date"],
    }

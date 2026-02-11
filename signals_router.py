from fastapi import APIRouter, Query
from signals import compute_all_signals

router = APIRouter()

@router.get("/signals")
async def get_signals(min_confidence: float = 0.0):
    signals = compute_all_signals()
    if min_confidence > 0:
        signals = [
            s for s in signals
            if (s.get("confidence") or 0) >= min_confidence
        ]
    return {"signals": signals}

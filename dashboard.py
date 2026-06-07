# =========================================================
# dashboard.py — ENTERPRISE DASHBOARD MODULE v1.1
# =========================================================
# 🔹 LEE SOLO DESDE /data/predictions
# 🔹 signals.py NO es fuente de análisis
# 🔹 Frontend usa predictions como VERDAD
# 🔹 signals = visor secundario (pestaña señales)
# 🔹 ESTE ARCHIVO NO MONTA FASTAPI (solo router)
# 🔹 RATE LIMIT SOLO EN ENDPOINT PESADO (summary)
#
# FIX v1.1:
#   [D1] universe(): kill switch retornaba executable=True —
#        score <= -0.40 significa CERRAR posición existente,
#        NO abrir nueva. Ahora executable=False + block_reason
#        diferenciado entre "kill_switch_close" (tiene posición)
#        y "kill_switch_no_open" (no tiene posición).
#   [D2] executability_preview(): creaba TradingOrchestrator()
#        sin capital real → CapitalGovernor usaba $1M del env
#        en vez de ~$85k reales → sizings incorrectos en frontend.
#        Ahora lee equity real de Alpaca antes de instanciar.
# =========================================================

import os
import json
import logging
import sys
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from functools import lru_cache
from trading_orchestrator import TradingOrchestrator
from real_performance import get_alpaca_real_metrics
from fastapi import (
    APIRouter,
    Query,
    HTTPException,
    Request,
    Depends,
)

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")

# =========================================================
# LOGGING
# =========================================================
def setup_logging():
    level = logging.INFO
    try:
        log_path = Path(DATA_PATH) / "dashboard.log"
        log_path.parent.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_path)
    except Exception:
        fh = logging.NullHandler()

    sh = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        handlers=[fh, sh],
    )

setup_logging()
logger = logging.getLogger("dashboard")

# =========================================================
# RATE LIMITER (SOLO PARA SUMMARY)
# =========================================================
class SimpleRateLimiter:
    def __init__(self, requests=30, per_seconds=60):
        self.requests = requests
        self.per_seconds = per_seconds
        self.buckets: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=requests * 2)
        )

    async def __call__(self, request: Request):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        q = self.buckets[ip]

        while q and now - q[0] > self.per_seconds:
            q.popleft()

        if len(q) >= self.requests:
            raise HTTPException(429, "Rate limit exceeded")

        q.append(now)
        return True

rate_limiter = SimpleRateLimiter()

# =========================================================
# ROUTER
# =========================================================
router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# =========================================================
# HELPERS
# =========================================================
@lru_cache(maxsize=2)
def list_tickers(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

async def _get_real_capital() -> float:
    """
    [D2] Lee equity real de Alpaca para que el CapitalGovernor
    use el capital correcto (~$85k) en vez del env ($1M).
    Fallback a FIXED_CAPITAL si Alpaca no responde.
    """
    try:
        from broker import get_engine
        engine  = get_engine()
        account = await engine.get_account()
        equity  = float(account.equity)
        if equity > 0:
            logger.info(f"💰 Capital real Alpaca (dashboard): ${equity:,.0f}")
            return equity
    except Exception as e:
        logger.warning(f"⚠️ No se pudo leer equity Alpaca: {e} → usando env")
    return float(os.getenv("FIXED_CAPITAL", 1_000_000))

# =========================================================
# STATUS
# =========================================================
@router.get("/status")
async def status():
    pred = Path(DATA_PATH) / "predictions"
    return {
        "status": "ok",
        "predictions_exists": pred.exists(),
        "n_tickers": len(list_tickers(str(pred))),
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================================================
# TICKERS
# =========================================================
@router.get("/tickers")
async def tickers():
    pred_path = Path(DATA_PATH) / "predictions"
    return {
        "tickers": list_tickers(str(pred_path))
    }

# =========================================================
# SUMMARY (CON RATE LIMIT)
# =========================================================
@router.get("/predictions/summary")
async def prediction_summary(
    ticker: str,
    limit: int = Query(60, ge=1, le=500),
    _: Any = Depends(rate_limiter),
):
    pred_dir = Path(DATA_PATH) / "predictions" / ticker

    if not pred_dir.exists():
        raise HTTPException(404, f"No predictions for {ticker}")

    files = sorted(pred_dir.glob("*.json"))
    if not files:
        raise HTTPException(404, f"No prediction files for {ticker}")

    data = []
    for fp in files[-limit:]:
        obj = load_json(fp)
        if not obj:
            continue

        p = obj.get("prediction")
        if not isinstance(p, dict):
            continue

        data.append({
            "date_base":      p.get("date_base"),
            "price_now":      safe_float(p.get("price_now")),
            "price_pred":     safe_float(p.get("price_pred")),
            "ret_ens_pct":    safe_float(p.get("ret_ens_pct")),
            "recommendation": p.get("recommendation"),
        })

    if not data:
        raise HTTPException(404, "No usable prediction data")

    return {
        "ticker": ticker,
        "count":  len(data),
        "data":   data,
    }

# =========================================================
# LATEST
# =========================================================
@router.get("/latest/{ticker}")
async def latest_snapshot(ticker: str):
    pred_dir = Path(DATA_PATH) / "predictions" / ticker

    if not pred_dir.exists():
        raise HTTPException(404, f"No predictions for {ticker}")

    files = sorted(pred_dir.glob("*.json"))
    if not files:
        raise HTTPException(404, f"No prediction files for {ticker}")

    last = load_json(files[-1])
    if not last:
        raise HTTPException(404, "Invalid prediction file")

    return {"ticker": ticker, "latest": last}

# =========================================================
# EVALUATION LATEST
# =========================================================
@router.get("/evaluation-latest/{ticker}")
async def evaluation_latest(ticker: str):
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise HTTPException(400, "ticker requerido")

    eval_dir = Path(DATA_PATH) / "evaluations" / ticker

    if not eval_dir.exists():
        raise HTTPException(404, f"No evaluations for {ticker}")

    files = sorted(eval_dir.glob("*.json"), key=lambda p: p.name)
    if not files:
        raise HTTPException(404, f"No evaluation files for {ticker}")

    last_fp = files[-1]
    last    = load_json(last_fp)
    if not last:
        raise HTTPException(404, "Invalid evaluation file")

    return {
        "ticker":     ticker,
        "evaluation": last,
        "file":       last_fp.name,
    }

# =========================================================
# SCREENER
# =========================================================
@router.get("/screener")
async def screener():
    p = Path(DATA_PATH) / "screener_candidates.json"
    if not p.exists():
        raise HTTPException(404, "Screener not available")

    data = load_json(p)
    if not data:
        raise HTTPException(500, "Invalid screener file")

    return data

# =========================================================
# MARKET CONTEXT
# =========================================================
@router.get("/market-context")
async def market_context():
    p = Path(DATA_PATH) / "market_context.json"

    if not p.exists():
        raise HTTPException(404, "Market context not available")

    data = load_json(p)
    if not data:
        raise HTTPException(500, "Invalid market context file")

    return data

# =========================================================
# EXECUTABILITY PREVIEW
# [D2] Lee capital real de Alpaca antes de instanciar orchestrator
# =========================================================
@router.get("/executability-preview")
async def executability_preview():
    market_ctx_path = Path(DATA_PATH) / "market_context.json"
    if not market_ctx_path.exists():
        raise HTTPException(404, "market_context.json not found")

    market_ctx = json.loads(market_ctx_path.read_text())

    # [D2] Capital real para sizing correcto
    real_capital = await _get_real_capital()
    orchestrator = TradingOrchestrator()
    orchestrator.fixed_capital    = real_capital
    orchestrator._fallback_capital = real_capital

    return await orchestrator.preview_executability(market_ctx=market_ctx)

# =========================================================
# UNIVERSE
# [D1] Kill switch: executable=False — score <= -0.40 significa
#      CERRAR posición existente, NO abrir nueva.
# =========================================================
@router.get("/universe")
async def universe():
    from portfolio_store import load_positions

    ALPHA_FILE      = Path(DATA_PATH) / "alpha_last.json"
    market_ctx_path = Path(DATA_PATH) / "market_context.json"

    alpha_data = load_json(ALPHA_FILE) or {}
    alpha_map  = alpha_data.get("results", {})

    positions_list = load_positions()
    positions = {
        p.get("ticker", "").upper(): p
        for p in positions_list
        if "ticker" in p
    }

    market_ctx        = load_json(market_ctx_path) or {}
    mode              = market_ctx.get("market_mode", "neutral")
    thresholds        = {"growth": 0.65, "neutral": 0.75, "defensive": 0.85}
    current_threshold = thresholds.get(mode, 0.75)

    rows = []

    universe_tickers = set(alpha_map.keys()) | set(positions.keys())

    for ticker in universe_tickers:
        data  = alpha_map.get(ticker, {})
        score = data.get("alpha_score")

        is_executable = False
        block_reason  = None
        has_position  = ticker.upper() in positions

        if score is None:
            block_reason = "no_alpha"

        elif score <= -0.40:
            # [D1] Kill switch: NO es apertura — es señal de cierre
            # executable=False porque no se debe ABRIR esta posición
            # El orchestrator cerrará la posición existente si la hay
            is_executable = False
            block_reason  = "kill_switch_close" if has_position else "kill_switch_no_open"

        elif score >= current_threshold:
            is_executable = True

        else:
            block_reason = "alpha_below_threshold"

        # Si el alpha_engine ya generó un motivo explícito
        if data.get("reason"):
            block_reason = data["reason"]

        rows.append({
            "ticker":       ticker,
            "alpha":        score,
            "confidence":   data.get("components", {}).get("confidence"),
            "positionValue": positions.get(ticker.upper(), {}).get("market_value", 0),
            "executable":   is_executable,
            "block_reason": block_reason,
            "has_position": has_position,
            "mode_context": mode,
        })

    rows.sort(
        key=lambda x: x["alpha"] if x["alpha"] is not None else -999,
        reverse=True,
    )
    return {"rows": rows}

# =========================================================
# REAL PERFORMANCE
# =========================================================
@router.get("/real-execution")
async def real_execution():
    try:
        metrics = await get_alpaca_real_metrics()
        if metrics.get("status") == "error":
            raise HTTPException(500, metrics.get("error"))
        return metrics
    except Exception as e:
        logger.error(f"Error en endpoint real-execution: {e}")
        raise HTTPException(500, str(e))
        

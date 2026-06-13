# =========================================================
# dashboard.py — ENTERPRISE DASHBOARD MODULE (PRODUCCIÓN)
# =========================================================
# 🔹 LEE SOLO DESDE /data/predictions
# 🔹 signals.py NO es fuente de análisis
# 🔹 Frontend usa predictions como VERDAD
# 🔹 signals = visor secundario (pestaña señales)
# 🔹 ESTE ARCHIVO NO MONTA FASTAPI (solo router)
# 🔹 RATE LIMIT SOLO EN ENDPOINT PESADO (summary)
#
# FIX:
#   [D1] universe(): kill switch retornaba executable=True — corregido
#   [D2] executability_preview(): capital real de Alpaca
#   [D3] universe(): agregado tickers.json como fuente principal
#   [D4] latest_snapshot(): enriquece historical con evaluaciones
#        limpias reales desde /data/evaluations/.
#        Antes: historical.hit_rate_mean siempre null (campo del
#        modelo que nunca se calculaba en producción).
#        Ahora: lee evaluaciones con legacy_bad_horizon=False,
#        price_real válido, hit_sign no nulo → hit_rate real.
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
    try:
        from broker import get_engine
        engine  = get_engine()
        account = await engine.get_account()
        equity  = float(account.equity)
        if equity > 0:
            return equity
    except Exception as e:
        logger.warning(f"⚠️ No se pudo leer equity Alpaca: {e} → usando env")
    return float(os.getenv("FIXED_CAPITAL", 1_000_000))


# =========================================================
# [D4] HELPER — hit rate real desde evaluaciones limpias
# =========================================================
def _get_real_hit_rate(ticker: str) -> Optional[Dict[str, Any]]:
    """
    [D4] Lee evaluaciones limpias del ticker y calcula hit_rate real.
    Solo usa evaluaciones con:
      - legacy_bad_horizon=False (evaluadas con horizonte correcto)
      - price_real no None (tienen precio real de Alpaca)
      - hit_sign no None (se pudo calcular dirección)

    Retorna dict compatible con el campo 'historical' del frontend,
    o None si no hay evaluaciones limpias suficientes.
    """
    eval_dir = Path(DATA_PATH) / "evaluations" / ticker.upper()
    if not eval_dir.exists():
        return None

    hits:   list = []
    errors: list = []

    for f in eval_dir.glob("*.json"):
        ev = load_json(f)
        if not ev:
            continue
        # Filtro estricto: solo evaluaciones limpias
        if ev.get("legacy_bad_horizon") is True:
            continue
        if ev.get("price_real") is None:
            continue
        if ev.get("alpaca_unsupported") is True:
            continue
        if ev.get("hit_sign") is None:
            continue

        hits.append(1.0 if ev["hit_sign"] else 0.0)
        err = ev.get("error_return_pct")
        if err is not None:
            errors.append(float(err))

    if not hits:
        return None

    return {
        "hit_rate_mean": round(sum(hits) / len(hits), 4),
        "mae_mean":      round(sum(errors) / len(errors), 4) if errors else None,
        "rmse_mean":     None,
        "n_windows":     len(hits),
        "source":        "evaluations_v2",  # distingue del campo legacy null
    }


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
# [D4] Enriquece historical con evaluaciones limpias reales
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

    # [D4] Enriquecer historical con hit_rate real de evaluaciones limpias
    real_metrics = _get_real_hit_rate(ticker)
    if real_metrics:
        # Preservar campos que el modelo sí calcula (pca_dims, n_features)
        existing = last.get("historical") or {}
        real_metrics["pca_dims"]   = existing.get("pca_dims")
        real_metrics["n_features"] = existing.get("n_features")
        last["historical"] = real_metrics
        logger.debug(
            f"[D4] {ticker} hit_rate={real_metrics['hit_rate_mean']:.1%} "
            f"n={real_metrics['n_windows']} evaluaciones limpias"
        )
    else:
        # Sin evaluaciones limpias — mantener estructura pero indicar fuente
        existing = last.get("historical") or {}
        existing["source"] = "model_legacy"
        last["historical"] = existing

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
# =========================================================
@router.get("/executability-preview")
async def executability_preview():
    market_ctx_path = Path(DATA_PATH) / "market_context.json"
    if not market_ctx_path.exists():
        raise HTTPException(404, "market_context.json not found")

    market_ctx = json.loads(market_ctx_path.read_text())

    real_capital = await _get_real_capital()
    orchestrator = TradingOrchestrator()
    orchestrator.fixed_capital     = real_capital
    orchestrator._fallback_capital = real_capital

    return await orchestrator.preview_executability(market_ctx=market_ctx)

# =========================================================
# UNIVERSE
# [D3] tickers.json como fuente principal del universo completo
# =========================================================
@router.get("/universe")
async def universe():
    from portfolio_store import load_positions

    ALPHA_FILE      = Path(DATA_PATH) / "alpha_last.json"
    TICKERS_FILE    = Path(DATA_PATH) / "tickers.json"
    market_ctx_path = Path(DATA_PATH) / "market_context.json"

    try:
        tickers_universe = set(json.loads(TICKERS_FILE.read_text()) or [])
    except Exception:
        tickers_universe = set()

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

    universe_tickers = tickers_universe | set(alpha_map.keys()) | set(positions.keys())

    rows = []

    for ticker in universe_tickers:
        data  = alpha_map.get(ticker, {})
        score = data.get("alpha_score")

        is_executable = False
        block_reason  = None
        has_position  = ticker.upper() in positions

        if score is None:
            block_reason = "no_alpha"
        elif score <= -0.40:
            is_executable = False
            block_reason  = "kill_switch_close" if has_position else "kill_switch_no_open"
        elif score >= current_threshold:
            is_executable = True
        else:
            block_reason = "alpha_below_threshold"

        if data.get("reason"):
            block_reason = data["reason"]

        rows.append({
            "ticker":        ticker,
            "alpha":         score,
            "confidence":    data.get("components", {}).get("confidence"),
            "positionValue": positions.get(ticker.upper(), {}).get("market_value", 0),
            "executable":    is_executable,
            "block_reason":  block_reason,
            "has_position":  has_position,
            "mode_context":  mode,
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

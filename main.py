# main.py — TRADING SUITE ENTERPRISE (CORREGIDO Y PRODUCTION-READY)
# ✅ Integra model.py + evaluator.py + signals.py + broker.py + dashboard.py
# ✅ Healthchecks Kubernetes/Render (live/ready/health)
# ✅ Rate limiting memory-safe (+ cleanup básico)
# ✅ CORS seguro (sin wildcard+credentials)
# ✅ Filtro de señales corregido (precedencia AND/OR)
# ✅ Auto-execute usa compute_all_signals directo (no llama endpoint interno)
# ✅ Escritura de predicciones: filename único (evita overwrite)
# ✅ Render/Vercel/Railway compatible

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
from fastapi import FastAPI, Query, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# Config
# =========================================================
DATA_PATH = os.getenv("DATA_PATH", "/data")
PORT = int(os.getenv("PORT", "8000"))

# Rate limiting
RL_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RL_PER_SECONDS = int(os.getenv("RATE_LIMIT_PER_SECONDS", "60"))
RL_MAX_IPS = int(os.getenv("RATE_LIMIT_MAX_IPS", "5000"))

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")  # comma-separated or "*"
ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

# Signals defaults
DEFAULT_WINDOW = int(os.getenv("SIGNAL_WINDOW", "30"))
DEFAULT_MIN_CONF = float(os.getenv("SIGNAL_MIN_CONFIDENCE", "0.4"))

# =========================================================
# Rate limiter memory-safe (+ cleanup básico)
# =========================================================
class SimpleRateLimiter:
    def __init__(self, requests: int = 100, per_seconds: int = 60, max_ips: int = 5000):
        self.requests = requests
        self.per_seconds = per_seconds
        self.max_ips = max_ips
        self.buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=requests * 2))

    async def __call__(self, request: Request):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        q = self.buckets[client_ip]

        while q and now - q[0] > self.per_seconds:
            q.popleft()

        if len(q) >= self.requests:
            raise HTTPException(status_code=429, detail=f"Rate limit {self.requests}/{self.per_seconds}s")

        q.append(now)

        # cleanup cuando crece demasiado
        if len(self.buckets) > self.max_ips:
            self._cleanup(now)

        return True

    def _cleanup(self, now: float):
        # borra IPs inactivas
        to_delete = []
        for ip, q in self.buckets.items():
            if not q or now - q[-1] > self.per_seconds * 2:
                to_delete.append(ip)
        for ip in to_delete[: max(100, len(to_delete) // 2)]:
            self.buckets.pop(ip, None)

rate_limiter = SimpleRateLimiter(RL_REQUESTS, RL_PER_SECONDS, RL_MAX_IPS)

# =========================================================
# Logging enterprise
# =========================================================
def setup_logging():
    level = logging.INFO
    data_path = Path(DATA_PATH)
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        fh = logging.FileHandler(data_path / "main.log")
    except Exception:
        fh = logging.NullHandler()
    sh = logging.StreamHandler()

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        handlers=[fh, sh]
    )

setup_logging()
logger = logging.getLogger(__name__)

# =========================================================
# FastAPI
# =========================================================
app = FastAPI(
    title="🚀 Trading Suite Enterprise",
    description="ML Pipeline + Signals + Trading",
    version="2.0.1"
)

# CORS seguro:
# - Si allow_origins="*" => allow_credentials debe ser False.
# - Si quieres cookies/auth, define dominios explícitos en CORS_ORIGINS.
if CORS_ORIGINS.strip() == "*":
    allow_origins = ["*"]
    allow_credentials = False
else:
    allow_origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
    allow_credentials = ALLOW_CREDENTIALS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Load modules (best-effort)
# =========================================================
ALL_MODULES_LOADED = False

run_model = None
evaluate_all = None
compute_signal = None
compute_all_signals = None
get_signals = None

dashboard_router = None
broker_router = None

try:
    from model import run_model as _run_model
    from evaluator import evaluate_all as _evaluate_all
    from signals import compute_signal as _compute_signal
    from signals import compute_all_signals as _compute_all_signals
    from signals import get_signals as _get_signals
    run_model = _run_model
    evaluate_all = _evaluate_all
    compute_signal = _compute_signal
    compute_all_signals = _compute_all_signals
    get_signals = _get_signals
    ALL_MODULES_LOADED = True
    logger.info("✅ Core modules loaded: model + evaluator + signals")
except Exception as e:
    logger.warning(f"⚠️ Core modules missing: {e}")

# Routers (dashboard/broker)
try:
    from dashboard import router as _dashboard_router
    dashboard_router = _dashboard_router
    logger.info("✅ dashboard router loaded")
except Exception as e:
    logger.warning(f"⚠️ dashboard router missing: {e}")

try:
    from broker import router as _broker_router
    broker_router = _broker_router
    logger.info("✅ broker router loaded")
except Exception as e:
    logger.warning(f"⚠️ broker router missing: {e}")

# Include routers
# IMPORTANTE:
# - dashboard.py ya tiene prefix "/dashboard" dentro; broker.py ya tiene "/trading".
# - Por lo tanto NO agregamos prefix aquí para evitar duplicar rutas.
if dashboard_router is not None:
    app.include_router(dashboard_router)
if broker_router is not None:
    app.include_router(broker_router)

# =========================================================
# Healthchecks
# =========================================================
@app.get("/live")
async def live():
    return {"status": "live"}

@app.get("/ready")
async def ready(_: Any = Depends(rate_limiter)):
    data_path = Path(DATA_PATH)
    pred_path = data_path / "predictions"
    eval_path = data_path / "evaluations"

    data_ok = data_path.exists()
    pred_ok = pred_path.exists()
    eval_ok = eval_path.exists()

    return {
        "status": "ready" if data_ok else "not_ready",
        "data_path_ok": data_ok,
        "predictions_path_ok": pred_ok,
        "evaluations_path_ok": eval_ok,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/health")
async def health(_: Any = Depends(rate_limiter)):
    issues = []

    data_path = Path(DATA_PATH)
    if not data_path.exists():
        issues.append("DATA_PATH missing")

    pred_path = data_path / "predictions"
    eval_path = data_path / "evaluations"

    n_tickers = 0
    if pred_path.exists():
        try:
            n_tickers = len([d for d in pred_path.iterdir() if d.is_dir()])
        except Exception:
            n_tickers = 0

    signals_ok = compute_all_signals is not None
    if not signals_ok:
        issues.append("signals unavailable")

    broker_ok = broker_router is not None

    status = "healthy" if not issues else "degraded"
    return {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "n_tickers": n_tickers,
        "signals_ok": signals_ok,
        "broker_ok": broker_ok,
        "issues": issues,
    }

# =========================================================
# Main endpoints
# =========================================================
@app.get("/predict")
async def predict(
    ticker: str = Query("SPY"),
    horizon: int = Query(10, ge=1, le=30),
    theta: float = Query(0.75, ge=0.1, le=1.0),
    _: Any = Depends(rate_limiter),
):
    if not ALL_MODULES_LOADED or run_model is None:
        raise HTTPException(status_code=503, detail="ML modules not loaded")

    try:
        t0 = time.time()
        result = run_model(ticker=ticker, horizon=horizon, theta=theta)
        latency = time.time() - t0

        return {
            "ok": True,
            "timestamp": datetime.utcnow().isoformat(),
            "latency_s": round(latency, 3),
            "ticker": ticker,
            "recommendation": result["prediction"]["recommendation"],
            "ret_ens_pct": result["prediction"]["ret_ens_pct"],
            "price_pred": result["prediction"]["price_pred"],
            "full_result": result,
        }
    except Exception as e:
        logger.error(f"Predict {ticker} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals")
async def signals(
    window: int = Query(DEFAULT_WINDOW, ge=7, le=365),
    min_confidence: float = Query(DEFAULT_MIN_CONF, ge=0.0, le=1.0),
    strong_only: bool = Query(False),
    limit: int = Query(20, ge=1, le=200),
    _: Any = Depends(rate_limiter),
):
    if compute_all_signals is None:
        return {"ok": False, "error": "signals module not loaded", "signals": [], "count": 0}

    try:
        t0 = time.time()
        signals_raw = compute_all_signals(window)

        # ✅ FIX: precedencia AND/OR (paréntesis)
        filtered = [
            s for s in signals_raw
            if (
                (not strong_only or s.get("quality") == "🔥 STRONG")
                and (s.get("confidence") or 0) >= min_confidence
            )
        ]

        filtered.sort(
            key=lambda s: (
                -(s.get("confidence") or 0),
                -(s.get("signal_strength") or 0),
                -(s.get("ret_ens_pct") or 0),
                s.get("ticker", ""),
            )
        )

        out = filtered[:limit]
        latency = time.time() - t0

        return {
            "ok": True,
            "timestamp": datetime.utcnow().isoformat(),
            "latency_s": round(latency, 3),
            "window": window,
            "min_confidence": min_confidence,
            "strong_only": strong_only,
            "count": len(out),
            "signals": out,
        }
    except Exception as e:
        logger.error(f"Signals failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trading/auto-execute")
async def auto_execute(
    max_trades: int = Query(3, ge=1, le=10),
    strong_only: bool = Query(True),
    window: int = Query(DEFAULT_WINDOW, ge=7, le=365),
    min_confidence: float = Query(max(DEFAULT_MIN_CONF, 0.7), ge=0.0, le=1.0),
    _: Any = Depends(rate_limiter),
):
    """
    🚀 AUTO-TRADE: ejecuta TOP señales usando:
      - compute_all_signals() (sin llamar endpoint interno)
      - broker.get_trading_engine().execute_signal()
    """
    if broker_router is None:
        return {"ok": False, "status": "broker_not_loaded"}

    if compute_all_signals is None:
        return {"ok": False, "status": "signals_not_loaded"}

    try:
        from broker import get_trading_engine  # import local seguro
        engine = get_trading_engine()

        signals_raw = compute_all_signals(window)

        candidates = [
            s for s in signals_raw
            if (
                (not strong_only or s.get("quality") == "🔥 STRONG")
                and (s.get("confidence") or 0) >= min_confidence
            )
        ]

        candidates.sort(
            key=lambda s: (
                -(s.get("confidence") or 0),
                -(s.get("signal_strength") or 0),
                -(s.get("ret_ens_pct") or 0),
            )
        )

        results = []
        for s in candidates[:max_trades]:
            res = engine.execute_signal(s)
            # res es Pydantic model en nuestro broker.py corregido
            results.append(res.model_dump() if hasattr(res, "model_dump") else dict(res))

        trades_executed = len([r for r in results if r.get("status") == "executed"])

        return {
            "ok": True,
            "timestamp": datetime.utcnow().isoformat(),
            "max_trades": max_trades,
            "signals_candidates": len(candidates),
            "trades_executed": trades_executed,
            "results": results,
        }
    except Exception as e:
        logger.error(f"Auto-execute failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# Batch operations (cron)
# =========================================================
@app.get("/batch/predict-all")
async def batch_predict_all(
    limit: int = Query(20, ge=1, le=500),
    _: Any = Depends(rate_limiter),
):
    """
    Cron: Predice TODOS los tickers (limitado).
    Guarda con nombre único para evitar overwrite.
    """
    if run_model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        tickers_path = Path("tickers.json")
        if not tickers_path.exists():
            raise HTTPException(status_code=404, detail="tickers.json missing")

        with open(tickers_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tickers = data.get("tickers", data) if isinstance(data, dict) else data
        if not isinstance(tickers, list):
            raise HTTPException(status_code=400, detail="tickers.json format invalid")

        results = []
        for ticker in tickers[:limit]:
            try:
                result = run_model(ticker=ticker)

                pred_dir = Path(DATA_PATH) / "predictions" / ticker
                pred_dir.mkdir(parents=True, exist_ok=True)

                # ✅ filename único (evita overwrite)
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                fp = pred_dir / f"{ts}.json"
                fp.write_text(json.dumps(result), encoding="utf-8")

                results.append({"ticker": ticker, "status": "saved", "file": str(fp)})
            except Exception as e:
                results.append({"ticker": ticker, "status": "failed", "error": str(e)})

        return {"ok": True, "processed": len(results), "results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/batch/evaluate-all")
async def batch_evaluate_all(_: Any = Depends(rate_limiter)):
    """Cron: Evalúa TODAS las predicciones."""
    if evaluate_all is None:
        return {"ok": False, "error": "evaluator not loaded"}
    try:
        result = evaluate_all()
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# Metrics / Overview
# =========================================================
@app.get("/metrics")
async def metrics_overview(_: Any = Depends(rate_limiter)):
    """Resumen ejecutivo (ligero)."""
    try:
        data_path = Path(DATA_PATH)
        pred_path = data_path / "predictions"
        eval_path = data_path / "evaluations"

        n_tickers_pred = len([d for d in pred_path.iterdir() if d.is_dir()]) if pred_path.exists() else 0
        n_tickers_eval = len([d for d in eval_path.iterdir() if d.is_dir()]) if eval_path.exists() else 0

        # Sample hit-rate (último archivo por ticker en sample)
        total_hits = 0
        total_evals = 0

        if eval_path.exists():
            for ticker_dir in list(eval_path.iterdir())[:10]:
                if not ticker_dir.is_dir():
                    continue
                files = sorted(ticker_dir.glob("*.json"))
                if not files:
                    continue
                try:
                    with open(files[-1], "r", encoding="utf-8") as f:
                        data = json.load(f)
                    total_hits += 1 if data.get("decision_correct") else 0
                    total_evals += 1
                except Exception:
                    continue

        hit_rate = round(total_hits / max(total_evals, 1), 4)

        return {
            "ok": True,
            "n_tickers_predictions": n_tickers_pred,
            "n_tickers_evaluations": n_tickers_eval,
            "sample_hit_rate": hit_rate,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# Root
# =========================================================
@app.get("/")
async def root():
    return {
        "service": "Trading Suite Enterprise",
        "version": "2.0.1",
        "endpoints": {
            "predict": "/predict?ticker=SPY",
            "signals": "/signals?strong_only=true",
            "trading_status": "/trading/status",
            "auto_execute": "/trading/auto-execute?max_trades=3",
            "health": "/health",
            "metrics": "/metrics",
        },
        "status": "🚀 PRODUCTION READY",
    }

# =========================================================
# Lifecycle
# =========================================================
@app.on_event("startup")
async def startup():
    logger.info("🚀 Trading Suite Enterprise starting...")
    if ALL_MODULES_LOADED:
        logger.info("✅ Core stack loaded")
    else:
        logger.warning("⚠️ Partial stack (missing core modules)")

@app.on_event("shutdown")
async def shutdown():
    logger.info("🛑 Trading Suite shutting down...")

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
# =========================
# AUTO-INIT tickers.json (solo si no existe)
# =========================
from pathlib import Path
import json

DATA_PATH = Path("/data")
DATA_PATH.mkdir(exist_ok=True)

tickers_file = DATA_PATH / "tickers.json"

if not tickers_file.exists():
    tickers_file.write_text(
        json.dumps(["JNJ", "KO", "PG", "MCD", "SPY"], indent=2)
    )

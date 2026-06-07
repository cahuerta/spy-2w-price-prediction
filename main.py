# =========================================================
# main.py — TRADING SUITE ENTERPRISE v2.9.4 PRODUCCIÓN
# =========================================================
# ✔ Runtime de trading (NO batch)
# ✔ NO decide mercado
# ✔ NO ejecuta modelos
# ✔ NO ejecuta pipeline
# ✔ RECIBE resultados del pipeline
# ✔ GRABA a disco (fuente única)
# ✔ Ejecuta SOLO TradingOrchestrator
# ✔ 🔥 MERGE DE TICKERS EN STARTUP (NUNCA BORRA)
# ✔ 🕐 SCHEDULER INTERNO (reemplaza cron externo)
#
# v2.9.2:
#   [M1] /internal/trading/monitor-close — cierre defensivo horario
#   [M2] /internal/darwin/run            — ciclo evolutivo manual
#   [M3] /internal/darwin/status         — estado del Darwin Engine
#
# v2.9.3:
#   [M4] /internal/pipeline/last — agregada autenticación X-PIPELINE-KEY
#
# v2.9.4:
#   [M5] merge_tickers_on_startup: filtra tickers fantasma validando
#        contra anchor_universe.json — tickers que no existen en el
#        anchor universe se loggean como advertencia pero NO se agregan
#        al universe de predicción (evita OPENs silenciosos ignorados).
# =========================================================

import os
import json
import logging
import signal
import asyncio
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Set

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from signals_router import router as signals_router
from dashboard import router as dashboard_router
from pipeline_router import router as pipeline_router

from market_orchestrator import MarketOrchestrationContext
from trading_orchestrator import TradingOrchestrator
from debug_data_router import router as debug_router
from admin_tickers_router import router as admin_router
from broker import router as trading_router
from alpha_router import router as alpha_router
from performance_router import router as performance_router
from analysis_router import router as analysis_router
from execution_analyzer import router as execution_router
from scheduler import start_scheduler


# =========================================================
# CONFIG
# =========================================================
class Config:
    DATA_PATH    = Path(os.getenv("DATA_PATH", "/data"))
    PORT         = int(os.getenv("PORT", "8000"))
    PIPELINE_KEY = os.getenv("PIPELINE_KEY")
    LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO").upper()

config = Config()

DATA_PATH           = config.DATA_PATH
TICKERS_FILE        = DATA_PATH / "tickers.json"
REPO_TICKERS_FILE   = Path(__file__).resolve().parent / "tickers.json"
MARKET_CTX_FILE     = DATA_PATH / "market_context.json"
SCREENER_FILE       = DATA_PATH / "screener_candidates.json"
PIPELINE_AUDIT_FILE = DATA_PATH / "last_pipeline.json"
ANCHOR_FILE         = Path(os.getenv("ANCHOR_FILE", str(Path(__file__).resolve().parent / "anchor_universe.json")))

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trading_suite")

# =========================================================
# HELPERS DISCO
# =========================================================
def save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())

# =========================================================
# [M5] CARGAR ANCHOR UNIVERSE COMO REFERENCIA
# =========================================================
def _load_anchor_tickers() -> Set[str]:
    """
    [M5] Carga los tickers del anchor_universe.json como set de referencia.
    Se usa para validar tickers en merge_tickers_on_startup.
    Retorna set vacío si el archivo no existe — en ese caso no se filtra nada.
    """
    try:
        if ANCHOR_FILE.exists():
            data = json.loads(ANCHOR_FILE.read_text())
            if isinstance(data, list):
                tickers = {a["ticker"].upper() for a in data if "ticker" in a}
                logger.info(f"⚓ Anchor reference: {len(tickers)} tickers")
                return tickers
    except Exception as e:
        logger.warning(f"⚠️ No se pudo leer anchor_universe.json: {e}")
    return set()

# =========================================================
# 🔥 MERGE DE TICKERS (STARTUP)
# =========================================================
def merge_tickers_on_startup():
    """
    [M5] Merge de tickers en startup con validación contra anchor_universe.
    Tickers fantasma (no en anchor universe) se loggean pero no se agregan.
    Si anchor_universe.json no existe, se comporta igual que antes (sin filtro).
    """
    logger.info("🔧 Merging tickers on startup")

    disk_tickers: List[str] = load_json(TICKERS_FILE, [])

    if REPO_TICKERS_FILE.exists():
        try:
            repo_raw = json.loads(REPO_TICKERS_FILE.read_text())
            if isinstance(repo_raw, list):
                base_tickers = [str(t).strip() for t in repo_raw if isinstance(t, str)]
            elif isinstance(repo_raw, dict) and "tickers" in repo_raw:
                base_tickers = [str(t).strip() for t in repo_raw["tickers"] if isinstance(t, str)]
            else:
                base_tickers = []
        except Exception:
            base_tickers = []
    else:
        base_tickers = []

    # [M5] Validar contra anchor universe
    anchor_tickers = _load_anchor_tickers()

    if anchor_tickers:
        # Separar válidos de fantasma
        valid_base   = [t for t in base_tickers if t.upper() in anchor_tickers]
        phantom      = [t for t in base_tickers if t.upper() not in anchor_tickers]

        if phantom:
            logger.warning(
                f"⚠️ Tickers fantasma ignorados ({len(phantom)}) — "
                f"no están en anchor_universe.json: {phantom}"
            )

        base_tickers = valid_base

    merged = sorted(set(disk_tickers) | set(base_tickers))

    if merged != disk_tickers:
        save_json(TICKERS_FILE, merged)
        logger.info(
            f"📈 tickers.json actualizado | total={len(merged)} "
            f"(antes={len(disk_tickers)})"
        )
    else:
        logger.info("✔ tickers.json ya estaba actualizado")

# =========================================================
# MARKET CONTEXT
# =========================================================
def load_market_context() -> MarketOrchestrationContext:
    if not MARKET_CTX_FILE.exists():
        raise RuntimeError("market_context.json no existe")
    data = json.loads(MARKET_CTX_FILE.read_text())
    return MarketOrchestrationContext(**data)

# =========================================================
# FASTAPI APP
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Trading Suite v2.9.4 START")
    merge_tickers_on_startup()
    start_scheduler()
    yield
    logger.info("🛑 Trading Suite v2.9.4 STOP")

app = FastAPI(
    title="Trading Suite Enterprise",
    version="2.9.4",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# ROUTERS
# =========================================================
app.include_router(dashboard_router)
app.include_router(pipeline_router)
app.include_router(signals_router)
app.include_router(debug_router)
app.include_router(admin_router)
app.include_router(trading_router)
app.include_router(alpha_router)
app.include_router(performance_router)
app.include_router(analysis_router)
app.include_router(execution_router)

# =========================================================
# PIPELINE COMMIT
# =========================================================
@app.post("/internal/pipeline/commit")
async def pipeline_commit(payload: Dict[str, Any], request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    logger.info("💾 Committing pipeline payload")

    try:
        if "screener" in payload:
            save_json(SCREENER_FILE, payload["screener"])
        if "market_ctx" in payload:
            save_json(MARKET_CTX_FILE, payload["market_ctx"])
        save_json(PIPELINE_AUDIT_FILE, payload)
    except Exception as e:
        logger.error("❌ Commit failed")
        raise HTTPException(500, str(e))

    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# =========================================================
# TRADING
# =========================================================
@app.post("/internal/trading/run")
async def trading_run(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    market_ctx   = load_market_context()
    orchestrator = TradingOrchestrator()
    result       = await orchestrator.run(market_ctx.to_dict())

    return {
        "status":      "ok",
        "market_mode": market_ctx.market_mode,
        "result":      result,
        "timestamp":   datetime.utcnow().isoformat(),
    }

# =========================================================
# [M1] MONITOR CLOSE — cierre defensivo horario
# =========================================================
@app.post("/internal/trading/monitor-close")
async def monitor_close(payload: Dict[str, Any], request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    tickers = payload.get("tickers", [])
    reason  = payload.get("reason", "monitor_close")

    if not tickers:
        return {"status": "ok", "closed": [], "message": "sin tickers"}

    logger.info(f"🔴 Monitor close | {tickers} | {reason}")

    orchestrator = TradingOrchestrator()

    try:
        from portfolio_store import load_positions
        positions_snapshot = load_positions()
    except Exception:
        positions_snapshot = []

    try:
        import json as _json
        from pathlib import Path as _Path
        _alpha_file = _Path(os.getenv("DATA_PATH", "/data")) / "alpha_last.json"
        _alpha_data = _json.loads(_alpha_file.read_text()) if _alpha_file.exists() else {}
        alpha_map   = {
            t.upper(): d.get("alpha_score", 0.0)
            for t, d in _alpha_data.get("results", {}).items()
            if isinstance(d, dict)
        }
    except Exception:
        alpha_map = {}

    closed = []
    errors = []

    for ticker in tickers:
        try:
            result = await asyncio.wait_for(
                orchestrator.broker.execute_decision({
                    "action": "CLOSE",
                    "ticker": ticker,
                    "reason": reason,
                }),
                timeout=30,
            )
            closed.append(ticker)
            logger.info(f"⚰️ Monitor closed {ticker}")

            try:
                from positions_meta import remove_entry
                remove_entry(ticker)
            except Exception:
                pass

            try:
                from darwin_engine.trade_tracker import register_close

                exit_price = 0.0
                if result and hasattr(result, "filled_avg_price") and result.filled_avg_price:
                    exit_price = float(result.filled_avg_price)
                elif result and isinstance(result, dict):
                    exit_price = float(
                        result.get("filled_avg_price")
                        or result.get("fill_price")
                        or result.get("price")
                        or 0
                    )

                if exit_price <= 0:
                    exit_price = next(
                        (
                            float(p.get("price_now") or p.get("current_price") or p.get("avg_entry_price") or 0)
                            for p in positions_snapshot
                            if str(p.get("ticker", "")).upper() == ticker.upper()
                        ),
                        0.0,
                    )

                alpha_at_close = alpha_map.get(ticker.upper(), 0.0)
                logger.info(f"📝 Darwin monitor-close {ticker} @ ${exit_price:.2f}")

                register_close(
                    ticker      = ticker,
                    exit_price  = exit_price,
                    reason      = reason,
                    alpha_score = alpha_at_close,
                )
            except Exception as _de:
                logger.warning(f"⚠️ darwin monitor-close {ticker}: {_de}")

        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
            logger.error(f"❌ Monitor close {ticker}: {e}")

    return {
        "status":    "ok",
        "closed":    closed,
        "errors":    errors,
        "requested": tickers,
        "timestamp": datetime.utcnow().isoformat(),
    }


# =========================================================
# [M2] DARWIN RUN — ciclo evolutivo manual
# =========================================================
@app.post("/internal/darwin/run")
async def darwin_run(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    try:
        body    = await request.json()
        dry_run = body.get("dry_run", False)
    except Exception:
        dry_run = False

    results = {}

    try:
        from darwin_engine.trade_tracker import resolve_pending_trades
        resolved = resolve_pending_trades()
        results["trades_resolved"] = len(resolved)
        logger.info(f"🧬 Darwin resolve: {len(resolved)} trades resueltos")
    except Exception as e:
        results["trades_resolved_error"] = str(e)

    try:
        from darwin_engine.arena import run_evolution_cycle
        executor_result = run_evolution_cycle(dry_run=dry_run)
        results["executor"] = {
            "champion":       executor_result.get("champion_after"),
            "was_promoted":   executor_result.get("was_promoted"),
            "new_generation": executor_result.get("new_generation", []),
        }
    except Exception as e:
        results["executor_error"] = str(e)

    try:
        from darwin_engine.predictor_arena import run_predictor_evolution
        predictor_result = run_predictor_evolution(dry_run=dry_run)
        results["predictors"] = {
            "h_evaluated": predictor_result.get("h_evaluated"),
            "promotions":  predictor_result.get("promotions"),
            "ranking":     predictor_result.get("ranking", []),
        }
    except Exception as e:
        results["predictors_error"] = str(e)

    return {
        "status":    "ok",
        "dry_run":   dry_run,
        "timestamp": datetime.utcnow().isoformat(),
        "results":   results,
    }

# =========================================================
# [M3] DARWIN STATUS
# =========================================================
@app.get("/internal/darwin/status")
async def darwin_status(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    status = {}

    try:
        from darwin_engine.trade_tracker import get_tracker_summary
        status["tracker"] = get_tracker_summary()
    except Exception as e:
        status["tracker_error"] = str(e)

    try:
        from darwin_engine.predictor_genome import PredictorGenome
        predictors = {}
        for h in range(1, 11):
            genome = PredictorGenome.load_champion(h)
            predictors[f"H{h}"] = {
                "genome_id":  genome.genome_id,
                "generation": genome.generation,
                "hit_rate":   genome.hit_rate,
                "n_evals":    genome.data.get("n_evaluations", 0),
            }
        status["predictors"] = predictors
    except Exception as e:
        status["predictors_error"] = str(e)

    try:
        from darwin_engine.arena import get_champion
        champion = get_champion()
        if champion:
            status["executor"] = {
                "genome_id":  champion.genome_id,
                "generation": champion.generation,
                "fitness":    champion.fitness,
            }
    except Exception as e:
        status["executor_error"] = str(e)

    return {
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "darwin":    status,
    }

# =========================================================
# INTERNAL INSPECTION
# =========================================================
@app.get("/internal/pipeline-last")
async def internal_pipeline_last(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")
    return load_json(PIPELINE_AUDIT_FILE, {})


@app.get("/internal/pipeline/last")
async def get_last_pipeline(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")
    if not PIPELINE_AUDIT_FILE.exists():
        return {"status": "no_pipeline_executed"}
    return load_json(PIPELINE_AUDIT_FILE, {})


@app.get("/internal/trades-today")
async def internal_trades_today(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")

    today = datetime.utcnow().strftime("%Y-%m-%d")
    fp    = DATA_PATH / "trades" / f"trades_{today}.jsonl"

    if not fp.exists():
        return []

    out = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


@app.get("/internal/positions")
async def internal_positions(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403, "Invalid pipeline key")
    from portfolio_store import load_positions
    return load_positions()


# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "status":  "ok",
        "service": "spy-2w-price-prediction",
        "env":     "production"
    }

@app.post("/internal/debug/fix-entry-dates")
async def fix_entry_dates(request: Request):
    if request.headers.get("X-PIPELINE-KEY") != config.PIPELINE_KEY:
        raise HTTPException(403)

    try:
        from positions_meta import get_all, set_entry_date
        from portfolio_store import load_positions

        meta    = get_all()
        fixed   = []
        skipped = []

        ESTIMATED_DATES = {
            "GLW":  "2026-04-21",
            "JNJ":  "2026-04-21",
            "VZ":   "2026-04-21",
            "WMT":  "2026-04-21",
            "GILD": "2026-04-21",
            "PG":   "2026-04-21",
            "KO":   "2026-04-21",
            "AMAT": "2026-04-22",
            "CEG":  "2026-04-27",
            "FDX":  "2026-04-27",
            "CAT":  "2026-05-05",
            "DOW":  "2026-05-05",
            "MPC":  "2026-05-05",
        }

        for ticker, data in meta.items():
            if not data.get("entry_date"):
                date = ESTIMATED_DATES.get(ticker.upper(), "2026-04-21")
                set_entry_date(ticker, date)
                fixed.append(f"{ticker} → {date}")
            else:
                skipped.append(ticker)

        return {"status": "ok", "fixed": fixed, "skipped": skipped}

    except Exception as e:
        return {"error": str(e)}

# =========================================================
# SHUTDOWN
# =========================================================
def handle_shutdown(signum, frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.PORT,
        log_level="error",
    )
    

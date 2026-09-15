# =========================================================
# performance_router.py — V6
# =========================================================
# V4 → V5:
#   [M1] model-quality agrega métricas de tendencia reciente:
#        - hit_rate_7d_pct   : últimos 7 días
#        - hit_rate_14d_pct  : últimos 14 días
#        - hit_rate_30d_pct  : últimos 30 días
#        - trend             : "mejorando" | "empeorando" | "estable"
#        Permite ver si el fix del theta está funcionando
#        sin resetear el historial completo.
#
# V5 → V6:
#   [M2] (2026-09-15) FIX CRASH OOM — /model-quality dejó de escanear
#        y parsear los ~15,000 archivos de /data/evaluations/ en cada
#        request. Esa lectura completa, disparada cada vez que alguien
#        abría el dashboard, fue la causa confirmada de un "Ran out of
#        memory (used over 512MB)" en Render.
#        Ahora lee /data/model_quality_history.json — un registro
#        pequeño POR DÍA (hits/total agregados), que evaluator.py
#        [E10] actualiza automáticamente al terminar cada corrida.
#        Se conserva el histórico completo día por día (necesario para
#        ver si el sistema mejora con el tiempo) sin volver a abrir
#        ningún archivo de evaluación individual desde el dashboard.
# =========================================================

import os
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter
from broker import get_engine

DATA_PATH                  = Path(os.getenv("DATA_PATH", "/data"))
META_FILE                  = DATA_PATH / "account_meta.json"
EQUITY_SNAPSHOTS_FILE      = DATA_PATH / "equity_snapshots.json"
MODEL_QUALITY_HISTORY_FILE = DATA_PATH / "model_quality_history.json"  # [M2]

router = APIRouter(prefix="/dashboard", tags=["performance"])


# =========================================================
# HELPERS
# =========================================================

def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def list_evaluation_files() -> List[Path]:
    """
    Restaurada tal cual estaba en V5 — [M2] ya no se usa dentro de
    /model-quality (ver nota de arriba), pero execution_analyzer.py
    la importa directo desde este módulo. Eliminarla rompió el deploy
    (ImportError en main.py al cargar execution_analyzer). Se mantiene
    aquí sin cambios para no romper ese otro consumidor.
    """
    root = DATA_PATH / "evaluations"
    if not root.exists():
        return []
    files = []
    for ticker_dir in root.iterdir():
        if ticker_dir.is_dir():
            files.extend(ticker_dir.glob("*.json"))
    return sorted(files)


# =========================================================
# SNAPSHOT DIARIO
# =========================================================

def record_equity_snapshot(equity: float) -> None:
    today     = datetime.now(timezone.utc).date().isoformat()
    snapshots = load_json(EQUITY_SNAPSHOTS_FILE) or {}
    if today not in snapshots:
        snapshots[today] = round(equity, 2)
        save_json(EQUITY_SNAPSHOTS_FILE, snapshots)


# =========================================================
# BLOQUE BROKER — /dashboard/performance
# =========================================================

@router.get("/performance")
async def performance():
    try:
        engine  = get_engine()
        account = await engine.get_account()
        equity  = float(account.equity)
    except Exception:
        equity = None

    meta = load_json(META_FILE)

    if not meta and equity is not None:
        meta = {
            "initial_equity":  equity,
            "start_date":      datetime.now(timezone.utc).isoformat(),
            "high_water_mark": equity,
        }
        save_json(META_FILE, meta)

    total_return_pct = None
    drawdown_pct     = None
    high_water_mark  = None

    if meta and equity is not None:
        initial_equity  = float(meta["initial_equity"])
        high_water_mark = float(meta.get("high_water_mark", equity))

        if equity > high_water_mark:
            high_water_mark          = equity
            meta["high_water_mark"]  = equity
            save_json(META_FILE, meta)

        total_return_pct = round((equity - initial_equity) / initial_equity * 100, 2)
        drawdown_pct     = round((equity - high_water_mark) / high_water_mark * 100, 2)

        record_equity_snapshot(equity)

    return {
        "equity":           equity,
        "initial_equity":   float(meta["initial_equity"]) if meta else None,
        "total_return_pct": total_return_pct,
        "drawdown_pct":     drawdown_pct,
        "high_water_mark":  high_water_mark,
        "since":            meta["start_date"] if meta else None,
    }


# =========================================================
# BLOQUE BROKER — /dashboard/equity-curve
# =========================================================

@router.get("/equity-curve")
async def equity_curve():
    curve  = []
    source = "alpaca"

    try:
        engine  = get_engine()
        history = engine.client.get_portfolio_history(
            period="1M",
            timeframe="1D",
            intraday_reporting="market_hours",
        )
        if history and history.equity:
            for ts, eq in zip(history.timestamp, history.equity):
                if eq is None or eq == 0:
                    continue
                date_str = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
                curve.append({"date": date_str, "equity": round(float(eq), 2)})
            source = "alpaca"
    except Exception:
        source = "snapshots"

    if not curve:
        snapshots = load_json(EQUITY_SNAPSHOTS_FILE) or {}
        for date in sorted(snapshots.keys()):
            curve.append({"date": date, "equity": snapshots[date]})
        source = "snapshots"

    if curve:
        base = curve[0]["equity"]
        for point in curve:
            point["return_pct"] = round(
                (point["equity"] - base) / base * 100, 2
            ) if base > 0 else 0.0

    meta           = load_json(META_FILE)
    initial_equity = float(meta["initial_equity"]) if meta else None
    current_equity = curve[-1]["equity"] if curve else None

    return {
        "curve":          curve,
        "initial_equity": initial_equity,
        "current_equity": current_equity,
        "n_days":         len(curve),
        "source":         source,
        "updated_at":     datetime.now(timezone.utc).isoformat(),
    }


# =========================================================
# [M2] BLOQUE MODELO — /dashboard/model-quality
# Lee el historial diario liviano (model_quality_history.json) en vez
# de reprocesar los ~15,000 archivos de /data/evaluations/. El
# historial lo mantiene evaluator.py::_update_daily_history() [E10],
# un registro por día con hits/total ya agregados.
# =========================================================

def _pct(hits: int, total: int) -> Optional[float]:
    return round(hits / total * 100, 2) if total > 0 else None


def _avg_error(sum_error: float, n_error: int) -> Optional[float]:
    return round(sum_error / n_error, 2) if n_error > 0 else None


@router.get("/model-quality")
async def model_quality():

    history = load_json(MODEL_QUALITY_HISTORY_FILE) or []

    empty = {
        "hit_rate_direction_pct": None,
        "hit_rate_7d_pct":        None,
        "hit_rate_14d_pct":       None,
        "hit_rate_30d_pct":       None,
        "trend":                  None,
        "avg_error_pct":          None,
        "evaluated":              0,
        "pending":                None,  # [M2] ya no se calcula acá (requeriría escanear predictions/)
        "total":                  0,
        "by_recommendation":      {},
        "by_horizon":             {},
    }

    if not history:
        return empty

    # ── Ventanas temporales ──────────────────────────────────────
    now     = datetime.now(timezone.utc).date()
    cut_7d  = (now - timedelta(days=7)).isoformat()
    cut_14d = (now - timedelta(days=14)).isoformat()
    cut_30d = (now - timedelta(days=30)).isoformat()

    def _window_hit_rate(cutoff: str) -> Optional[float]:
        hits  = sum(e.get("hits_dir", 0)  for e in history if e.get("date", "") >= cutoff)
        total = sum(e.get("total_dir", 0) for e in history if e.get("date", "") >= cutoff)
        return _pct(hits, total) if total >= 5 else None

    hr_7d  = _window_hit_rate(cut_7d)
    hr_14d = _window_hit_rate(cut_14d)
    hr_30d = _window_hit_rate(cut_30d)

    # ── Global histórico (suma de todos los días guardados) ──────
    total_hits  = sum(e.get("hits_dir", 0)  for e in history)
    total_dir   = sum(e.get("total_dir", 0) for e in history)
    total_sum_e = sum(e.get("sum_error", 0.0) for e in history)
    total_n_e   = sum(e.get("n_error", 0)     for e in history)

    hit_rate_all = _pct(total_hits, total_dir)
    avg_error    = _avg_error(total_sum_e, total_n_e)

    # ── Tendencia: compara 7d vs 30d (o vs histórico) ────────────
    trend = None
    ref   = hr_30d if hr_30d is not None else hit_rate_all
    if hr_7d is not None and ref is not None:
        diff = hr_7d - ref
        if diff >= 3:
            trend = "mejorando"
        elif diff <= -3:
            trend = "empeorando"
        else:
            trend = "estable"

    # ── by_recommendation / by_horizon — suma across TODO el historial ──
    by_rec: Dict[str, Dict[str, Any]] = {}
    by_horizon: Dict[str, Dict[str, Any]] = {}

    for e in history:
        for rec, s in (e.get("by_recommendation") or {}).items():
            acc = by_rec.setdefault(rec, {"total": 0, "hits": 0, "sum_error": 0.0, "n_error": 0})
            acc["total"]     += s.get("total", 0)
            acc["hits"]      += s.get("hits", 0)
            acc["sum_error"] += s.get("sum_error", 0.0)
            acc["n_error"]   += s.get("n_error", 0)

        for hkey, s in (e.get("by_horizon") or {}).items():
            acc = by_horizon.setdefault(hkey, {"total": 0, "hits": 0, "sum_error": 0.0, "n_error": 0})
            acc["total"]     += s.get("total", 0)
            acc["hits"]      += s.get("hits", 0)
            acc["sum_error"] += s.get("sum_error", 0.0)
            acc["n_error"]   += s.get("n_error", 0)

    by_rec_out = {
        rec: {
            "total":         s["total"],
            "hit_rate_pct":  _pct(s["hits"], s["total"]),
            "avg_error_pct": _avg_error(s["sum_error"], s["n_error"]),
        }
        for rec, s in by_rec.items()
    }

    by_horizon_out = {
        hkey: {
            "total":         s["total"],
            "hit_rate_pct":  _pct(s["hits"], s["total"]),
            "avg_error_pct": _avg_error(s["sum_error"], s["n_error"]),
        }
        for hkey, s in sorted(by_horizon.items())
    }

    return {
        # Global histórico
        "hit_rate_direction_pct": hit_rate_all,
        "avg_error_pct":          avg_error,
        "evaluated":              total_dir,
        "pending":                None,  # [M2] ver nota en `empty`
        "total":                  total_dir,

        # Ventanas recientes
        "hit_rate_7d_pct":        hr_7d,
        "hit_rate_14d_pct":       hr_14d,
        "hit_rate_30d_pct":       hr_30d,
        "trend":                  trend,
        "recent_window_sizes": {
            "7d":  sum(1 for e in history if e.get("date", "") >= cut_7d),
            "14d": sum(1 for e in history if e.get("date", "") >= cut_14d),
            "30d": sum(1 for e in history if e.get("date", "") >= cut_30d),
        },

        "by_recommendation": by_rec_out,
        "by_horizon":        by_horizon_out,
        "updated_at":        datetime.now(timezone.utc).isoformat(),
    }

# =========================================================
# performance_router.py — V4
# =========================================================
# Dos bloques completamente separados:
#
# [BROKER]  → todo viene de Alpaca (equity real)
#   - /dashboard/performance       → KPIs reales
#   - /dashboard/equity-curve      → historial real (get_portfolio_history
#                                    con fallback a snapshots en disco)
#
# [MODELO]  → todo viene de /data/evaluations/
#   - /dashboard/model-quality     → hit rate direccional, error,
#                                    desglose H1→H10
#
# NUNCA se mezclan las dos fuentes.
# =========================================================

import os
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from fastapi import APIRouter

from broker import get_engine

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
META_FILE = DATA_PATH / "account_meta.json"
EQUITY_SNAPSHOTS_FILE = DATA_PATH / "equity_snapshots.json"  # fallback diario

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
    root = DATA_PATH / "evaluations"
    if not root.exists():
        return []
    files = []
    for ticker_dir in root.iterdir():
        if ticker_dir.is_dir():
            files.extend(ticker_dir.glob("*.json"))
    return sorted(files)


# =========================================================
# SNAPSHOT DIARIO (fallback para equity curve)
# Guarda equity de hoy en disco si aún no existe entrada para hoy
# =========================================================

def record_equity_snapshot(equity: float) -> None:
    """Persiste un punto diario de equity para construir curva histórica."""
    today = datetime.now(timezone.utc).date().isoformat()
    snapshots: Dict = load_json(EQUITY_SNAPSHOTS_FILE) or {}

    # Solo graba una vez por día (no sobreescribe si ya existe)
    if today not in snapshots:
        snapshots[today] = round(equity, 2)
        save_json(EQUITY_SNAPSHOTS_FILE, snapshots)


# =========================================================
# BLOQUE BROKER — /dashboard/performance
# =========================================================

@router.get("/performance")
async def performance():
    """KPIs reales desde Alpaca. Sin mezcla con evaluaciones."""

    try:
        engine = get_engine()
        account = await engine.get_account()
        equity = float(account.equity)
    except Exception:
        equity = None

    # ── account_meta: guarda initial_equity la primera vez ──────
    meta = load_json(META_FILE)

    if not meta and equity is not None:
        meta = {
            "initial_equity": equity,
            "start_date": datetime.now(timezone.utc).isoformat(),
            "high_water_mark": equity,
        }
        save_json(META_FILE, meta)

    total_return_pct = None
    drawdown_pct = None
    high_water_mark = None

    if meta and equity is not None:
        initial_equity = float(meta["initial_equity"])
        high_water_mark = float(meta.get("high_water_mark", equity))

        # Actualiza high water mark
        if equity > high_water_mark:
            high_water_mark = equity
            meta["high_water_mark"] = equity
            save_json(META_FILE, meta)

        total_return_pct = round(
            (equity - initial_equity) / initial_equity * 100, 2
        )
        drawdown_pct = round(
            (equity - high_water_mark) / high_water_mark * 100, 2
        )

        # Registra snapshot diario para fallback de equity curve
        record_equity_snapshot(equity)

    return {
        "equity":            equity,
        "initial_equity":    float(meta["initial_equity"]) if meta else None,
        "total_return_pct":  total_return_pct,
        "drawdown_pct":      drawdown_pct,
        "high_water_mark":   high_water_mark,
        "since":             meta["start_date"] if meta else None,
    }


# =========================================================
# BLOQUE BROKER — /dashboard/equity-curve
# Fuente 1: Alpaca get_portfolio_history()
# Fuente 2 (fallback): snapshots diarios guardados en disco
# =========================================================

@router.get("/equity-curve")
async def equity_curve():
    """
    Curva de equity real desde Alpaca.
    Fallback a snapshots en disco si Alpaca no está disponible.
    """
    curve = []
    source = "alpaca"

    # ── Intento 1: Alpaca portfolio history ─────────────────────
    try:
        engine = get_engine()
        # get_portfolio_history devuelve timestamps + equity points
        history = engine.client.get_portfolio_history(
            period="1M",          # último mes
            timeframe="1D",       # un punto por día
            intraday_reporting="market_hours",
        )

        if history and history.equity:
            for ts, eq in zip(history.timestamp, history.equity):
                if eq is None or eq == 0:
                    continue
                date_str = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
                curve.append({
                    "date":   date_str,
                    "equity": round(float(eq), 2),
                })
            source = "alpaca"

    except Exception as e:
        # Alpaca paper puede no soportar portfolio history → fallback
        source = "snapshots"

    # ── Intento 2: snapshots en disco ───────────────────────────
    if not curve:
        snapshots: Dict = load_json(EQUITY_SNAPSHOTS_FILE) or {}
        for date in sorted(snapshots.keys()):
            curve.append({
                "date":   date,
                "equity": snapshots[date],
            })
        source = "snapshots"

    # ── Calcular return_pct por punto ───────────────────────────
    if curve:
        base = curve[0]["equity"]
        for point in curve:
            point["return_pct"] = round(
                (point["equity"] - base) / base * 100, 2
            ) if base > 0 else 0.0

    meta = load_json(META_FILE)
    initial_equity = float(meta["initial_equity"]) if meta else None
    current_equity = curve[-1]["equity"] if curve else None

    return {
        "curve":           curve,
        "initial_equity":  initial_equity,
        "current_equity":  current_equity,
        "n_days":          len(curve),
        "source":          source,   # "alpaca" o "snapshots"
        "updated_at":      datetime.now(timezone.utc).isoformat(),
    }


# =========================================================
# BLOQUE MODELO — /dashboard/model-quality
# Solo evalúa qué tan bien predice la DIRECCIÓN el modelo.
# No mezcla con PnL real del broker.
# =========================================================

@router.get("/model-quality")
async def model_quality():
    """
    Calidad predictiva del modelo, independiente del PnL real.

    Métricas:
    - hit_rate_direction_pct : % de predicciones que acertaron la dirección
    - avg_error_pct          : error promedio de retorno predicho vs real
    - by_horizon             : desglose H1→H10 (qué horizonte predice mejor)
    - by_recommendation      : COMPRA / VENDE / MANTÉN por separado
    - evaluated / pending    : cobertura del modelo
    """

    files = list_evaluation_files()

    empty = {
        "hit_rate_direction_pct": None,
        "avg_error_pct":          None,
        "evaluated":              0,
        "pending":                0,
        "total":                  0,
        "by_recommendation":      {},
        "by_horizon":             {},
    }

    if not files:
        return empty

    total       = 0
    evaluated   = 0
    hits_dir    = 0
    errors      = []

    # Por recomendación
    rec_stats: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "hits": 0, "errors": []})

    # Por horizonte H1→H10 (desde models_diagnostics)
    horizon_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "total": 0, "errors": []})

    for f in files:
        data = load_json(f)
        if not data:
            continue

        total += 1

        if data.get("real_return_pct") is None:
            continue

        evaluated += 1

        rec = (data.get("recommendation") or "HOLD").strip().upper()
        real_ret  = float(data["real_return_pct"])
        pred_ret  = float(data.get("predicted_return_pct", 0))

        # ── Hit direccional (signo correcto) ────────────────────
        hit = bool(np.sign(real_ret) == np.sign(pred_ret)) if pred_ret != 0 else False
        if hit:
            hits_dir += 1

        # ── Error de retorno ────────────────────────────────────
        if data.get("error_return_pct") is not None:
            err = float(data["error_return_pct"])
            errors.append(err)
            rec_stats[rec]["errors"].append(err)

        rec_stats[rec]["total"] += 1
        if hit:
            rec_stats[rec]["hits"] += 1

        # ── Desglose por horizonte H1→H10 ───────────────────────
        models_diag = data.get("models_diagnostics") or {}
        for hkey, hdata in models_diag.items():
            if not isinstance(hdata, dict):
                continue
            horizon_stats[hkey]["total"] += 1
            if hdata.get("hit_sign"):
                horizon_stats[hkey]["hits"] += 1
            if hdata.get("error_pct") is not None:
                horizon_stats[hkey]["errors"].append(float(hdata["error_pct"]))

    pending = total - evaluated

    hit_rate_dir = round((hits_dir / evaluated) * 100, 2) if evaluated > 0 else None
    avg_error    = round(float(np.mean(errors)), 2) if errors else None

    # ── Formatear by_recommendation ─────────────────────────────
    by_rec = {}
    for rec, s in rec_stats.items():
        by_rec[rec] = {
            "total":            s["total"],
            "hit_rate_pct":     round((s["hits"] / s["total"]) * 100, 1) if s["total"] else None,
            "avg_error_pct":    round(float(np.mean(s["errors"])), 2) if s["errors"] else None,
        }

    # ── Formatear by_horizon ────────────────────────────────────
    by_horizon = {}
    for hkey in sorted(horizon_stats.keys()):
        s = horizon_stats[hkey]
        by_horizon[hkey] = {
            "total":         s["total"],
            "hit_rate_pct":  round((s["hits"] / s["total"]) * 100, 1) if s["total"] else None,
            "avg_error_pct": round(float(np.mean(s["errors"])), 2) if s["errors"] else None,
        }

    return {
        "hit_rate_direction_pct": hit_rate_dir,
        "avg_error_pct":          avg_error,
        "evaluated":              evaluated,
        "pending":                pending,
        "total":                  total,
        "by_recommendation":      by_rec,
        "by_horizon":             by_horizon,
        "updated_at":             datetime.now(timezone.utc).isoformat(),
    }

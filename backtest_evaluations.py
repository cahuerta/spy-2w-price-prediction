# =========================================================
# backtest_evaluations.py — SHARPE/DRAWDOWN REAL DESDE EVALUACIONES
# =========================================================
# Construye una curva de equity simulada a partir de las
# evaluaciones YA EXISTENTES en /data/evaluations/ — sin
# retraining, sin backtest walk-forward, sin look-ahead bias
# (las evaluaciones ya fueron generadas prediction-first).
#
# Responde: "el hit_rate de 51-54% en H4-H9, ¿se traduce en
# Sharpe > 1.0 real, o es ruido?" — usando datos que el sistema
# ya generó en producción.
#
# Dos niveles de análisis:
#   1. ENSEMBLE — usa 'recommendation' (COMPRA/VENDE), ignora
#      MANTÉN (el sistema no habría operado ahí).
#   2. POR HORIZONTE (H1-H10) — usa models_diagnostics, dirección
#      implícita = signo de pred_return.
# =========================================================

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np

logger = logging.getLogger("backtest_evaluations")
logging.basicConfig(level=logging.INFO)

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
EVAL_ROOT = DATA_PATH / "evaluations"

TRADING_DAYS_YEAR = 252


# =========================================================
# HELPERS
# =========================================================

def _load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sharpe(returns: List[float]) -> Optional[float]:
    """Sharpe anualizado. None si <5 muestras o std=0."""
    if len(returns) < 5:
        return None
    arr = np.array(returns, dtype=float)
    std = arr.std()
    if std < 1e-9:
        return None
    return float(arr.mean() / std * np.sqrt(TRADING_DAYS_YEAR))


def _max_drawdown(returns: List[float]) -> Optional[float]:
    """Max drawdown desde retornos porcentuales encadenados."""
    if not returns:
        return None
    equity = np.cumprod(1 + np.array(returns, dtype=float) / 100)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    return float(abs(dd.min()))


def _win_rate(returns: List[float]) -> Optional[float]:
    if not returns:
        return None
    return float(np.mean([r > 0 for r in returns]))


# =========================================================
# RESULTADO POR GRUPO (ensemble o un horizonte)
# =========================================================

@dataclass
class BacktestResult:
    label:          str
    n_trades:       int
    n_evaluations:  int          # incluye señales débiles descartadas
    sharpe:         Optional[float]
    max_drawdown:   Optional[float]
    win_rate:       Optional[float]
    total_return_pct: Optional[float]
    avg_return_pct:   Optional[float]


def _compute_result(label: str, dated_returns: List[tuple], n_evaluations: int) -> BacktestResult:
    """dated_returns: lista de (fecha_str, retorno_pct), se ordena cronológicamente."""
    dated_returns.sort(key=lambda x: x[0])
    returns = [r for _, r in dated_returns]

    total_return = None
    if returns:
        equity = np.cumprod(1 + np.array(returns) / 100)
        total_return = float((equity[-1] - 1) * 100)

    return BacktestResult(
        label=label,
        n_trades=len(returns),
        n_evaluations=n_evaluations,
        sharpe=_sharpe(returns),
        max_drawdown=_max_drawdown(returns),
        win_rate=_win_rate(returns),
        total_return_pct=round(total_return, 2) if total_return is not None else None,
        avg_return_pct=round(float(np.mean(returns)), 4) if returns else None,
    )


# =========================================================
# CARGA Y PROCESAMIENTO DE EVALUACIONES
# =========================================================

def _iter_evaluations():
    """Generador: recorre todas las evaluaciones en disco."""
    if not EVAL_ROOT.exists():
        return
    for ticker_dir in sorted(EVAL_ROOT.iterdir()):
        if not ticker_dir.is_dir():
            continue
        for f in sorted(ticker_dir.glob("*.json")):
            data = _load_json(f)
            if data:
                yield ticker_dir.name, data


def run_backtest_from_evaluations() -> Dict[str, Any]:
    """
    Construye curvas de equity simuladas desde evaluaciones existentes.
    Retorna: {"ensemble": BacktestResult, "by_horizon": {H1: ..., H2: ...}}
    """
    ensemble_returns: List[tuple] = []
    ensemble_n_eval = 0

    horizon_returns: Dict[str, List[tuple]] = defaultdict(list)
    horizon_n_eval:  Dict[str, int] = defaultdict(int)

    for ticker, ev in _iter_evaluations():
        # ── Filtro de calidad — igual criterio que signals.py [S6] ──
        if ev.get("legacy_bad_horizon") is True:
            continue
        if ev.get("alpaca_unsupported") is True:
            continue

        pred_date = ev.get("prediction_date", "")

        # ── ENSEMBLE (recommendation COMPRA/VENDE) ──
        rec = (ev.get("recommendation") or "").upper()
        real_ret = ev.get("real_return_pct")
        if real_ret is not None and rec in ("COMPRA", "VENDE"):
            ensemble_n_eval += 1
            direction = 1 if rec == "COMPRA" else -1
            ensemble_returns.append((pred_date, direction * float(real_ret)))

        # ── POR HORIZONTE (models_diagnostics) ──
        diag = ev.get("models_diagnostics") or {}
        for hkey, hdata in diag.items():
            if not isinstance(hdata, dict):
                continue
            pred_ret = hdata.get("pred_return")
            real_ret_h = hdata.get("real_return")
            if pred_ret is None or real_ret_h is None:
                continue

            horizon_n_eval[hkey] += 1

            # Señal débil (mismo umbral que evaluator.py) → no simular trade
            if abs(float(pred_ret)) < 0.05:
                continue

            direction = 1 if float(pred_ret) > 0 else -1
            horizon_returns[hkey].append((pred_date, direction * float(real_ret_h)))

    result = {
        "ensemble": asdict(_compute_result("Ensemble (COMPRA/VENDE)", ensemble_returns, ensemble_n_eval)),
        "by_horizon": {},
    }

    for h in [f"H{i}" for i in range(1, 11)]:
        rets = horizon_returns.get(h, [])
        n_ev = horizon_n_eval.get(h, 0)
        result["by_horizon"][h] = asdict(_compute_result(h, rets, n_ev))

    result["generated_at"] = datetime.utcnow().isoformat()
    return result


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    result = run_backtest_from_evaluations()

    print("\n=== ENSEMBLE (COMPRA/VENDE) ===")
    e = result["ensemble"]
    print(f"Trades: {e['n_trades']} (de {e['n_evaluations']} evaluaciones)")
    print(f"Sharpe: {e['sharpe']}")
    print(f"Max Drawdown: {e['max_drawdown']}")
    print(f"Win Rate: {e['win_rate']}")
    print(f"Retorno Total: {e['total_return_pct']}%")

    print("\n=== POR HORIZONTE ===")
    for h, r in result["by_horizon"].items():
        if r["n_trades"] == 0:
            continue
        print(
            f"{h}: trades={r['n_trades']} sharpe={r['sharpe']} "
            f"dd={r['max_drawdown']} win={r['win_rate']} "
            f"ret_total={r['total_return_pct']}%"
        )

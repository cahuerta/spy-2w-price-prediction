# =========================================================
# backtest_evaluations.py — SHARPE/DRAWDOWN REAL DESDE EVALUACIONES
# =========================================================
# Construye una curva de equity simulada a partir de las
# evaluaciones YA EXISTENTES en /data/evaluations/ — sin
# retraining, sin backtest walk-forward, sin look-ahead bias.
#
# FIX v1.2 — CORRECCIÓN MATEMÁTICA DE FONDO:
#   [BF2] Los retornos evaluados son MULTI-DÍA (ej. H9 = retorno
#         acumulado en 9 días), pero se estaban insertando en la
#         curva de equity como si fueran retornos de 1 solo día.
#         Esto no es un problema de anualización (sqrt(N)) — es un
#         error de UNIDADES: componer un retorno de 9 días como si
#         fuera diario infla la curva exponencialmente sin sentido,
#         sin importar qué factor de anualización se use después.
#
#         FIX correcto: convertir cada retorno multi-día a su tasa
#         diaria equivalente compuesta (CAGR estándar) ANTES de
#         insertarlo en la curva:
#             r_diario = (1 + R_total/100)^(1/horizonte_dias) - 1
#         Con esto, cada punto de la serie es una tasa diaria
#         genuina, y sqrt(252) para anualizar el Sharpe vuelve a
#         ser matemáticamente correcto (antes se combinaban DOS
#         errores: unidades incorrectas + anualización sobre esas
#         unidades incorrectas).
#
#   LIMITACIÓN CONOCIDA, NO RESUELTA AQUÍ (y no trivial de resolver):
#         Los horizontes se solapan en el tiempo — un trade que
#         entra el día N y otro que entra el día N+1 con el mismo
#         horizonte comparten casi todo el mismo período de mercado.
#         Esto genera autocorrelación entre observaciones consecutivas
#         de la serie, y el Sharpe estándar asume independencia. La
#         corrección rigurosa (ajuste tipo Newey-West sobre el error
#         estándar) es matemática adicional real, pendiente para el
#         backtest walk-forward formal — NO improvisada aquí. Este
#         fix corrige la magnitud (unidades), no la autocorrelación.
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
DEFAULT_HORIZON_DAYS = 10  # fallback si una evaluación no trae su horizonte


# =========================================================
# HELPERS
# =========================================================

def _load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _daily_equivalent_return(total_return_pct: float, horizon_days: int) -> float:
    """
    [BF2] Convierte un retorno total sobre `horizon_days` días a su
    tasa diaria equivalente compuesta (CAGR estándar):
        (1 + r)^horizon_days = 1 + total_return_pct/100
        r = (1 + total_return_pct/100)^(1/horizon_days) - 1

    Retorna la tasa en formato porcentual (ej. 0.5 = 0.5% diario).
    """
    horizon_days = max(1, int(horizon_days))
    base = 1.0 + (total_return_pct / 100.0)
    # Protección: una pérdida total (-100% o peor, dato corrupto)
    # no tiene raíz real definida — se trata como pérdida total.
    if base <= 0:
        return -100.0
    daily_factor = base ** (1.0 / horizon_days)
    return (daily_factor - 1.0) * 100.0


def _sharpe(returns: List[float]) -> Optional[float]:
    """
    Sharpe anualizado con sqrt(252). Válido porque, tras [BF2],
    `returns` son tasas diarias equivalentes genuinas — no
    retornos multi-día sin convertir.
    """
    if len(returns) < 5:
        return None
    arr = np.array(returns, dtype=float)
    std = arr.std()
    if std < 1e-9:
        return None
    return float(arr.mean() / std * np.sqrt(TRADING_DAYS_YEAR))


def _max_drawdown(returns: List[float]) -> Optional[float]:
    """Max drawdown sobre la curva de equity de tasas diarias equivalentes."""
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
    label:            str
    n_trades:         int          # trades individuales reales
    n_days:           int          # puntos en la serie diaria agregada
    n_evaluations:    int          # incluye señales débiles descartadas
    sharpe:           Optional[float]
    max_drawdown:     Optional[float]
    win_rate:         Optional[float]
    total_return_pct: Optional[float]
    avg_daily_return_pct: Optional[float]


def _compute_result(
    label: str,
    dated_returns: List[tuple],   # (fecha_str, total_return_pct, horizon_days)
    n_evaluations: int,
) -> BacktestResult:
    """
    [BF2] Cada elemento de dated_returns trae su propio horizon_days.
    Se convierte a tasa diaria equivalente ANTES de agrupar por
    fecha y encadenar — así la curva de equity está en unidades
    correctas y consistentes.
    """
    # win_rate se calcula sobre el retorno TOTAL de cada trade
    # (tiene sentido a nivel de trade individual, sin conversión)
    trade_level_returns = [r for _, r, _ in dated_returns]

    by_date = defaultdict(list)
    for date_str, total_ret, horizon_days in dated_returns:
        if not date_str:
            continue
        daily_equiv = _daily_equivalent_return(total_ret, horizon_days)
        by_date[date_str].append(daily_equiv)

    # Una fecha = un punto de la serie = promedio de las tasas
    # diarias equivalentes de todos los trades entrados ese día
    daily_series = sorted(
        (date_str, float(np.mean(vals))) for date_str, vals in by_date.items()
    )
    daily_returns = [r for _, r in daily_series]

    total_return = None
    if daily_returns:
        equity = np.cumprod(1 + np.array(daily_returns) / 100)
        total_return = float((equity[-1] - 1) * 100)

    return BacktestResult(
        label=label,
        n_trades=len(dated_returns),
        n_days=len(daily_returns),
        n_evaluations=n_evaluations,
        sharpe=_sharpe(daily_returns),
        max_drawdown=_max_drawdown(daily_returns),
        win_rate=_win_rate(trade_level_returns),
        total_return_pct=round(total_return, 2) if total_return is not None else None,
        avg_daily_return_pct=round(float(np.mean(daily_returns)), 4) if daily_returns else None,
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
    # (fecha, retorno_total_pct, horizon_days)
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
            h_days = int(ev.get("evaluation_horizon_days") or DEFAULT_HORIZON_DAYS)
            ensemble_returns.append((pred_date, direction * float(real_ret), h_days))

        # ── POR HORIZONTE (models_diagnostics) ──
        diag = ev.get("models_diagnostics") or {}
        for hkey, hdata in diag.items():
            if not isinstance(hdata, dict):
                continue
            pred_ret   = hdata.get("pred_return")
            real_ret_h = hdata.get("real_return")
            if pred_ret is None or real_ret_h is None:
                continue

            horizon_n_eval[hkey] += 1

            # Señal débil (mismo umbral que evaluator.py) → no simular trade
            if abs(float(pred_ret)) < 0.05:
                continue

            direction = 1 if float(pred_ret) > 0 else -1
            # h_days real de este horizonte, ej. "H9" → 9
            try:
                h_days = int(hkey.replace("H", ""))
            except ValueError:
                h_days = DEFAULT_HORIZON_DAYS

            horizon_returns[hkey].append((pred_date, direction * float(real_ret_h), h_days))

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
    print(f"Trades: {e['n_trades']} (de {e['n_evaluations']} evaluaciones) | Días agregados: {e['n_days']}")
    print(f"Sharpe: {e['sharpe']}")
    print(f"Max Drawdown: {e['max_drawdown']}")
    print(f"Win Rate: {e['win_rate']}")
    print(f"Retorno Total: {e['total_return_pct']}%")

    print("\n=== POR HORIZONTE ===")
    for h, r in result["by_horizon"].items():
        if r["n_trades"] == 0:
            continue
        print(
            f"{h}: trades={r['n_trades']} días={r['n_days']} "
            f"sharpe={r['sharpe']} dd={r['max_drawdown']} "
            f"win={r['win_rate']} ret_total={r['total_return_pct']}%"
        )

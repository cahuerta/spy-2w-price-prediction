# =========================================================
# backtest_real_trades.py — SHARPE/DRAWDOWN REAL DE LA CUENTA
# =========================================================
# A diferencia de backtest_evaluations.py (que mide el potencial
# TEÓRICO del modelo si se le creyera ciegamente), este archivo
# mide lo que REALMENTE pasó con el capital: lee
# /data/darwin/trades/*.json — trades que ya pasaron por TODOS
# los filtros de decisión (PM Growth/Neutral/Defensive, Capital
# Governor, Circuit Breaker, Alpha Kill Switch, timing de entrada
# del intraday tracker) antes de ejecutarse o no.
#
# Reutiliza la MISMA lógica matemática ya validada en
# backtest_evaluations.py (conversión a tasa diaria equivalente,
# Sharpe, drawdown) — sin reimplementar ni divergir el cálculo.
#
# Diferencia clave de fuente de datos:
#   backtest_evaluations.py → /data/evaluations/  (predicción del
#     modelo vs precio real, TODAS las señales, sin filtro humano
#     ni de sistema — "si le creyéramos ciegamente")
#   backtest_real_trades.py → /data/darwin/trades/ (solo lo que
#     realmente se ejecutó y se cerró en Alpaca — "lo que pasó
#     de verdad con el capital")
#
# Con ambos corriendo se puede comparar potencial teórico vs
# resultado real, y esa diferencia mide cuánto está ayudando (o
# restando) todo el aparato de decisión (PM/Governor/Kill Switch/
# timing) por encima de la señal cruda del modelo.
# =========================================================

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np

# Reutiliza la matemática ya validada — no se reimplementa
from backtest_evaluations import (
    BacktestResult,
    _daily_equivalent_return,
    _sharpe,
    _max_drawdown,
    _win_rate,
)

logger = logging.getLogger("backtest_real_trades")
logging.basicConfig(level=logging.INFO)

DATA_PATH   = Path(os.getenv("DATA_PATH", "/data"))
TRACKER_DIR = DATA_PATH / "darwin" / "trades"


# =========================================================
# HELPERS
# =========================================================

def _load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_trades():
    """Generador: recorre todos los trades registrados en disco."""
    if not TRACKER_DIR.exists():
        return
    for f in sorted(TRACKER_DIR.glob("*.json")):
        data = _load_json(f)
        if data:
            yield data


# =========================================================
# CÁLCULO POR GRUPO (todos los trades, o desglose por horizonte)
# =========================================================

def _compute_result(
    label: str,
    dated_returns: List[tuple],   # (fecha_str, pnl_real_pct, dias_reales)
    n_total_trades: int,
) -> BacktestResult:
    """
    Misma conversión que backtest_evaluations.py: cada trade se
    convierte a tasa diaria equivalente ANTES de agrupar por fecha
    y encadenar. Usa días_held REALES (no horizon_days teórico) —
    el sistema puede cerrar antes o después del horizonte original,
    y lo que importa para el Sharpe real es cuánto tiempo el
    capital estuvo efectivamente comprometido en ese trade.
    """
    trade_level_returns = [r for _, r, _ in dated_returns]

    by_date = defaultdict(list)
    for date_str, pnl_real, days_held in dated_returns:
        if not date_str:
            continue
        # Trade cerrado el mismo día → ya es una tasa diaria, sin
        # necesidad de convertir (horizonte=1 evita distorsión)
        h = max(1, int(days_held))
        daily_equiv = _daily_equivalent_return(pnl_real, h)
        by_date[date_str].append(daily_equiv)

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
        n_evaluations=n_total_trades,
        sharpe=_sharpe(daily_returns),
        max_drawdown=_max_drawdown(daily_returns),
        win_rate=_win_rate(trade_level_returns),
        total_return_pct=round(total_return, 2) if total_return is not None else None,
        avg_daily_return_pct=round(float(np.mean(daily_returns)), 4) if daily_returns else None,
    )


# =========================================================
# CORE
# =========================================================

def run_backtest_from_real_trades() -> Dict[str, Any]:
    """
    Construye la curva de equity REAL desde /data/darwin/trades/.
    Solo cuenta trades que efectivamente se cerraron (tienen
    pnl_real_pct) — independiente de si ya se resolvió contra el
    precio al horizonte teórico o no.

    Retorna:
      "real_account"  → todos los trades reales, sin distinguir horizonte
      "by_dominant_h" → desglosado por el horizonte dominante que
                         originó cada trade (trade["dominant_h"])
      "closed_early_stats" → cuánto costó/ayudó cerrar antes del
                         horizonte (usa oportunidad_pct si existe)
    """
    all_returns: List[tuple] = []
    n_total = 0

    by_horizon_returns: Dict[str, List[tuple]] = defaultdict(list)

    oportunidad_perdida = []   # cerró antes y dejó dinero sobre la mesa
    oportunidad_ganada  = []   # cerró antes y evitó una pérdida mayor

    for trade in _iter_trades():
        pnl_real = trade.get("pnl_real_pct")
        if pnl_real is None:
            continue  # trade aún abierto, no cerrado — no cuenta

        n_total += 1

        exit_date  = trade.get("exit_date") or trade.get("resolved_at", "")[:10]
        days_held  = trade.get("days_held")
        if days_held is None:
            days_held = 1

        all_returns.append((exit_date, float(pnl_real), int(days_held)))

        dominant_h = trade.get("dominant_h") or "unknown"
        by_horizon_returns[dominant_h].append((exit_date, float(pnl_real), int(days_held)))

        # Estadística de timing — solo si ya se resolvió contra el horizonte
        if trade.get("status") in ("resolved",) and trade.get("closed_before_horizon"):
            op = trade.get("oportunidad_pct")
            if op is not None:
                if op > 0:
                    oportunidad_perdida.append(abs(op))
                elif op < 0:
                    oportunidad_ganada.append(abs(op))

    result = {
        "real_account": asdict(_compute_result("Cuenta real (todos los trades cerrados)", all_returns, n_total)),
        "by_dominant_h": {},
        "closed_early_stats": {
            "n_oportunidad_perdida": len(oportunidad_perdida),
            "avg_oportunidad_perdida_pct": round(float(np.mean(oportunidad_perdida)), 2) if oportunidad_perdida else None,
            "n_oportunidad_ganada": len(oportunidad_ganada),
            "avg_oportunidad_ganada_pct": round(float(np.mean(oportunidad_ganada)), 2) if oportunidad_ganada else None,
        },
    }

    for h in sorted(by_horizon_returns.keys()):
        rets = by_horizon_returns[h]
        result["by_dominant_h"][h] = asdict(_compute_result(h, rets, len(rets)))

    result["generated_at"] = datetime.utcnow().isoformat()
    return result


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    result = run_backtest_from_real_trades()

    print("\n=== CUENTA REAL (todos los trades cerrados) ===")
    r = result["real_account"]
    print(f"Trades: {r['n_trades']} | Días agregados: {r['n_days']}")
    print(f"Sharpe: {r['sharpe']}")
    print(f"Max Drawdown: {r['max_drawdown']}")
    print(f"Win Rate: {r['win_rate']}")
    print(f"Retorno Total: {r['total_return_pct']}%")

    print("\n=== POR HORIZONTE DOMINANTE ===")
    for h, res in result["by_dominant_h"].items():
        if res["n_trades"] == 0:
            continue
        print(
            f"{h}: trades={res['n_trades']} días={res['n_days']} "
            f"sharpe={res['sharpe']} dd={res['max_drawdown']} "
            f"win={res['win_rate']} ret_total={res['total_return_pct']}%"
        )

    print("\n=== IMPACTO DE CERRAR ANTES DEL HORIZONTE ===")
    ce = result["closed_early_stats"]
    print(f"Veces que dejó dinero sobre la mesa: {ce['n_oportunidad_perdida']} (promedio {ce['avg_oportunidad_perdida_pct']}%)")
    print(f"Veces que evitó pérdida mayor: {ce['n_oportunidad_ganada']} (promedio {ce['avg_oportunidad_ganada_pct']}%)")

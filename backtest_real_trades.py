# =========================================================
# backtest_real_trades.py — SHARPE/DRAWDOWN REAL DE LA CUENTA
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

from backtest_evaluations import (
    BacktestResult,
    _daily_equivalent_return,
    _sharpe_classic,
    _sharpe_newey_west,
    _max_drawdown,
    _win_rate,
)

logger = logging.getLogger("backtest_real_trades")
logging.basicConfig(level=logging.INFO)

DATA_PATH   = Path(os.getenv("DATA_PATH", "/data"))
TRACKER_DIR = DATA_PATH / "darwin" / "trades"


def _load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_trades():
    if not TRACKER_DIR.exists():
        return
    for f in sorted(TRACKER_DIR.glob("*.json")):
        data = _load_json(f)
        if data:
            yield data


def _compute_result(
    label: str,
    dated_returns: List[tuple],
    n_total_trades: int,
) -> BacktestResult:
    trade_level_returns = [r for _, r, _ in dated_returns]

    by_date = defaultdict(list)
    for date_str, pnl_real, days_held in dated_returns:
        if not date_str:
            continue
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
        sharpe_classic=_sharpe_classic(daily_returns),
        sharpe_newey_west=_sharpe_newey_west(daily_returns),
        max_drawdown=_max_drawdown(daily_returns),
        win_rate=_win_rate(trade_level_returns),
        total_return_pct=round(total_return, 2) if total_return is not None else None,
        avg_daily_return_pct=round(float(np.mean(daily_returns)), 4) if daily_returns else None,
    )


def run_backtest_from_real_trades() -> Dict[str, Any]:
    all_returns: List[tuple] = []
    n_total = 0

    by_horizon_returns: Dict[str, List[tuple]] = defaultdict(list)

    oportunidad_perdida = []
    oportunidad_ganada  = []

    for trade in _iter_trades():
        pnl_real = trade.get("pnl_real_pct")
        if pnl_real is None:
            continue

        n_total += 1

        exit_date  = trade.get("exit_date") or trade.get("resolved_at", "")[:10]
        days_held  = trade.get("days_held")
        if days_held is None:
            days_held = 1

        all_returns.append((exit_date, float(pnl_real), int(days_held)))

        dominant_h = trade.get("dominant_h") or "unknown"
        by_horizon_returns[dominant_h].append((exit_date, float(pnl_real), int(days_held)))

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


if __name__ == "__main__":
    result = run_backtest_from_real_trades()

    print("\n=== CUENTA REAL (todos los trades cerrados) ===")
    r = result["real_account"]
    print(f"Trades: {r['n_trades']} | Días agregados: {r['n_days']}")
    print(f"Sharpe clásico:     {r['sharpe_classic']}")
    print(f"Sharpe Newey-West:  {r['sharpe_newey_west']}")
    print(f"Max Drawdown: {r['max_drawdown']}")
    print(f"Win Rate: {r['win_rate']}")
    print(f"Retorno Total: {r['total_return_pct']}%")

    print("\n=== POR HORIZONTE DOMINANTE ===")
    for h, res in result["by_dominant_h"].items():
        if res["n_trades"] == 0:
            continue
        print(
            f"{h}: trades={res['n_trades']} días={res['n_days']} "
            f"sharpe_clas={res['sharpe_classic']} sharpe_nw={res['sharpe_newey_west']} "
            f"dd={res['max_drawdown']} win={res['win_rate']} ret_total={res['total_return_pct']}%"
        )

    print("\n=== IMPACTO DE CERRAR ANTES DEL HORIZONTE ===")
    ce = result["closed_early_stats"]
    print(f"Veces que dejó dinero sobre la mesa: {ce['n_oportunidad_perdida']} (promedio {ce['avg_oportunidad_perdida_pct']}%)")
    print(f"Veces que evitó pérdida mayor: {ce['n_oportunidad_ganada']} (promedio {ce['avg_oportunidad_ganada_pct']}%)")

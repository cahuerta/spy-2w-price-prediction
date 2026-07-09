# =========================================================
# backtest_evaluations.py — SHARPE/DRAWDOWN REAL DESDE EVALUACIONES
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
DEFAULT_HORIZON_DAYS = 10


def _load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _daily_equivalent_return(total_return_pct: float, horizon_days: int) -> float:
    horizon_days = max(1, int(horizon_days))
    base = 1.0 + (total_return_pct / 100.0)
    if base <= 0:
        return -100.0
    daily_factor = base ** (1.0 / horizon_days)
    return (daily_factor - 1.0) * 100.0


def _sharpe_classic(returns: List[float]) -> Optional[float]:
    if len(returns) < 5:
        return None
    arr = np.array(returns, dtype=float)
    std = arr.std()
    if std < 1e-9:
        return None
    return float(arr.mean() / std * np.sqrt(TRADING_DAYS_YEAR))


def _sharpe_newey_west(returns: List[float], max_lag: Optional[int] = None) -> Optional[float]:
    if len(returns) < 10:
        return None
    arr = np.array(returns, dtype=float)
    n = len(arr)
    mean = arr.mean()
    demeaned = arr - mean

    if max_lag is None:
        max_lag = max(1, int(4 * (n / 100) ** (2 / 9)))

    gamma_0 = float(np.sum(demeaned ** 2) / n)
    nw_var = gamma_0
    for lag in range(1, max_lag + 1):
        gamma_lag = float(np.sum(demeaned[lag:] * demeaned[:-lag]) / n)
        weight = 1 - lag / (max_lag + 1)
        nw_var += 2 * weight * gamma_lag

    nw_std = np.sqrt(max(nw_var, 1e-12))
    if nw_std < 1e-9:
        return None
    return float(mean / nw_std * np.sqrt(TRADING_DAYS_YEAR))


def _max_drawdown(returns: List[float]) -> Optional[float]:
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


@dataclass
class BacktestResult:
    label:                 str
    n_trades:              int
    n_days:                int
    n_evaluations:         int
    sharpe_classic:        Optional[float]
    sharpe_newey_west:     Optional[float]
    max_drawdown:          Optional[float]
    win_rate:              Optional[float]
    total_return_pct:      Optional[float]
    avg_daily_return_pct:  Optional[float]


def _compute_result(
    label: str,
    dated_returns: List[tuple],
    n_evaluations: int,
) -> BacktestResult:
    trade_level_returns = [r for _, r, _ in dated_returns]

    by_date = defaultdict(list)
    for date_str, total_ret, horizon_days in dated_returns:
        if not date_str:
            continue
        daily_equiv = _daily_equivalent_return(total_ret, horizon_days)
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
        n_evaluations=n_evaluations,
        sharpe_classic=_sharpe_classic(daily_returns),
        sharpe_newey_west=_sharpe_newey_west(daily_returns),
        max_drawdown=_max_drawdown(daily_returns),
        win_rate=_win_rate(trade_level_returns),
        total_return_pct=round(total_return, 2) if total_return is not None else None,
        avg_daily_return_pct=round(float(np.mean(daily_returns)), 4) if daily_returns else None,
    )


def _iter_evaluations():
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
    ensemble_returns: List[tuple] = []
    ensemble_n_eval = 0

    horizon_returns: Dict[str, List[tuple]] = defaultdict(list)
    horizon_n_eval:  Dict[str, int] = defaultdict(int)

    for ticker, ev in _iter_evaluations():
        if ev.get("legacy_bad_horizon") is True:
            continue
        if ev.get("alpaca_unsupported") is True:
            continue

        pred_date = ev.get("prediction_date", "")

        rec = (ev.get("recommendation") or "").upper()
        real_ret = ev.get("real_return_pct")
        if real_ret is not None and rec in ("COMPRA", "VENDE"):
            ensemble_n_eval += 1
            direction = 1 if rec == "COMPRA" else -1
            h_days = int(ev.get("evaluation_horizon_days") or DEFAULT_HORIZON_DAYS)
            ensemble_returns.append((pred_date, direction * float(real_ret), h_days))

        diag = ev.get("models_diagnostics") or {}
        for hkey, hdata in diag.items():
            if not isinstance(hdata, dict):
                continue
            pred_ret   = hdata.get("pred_return")
            real_ret_h = hdata.get("real_return")
            if pred_ret is None or real_ret_h is None:
                continue

            horizon_n_eval[hkey] += 1

            if abs(float(pred_ret)) < 0.05:
                continue

            direction = 1 if float(pred_ret) > 0 else -1
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


if __name__ == "__main__":
    result = run_backtest_from_evaluations()

    print("\n=== ENSEMBLE (COMPRA/VENDE) ===")
    e = result["ensemble"]
    print(f"Trades: {e['n_trades']} (de {e['n_evaluations']} evaluaciones) | Días agregados: {e['n_days']}")
    print(f"Sharpe clásico:     {e['sharpe_classic']}")
    print(f"Sharpe Newey-West:  {e['sharpe_newey_west']}")
    print(f"Max Drawdown: {e['max_drawdown']}")
    print(f"Win Rate: {e['win_rate']}")
    print(f"Retorno Total: {e['total_return_pct']}%")

    print("\n=== POR HORIZONTE ===")
    for h, r in result["by_horizon"].items():
        if r["n_trades"] == 0:
            continue
        print(
            f"{h}: trades={r['n_trades']} días={r['n_days']} "
            f"sharpe_clas={r['sharpe_classic']} sharpe_nw={r['sharpe_newey_west']} "
            f"dd={r['max_drawdown']} win={r['win_rate']} ret_total={r['total_return_pct']}%"
        )

# =========================================================
# analyze_signal.py — ¿HAY SEÑAL? ¿SE PUEDE GANAR PLATA?
# =========================================================
#
# Responde con los datos reales de /data, sin descargar nada:
#
#   1. SKILL DE CADA PREDICTOR H1-H10 (salida CRUDA del predictor,
#      leída de predictions/{T}/{fecha}.json → models_diagnostics).
#      Se evita a propósito el pred_return de evaluations/…/H{h},
#      porque ese sale de price_curve, que está distorsionada por los
#      pesos de horizonte (ver master_orchestrator._weighted_price_curve).
#
#   2. SKILL DE LA PREDICCIÓN FINAL (predicted_return_pct / real_return_pct
#      de la evaluación a 10 días) — la que usa el trading.
#
#   3. TRADES REALES de Darwin (darwin/trades/*.json).
#
# Métricas y por qué:
#   - hit_rate vs base_rate: acertar el signo 55% no vale nada si el
#     mercado subió el 58% de las veces (base_rate = "siempre compro").
#   - IC (Spearman por fecha entre predicción y retorno real, entre
#     tickers): mide si el modelo ORDENA bien las acciones, que es lo
#     que importa para elegir cuáles comprar.
#   - top_vs_universo: retorno del quintil con mejor predicción menos
#     el promedio del universo ese día. Es la plata extra que da elegir
#     con el modelo en vez de comprar todo.
#   - t-stats con fechas NO solapadas (una cada h días): las
#     evaluaciones de días seguidos a horizonte h comparten h-1 días de
#     retorno, y las acciones del mismo día se mueven juntas, así que
#     contar cada (ticker, día) como independiente infla la muestra
#     ~10-50x. Aquí la unidad independiente es la FECHA.
#
# Uso:
#   python analyze_signal.py            → imprime reporte
#   python analyze_signal.py --json     → imprime JSON
#   GET /dashboard/signal-analysis      → mismo JSON vía API
# =========================================================

import os
import sys
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
EVAL_ROOT = DATA_PATH / "evaluations"
PRED_ROOT = DATA_PATH / "predictions"
TRADES_DIR = DATA_PATH / "darwin" / "trades"

MIN_TICKERS_PER_DATE = 10   # mínimo de tickers para calcular IC/quintiles de una fecha
TOP_FRACTION = 0.2          # quintil superior


def _load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _tstat(xs: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    if var <= 0:
        return None
    return m / math.sqrt(var / n)


def _ranks(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = _mean(ra), _mean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va <= 0 or vb <= 0:
        return None
    return cov / math.sqrt(va * vb)


def _non_overlapping(dates: List[str], step: int) -> List[str]:
    """Toma una fecha cada `step` fechas (ordenadas) → muestras sin solape."""
    ds = sorted(dates)
    return ds[::max(1, step)]


def _skill(rows: List[Dict[str, Any]], horizon: int) -> Dict[str, Any]:
    """
    rows: [{date, ticker, pred, real}] con retornos en %.
    """
    out: Dict[str, Any] = {"horizon_days": horizon, "n_obs": len(rows)}
    if not rows:
        return out

    by_date: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)
    out["n_dates"] = len(by_date)
    out["n_tickers"] = len({r["ticker"] for r in rows})

    # Hit rate (excluye predicciones ~0, igual que evaluator HIT_SIGN_MIN_PCT)
    signed = [r for r in rows if abs(r["pred"]) >= 0.05 and r["real"] != 0]
    if signed:
        out["hit_rate"] = round(sum((r["pred"] > 0) == (r["real"] > 0) for r in signed) / len(signed), 4)
        out["pct_pred_up"] = round(sum(r["pred"] > 0 for r in signed) / len(signed), 4)
    out["base_rate_up"] = round(sum(r["real"] > 0 for r in rows) / len(rows), 4)
    out["mean_real_ret_pct"] = round(_mean([r["real"] for r in rows]), 4)

    # Métricas por fecha (cross-section)
    ic_by_date: Dict[str, float] = {}
    top_excess_by_date: Dict[str, float] = {}
    long_short_by_date: Dict[str, float] = {}
    for d, rs in by_date.items():
        if len(rs) < MIN_TICKERS_PER_DATE:
            continue
        preds = [r["pred"] for r in rs]
        reals = [r["real"] for r in rs]
        ic = _spearman(preds, reals)
        if ic is not None:
            ic_by_date[d] = ic
        srt = sorted(rs, key=lambda r: r["pred"])
        k = max(1, int(len(srt) * TOP_FRACTION))
        top = _mean([r["real"] for r in srt[-k:]])
        bot = _mean([r["real"] for r in srt[:k]])
        uni = _mean(reals)
        top_excess_by_date[d] = top - uni
        long_short_by_date[d] = top - bot

    def _summ(series: Dict[str, float], name: str, digits: int = 4):
        if not series:
            return
        all_vals = list(series.values())
        indep_dates = _non_overlapping(list(series.keys()), horizon)
        indep = [series[d] for d in indep_dates]
        out[f"{name}_mean"] = round(_mean(all_vals), digits)
        out[f"{name}_pct_dates_positive"] = round(sum(v > 0 for v in all_vals) / len(all_vals), 4)
        t = _tstat(indep)
        out[f"{name}_tstat_indep"] = round(t, 2) if t is not None else None
        out[f"{name}_n_indep_dates"] = len(indep)

    _summ(ic_by_date, "ic")
    _summ(top_excess_by_date, "top_vs_universe_pct")
    _summ(long_short_by_date, "long_short_pct")
    return out


def _verdict(s: Dict[str, Any]) -> str:
    t_ic = s.get("ic_tstat_indep")
    t_top = s.get("top_vs_universe_pct_tstat_indep")
    if t_ic is None or t_top is None:
        return "SIN DATOS SUFICIENTES"
    if t_ic >= 2 and t_top >= 2:
        return "SEÑAL POSITIVA (significativa)"
    if t_ic <= -2 and t_top <= -2:
        return "SEÑAL INVERTIDA (significativa) — el modelo ordena al revés"
    if abs(t_ic) < 1 and abs(t_top) < 1:
        return "SIN SEÑAL — indistinguible del azar"
    return "DÉBIL / NO CONCLUYENTE"


# =========================================================
# 1-2. PREDICTORES CRUDOS Y PREDICCIÓN FINAL
# =========================================================
def _collect_rows() -> Dict[str, List[Dict[str, Any]]]:
    rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not EVAL_ROOT.exists():
        return rows

    for tdir in EVAL_ROOT.iterdir():
        if not tdir.is_dir():
            continue
        ticker = tdir.name.upper()
        for f in tdir.glob("*.json"):
            ev = _load_json(f)
            if not ev or ev.get("legacy_bad_horizon") or ev.get("alpaca_unsupported"):
                continue
            date = str(ev.get("prediction_date") or f.stem)[:10]

            # Predicción final (la que usa el trading)
            pred_f = _num(ev.get("predicted_return_pct"))
            real_f = _num(ev.get("real_return_pct"))
            h_f = int(ev.get("evaluation_horizon_days") or 10)
            if pred_f is not None and real_f is not None:
                rows["FINAL"].append({"date": date, "ticker": ticker, "pred": pred_f, "real": real_f, "h": h_f})

            # Predictores crudos H1-H10: pred de predictions/, real de la evaluación
            diag_eval = ev.get("models_diagnostics") or {}
            pred_file = _load_json(PRED_ROOT / ticker / f"{date}.json") or {}
            diag_pred = pred_file.get("models_diagnostics") or {}
            for h in range(1, 11):
                key = f"H{h}"
                real_h = _num((diag_eval.get(key) or {}).get("real_return"))
                if real_h is None:
                    continue
                raw_pred = _num((diag_pred.get(key) or {}).get("pred_return"))
                if raw_pred is not None:
                    rows[f"{key}_raw"].append({"date": date, "ticker": ticker, "pred": raw_pred, "real": real_h})
                curve_pred = _num((diag_eval.get(key) or {}).get("pred_return"))
                if curve_pred is not None:
                    rows[f"{key}_curve"].append({"date": date, "ticker": ticker, "pred": curve_pred, "real": real_h})
    return rows


# =========================================================
# 3. TRADES REALES
# =========================================================
def _real_trades() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not TRADES_DIR.exists():
        return {"error": f"{TRADES_DIR} no existe"}

    closed, opp = [], []
    by_reason: Dict[str, List[float]] = defaultdict(list)
    n_open = 0
    for f in TRADES_DIR.glob("*.json"):
        t = _load_json(f)
        if not t:
            continue
        if t.get("status") == "open":
            n_open += 1
            continue
        pnl = _num(t.get("pnl_real_pct"))
        if pnl is None:
            continue
        closed.append({"pnl": pnl, "exit": str(t.get("exit_date") or "")[:10]})
        by_reason[str(t.get("reason_close") or "UNKNOWN")].append(pnl)
        teo = _num(t.get("pnl_teorico_pct"))
        if teo is not None:
            opp.append(teo - pnl)

    out["n_open"] = n_open
    out["n_closed"] = len(closed)
    if not closed:
        return out

    pnls = [c["pnl"] for c in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    out["mean_pnl_pct"] = round(_mean(pnls), 4)
    out["win_rate"] = round(len(wins) / len(pnls), 4)
    out["avg_win_pct"] = round(_mean(wins), 4) if wins else None
    out["avg_loss_pct"] = round(_mean(losses), 4) if losses else None
    out["profit_factor"] = round(sum(wins) / abs(sum(losses)), 3) if losses and sum(losses) != 0 else None

    # t-stat agrupando por día de salida (trades del mismo día están correlacionados)
    by_exit: Dict[str, List[float]] = defaultdict(list)
    for c in closed:
        by_exit[c["exit"]].append(c["pnl"])
    daily = [_mean(v) for v in by_exit.values()]
    t = _tstat(daily)
    out["tstat_by_exit_day"] = round(t, 2) if t is not None else None
    out["n_exit_days"] = len(daily)

    if opp:
        out["mean_holding_to_horizon_minus_real_pct"] = round(_mean(opp), 4)

    out["by_reason"] = {
        r: {"n": len(v), "mean_pnl_pct": round(_mean(v), 4), "win_rate": round(sum(p > 0 for p in v) / len(v), 4)}
        for r, v in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))
    }
    return out


# =========================================================
# RUN
# =========================================================
def run_signal_analysis() -> Dict[str, Any]:
    rows = _collect_rows()
    predictors: Dict[str, Any] = {}

    if rows.get("FINAL"):
        h = max(set(r["h"] for r in rows["FINAL"]), key=[r["h"] for r in rows["FINAL"]].count)
        s = _skill(rows["FINAL"], h)
        s["verdict"] = _verdict(s)
        predictors["FINAL"] = s

    for h in range(1, 11):
        for kind in ("raw", "curve"):
            key = f"H{h}_{kind}"
            if rows.get(key):
                s = _skill(rows[key], h)
                s["verdict"] = _verdict(s)
                predictors[key] = s

    return {
        "data_path": str(DATA_PATH),
        "predictors": predictors,
        "real_trades": _real_trades(),
        "notes": [
            "H{h}_raw = salida directa del predictor; H{h}_curve = punto de price_curve (lo que mide Darwin hoy).",
            "t-stat con fechas no solapadas; |t| >= 2 ≈ significativo.",
            "top_vs_universe_pct = % extra por elegir el quintil top del modelo vs comprar todo el universo.",
        ],
    }


def _print_report(res: Dict[str, Any]) -> None:
    print("=" * 78)
    print("ANÁLISIS DE SEÑAL")
    print("=" * 78)
    hdr = f"{'modelo':<10}{'n':>7}{'fechas':>7}{'hit':>7}{'base↑':>7}{'IC':>8}{'t(IC)':>7}{'top-uni%':>9}{'t':>6}  veredicto"
    print(hdr)
    print("-" * len(hdr))
    for k, s in res["predictors"].items():
        def g(x, fmt):
            v = s.get(x)
            return format(v, fmt) if isinstance(v, (int, float)) else "-"
        print(
            f"{k:<10}{s.get('n_obs', 0):>7}{s.get('n_dates', 0):>7}"
            f"{g('hit_rate', '.3f'):>7}{g('base_rate_up', '.3f'):>7}"
            f"{g('ic_mean', '.4f'):>8}{g('ic_tstat_indep', '.2f'):>7}"
            f"{g('top_vs_universe_pct_mean', '.3f'):>9}{g('top_vs_universe_pct_tstat_indep', '.2f'):>6}"
            f"  {s.get('verdict', '')}"
        )
    print()
    print("TRADES REALES")
    print(json.dumps(res["real_trades"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    result = run_signal_analysis()
    if "--json" in sys.argv:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_report(result)

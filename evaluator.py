import os
import json
import logging
from datetime import datetime, timedelta, date as date_type
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import concurrent.futures
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ======================================================
# CONFIG
# ======================================================

DATA_PATH     = os.getenv("DATA_PATH", "/data")
MAX_WORKERS   = min(int(os.getenv("EVAL_MAX_WORKERS", "4")), 16)
ALPACA_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
EVAL_HORIZON  = int(os.getenv("EVAL_HORIZON", "10"))
PRICE_CACHE: Dict[str, float] = {}

# Umbral mínimo de señal direccional para evaluar hit_sign
# Predicciones con |pred| < umbral → hit_sign=None (sin señal)
HIT_SIGN_MIN_PCT = float(os.getenv("HIT_SIGN_MIN_PCT", "0.05"))

# ======================================================
# FIXES v2.3:
#
#   [E1] nth_business_day(start, n): días hábiles estrictos.
#   [E2] get_price_at_date: precio exacto, fallback hábil anterior.
#   [E3] evaluate_models: cada Hx en su n-ésimo día hábil exacto.
#   [E4] Tickers sin soporte Alpaca marcados alpaca_unsupported.
#   [E5] Trazabilidad: evaluation_horizon_days, evaluation_target_date.
#   [E6] legacy_bad_horizon=False en evaluaciones nuevas.
#        evaluate_all recalcula si legacy_bad_horizon != False.
#
#   [E7] FIX hit_sign con pred≈0:
#        np.sign(0.0) = 0 → nunca igual a np.sign(±real) → siempre False.
#        Esto penalizaba injustamente tickers con predicciones débiles
#        (MCD, SO, KR, EQIX) dándoles hit_rate artificialmente bajo.
#        Solución: si |predicted_return| < HIT_SIGN_MIN_PCT → hit_sign=None.
#        None es excluido del cálculo de rolling_metrics en signals.py.
#        Impacto medido antes del fix:
#          SO:   26.9% → 50.0% (sin pred≈0)
#          KR:   24.1% → 40.6%
#          EQIX: 24.6% → 37.1%
#          MCD:  20.3% → 27.5%
#          BALL: 30.9% → 36.4%
#        TTWO y GOOGL no mejoran → genuinamente malos predictores.
#
#   [E7b] Mismo fix aplicado en evaluate_models para hit_sign de H1-H10.
# ======================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALPACA_UNSUPPORTED_SUFFIXES = (
    ".PA", ".DE", ".SW", ".TO", ".MC", ".AS", ".MI", ".BR",
    ".SN", ".SX", ".L", ".HK", ".T",
)
ALPACA_UNSUPPORTED_EXACT = {
    "OR.PA", "AIR.PA", "NOVN.SW", "ENB.TO", "MFC.TO",
    "IBE.MC", "SAN.MC", "DTE.DE", "NESN.SW", "RY.TO",
    "TTE.PA", "BNP.PA", "ABI.BR", "ASML.AS", "BF-B",
}


def _is_alpaca_unsupported(ticker: str) -> bool:
    t = ticker.upper()
    if t in ALPACA_UNSUPPORTED_EXACT:
        return True
    return any(t.endswith(s) for s in ALPACA_UNSUPPORTED_SUFFIXES)


# ======================================================
# DÍAS HÁBILES
# ======================================================

def nth_business_day(start: date_type, n: int) -> date_type:
    """
    [E1] Retorna el n-ésimo día hábil (lun-vie) después de start.
    start no cuenta — se empieza a contar desde el día siguiente.
    """
    current = start
    count   = 0
    while count < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return current


def prev_business_day(d: date_type) -> date_type:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ======================================================
# STRUCT
# ======================================================

@dataclass
class EvaluationResult:
    meta: Dict[str, Any]
    prediction_date: str
    evaluation_date: str
    evaluation_horizon_days: int
    evaluation_target_date: str
    price_now: float
    price_pred: float
    price_real: Optional[float]
    predicted_return_pct: float
    real_return_pct: Optional[float]
    error_price_pct: Optional[float]
    error_return_pct: Optional[float]
    hit_sign: Optional[bool]
    hit_threshold: Optional[bool]
    recommendation: str
    decision_correct: Optional[bool]
    evaluated_at: str
    alpaca_unsupported: bool = False
    legacy_bad_horizon: bool = False
    weak_signal: bool = False        # [E7] True si |pred| < HIT_SIGN_MIN_PCT
    models_diagnostics: Dict[str, Any] = None
    models_summary: Dict[str, Any] = None


# ======================================================
# UTILS
# ======================================================

def load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"load_json failed {path}: {e}")
        return {}


def save_json(path: Path, data: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"save_json failed {path}: {e}")
        return False


def parse_date(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _eval_needs_recalculation(eval_path: Path) -> bool:
    """
    [E6] True si la evaluación debe crearse o recalcularse.
    También recalcula evaluaciones v2.2 que no tienen el fix E7
    (no tienen el campo 'weak_signal').
    """
    if not eval_path.exists():
        return True
    try:
        d = json.loads(eval_path.read_text())
        # Si está marcada como mala → recalcular
        if d.get("legacy_bad_horizon") is not False:
            return True
        # Si no tiene el campo weak_signal → es v2.2 sin fix E7 → recalcular
        if "weak_signal" not in d:
            return True
        return False
    except Exception:
        return True


# ======================================================
# [E7] HIT SIGN CON UMBRAL DE SEÑAL
# ======================================================

def _calc_hit_sign(
    predicted_return: float,
    real_return: float,
) -> Tuple[Optional[bool], bool]:
    """
    [E7] Calcula hit_sign con umbral de señal mínima.

    Si |predicted_return| < HIT_SIGN_MIN_PCT:
      - hit_sign = None (señal demasiado débil para evaluar dirección)
      - weak_signal = True
      Antes: np.sign(0.0) = 0 → siempre False → hit_rate artificialmente bajo

    Si |predicted_return| >= HIT_SIGN_MIN_PCT:
      - hit_sign = bool(sign(pred) == sign(real))
      - weak_signal = False

    Returns: (hit_sign, weak_signal)
    """
    if abs(predicted_return) < HIT_SIGN_MIN_PCT:
        return None, True
    return bool(np.sign(predicted_return) == np.sign(real_return)), False


# ======================================================
# PRECIO REAL
# ======================================================

_alpaca_client: Optional[StockHistoricalDataClient] = None


def _get_alpaca_client() -> StockHistoricalDataClient:
    global _alpaca_client
    if _alpaca_client is None:
        _alpaca_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    return _alpaca_client


def get_price_at_date(ticker: str, target_date: date_type) -> Optional[float]:
    """
    [E2] Precio de cierre en target_date.
    Fallback al día hábil anterior si es feriado o fin de semana.
    """
    key = f"{ticker}_{target_date}"
    if key in PRICE_CACHE:
        return PRICE_CACHE[key]

    if _is_alpaca_unsupported(ticker):
        return None

    try:
        client  = _get_alpaca_client()
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=target_date - timedelta(days=7),
            end=target_date + timedelta(days=1),
        )
        bars = client.get_stock_bars(request).df
        if bars is None or bars.empty:
            return None

        bars = bars.reset_index()
        bars["date"] = bars["timestamp"].dt.date
        bars = bars[bars["date"] <= target_date]
        if bars.empty:
            return None

        price = float(bars["close"].iloc[-1])
        PRICE_CACHE[key] = price
        return price
    except Exception as e:
        logger.warning(f"get_price_at_date({ticker}, {target_date}): {e}")
        return None


# ======================================================
# EVALUACIÓN MODELOS H1-H10
# ======================================================

def evaluate_models(
    ticker: str,
    pred_date: date_type,
) -> Dict[str, Any]:
    """
    [E3] Para cada horizonte h (1 a 10):
      - h-ésimo día hábil después de pred_date
      - price_path[h-1] vs precio real en esa fecha
      - [E7b] hit_sign=None si pred≈0
    """
    models: Dict[str, Any] = {}
    pred_root = Path(DATA_PATH) / "predictions" / ticker
    pred_file = pred_root / f"{pred_date.strftime('%Y-%m-%d')}.json"

    if not pred_file.exists():
        return {}

    pred_data = load_json(pred_file)
    old_p     = pred_data.get("prediction", {})
    curve     = pred_data.get("price_curve", {}).get("price_path", [])
    price_now = float(old_p.get("price_now", 0))

    if price_now <= 0 or not curve:
        return {}

    for h in range(1, 11):
        if len(curve) < h:
            continue

        target     = nth_business_day(pred_date, h)
        price_real = get_price_at_date(ticker, target)
        if not price_real:
            continue

        price_pred = float(curve[h - 1])
        pred_ret   = (price_pred / price_now - 1) * 100
        real_ret   = (price_real / price_now - 1) * 100

        # [E7b] Umbral de señal para H diagnósticos
        h_hit_sign, h_weak = _calc_hit_sign(pred_ret, real_ret)

        models[f"H{h}"] = {
            "target_date":  str(target),
            "pred_price":   round(price_pred, 4),
            "pred_return":  round(pred_ret,   4),
            "real_price":   round(price_real, 4),
            "real_return":  round(real_ret,   4),
            "error_pct":    round(abs(pred_ret - real_ret), 4),
            "hit_sign":     h_hit_sign,   # None si señal débil
            "weak_signal":  h_weak,
        }

    return models


def summarize_models(models: Dict[str, Any]) -> Dict[str, Any]:
    errors = {k: v["error_pct"] for k, v in models.items() if v.get("error_pct") is not None}
    if not errors:
        return {"best_model": None, "worst_model": None, "mean_error": 0}
    return {
        "best_model":  min(errors, key=errors.get),
        "worst_model": max(errors, key=errors.get),
        "mean_error":  round(float(np.mean(list(errors.values()))), 4),
    }


# ======================================================
# EVALUACIÓN INDIVIDUAL
# ======================================================

def evaluate_prediction(
    prediction_path: Path,
    horizon_days: int = None,
) -> Optional[EvaluationResult]:
    """
    Evalúa la predicción en su horizonte_days-ésimo día hábil.
    [E7] hit_sign=None si |predicted_return| < HIT_SIGN_MIN_PCT.
    """
    pred = load_json(prediction_path)
    if "meta" not in pred or "prediction" not in pred:
        return None

    meta   = pred["meta"]
    p      = pred["prediction"]
    ticker = meta["ticker"]
    theta  = float(meta.get("theta", 0.75))

    h_days      = horizon_days or int(meta.get("horizon_days", EVAL_HORIZON))
    pred_date   = parse_date(prediction_path.stem)
    if pred_date is None:
        return None

    target_date = nth_business_day(pred_date, h_days)
    unsupported = _is_alpaca_unsupported(ticker)

    real_price       = get_price_at_date(ticker, target_date) if not unsupported else None
    price_now        = float(p.get("price_now", 0))
    price_pred       = float(p.get("price_pred", 0))
    predicted_return = float(p.get("ret_ens_pct", 0))
    rec              = p.get("recommendation", "HOLD")

    real_return = hit_sign = hit_threshold = decision_correct = None
    error_price_pct = error_return_pct = None
    weak_signal = False

    if real_price and price_now > 0:
        real_return = (real_price / price_now - 1) * 100.0

        # [E7] Umbral de señal — excluye pred≈0 del hit_rate
        hit_sign, weak_signal = _calc_hit_sign(predicted_return, real_return)

        hit_threshold    = bool(abs(real_return) >= theta)
        error_price_pct  = abs(price_pred / real_price - 1) * 100.0
        error_return_pct = abs(predicted_return - real_return)

        if rec == "COMPRA":
            decision_correct = real_return >= 0
        elif rec == "VENDE":
            decision_correct = real_return <= 0
        else:
            decision_correct = abs(real_return) < theta

    models_diag = evaluate_models(ticker, pred_date) if not unsupported else {}
    models_sum  = summarize_models(models_diag)

    return EvaluationResult(
        meta=meta,
        prediction_date=str(pred_date),
        evaluation_date=str(target_date),
        evaluation_horizon_days=h_days,
        evaluation_target_date=str(target_date),
        price_now=price_now,
        price_pred=price_pred,
        price_real=real_price,
        predicted_return_pct=predicted_return,
        real_return_pct=real_return,
        error_price_pct=error_price_pct,
        error_return_pct=error_return_pct,
        hit_sign=hit_sign,
        hit_threshold=hit_threshold,
        recommendation=rec,
        decision_correct=decision_correct,
        evaluated_at=datetime.utcnow().isoformat(),
        alpaca_unsupported=unsupported,
        legacy_bad_horizon=False,
        weak_signal=weak_signal,
        models_diagnostics=models_diag,
        models_summary=models_sum,
    )


# ======================================================
# EVALUACIÓN MASIVA
# ======================================================

def evaluate_all(
    ticker: Optional[str] = None,
    max_workers: Optional[int] = None,
    dry_run: bool = False,
    horizon: int = EVAL_HORIZON,
) -> Dict[str, Any]:
    """
    Para cada predicción:
      1. Calcula su fecha objetivo (nth_business_day)
      2. Solo evalúa si esa fecha ya pasó
      3. Recalcula si legacy_bad_horizon != False o falta weak_signal
    """
    pred_root = Path(DATA_PATH) / "predictions"
    eval_root = Path(DATA_PATH) / "evaluations"
    results   = {"evaluated": [], "skipped": [], "errors": [], "summary": {}}

    if not pred_root.exists():
        return results

    today = datetime.utcnow().date()

    ticker_dirs = (
        [pred_root / ticker]
        if ticker
        else [d for d in pred_root.iterdir() if d.is_dir()]
    )

    pending: List[Tuple] = []
    for td in ticker_dirs:
        if not td.is_dir():
            continue

        for pred_file in sorted(td.glob("*.json")):
            pred_date = parse_date(pred_file.stem)
            if pred_date is None:
                continue

            try:
                d      = load_json(pred_file)
                h_days = int(d.get("meta", {}).get("horizon_days", horizon))
            except Exception:
                h_days = horizon

            target_date = nth_business_day(pred_date, h_days)

            if target_date > today:
                continue

            eval_file = eval_root / td.name / f"{pred_file.stem}.json"

            if not _eval_needs_recalculation(eval_file):
                continue

            pending.append((pred_file, h_days))

    if dry_run:
        return {"pending": len(pending)}

    logger.info(f"📊 Evaluator v2.3 | {len(pending)} predicciones a evaluar/recalcular")

    workers = max_workers or MAX_WORKERS
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(evaluate_prediction, f, h): (f, h)
            for f, h in pending
        }
        for fut in concurrent.futures.as_completed(future_map):
            f, h = future_map[fut]
            try:
                ev = fut.result()
                if ev:
                    save_json(eval_root / f.parent.name / f.name, asdict(ev))
                    results["evaluated"].append(str(f))
                else:
                    results["skipped"].append(str(f))
            except Exception as e:
                logger.error(f"Error evaluating {f}: {e}")
                results["errors"].append(str(f))

    results["summary"] = {
        "evaluated": len(results["evaluated"]),
        "skipped":   len(results["skipped"]),
        "errors":    len(results["errors"]),
    }
    logger.info(f"✅ Evaluator v2.3 COMPLETADO | {results['summary']}")
    return results


def evaluate_all_compat(ticker: Optional[str] = None) -> Dict[str, List[str]]:
    r = evaluate_all(ticker)
    return {"evaluated": r["evaluated"], "skipped": r["skipped"]}

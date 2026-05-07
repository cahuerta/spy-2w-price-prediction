"""
darwin_engine/trade_tracker.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIXES v1.1:
  [T1] register_open ahora verifica que el ticker sea ejecutable
       antes de registrar — evita trades fantasma de OR.PA, AIR.PA, etc.
  [T2] register_open retorna None si el ticker no es ejecutable
       → el orchestrator debe llamarlo SOLO después de fill confirmado
  [T3] resolve_pending_trades salta tickers europeos sin precio Alpaca
       → evita que queden pendientes para siempre
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger("trade_tracker")

# ══════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════

DATA_PATH    = Path(os.getenv("DATA_PATH", "/data"))
TRACKER_DIR  = DATA_PATH / "darwin" / "trades"
PRED_DIR     = DATA_PATH / "predictions"

ALPACA_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")

# [T1] Tickers no ejecutables en Alpaca paper
# Se mantienen en anchor_universe para cuando se migre a broker internacional
# pero NO deben registrarse en Darwin (nunca se ejecutan → datos basura)
ALPACA_UNSUPPORTED = {
    "OR.PA", "AIR.PA", "NOVN.SW", "ENB.TO", "MFC.TO",
    "IBE.MC", "SAN.MC", "DTE.DE", "NESN.SW", "RY.TO",
    "TTE.PA", "BNP.PA", "ABI.BR", "ASML.AS",
}


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def _load_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"load_json failed {path}: {e}")
        return {}


def _save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)


def _get_alpaca_client():
    if not ALPACA_KEY or not ALPACA_SECRET:
        return None
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        return StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    except Exception as e:
        logger.warning(f"Alpaca client error: {e}")
        return None


def _get_price_at_date(ticker: str, target_date) -> Optional[float]:
    """Obtiene precio de cierre en una fecha específica desde Alpaca."""
    # [T3] Saltar tickers no soportados
    if ticker.upper() in ALPACA_UNSUPPORTED:
        logger.info(f"⏭ _get_price_at_date skip {ticker} — no soportado en Alpaca")
        return None

    client = _get_alpaca_client()
    if not client:
        return None
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=target_date - timedelta(days=5),
            end=target_date + timedelta(days=1),
        )
        bars = client.get_stock_bars(req).df
        if bars is None or bars.empty:
            return None
        bars = bars.reset_index()
        bars["date"] = bars["timestamp"].dt.date
        row = bars[bars["date"] <= target_date]
        if row.empty:
            return None
        return float(row["close"].iloc[-1])
    except Exception as e:
        logger.warning(f"_get_price_at_date({ticker}, {target_date}): {e}")
        return None


# ══════════════════════════════════════════════════════
# LEER SEÑALES H1-H10 DE LA PREDICCIÓN ACTUAL
# ══════════════════════════════════════════════════════

def _read_h_signals(ticker: str, entry_date) -> Dict[str, Any]:
    """
    Lee el JSON de predicción del día de entrada para obtener
    la señal, confianza y retorno esperado de cada H.
    """
    date_str  = entry_date.strftime("%Y-%m-%d")
    pred_path = PRED_DIR / ticker / f"{date_str}.json"

    if not pred_path.exists():
        ticker_dir = PRED_DIR / ticker
        if not ticker_dir.exists():
            return {}
        candidates = sorted(ticker_dir.glob("*.json"))
        if not candidates:
            return {}
        pred_path = candidates[-1]

    pred = _load_json(pred_path)
    if not pred:
        return {}

    p = pred.get("prediction", {})

    signals = {
        "main": {
            "recommendation": p.get("recommendation", "MANTÉN"),
            "ret_ens_pct":    float(p.get("ret_ens_pct", 0)),
            "confidence":     float(p.get("confidence", 0.5)),
            "theta_dynamic":  float(p.get("theta_dynamic_pct", 1.0)),
            "price_now":      float(p.get("price_now", 0)),
        }
    }

    models = pred.get("models_diagnostics", {})
    for h_key, h_data in models.items():
        if isinstance(h_data, dict):
            signals[h_key] = {
                "pred_return": h_data.get("pred_return", 0),
                "hit_sign":    h_data.get("hit_sign"),
                "error_pct":   h_data.get("error_pct"),
            }

    return signals


# ══════════════════════════════════════════════════════
# API PÚBLICA: REGISTRAR APERTURA
# ══════════════════════════════════════════════════════

def register_open(
    ticker: str,
    entry_price: float,
    shares: int,
    reason: str,
    alpha_score: float,
    executor_genome_id: str  = "executor_v1",
    predictor_genome_id: str = "predictor_v1",
) -> Optional[Dict[str, Any]]:
    """
    Llamar desde trading_orchestrator SOLO después de confirmar fill.

    [T1] Retorna None y no registra si el ticker no es ejecutable
         en Alpaca paper (tickers europeos/canadienses).

    IMPORTANTE: llamar solo cuando result.get("status") in ("executed", "filled")
    """

    # [T1] No registrar tickers no ejecutables en Alpaca
    ticker_upper = ticker.upper()
    if ticker_upper in ALPACA_UNSUPPORTED:
        logger.info(
            f"⏭ register_open skip {ticker} — "
            f"no soportado en Alpaca paper (se registrará cuando se migre a broker internacional)"
        )
        return None

    # [T1] Detectar por sufijo europeo/canadiense aunque no esté en la lista
    unsupported_suffixes = (".PA", ".DE", ".SW", ".TO", ".MC", ".AS", ".MI", ".BR")
    if any(ticker_upper.endswith(s) for s in unsupported_suffixes):
        logger.info(f"⏭ register_open skip {ticker} — sufijo europeo/canadiense no soportado")
        return None

    now        = datetime.now(timezone.utc)
    entry_date = now.date()

    h_signals = _read_h_signals(ticker, entry_date)

    # Horizonte dominante
    dominant_h   = "H6"
    dominant_ret = 0.0
    for h in [f"H{i}" for i in range(10, 0, -1)]:
        sig = h_signals.get(h, {})
        ret = abs(sig.get("pred_return", 0))
        if ret > dominant_ret:
            dominant_ret = ret
            dominant_h   = h

    horizon_days = int(dominant_h[1:]) if dominant_h[1:].isdigit() else 6

    trade = {
        "ticker":               ticker,
        "trade_id":             f"{ticker}_{entry_date}_{now.strftime('%H%M%S')}",
        "status":               "open",
        "predictor_genome_id":  predictor_genome_id,
        "executor_genome_id":   executor_genome_id,
        "entry_price":          round(entry_price, 4),
        "entry_date":           str(entry_date),
        "entry_timestamp":      now.isoformat(),
        "shares":               shares,
        "alpha_score_entry":    round(alpha_score, 4),
        "reason_open":          reason,
        "h_signals_at_entry":   h_signals,
        "dominant_h":           dominant_h,
        "horizon_days":         horizon_days,
        "horizon_target_date":  str(entry_date + timedelta(days=horizon_days)),
        "exit_price":           None,
        "exit_date":            None,
        "exit_timestamp":       None,
        "reason_close":         None,
        "alpha_score_exit":     None,
        "days_held":            None,
        "pnl_real_pct":         None,
        "pnl_real_usd":         None,
        "price_at_horizon":     None,
        "pnl_teorico_pct":      None,
        "oportunidad_pct":      None,
        "closed_before_horizon": None,
        "predictor_was_right":  None,
        "executor_was_right":   None,
    }

    path = TRACKER_DIR / f"{trade['trade_id']}.json"
    _save_json(path, trade)
    logger.info(
        f"📝 TRADE OPEN registrado | {ticker} | "
        f"entrada=${entry_price:.2f} | dominante={dominant_h} | "
        f"horizonte={horizon_days}d"
    )
    return trade


# ══════════════════════════════════════════════════════
# API PÚBLICA: REGISTRAR CIERRE
# ══════════════════════════════════════════════════════

def register_close(
    ticker: str,
    exit_price: float,
    reason: str,
    alpha_score: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Llamar desde trading_orchestrator cuando se ejecuta un CLOSE exitoso.
    """
    # [T1] No procesar tickers europeos
    ticker_upper = ticker.upper()
    if ticker_upper in ALPACA_UNSUPPORTED:
        return None
    unsupported_suffixes = (".PA", ".DE", ".SW", ".TO", ".MC", ".AS", ".MI", ".BR")
    if any(ticker_upper.endswith(s) for s in unsupported_suffixes):
        return None

    trade = _find_open_trade(ticker)
    if not trade:
        logger.warning(f"⚠️ register_close: no se encontró trade abierto para {ticker}")
        return None

    now       = datetime.now(timezone.utc)
    exit_date = now.date()

    entry_price  = float(trade["entry_price"])
    entry_date   = trade["entry_date"]
    horizon_days = int(trade.get("horizon_days", 6))

    recommendation = trade.get("h_signals_at_entry", {}).get("main", {}).get("recommendation", "COMPRA")
    if recommendation == "VENDE":
        pnl_real_pct = (entry_price - exit_price) / entry_price * 100
    else:
        pnl_real_pct = (exit_price - entry_price) / entry_price * 100

    pnl_real_usd = pnl_real_pct / 100 * entry_price * int(trade.get("shares", 0))

    from datetime import date as date_type
    entry_date_obj = (
        entry_date if isinstance(entry_date, date_type)
        else datetime.strptime(str(entry_date), "%Y-%m-%d").date()
    )
    days_held = (exit_date - entry_date_obj).days

    trade.update({
        "status":               "closed_pending_resolution",
        "exit_price":           round(exit_price, 4),
        "exit_date":            str(exit_date),
        "exit_timestamp":       now.isoformat(),
        "reason_close":         reason,
        "alpha_score_exit":     round(alpha_score, 4),
        "days_held":            days_held,
        "pnl_real_pct":         round(pnl_real_pct, 4),
        "pnl_real_usd":         round(pnl_real_usd, 2),
        "closed_before_horizon": days_held < horizon_days,
    })

    path = TRACKER_DIR / f"{trade['trade_id']}.json"
    _save_json(path, trade)
    logger.info(
        f"📝 TRADE CLOSE registrado | {ticker} | "
        f"salida=${exit_price:.2f} | PnL={pnl_real_pct:+.2f}% | "
        f"días={days_held}/{horizon_days} | razón={reason}"
    )
    return trade


# ══════════════════════════════════════════════════════
# RESOLUCIÓN: PRECIO AL HORIZONTE TEÓRICO
# ══════════════════════════════════════════════════════

def resolve_pending_trades() -> List[Dict]:
    """
    Cron diario: busca trades cerrados cuyo horizonte ya venció
    y obtiene el precio real al horizonte para calcular oportunidad.
    """
    if not TRACKER_DIR.exists():
        return []

    resolved = []

    for path in TRACKER_DIR.glob("*.json"):
        trade = _load_json(path)
        if trade.get("status") != "closed_pending_resolution":
            continue

        horizon_date_str = trade.get("horizon_target_date")
        if not horizon_date_str:
            continue

        from datetime import date as date_type
        horizon_date = datetime.strptime(horizon_date_str, "%Y-%m-%d").date()
        today        = datetime.now(timezone.utc).date()

        if today < horizon_date:
            continue

        ticker      = trade["ticker"]

        # [T3] Saltar tickers europeos — nunca tendrán precio en Alpaca
        ticker_upper = ticker.upper()
        unsupported_suffixes = (".PA", ".DE", ".SW", ".TO", ".MC", ".AS", ".MI", ".BR")
        if ticker_upper in ALPACA_UNSUPPORTED or any(ticker_upper.endswith(s) for s in unsupported_suffixes):
            # Marcar como resuelto con datos vacíos para no quedar pendiente siempre
            trade.update({
                "status":              "resolved_unsupported",
                "price_at_horizon":    None,
                "pnl_teorico_pct":     None,
                "oportunidad_pct":     None,
                "predictor_was_right": None,
                "executor_was_right":  None,
                "resolved_at":         datetime.now(timezone.utc).isoformat(),
                "resolve_note":        "ticker no soportado en Alpaca paper",
            })
            _save_json(path, trade)
            logger.info(f"⏭ SKIP resolve {ticker} — europeo/canadiense")
            continue

        entry_price = float(trade["entry_price"])
        pnl_real    = float(trade.get("pnl_real_pct", 0))

        price_at_horizon = _get_price_at_date(ticker, horizon_date)
        if not price_at_horizon:
            logger.warning(f"⚠️ resolve: no precio horizonte {ticker} {horizon_date}")
            continue

        recommendation = trade.get("h_signals_at_entry", {}).get("main", {}).get("recommendation", "COMPRA")
        if recommendation == "VENDE":
            pnl_teorico = (entry_price - price_at_horizon) / entry_price * 100
        else:
            pnl_teorico = (price_at_horizon - entry_price) / entry_price * 100

        oportunidad     = pnl_teorico - pnl_real
        predictor_right = pnl_teorico > 0
        executor_right  = pnl_real >= pnl_teorico if trade.get("closed_before_horizon") else True

        trade.update({
            "status":               "resolved",
            "price_at_horizon":     round(price_at_horizon, 4),
            "pnl_teorico_pct":      round(pnl_teorico, 4),
            "oportunidad_pct":      round(oportunidad, 4),
            "predictor_was_right":  predictor_right,
            "executor_was_right":   executor_right,
            "resolved_at":          datetime.now(timezone.utc).isoformat(),
        })

        _save_json(path, trade)
        resolved.append(trade)

        logger.info(
            f"✅ RESUELTO | {ticker} | "
            f"PnL real={pnl_real:+.2f}% | "
            f"PnL teórico={pnl_teorico:+.2f}% | "
            f"Oportunidad={oportunidad:+.2f}% | "
            f"Predictor={'✅' if predictor_right else '❌'} | "
            f"Executor={'✅' if executor_right else '❌'}"
        )

    logger.info(f"🔍 resolve_pending_trades: {len(resolved)} trades resueltos")
    return resolved


# ══════════════════════════════════════════════════════
# HELPERS INTERNOS
# ══════════════════════════════════════════════════════

def _find_open_trade(ticker: str) -> Optional[Dict]:
    """Busca el trade abierto más reciente para un ticker."""
    if not TRACKER_DIR.exists():
        return None

    candidates = []
    for path in TRACKER_DIR.glob(f"{ticker}_*.json"):
        trade = _load_json(path)
        if trade.get("status") == "open" and trade.get("ticker") == ticker:
            candidates.append((path, trade))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1].get("entry_timestamp", ""), reverse=True)
    return candidates[0][1]


def get_resolved_trades(
    predictor_genome_id: str = None,
    executor_genome_id: str  = None,
    last_n: int = 200,
) -> List[Dict]:
    """Retorna trades resueltos, opcionalmente filtrados por genome."""
    if not TRACKER_DIR.exists():
        return []

    trades = []
    for path in TRACKER_DIR.glob("*.json"):
        trade = _load_json(path)
        if trade.get("status") != "resolved":
            continue
        if predictor_genome_id and trade.get("predictor_genome_id") != predictor_genome_id:
            continue
        if executor_genome_id and trade.get("executor_genome_id") != executor_genome_id:
            continue
        trades.append(trade)

    trades.sort(key=lambda t: t.get("resolved_at", ""), reverse=True)
    return trades[:last_n]


def get_tracker_summary() -> Dict[str, Any]:
    """Resumen del estado actual del tracker."""
    if not TRACKER_DIR.exists():
        return {"total": 0, "open": 0, "pending": 0, "resolved": 0}

    counts  = {"open": 0, "closed_pending_resolution": 0, "resolved": 0}
    pnl_list = []

    for path in TRACKER_DIR.glob("*.json"):
        trade  = _load_json(path)
        status = trade.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
        if status == "resolved" and trade.get("pnl_real_pct") is not None:
            pnl_list.append(float(trade["pnl_real_pct"]))

    avg_pnl  = float(np.mean(pnl_list))              if pnl_list else 0.0
    total_pnl = float(np.sum(pnl_list))              if pnl_list else 0.0
    win_rate  = float(np.mean([p > 0 for p in pnl_list])) if pnl_list else 0.0

    return {
        "total":          sum(counts.values()),
        "open":           counts.get("open", 0),
        "pending":        counts.get("closed_pending_resolution", 0),
        "resolved":       counts.get("resolved", 0),
        "avg_pnl_pct":    round(avg_pnl,   4),
        "total_pnl_pct":  round(total_pnl,  4),
        "win_rate":       round(win_rate,   4),
        "n_pnl_samples":  len(pnl_list),
      }
  

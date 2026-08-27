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

FIX v1.2:
  [T4] register_close valida exit_price > 0 antes de registrar —
       antes el broker podía devolver exit_price=0.0 (bug pre-v3.3)
       causando pnl_real_pct=-100% ficticio que contaminaba el
       fitness del Darwin Engine con nan.
  [T5] register_open valida entry_price > 0 y shares > 0 antes de
       registrar — evita trades fantasma con datos vacíos.

FIX v1.3:
  [T6] _get_price_at_date ahora cae a data_provider.get_price_history()
       (cascada Cache → Chile → EODHD → Yahoo → Twelve) cuando Alpaca
       falla o no tiene permiso de datos (ej. "subscription does not
       permit querying recent SIP data"). Antes esos tickers (ej. STT)
       quedaban en closed_pending_resolution para siempre, reintentando
       y fallando cada día sin nunca resolver ni aportar al fitness
       del Darwin Engine.

FIX v1.4 [AUD-P5]:
  [T7] Selección de dominant_h corregida: antes ganaba el horizonte
       con mayor |pred_return| sin importar si ese horizonte suele
       acertar. Ahora se pondera por hit_rate real del genoma
       campeón de cada horizonte (PredictorGenome.load_champion(h)):
       score = |pred_return| * max(0, hit_rate - 0.5).
       Horizontes SIN hit_rate calculado aún (None — nunca evaluados)
       quedan EXCLUIDOS de la competencia por dominante: no hay
       evidencia de que acierten, así que no deben competir solo
       por la magnitud de su retorno predicho. Si NINGÚN horizonte
       tiene hit_rate todavía (sistema recién arrancado, sin
       historial de evaluaciones), se usa "H6" como fallback —
       mismo comportamiento default que existía antes de este fix.
       Confirmado con el usuario: sin hit_rate no hay confianza en
       la predicción, por lo tanto no debe competir.

FIX v1.5 [AUD-P6] (auditoría 2026-08-24):
  [T8] register_close y resolve_pending_trades invertían el signo del
       PnL cuando `recommendation` (la etiqueta direccional de la
       predicción H10 del día de entrada) era "VENDE" — tratando el
       trade como si fuera una posición corta. Confirmado en
       trading_orchestrator.py: el sistema NUNCA abre cortos, todas
       las posiciones son compras de `shares` (positivo, real).
       `recommendation` es solo la etiqueta direccional que tenía la
       predicción ESE día, no el tipo de posición ejecutada — el
       alpha_score que decide la compra puede ser positivo aunque la
       predicción cruda de H10 diga "VENDE" ese mismo día. Esto podía
       invertir el signo real de ganancias/pérdidas en un subconjunto
       de trades, contaminando pnl_real_pct, pnl_teorico_pct, el
       fitness que decide el campeón de Darwin, y las métricas
       agregadas del tracker — coherente con la brecha no reconciliada
       entre el PnL que reporta Darwin (+$1,088) y la caída real de la
       cuenta (-$16,766) detectada en la auditoría.
       Fix: pnl_real_pct y pnl_teorico_pct se calculan SIEMPRE como
       posición larga, sin ninguna rama condicional por
       `recommendation`.
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

# [Problema 3] Comisión del broker como % del monto, para aislarla del
# movimiento de precio puro. 0.0 por defecto = sin cambio de comportamiento
# hasta que se configure el valor real del broker.
COMMISSION_PCT = float(os.getenv("BROKER_COMMISSION_PCT", "0.0"))
PRED_DIR     = DATA_PATH / "predictions"

ALPACA_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")

# [T1] Tickers no ejecutables en Alpaca paper
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


def _get_price_at_date_alpaca(ticker: str, target_date) -> Optional[float]:
    """Intenta obtener precio de cierre en una fecha específica desde Alpaca."""
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
        logger.warning(f"_get_price_at_date_alpaca({ticker}, {target_date}): {e}")
        return None


def _get_price_at_date_fallback(ticker: str, target_date) -> Optional[float]:
    """
    [T6] Fallback usando data_provider (cascada Cache → Chile → EODHD →
    Yahoo → Twelve). Se usa cuando Alpaca falla o no tiene permiso de
    datos (ej. límite de SIP data en suscripciones básicas).

    Detecta el esquema del DataFrame de forma defensiva, sin asumir
    nombres exactos de columnas (pueden venir de distintos providers).
    """
    try:
        from data_provider import get_price_history
    except Exception as e:
        logger.warning(f"_get_price_at_date_fallback: data_provider no disponible: {e}")
        return None

    try:
        df = get_price_history(ticker, period="1mo", interval="1d")
        if df is None or df.empty:
            return None

        df = df.reset_index()

        # Detectar columna de fecha (índice resateado, 'date', 'Date', 'timestamp', etc.)
        date_col = None
        for cand in ("date", "Date", "timestamp", "Timestamp", "index"):
            if cand in df.columns:
                date_col = cand
                break
        if date_col is None:
            # Última opción: primera columna si parece fecha/datetime
            first_col = df.columns[0]
            if pd_api_is_datetimeish(df[first_col]):
                date_col = first_col
        if date_col is None:
            logger.warning(f"_get_price_at_date_fallback({ticker}): no se identificó columna de fecha")
            return None

        # Normalizar a date (sin hora)
        df["_date_norm"] = pd_to_date(df[date_col])

        # Detectar columna de cierre
        close_col = None
        for cand in ("close", "Close", "Adj Close", "adj_close", "AdjClose"):
            if cand in df.columns:
                close_col = cand
                break
        if close_col is None:
            logger.warning(f"_get_price_at_date_fallback({ticker}): no se identificó columna de cierre")
            return None

        row = df[df["_date_norm"] <= target_date]
        if row.empty:
            return None

        price = row.iloc[-1][close_col]
        if price is None:
            return None
        return float(price)
    except Exception as e:
        logger.warning(f"_get_price_at_date_fallback({ticker}, {target_date}): {e}")
        return None


def pd_api_is_datetimeish(series) -> bool:
    """Heurística simple: ¿la serie parece de tipo fecha/datetime?"""
    try:
        import pandas as pd
        return pd.api.types.is_datetime64_any_dtype(series)
    except Exception:
        return False


def pd_to_date(series):
    """Convierte una serie (datetime, str, o índice) a objetos date puros."""
    import pandas as pd
    return pd.to_datetime(series).dt.date


def _get_price_at_date(ticker: str, target_date) -> Optional[float]:
    """
    Obtiene precio de cierre en una fecha específica.
    Orden: Alpaca primero (rápido, ya autenticado) → fallback a
    data_provider (Cache/Chile/EODHD/Yahoo/Twelve) si Alpaca falla
    o no tiene permiso de datos (ej. SIP data restringido).
    """
    if ticker.upper() in ALPACA_UNSUPPORTED:
        logger.info(f"⏭ _get_price_at_date skip {ticker} — no soportado en Alpaca")
        return None

    price = _get_price_at_date_alpaca(ticker, target_date)
    if price is not None:
        return price

    logger.info(f"↪️ _get_price_at_date {ticker} — Alpaca sin precio, probando fallback data_provider")
    price = _get_price_at_date_fallback(ticker, target_date)
    if price is not None:
        logger.info(f"✅ _get_price_at_date {ticker} resuelto vía fallback data_provider: ${price:.4f}")
        return price

    logger.warning(f"⚠️ _get_price_at_date {ticker} — sin precio en Alpaca ni en fallback")
    return None


# ══════════════════════════════════════════════════════
# LEER SEÑALES H1-H10 DE LA PREDICCIÓN ACTUAL
# ══════════════════════════════════════════════════════

def _read_h_signals(ticker: str, entry_date) -> Dict[str, Any]:
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
# [T7][AUD-P5] SELECCIÓN DE HORIZONTE DOMINANTE
# ══════════════════════════════════════════════════════

def _select_dominant_horizon(h_signals: Dict[str, Any]) -> str:
    """
    [T7] Selecciona el horizonte dominante ponderando por hit_rate real
    del genoma campeón de cada horizonte, no solo por la magnitud del
    retorno predicho.

    score(h) = |pred_return(h)| * max(0, hit_rate(h) - 0.5)

    Horizontes sin hit_rate calculado (None — el genoma nunca fue
    evaluado) quedan EXCLUIDOS de la competencia: sin evidencia de
    que acierten, no deben competir por dominante solo por gritar
    un retorno grande. Si NINGÚN horizonte tiene hit_rate todavía,
    se usa "H6" como fallback (mismo default que antes del fix).
    """
    try:
        from darwin_engine.predictor_genome import PredictorGenome
    except Exception as e:
        logger.warning(f"⚠️ _select_dominant_horizon: PredictorGenome no disponible ({e}) — usando H6")
        return "H6"

    dominant_h     = None
    dominant_score = -1.0

    for h in [f"H{i}" for i in range(1, 11)]:
        sig = h_signals.get(h, {})
        ret = abs(sig.get("pred_return", 0))
        if ret <= 0:
            continue

        try:
            genome   = PredictorGenome.load_champion(int(h[1:]))
            hit_rate = genome.hit_rate
        except Exception as e:
            logger.warning(f"⚠️ _select_dominant_horizon: no se pudo cargar champion {h}: {e}")
            hit_rate = None

        if hit_rate is None:
            # Sin evidencia de acierto → no compite por dominante
            continue

        score = ret * max(0.0, hit_rate - 0.5)
        if score > dominant_score:
            dominant_score = score
            dominant_h     = h

    if dominant_h is None:
        logger.info("ℹ️ _select_dominant_horizon: ningún horizonte con hit_rate calculado → fallback H6")
        return "H6"

    return dominant_h


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
    """
    # [T1] No registrar tickers no ejecutables en Alpaca
    ticker_upper = ticker.upper()
    if ticker_upper in ALPACA_UNSUPPORTED:
        logger.info(f"⏭ register_open skip {ticker} — no soportado en Alpaca paper")
        return None

    unsupported_suffixes = (".PA", ".DE", ".SW", ".TO", ".MC", ".AS", ".MI", ".BR")
    if any(ticker_upper.endswith(s) for s in unsupported_suffixes):
        logger.info(f"⏭ register_open skip {ticker} — sufijo europeo/canadiense no soportado")
        return None

    # [T5] Validar datos de entrada
    if not entry_price or float(entry_price) <= 0:
        logger.error(
            f"❌ register_open {ticker} entry_price={entry_price} inválido — "
            f"trade no registrado (fill no confirmado o precio vacío)"
        )
        return None

    if not shares or int(shares) <= 0:
        logger.error(
            f"❌ register_open {ticker} shares={shares} inválido — "
            f"trade no registrado"
        )
        return None

    now        = datetime.now(timezone.utc)
    entry_date = now.date()

    h_signals = _read_h_signals(ticker, entry_date)

    # [T7][AUD-P5] Horizonte dominante ponderado por hit_rate real
    dominant_h = _select_dominant_horizon(h_signals)

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

    [T8] pnl_real_pct se calcula SIEMPRE como posición larga. El sistema
    nunca abre cortos (trading_orchestrator.py solo compra `shares`
    positivos) — la etiqueta `recommendation` de h_signals_at_entry es
    solo la dirección que tenía la predicción H10 ESE día, no el tipo
    de posición realmente ejecutada, así que no debe invertir el signo
    del PnL.

    FIX (auditoría 2026-08-27, Problema 3):
    pnl_real_pct mezclaba en un solo número el movimiento de precio y
    la comisión del broker, sin dejar rastro de cuánto era cada cosa.
    Ahora se calcula pnl_gross_pct (precio puro, sin comisión) aparte,
    y pnl_real_pct = pnl_gross_pct - COMMISSION_PCT (configurable por
    env var BROKER_COMMISSION_PCT, default 0.0 — sin cambio de
    comportamiento si no se configura nada). Ambos quedan guardados en
    el trade para poder auditar la fricción por separado del error de
    modelo (que ya se mide en oportunidad_pct, calculado en
    resolve_pending_trades). También se marca closed_at_loss_time_exit
    cuando la razón de cierre es "non_anchor_time_exit" y el PnL fue
    negativo — el auditor encontró que esa razón de salida es la única
    con volumen relevante y PnL sistemáticamente negativo (-0.55%
    promedio); el campo solo la deja visible, no cambia la decisión.
    """
    # [T1] No procesar tickers europeos
    ticker_upper = ticker.upper()
    if ticker_upper in ALPACA_UNSUPPORTED:
        return None
    unsupported_suffixes = (".PA", ".DE", ".SW", ".TO", ".MC", ".AS", ".MI", ".BR")
    if any(ticker_upper.endswith(s) for s in unsupported_suffixes):
        return None

    # [T4] Validar exit_price antes de registrar
    if exit_price is None or float(exit_price) <= 0:
        logger.error(
            f"❌ register_close {ticker} exit_price={exit_price} inválido — "
            f"trade no registrado (evitar PnL=-100% ficticio en Darwin)"
        )
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

    # [T8] Siempre posición larga — el sistema no opera en corto.
    # [Problema 3] pnl_gross_pct = movimiento de precio puro, sin comisión.
    # pnl_real_pct = lo que efectivamente queda después de la comisión del
    # broker. Con COMMISSION_PCT=0.0 (default) ambos son iguales — no
    # cambia nada hasta que se configure la comisión real.
    pnl_gross_pct = (exit_price - entry_price) / entry_price * 100
    pnl_real_pct  = pnl_gross_pct - COMMISSION_PCT

    pnl_real_usd = pnl_real_pct / 100 * entry_price * int(trade.get("shares", 0))

    from datetime import date as date_type
    entry_date_obj = (
        entry_date if isinstance(entry_date, date_type)
        else datetime.strptime(str(entry_date), "%Y-%m-%d").date()
    )
    days_held = (exit_date - entry_date_obj).days

    # [Problema 3] Deja visible el patrón que encontró el auditor: la
    # única razón de salida con volumen relevante y PnL sistemáticamente
    # negativo. No cambia ninguna decisión, solo lo etiqueta para poder
    # filtrar/auditar después.
    closed_at_loss_time_exit = (reason == "non_anchor_time_exit" and pnl_real_pct < 0)

    trade.update({
        "status":                "closed_pending_resolution",
        "exit_price":            round(exit_price, 4),
        "exit_date":             str(exit_date),
        "exit_timestamp":        now.isoformat(),
        "reason_close":          reason,
        "alpha_score_exit":      round(alpha_score, 4),
        "days_held":             days_held,
        "pnl_gross_pct":         round(pnl_gross_pct, 4),
        "pnl_real_pct":          round(pnl_real_pct, 4),
        "pnl_real_usd":          round(pnl_real_usd, 2),
        "closed_before_horizon": days_held < horizon_days,
        "closed_at_loss_time_exit": closed_at_loss_time_exit,
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

    [T8] pnl_teorico_pct se calcula SIEMPRE como posición larga, por
    la misma razón que en register_close: el sistema no opera en corto.
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

        ticker = trade["ticker"]

        # [T3] Saltar tickers europeos — nunca tendrán precio en Alpaca
        ticker_upper = ticker.upper()
        unsupported_suffixes = (".PA", ".DE", ".SW", ".TO", ".MC", ".AS", ".MI", ".BR")
        if ticker_upper in ALPACA_UNSUPPORTED or any(ticker_upper.endswith(s) for s in unsupported_suffixes):
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
        # [Problema 3] Compat: trades cerrados antes de este fix no tienen
        # pnl_gross_pct — se usa pnl_real como equivalente (COMMISSION_PCT
        # era 0.0 de todas formas en esos casos).
        pnl_gross   = float(trade.get("pnl_gross_pct", pnl_real))

        price_at_horizon = _get_price_at_date(ticker, horizon_date)
        if not price_at_horizon:
            logger.warning(f"⚠️ resolve: no precio horizonte {ticker} {horizon_date}")
            continue

        # [T8] Siempre posición larga — el sistema no opera en corto.
        pnl_teorico = (price_at_horizon - entry_price) / entry_price * 100

        oportunidad     = pnl_teorico - pnl_real
        # [Problema 3] slippage_pct: fricción de ejecución pura (comisión +
        # spread de la orden de cierre), separada del error de timing del
        # modelo que ya mide oportunidad_pct.
        slippage_pct    = pnl_gross - pnl_teorico
        predictor_right = pnl_teorico > 0
        executor_right  = pnl_real >= pnl_teorico if trade.get("closed_before_horizon") else True

        trade.update({
            "status":               "resolved",
            "price_at_horizon":     round(price_at_horizon, 4),
            "pnl_teorico_pct":      round(pnl_teorico, 4),
            "oportunidad_pct":      round(oportunidad, 4),
            "slippage_pct":         round(slippage_pct, 4),
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
            f"Slippage={slippage_pct:+.2f}% | "
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

    counts   = {"open": 0, "closed_pending_resolution": 0, "resolved": 0}
    pnl_list = []

    for path in TRACKER_DIR.glob("*.json"):
        trade  = _load_json(path)
        status = trade.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
        if status == "resolved" and trade.get("pnl_real_pct") is not None:
            pnl_list.append(float(trade["pnl_real_pct"]))

    avg_pnl   = float(np.mean(pnl_list))                    if pnl_list else 0.0
    total_pnl = float(np.sum(pnl_list))                     if pnl_list else 0.0
    win_rate  = float(np.mean([p > 0 for p in pnl_list]))   if pnl_list else 0.0

    return {
        "total":         sum(counts.values()),
        "open":          counts.get("open", 0),
        "pending":       counts.get("closed_pending_resolution", 0),
        "resolved":      counts.get("resolved", 0),
        "avg_pnl_pct":   round(avg_pnl,   4),
        "total_pnl_pct": round(total_pnl,  4),
        "win_rate":      round(win_rate,   4),
        "n_pnl_samples": len(pnl_list),
  }

# =========================================================
# intraday_tracker.py — INTRADAY TRACKER v2.0
# =========================================================
# Corre cada 30 minutos durante horario de mercado.
#
# Dos funciones principales:
#
# 1. ENTRY OPTIMIZER (candidatos COMPRA)
#    Compara precio actual vs price_curve[día 1]
#    → entry_score → entrar_ahora true/false
#
# 2. POSITION MONITOR (posiciones abiertas)
#    Compara precio actual vs price_curve[día_actual]
#    → curve_status: ahead | on_track | lagging | diverging
#    → Informa al dashboard, no cierra posiciones
#
# Guarda en /data/intraday/{fecha}.json
# =========================================================

import os
import json
import logging
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger("intraday_tracker")
logging.basicConfig(level=logging.INFO)

# =========================================================
# CONFIG
# =========================================================

DATA_PATH     = Path(os.getenv("DATA_PATH", "/data"))
PRED_DIR      = DATA_PATH / "predictions"
INTRADAY_DIR  = DATA_PATH / "intraday"
ALPHA_FILE    = DATA_PATH / "alpha_last.json"
ALPACA_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")

ENTRY_THRESHOLD   = float(os.getenv("INTRADAY_ENTRY_THRESHOLD", "1.005"))
MIN_ALPHA_COMPRA  = float(os.getenv("INTRADAY_MIN_ALPHA",        "0.65"))

# Bandas de tolerancia para status de curva
AHEAD_THRESHOLD     = float(os.getenv("INTRADAY_AHEAD",     "1.015"))  # +1.5% sobre curva
LAGGING_THRESHOLD   = float(os.getenv("INTRADAY_LAGGING",   "0.985"))  # -1.5% bajo curva
DIVERGING_THRESHOLD = float(os.getenv("INTRADAY_DIVERGING", "0.970"))  # -3.0% bajo curva

_alpaca_client: Optional[StockHistoricalDataClient] = None


# =========================================================
# HELPERS
# =========================================================

def _get_alpaca_client() -> StockHistoricalDataClient:
    global _alpaca_client
    if _alpaca_client is None:
        _alpaca_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    return _alpaca_client


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"⚠️ load_json {path}: {e}")
        return {}


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# =========================================================
# PRECIO ACTUAL + MOMENTUM
# =========================================================

def _get_intraday_bars(ticker: str, lookback_minutes: int = 90) -> Optional[List[float]]:
    try:
        client = _get_alpaca_client()
        now    = datetime.now(timezone.utc)
        start  = now - timedelta(minutes=lookback_minutes)

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=start,
            end=now,
        )
        bars = client.get_stock_bars(request).df
        if bars is None or bars.empty:
            return None

        bars = bars.reset_index()
        if "close" not in bars.columns:
            return None

        closes = bars["close"].tolist()
        return closes if len(closes) >= 2 else None

    except Exception as e:
        logger.warning(f"⚠️ intraday bars {ticker}: {e}")
        return None


def _get_current_price(ticker: str) -> Optional[float]:
    bars = _get_intraday_bars(ticker, lookback_minutes=10)
    if bars:
        return float(bars[-1])
    return None


def _momentum_score(closes: List[float]) -> float:
    if not closes or len(closes) < 3:
        return 0.0
    last   = closes[-1]
    prev   = float(np.mean(closes[-4:-1]))
    if prev <= 0:
        return 0.0
    change = (last - prev) / prev
    return float(np.clip(change * 100, -1.0, 1.0))


# =========================================================
# CURVA ESPERADA
# =========================================================

def _get_price_curve(ticker: str) -> Optional[Dict]:
    ticker_dir = PRED_DIR / ticker.upper()
    if not ticker_dir.exists():
        return None
    candidates = sorted(ticker_dir.glob("*.json"))
    if not candidates:
        return None
    pred  = _load_json(candidates[-1])
    curve = pred.get("price_curve")
    if not curve or not curve.get("price_path"):
        return None
    return curve


def _get_expected_price_day1(ticker: str) -> Optional[float]:
    curve = _get_price_curve(ticker)
    if not curve:
        return None
    path = curve.get("price_path", [])
    return float(path[0]) if path else None


def _get_expected_price_for_day(ticker: str, dia: int) -> Optional[float]:
    """Precio esperado para el día N de la posición (1-based, clamp 1-9)."""
    curve = _get_price_curve(ticker)
    if not curve:
        return None
    path = curve.get("price_path", [])
    idx  = max(0, min(dia - 1, len(path) - 1))
    return float(path[idx]) if path else None


# =========================================================
# CANDIDATOS COMPRA
# =========================================================

def _get_compra_candidates() -> List[str]:
    alpha_data = _load_json(ALPHA_FILE)
    results    = alpha_data.get("results", {})
    candidates = []

    for ticker, data in results.items():
        if not isinstance(data, dict):
            continue
        score = float(data.get("alpha_score", 0))
        if score < MIN_ALPHA_COMPRA:
            continue
        ticker_dir = PRED_DIR / ticker.upper()
        if not ticker_dir.exists():
            continue
        pred_files = sorted(ticker_dir.glob("*.json"))
        if not pred_files:
            continue
        pred = _load_json(pred_files[-1])
        rec  = pred.get("prediction", {}).get("recommendation", "")
        if rec == "COMPRA":
            candidates.append(ticker.upper())

    return candidates


# =========================================================
# POSICIONES ABIERTAS
# =========================================================

def _get_open_positions_with_date() -> List[Dict]:
    try:
        from positions_meta import get_all
        meta  = get_all()
        today = datetime.now(timezone.utc).date()
        result = []
        for ticker, data in meta.items():
            entry_date_str = data.get("entry_date")
            if not entry_date_str:
                continue
            try:
                entry_date = date.fromisoformat(entry_date_str)
                dia_actual = (today - entry_date).days + 1
                dia_actual = max(1, min(dia_actual, 9))
                result.append({
                    "ticker":     ticker.upper(),
                    "entry_date": entry_date_str,
                    "dia_actual": dia_actual,
                })
            except Exception:
                continue
        return result
    except Exception as e:
        logger.warning(f"⚠️ positions_meta no disponible: {e}")
        return []


# =========================================================
# EVALUAR TIMING DE ENTRADA
# =========================================================

def _evaluate_entry_timing(ticker: str) -> Optional[Dict]:
    precio_actual   = _get_current_price(ticker)
    precio_esperado = _get_expected_price_day1(ticker)

    if not precio_actual or not precio_esperado or precio_esperado <= 0:
        return None

    tracking_ratio = precio_actual / precio_esperado
    bars           = _get_intraday_bars(ticker, lookback_minutes=90)
    momentum       = _momentum_score(bars) if bars else 0.0
    precio_score   = float(np.clip(0.5 + (1.0 - tracking_ratio) * 50, 0.0, 1.0))
    momentum_score = float(np.clip((momentum + 1.0) / 2.0, 0.0, 1.0))
    entry_score    = round(0.60 * precio_score + 0.40 * momentum_score, 3)

    precio_ok    = tracking_ratio <= ENTRY_THRESHOLD
    momentum_ok  = momentum > 0
    score_ok     = entry_score >= 0.55
    entrar_ahora = precio_ok and momentum_ok and score_ok

    razones = []
    if not precio_ok:
        razones.append(f"precio caro vs curva ({tracking_ratio:.3f})")
    if not momentum_ok:
        razones.append(f"momentum negativo ({momentum:.2f})")
    if not score_ok:
        razones.append(f"score insuficiente ({entry_score:.2f})")
    if entrar_ahora:
        razones.append(
            f"precio ok ({tracking_ratio:.3f}) + momentum {momentum:.2f} + score {entry_score:.2f}"
        )

    return {
        "ticker":               ticker,
        "tipo":                 "entry_candidate",
        "precio_actual":        round(precio_actual, 4),
        "precio_esperado_dia1": round(precio_esperado, 4),
        "tracking_ratio":       round(tracking_ratio, 4),
        "momentum":             round(momentum, 4),
        "entry_score":          entry_score,
        "entrar_ahora":         entrar_ahora,
        "razon":                " | ".join(razones),
    }


# =========================================================
# EVALUAR POSICIÓN ABIERTA VS CURVA
# =========================================================

def _evaluate_open_position(ticker: str, dia_actual: int, entry_date: str) -> Optional[Dict]:
    """
    Compara precio actual vs precio esperado para el día N de la curva.

    curve_status:
      ahead     → ratio >= 1.015  (va mejor de lo esperado)
      on_track  → ratio entre 0.985 y 1.015 (normal)
      lagging   → ratio entre 0.970 y 0.985 (por debajo)
      diverging → ratio < 0.970  (alerta)
    """
    precio_actual   = _get_current_price(ticker)
    precio_esperado = _get_expected_price_for_day(ticker, dia_actual)

    if not precio_actual or not precio_esperado or precio_esperado <= 0:
        return None

    tracking_ratio = precio_actual / precio_esperado
    bars           = _get_intraday_bars(ticker, lookback_minutes=90)
    momentum       = _momentum_score(bars) if bars else 0.0

    if tracking_ratio >= AHEAD_THRESHOLD:
        curve_status = "ahead"
    elif tracking_ratio >= LAGGING_THRESHOLD:
        curve_status = "on_track"
    elif tracking_ratio >= DIVERGING_THRESHOLD:
        curve_status = "lagging"
    else:
        curve_status = "diverging"

    # Retorno real vs precio original de entrada
    curve              = _get_price_curve(ticker)
    ret_vs_entrada     = None
    ret_esperado_total = None

    if curve:
        price_now_orig = float(curve.get("price_now", 0))
        path           = curve.get("price_path", [])
        if price_now_orig > 0:
            ret_vs_entrada = round((precio_actual / price_now_orig - 1) * 100, 2)
            if path:
                ret_esperado_total = round((path[-1] / price_now_orig - 1) * 100, 2)

    logger.info(
        f"📊 {ticker} día {dia_actual} | "
        f"actual={precio_actual:.2f} esperado={precio_esperado:.2f} "
        f"ratio={tracking_ratio:.3f} → {curve_status}"
    )

    return {
        "ticker":                 ticker,
        "tipo":                   "open_position",
        "dia_actual":             dia_actual,
        "entry_date":             entry_date,
        "precio_actual":          round(precio_actual, 4),
        "precio_esperado_hoy":    round(precio_esperado, 4),
        "tracking_ratio":         round(tracking_ratio, 4),
        "momentum":               round(momentum, 4),
        "curve_status":           curve_status,
        "ret_vs_entrada_pct":     ret_vs_entrada,
        "ret_esperado_total_pct": ret_esperado_total,
    }


# =========================================================
# RUN PRINCIPAL
# =========================================================

def run_intraday_tracker() -> Dict:
    ahora     = datetime.now(timezone.utc)
    fecha_str = ahora.date().isoformat()
    hora_str  = ahora.strftime("%H:%M")

    logger.info(f"📡 Intraday tracker v2.0 | {fecha_str} {hora_str} UTC")

    # ── 1. CANDIDATOS COMPRA ─────────────────────────────────
    candidatos = _get_compra_candidates()
    senales    = []
    entrar_now = []

    if candidatos:
        logger.info(f"🔍 Candidatos COMPRA: {candidatos}")
        for ticker in candidatos:
            try:
                resultado = _evaluate_entry_timing(ticker)
                if resultado:
                    senales.append(resultado)
                    if resultado["entrar_ahora"]:
                        entrar_now.append(ticker)
                        logger.info(f"✅ ENTRADA OK {ticker} | score={resultado['entry_score']}")
                    else:
                        logger.info(f"⏳ ESPERAR {ticker} | {resultado['razon']}")
            except Exception as e:
                logger.error(f"❌ Error entry {ticker}: {e}")
    else:
        logger.info("ℹ️  Sin candidatos COMPRA")

    # ── 2. POSICIONES ABIERTAS ───────────────────────────────
    posiciones_abiertas = _get_open_positions_with_date()
    monitor_posiciones  = []

    if posiciones_abiertas:
        logger.info(f"📊 Monitoreando {len(posiciones_abiertas)} posiciones")
        for pos in posiciones_abiertas:
            try:
                resultado = _evaluate_open_position(
                    pos["ticker"], pos["dia_actual"], pos["entry_date"]
                )
                if resultado:
                    monitor_posiciones.append(resultado)
            except Exception as e:
                logger.error(f"❌ Error monitor {pos['ticker']}: {e}")
    else:
        logger.info("ℹ️  Sin posiciones abiertas")

    # ── 3. GUARDAR SNAPSHOT ──────────────────────────────────
    snapshot_path = INTRADAY_DIR / f"{fecha_str}.json"
    snapshot      = _load_json(snapshot_path)

    snapshot[hora_str] = {
        "timestamp":          ahora.isoformat(),
        "senales":            senales,
        "entrar_ahora":       entrar_now,
        "monitor_posiciones": monitor_posiciones,
    }

    _save_json(snapshot_path, snapshot)

    logger.info(
        f"💾 Snapshot | compra={len(senales)} entrar={len(entrar_now)} "
        f"posiciones={len(monitor_posiciones)}"
    )

    return {
        "hora":               hora_str,
        "candidatos":         len(senales),
        "entrar_ahora":       entrar_now,
        "senales":            senales,
        "monitor_posiciones": monitor_posiciones,
    }


# =========================================================
# API PARA ORCHESTRATOR Y DASHBOARD
# =========================================================

def get_entry_signals_today() -> Dict[str, Dict]:
    """Mejor momento del día por ticker candidato COMPRA."""
    fecha_str     = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = INTRADAY_DIR / f"{fecha_str}.json"
    snapshot      = _load_json(snapshot_path)

    if not snapshot:
        return {}

    best_by_ticker: Dict[str, Dict] = {}
    for hora, data in snapshot.items():
        for senal in data.get("senales", []):
            ticker = senal.get("ticker")
            if not ticker:
                continue
            score = senal.get("entry_score", 0)
            if ticker not in best_by_ticker or score > best_by_ticker[ticker]["entry_score"]:
                best_by_ticker[ticker] = {**senal, "mejor_hora": hora}

    return best_by_ticker


def get_position_status_today() -> Dict[str, Dict]:
    """
    Último status de cada posición abierta monitoreada hoy.
    Llamable desde dashboard para mostrar estado vs curva.
    """
    fecha_str     = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = INTRADAY_DIR / f"{fecha_str}.json"
    snapshot      = _load_json(snapshot_path)

    if not snapshot:
        return {}

    latest: Dict[str, Dict] = {}
    for hora in sorted(snapshot.keys()):
        for pos in snapshot[hora].get("monitor_posiciones", []):
            ticker = pos.get("ticker")
            if ticker:
                latest[ticker] = {**pos, "ultima_hora": hora}

    return latest
    

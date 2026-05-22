# =========================================================
# intraday_tracker.py — INTRADAY TRACKER v2.3
# =========================================================
# v2.3:
#   Cierre por divergencia más robusto:
#   Requiere 2 desviaciones estándar bajo lower_band
#   para clasificar como "diverging" real.
#   Evita cierres prematuros por ruido intraday.
# =========================================================

import os
import json
import logging
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import yfinance as yf

logger = logging.getLogger("intraday_tracker")
logging.basicConfig(level=logging.INFO)

# =========================================================
# CONFIG
# =========================================================

DATA_PATH     = Path(os.getenv("DATA_PATH", "/data"))
PRED_DIR      = DATA_PATH / "predictions"
INTRADAY_DIR  = DATA_PATH / "intraday"
ALPHA_FILE    = DATA_PATH / "alpha_last.json"

MIN_ALPHA_COMPRA = float(os.getenv("INTRADAY_MIN_ALPHA", "0.65"))

ENTRY_THRESHOLD     = float(os.getenv("INTRADAY_ENTRY_THRESHOLD", "1.005"))
AHEAD_THRESHOLD     = float(os.getenv("INTRADAY_AHEAD",     "1.015"))
LAGGING_THRESHOLD   = float(os.getenv("INTRADAY_LAGGING",   "0.985"))
DIVERGING_THRESHOLD = float(os.getenv("INTRADAY_DIVERGING", "0.970"))

# Multiplicador de desviación para cierre real
DIVERGING_STD_MULT = float(os.getenv("INTRADAY_DIVERGING_STD_MULT", "2.0"))


# =========================================================
# HELPERS
# =========================================================

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
# PRECIO ACTUAL CON YAHOO FINANCE
# =========================================================

def _get_intraday_bars(ticker: str, lookback_minutes: int = 90) -> Optional[List[float]]:
    try:
        interval = "5m" if lookback_minutes <= 60 else "15m"
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="1d", interval=interval)
        if hist is None or hist.empty:
            return None
        closes = hist["Close"].dropna().tolist()
        return closes if len(closes) >= 2 else None
    except Exception as e:
        logger.warning(f"⚠️ intraday bars Yahoo {ticker}: {e}")
        return None


def _get_current_price(ticker: str) -> Optional[float]:
    try:
        tk = yf.Ticker(ticker)
        try:
            fi = tk.fast_info
            price = getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None)
            if price and float(price) > 0:
                return float(price)
        except Exception:
            pass
        hist = tk.history(period="1d", interval="5m")
        if hist is not None and not hist.empty:
            closes = hist["Close"].dropna()
            if len(closes) > 0:
                return float(closes.iloc[-1])
        return None
    except Exception as e:
        logger.warning(f"⚠️ current price Yahoo {ticker}: {e}")
        return None


def _momentum_score(closes: List[float]) -> float:
    if not closes or len(closes) < 3:
        return 0.0
    last = closes[-1]
    prev = float(np.mean(closes[-4:-1]))
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


def _get_band_for_day(curve: Dict, dia: int, band: str) -> Optional[float]:
    idx  = max(0, min(dia - 1, 8))
    data = curve.get(band, [])
    if data and idx < len(data):
        return float(data[idx])
    path = curve.get("price_path", [])
    return float(path[idx]) if path and idx < len(path) else None


def _get_expected_price_for_day(ticker: str, dia: int) -> Optional[float]:
    curve = _get_price_curve(ticker)
    if not curve:
        return None
    path = curve.get("price_path", [])
    idx  = max(0, min(dia - 1, len(path) - 1))
    return float(path[idx]) if path else None


def _get_expected_price_day1(ticker: str) -> Optional[float]:
    return _get_expected_price_for_day(ticker, 1)


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
    precio_actual = _get_current_price(ticker)
    if not precio_actual:
        return None

    curve           = _get_price_curve(ticker)
    precio_esperado = _get_expected_price_day1(ticker)

    if not precio_esperado or precio_esperado <= 0:
        return None

    tracking_ratio = precio_actual / precio_esperado
    bars           = _get_intraday_bars(ticker, lookback_minutes=90)
    momentum       = _momentum_score(bars) if bars else 0.0

    lower_band_dia1 = _get_band_for_day(curve, 1, "lower_band") if curve else None
    upper_band_dia1 = _get_band_for_day(curve, 1, "upper_band") if curve else None

    if lower_band_dia1 and upper_band_dia1:
        band_width   = upper_band_dia1 - lower_band_dia1
        precio_norm  = (precio_actual - lower_band_dia1) / band_width if band_width > 0 else 0.5
        precio_score = float(np.clip(1.0 - precio_norm, 0.0, 1.0))
        precio_ok    = precio_actual <= upper_band_dia1
        usar_bandas  = True
    else:
        precio_score = float(np.clip(0.5 + (1.0 - tracking_ratio) * 50, 0.0, 1.0))
        precio_ok    = tracking_ratio <= ENTRY_THRESHOLD
        usar_bandas  = False

    momentum_score = float(np.clip((momentum + 1.0) / 2.0, 0.0, 1.0))
    entry_score    = round(0.60 * precio_score + 0.40 * momentum_score, 3)

    momentum_ok  = momentum > 0
    score_ok     = entry_score >= 0.55
    entrar_ahora = precio_ok and momentum_ok and score_ok

    razones = []
    if not precio_ok:
        razones.append(
            f"precio sobre banda superior ({tracking_ratio:.3f})" if usar_bandas
            else f"precio caro vs curva ({tracking_ratio:.3f})"
        )
    if not momentum_ok:
        razones.append(f"momentum negativo ({momentum:.2f})")
    if not score_ok:
        razones.append(f"score insuficiente ({entry_score:.2f})")
    if entrar_ahora:
        razones.append(
            f"precio ok ({tracking_ratio:.3f}) + momentum {momentum:.2f} + score {entry_score:.2f}"
        )

    result = {
        "ticker":               ticker,
        "tipo":                 "entry_candidate",
        "precio_actual":        round(precio_actual, 4),
        "precio_esperado_dia1": round(precio_esperado, 4),
        "tracking_ratio":       round(tracking_ratio, 4),
        "momentum":             round(momentum, 4),
        "entry_score":          entry_score,
        "entrar_ahora":         entrar_ahora,
        "razon":                " | ".join(razones),
        "uso_bandas":           usar_bandas,
        "fuente_precio":        "yahoo",
    }

    if usar_bandas:
        result["lower_band_dia1"] = round(lower_band_dia1, 4)
        result["upper_band_dia1"] = round(upper_band_dia1, 4)

    if curve and curve.get("analysis"):
        result["curve_analysis"] = curve["analysis"]

    return result


# =========================================================
# EVALUAR POSICIÓN ABIERTA VS CURVA
# =========================================================

def _evaluate_open_position(ticker: str, dia_actual: int, entry_date: str) -> Optional[Dict]:
    precio_actual   = _get_current_price(ticker)
    precio_esperado = _get_expected_price_for_day(ticker, dia_actual)

    if not precio_actual or not precio_esperado or precio_esperado <= 0:
        return None

    tracking_ratio = precio_actual / precio_esperado
    bars           = _get_intraday_bars(ticker, lookback_minutes=90)
    momentum       = _momentum_score(bars) if bars else 0.0
    curve          = _get_price_curve(ticker)

    upper = _get_band_for_day(curve, dia_actual, "upper_band") if curve else None
    lower = _get_band_for_day(curve, dia_actual, "lower_band") if curve else None

    if upper and lower and lower > 0:
        usar_bandas = True
        std_band    = (upper - lower) / 3.0

        # =====================================================
        # v2.3 — Cierre robusto: requiere 2σ bajo lower_band
        # Antes cerraba al primer toque del lower_band
        # Ahora requiere caída significativa fuera del cono
        # =====================================================
        diverging_threshold = lower - (DIVERGING_STD_MULT * std_band)

        if precio_actual > upper:
            curve_status = "ahead"
        elif precio_actual >= lower:
            curve_status = "on_track"
        elif precio_actual >= diverging_threshold:
            curve_status = "lagging"       # entre lower y -2σ → esperar
        else:
            curve_status = "diverging"     # bajo -2σ → cierre real

    else:
        usar_bandas = False
        if tracking_ratio >= AHEAD_THRESHOLD:
            curve_status = "ahead"
        elif tracking_ratio >= LAGGING_THRESHOLD:
            curve_status = "on_track"
        elif tracking_ratio >= DIVERGING_THRESHOLD:
            curve_status = "lagging"
        else:
            curve_status = "diverging"

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
        + (f" [bandas]" if usar_bandas else " [fallback]")
        + f" [yahoo]"
    )

    result = {
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
        "uso_bandas":             usar_bandas,
        "fuente_precio":          "yahoo",
    }

    if usar_bandas:
        result["upper_band_hoy"]        = round(upper, 4)
        result["lower_band_hoy"]        = round(lower, 4)
        result["diverging_threshold"]   = round(diverging_threshold, 4)

    if curve and curve.get("analysis"):
        best_exit = curve["analysis"].get("best_exit_day")
        if best_exit:
            result["best_exit_day"]    = best_exit
            result["dias_para_salida"] = max(0, best_exit - dia_actual)

    return result


# =========================================================
# RUN PRINCIPAL
# =========================================================

def run_intraday_tracker() -> Dict:
    ahora     = datetime.now(timezone.utc)
    fecha_str = ahora.date().isoformat()
    hora_str  = ahora.strftime("%H:%M")

    logger.info(f"📡 Intraday tracker v2.2 (Yahoo) | {fecha_str} {hora_str} UTC")

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

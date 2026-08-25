# =========================================================
# intraday_tracker.py — INTRADAY TRACKER v3.1
# =========================================================
# v3.0 — Refactor arquitectural:
#   El tracker ya NO genera su propio universo.
#   Recibe listas de tickers desde el orchestrator
#   (decisiones ya tomadas por los PM).
#   Solo evalúa timing y retorna SUGERENCIAS.
#   La decisión final sigue siendo del orchestrator/CG.
#
#   Cambios principales v3.0:
#   - Eliminado: _get_compra_candidates() — universo propio
#   - Eliminado: run_intraday_tracker() — corría solo
#   - Nuevo: evaluar_timing_entrada(tickers) — recibe lista del PM
#   - Nuevo: evaluar_posiciones_abiertas(tickers) — recibe lista del PM
#   - Cierre: considera PnL actual antes de sugerir cerrar
#     (no cierra posiciones ganadoras por divergencia de curva)
#   - Retorna sugerencias, nunca decisiones finales
#
# v3.1 — Contexto de curva futura (informativo para el PM):
#   El tracker ahora analiza la curva COMPLETA desde dia_actual
#   en adelante y agrega contexto al resultado para que el PM
#   pueda tomar decisiones más informadas. NO cambia sugerencias.
#
#   Nuevos campos en resultado (posiciones abiertas):
#   - curva_futura: dict con análisis desde dia_actual en adelante
#     · slope_futura: "sube" | "baja" | "plana"
#     · ret_desde_hoy_a_peak_pct: retorno esperado hasta el máximo
#     · ret_desde_hoy_a_final_pct: retorno esperado al día final
#     · dias_hasta_peak: días restantes hasta el máximo de la curva
#     · precio_peak_esperado: precio máximo esperado en la curva
#     · precio_final_esperado: precio al día final de la curva
#     · en_zona_valle: True si precio actual está bajo lower_band hoy
#       (señal de potencial buy-the-dip para el PM)
#
#   Nuevos campos en resultado (timing entrada):
#   - curva_futura: igual que arriba pero desde día 1
#     · en_zona_valle: True si precio actual < lower_band día 1
#       (potencial entrada táctica en caída)
#
#   ARQUITECTURA: El tracker SOLO informa. El PM decide.
# =========================================================

import os
import json
import logging
from datetime import datetime, timezone, date
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

ENTRY_THRESHOLD     = float(os.getenv("INTRADAY_ENTRY_THRESHOLD", "1.005"))
AHEAD_THRESHOLD     = float(os.getenv("INTRADAY_AHEAD",           "1.015"))
LAGGING_THRESHOLD   = float(os.getenv("INTRADAY_LAGGING",         "0.985"))
DIVERGING_THRESHOLD = float(os.getenv("INTRADAY_DIVERGING",       "0.970"))

# Multiplicador de desviación para cierre real (2σ bajo lower_band)
# [AUD-P9] auditoría 2026-08-25: 2.0 era demasiado sensible a ruido
# intradía normal — subido a 3.0 (banda menos sensible).
DIVERGING_STD_MULT = float(os.getenv("INTRADAY_DIVERGING_STD_MULT", "3.0"))

# [AUD-P9] Piso de días mínimos antes de permitir CERRAR por divergencia
# de curva. dia_actual=1 es el día de apertura — sin este piso, 46/231
# trades (19.9%, según auditoría) cerraban con days_held=0. Excepción:
# si la pérdida ya supera SAME_DAY_STOP_LOSS_OVERRIDE_PCT, el cierre
# procede igual aunque sea el primer día (no proteger una caída fuerte).
MIN_HOLD_DAYS_BEFORE_CURVE_EXIT = int(os.getenv("INTRADAY_MIN_HOLD_DAYS", "1"))
SAME_DAY_STOP_LOSS_OVERRIDE_PCT = float(os.getenv("INTRADAY_SAME_DAY_STOP_LOSS", "4.0"))  # mismo % que STOP_LOSS_NEUTRAL_PCT en pm_neutral.py

# PnL mínimo para proteger con trailing en vez de cerrar directo
TRAILING_PNL_THRESHOLD = float(os.getenv("INTRADAY_TRAILING_PNL", "0.02"))  # 2%
# Caída desde máximo para activar trailing stop
TRAILING_DROP_PCT      = float(os.getenv("INTRADAY_TRAILING_DROP", "0.05"))  # 5%

# [N2] Diseño acordado 2026-08-25: si un ticker en cartera aparece en
# el top 15 BAJISTA del ranking de noticias (news_ranker.py), se usa
# un umbral de "diverging" más ajustado SOLO para esa posición, ese
# día — reacciona antes a la caída en vez de esperar el 3% normal.
# No toca el stop loss duro de los PM (última línea de defensa
# intacta) — solo el umbral de sugerencia de este tracker.
NEWS_RANKING_FILE    = DATA_PATH / "news_ranking.json"
NEWS_TIGHT_STD_MULT  = float(os.getenv("INTRADAY_NEWS_STD_MULT",  "1.0"))   # vs 3.0 normal
NEWS_TIGHT_THRESHOLD = float(os.getenv("INTRADAY_NEWS_THRESHOLD", "0.990"))  # vs 0.970 normal


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
# [N2] TICKERS CON NOTICIA BAJISTA — stop más ajustado
# =========================================================

def _load_bearish_news_tickers() -> set:
    """
    Lee news_ranking.json (generado por news_ranker.py) y retorna el
    conjunto de tickers en el top 15 bajista. Se usa para ajustar el
    umbral de "diverging" en posiciones abiertas — no filtra por si
    el ticker está en el universo, porque lo que importa acá es si
    TENEMOS la posición, no si es candidato nuevo.
    """
    data = _load_json(NEWS_RANKING_FILE)
    if not data:
        return set()
    return {
        str(item.get("ticker", "")).upper()
        for item in data.get("bearish", [])
        if item.get("ticker")
    }


# =========================================================
# PRECIO Y MOMENTUM — YAHOO FINANCE
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
            fi    = tk.fast_info
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


def _get_ret_final_esperado(curve: Dict) -> Optional[float]:
    """Retorno esperado total según precio final de la curva vs precio de entrada."""
    if not curve:
        return None
    price_now = float(curve.get("price_now", 0))
    path      = curve.get("price_path", [])
    if price_now > 0 and path:
        return round((path[-1] / price_now - 1) * 100, 2)
    return None


# =========================================================
# ANÁLISIS DE CURVA FUTURA — NUEVO v3.1
# =========================================================

def _analizar_curva_futura(
    curve: Dict,
    dia_actual: int,
    precio_actual: float,
) -> Optional[Dict]:
    """
    Analiza la curva de predicción desde dia_actual en adelante.
    Retorna contexto informativo para que el PM tome mejores decisiones.

    IMPORTANTE: Esta función solo INFORMA. No toma decisiones.
    El PM es quien decide qué hacer con este contexto.

    Retorna dict con:
    - slope_futura: "sube" | "baja" | "plana"
    - ret_desde_hoy_a_peak_pct: retorno esperado desde precio actual hasta el peak
    - ret_desde_hoy_a_final_pct: retorno esperado desde precio actual hasta día final
    - dias_hasta_peak: días restantes hasta el máximo de la curva
    - precio_peak_esperado: precio máximo en los días restantes
    - precio_final_esperado: precio al último día de la curva
    - dias_restantes: cuántos días quedan en la curva desde hoy
    - en_zona_valle: True si precio actual está bajo lower_band de hoy
      (señal potencial de compra en caída, informativo para el PM)
    - curva_segmento: lista de precios esperados desde dia_actual al final
    """
    if not curve or not precio_actual or precio_actual <= 0:
        return None

    path = curve.get("price_path", [])
    if not path or len(path) < 2:
        return None

    # Segmento de la curva desde dia_actual en adelante
    # dia_actual es 1-based; idx es 0-based
    idx_inicio = max(0, min(dia_actual - 1, len(path) - 1))
    segmento   = path[idx_inicio:]  # precios desde hoy hasta el final

    if not segmento:
        return None

    precio_final_esperado = float(segmento[-1])
    dias_restantes        = len(segmento)

    # Peak: máximo en el segmento futuro
    idx_peak_rel          = int(np.argmax(segmento))
    precio_peak_esperado  = float(segmento[idx_peak_rel])
    dias_hasta_peak       = idx_peak_rel  # 0 = ya estamos en el peak

    # Retornos esperados desde precio actual
    ret_a_peak  = round((precio_peak_esperado  / precio_actual - 1) * 100, 2) if precio_actual > 0 else None
    ret_a_final = round((precio_final_esperado / precio_actual - 1) * 100, 2) if precio_actual > 0 else None

    # Slope: tendencia general del segmento futuro
    # Comparamos promedio primera mitad vs segunda mitad
    mid = max(1, len(segmento) // 2)
    avg_first = float(np.mean(segmento[:mid]))
    avg_last  = float(np.mean(segmento[mid:]))
    diff_pct  = (avg_last - avg_first) / avg_first * 100 if avg_first > 0 else 0.0

    if diff_pct > 1.0:
        slope_futura = "sube"
    elif diff_pct < -1.0:
        slope_futura = "baja"
    else:
        slope_futura = "plana"

    # ¿Precio actual está en zona de valle? (bajo lower_band del día actual)
    lower_hoy = _get_band_for_day(curve, dia_actual, "lower_band")
    en_zona_valle = bool(lower_hoy and precio_actual < lower_hoy)

    return {
        "slope_futura":             slope_futura,
        "ret_desde_hoy_a_peak_pct": ret_a_peak,
        "ret_desde_hoy_a_final_pct": ret_a_final,
        "dias_hasta_peak":          dias_hasta_peak,
        "precio_peak_esperado":     round(precio_peak_esperado, 4),
        "precio_final_esperado":    round(precio_final_esperado, 4),
        "dias_restantes":           dias_restantes,
        "en_zona_valle":            en_zona_valle,
        "curva_segmento":           [round(p, 4) for p in segmento],
    }


# =========================================================
# POSICIONES ABIERTAS — desde positions_meta
# =========================================================

def _get_open_positions_with_date(tickers_filter: Optional[List[str]] = None) -> List[Dict]:
    """
    Retorna posiciones abiertas con su día actual.
    Si tickers_filter es provisto, solo retorna esos tickers.
    """
    try:
        from positions_meta import get_all
        meta  = get_all()
        today = datetime.now(timezone.utc).date()
        result = []
        for ticker, data in meta.items():
            ticker_up = ticker.upper()
            if tickers_filter and ticker_up not in [t.upper() for t in tickers_filter]:
                continue
            entry_date_str = data.get("entry_date")
            if not entry_date_str:
                continue
            try:
                entry_date = date.fromisoformat(entry_date_str)
                dia_actual = (today - entry_date).days + 1
                dia_actual = max(1, min(dia_actual, 9))
                result.append({
                    "ticker":     ticker_up,
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
    """
    Evalúa si es buen momento intraday para entrar a un ticker.
    El ticker YA fue decidido por el PM — aquí solo evaluamos timing.
    Retorna sugerencia, NO decisión final.

    v3.1: agrega curva_futura con contexto desde día 1 en adelante,
    incluyendo en_zona_valle para señales de buy-the-dip.
    """
    precio_actual = _get_current_price(ticker)
    if not precio_actual:
        return None

    curve           = _get_price_curve(ticker)
    precio_esperado = _get_expected_price_day1(ticker)

    if not precio_esperado or precio_esperado <= 0:
        return None

    tracking_ratio  = precio_actual / precio_esperado
    bars            = _get_intraday_bars(ticker, lookback_minutes=90)
    momentum        = _momentum_score(bars) if bars else 0.0
    ret_final_esp   = _get_ret_final_esperado(curve) if curve else None

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

    momentum_ok    = momentum > 0
    score_ok       = entry_score >= 0.55
    timing_ok      = precio_ok and momentum_ok and score_ok

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
    if timing_ok:
        razones.append(
            f"precio ok ({tracking_ratio:.3f}) + momentum {momentum:.2f} + score {entry_score:.2f}"
        )

    sugerencia = "ENTRAR" if timing_ok else "ESPERAR"

    logger.info(
        f"{'✅' if timing_ok else '⏳'} [{sugerencia}] {ticker} | "
        f"score={entry_score} momentum={momentum:.2f} | {' | '.join(razones)}"
    )

    result = {
        "ticker":               ticker,
        "tipo":                 "entry_candidate",
        "precio_actual":        round(precio_actual, 4),
        "precio_esperado_dia1": round(precio_esperado, 4),
        "tracking_ratio":       round(tracking_ratio, 4),
        "momentum":             round(momentum, 4),
        "entry_score":          entry_score,
        "timing_ok":            timing_ok,
        "sugerencia":           sugerencia,       # ENTRAR / ESPERAR
        "razon":                " | ".join(razones),
        "ret_final_esperado":   ret_final_esp,    # informativo para el orchestrator
        "uso_bandas":           usar_bandas,
        "fuente_precio":        "yahoo",
    }

    if usar_bandas:
        result["lower_band_dia1"] = round(lower_band_dia1, 4)
        result["upper_band_dia1"] = round(upper_band_dia1, 4)

    if curve and curve.get("analysis"):
        result["curve_analysis"] = curve["analysis"]

    # ── v3.1: contexto de curva futura desde día 1 ──────────
    # Informativo para el PM. No afecta sugerencia.
    curva_futura = _analizar_curva_futura(curve, 1, precio_actual) if curve else None
    if curva_futura:
        result["curva_futura"] = curva_futura

    return result


# =========================================================
# EVALUAR POSICIÓN ABIERTA VS CURVA
# =========================================================

def _evaluate_open_position(
    ticker: str,
    dia_actual: int,
    entry_date: str,
    bearish_news_tickers: Optional[set] = None,
) -> Optional[Dict]:
    """
    Evalúa si una posición abierta debe cerrarse ahora o mantenerse.
    Considera PnL actual: no sugiere cerrar posiciones ganadoras
    solo por divergencia de curva (evita cortar ganancias como AMAT +18%).
    Retorna sugerencia, NO decisión final.

    v3.1: agrega curva_futura con contexto desde dia_actual en adelante.
    El PM puede usar slope_futura, en_zona_valle y ret_desde_hoy_a_peak_pct
    para decidir si un cierre divergente tiene sentido estratégico
    (ej: diverging + PnL positivo + slope_futura=baja → PM puede cerrar
    tomando ganancia antes de caída esperada).

    [N2] Si `bearish_news_tickers` es None, se carga fresco desde
    news_ranking.json (llamada standalone). Si el ticker aparece ahí,
    se usa un umbral de "diverging" más ajustado (reacciona antes)
    solo para esta evaluación.
    """
    if bearish_news_tickers is None:
        bearish_news_tickers = _load_bearish_news_tickers()

    has_bearish_news = ticker.upper() in bearish_news_tickers
    std_mult_used  = NEWS_TIGHT_STD_MULT  if has_bearish_news else DIVERGING_STD_MULT
    threshold_used = NEWS_TIGHT_THRESHOLD if has_bearish_news else DIVERGING_THRESHOLD

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

    # ─── Estado vs curva ──────────────────────────────────
    if upper and lower and lower > 0:
        usar_bandas         = True
        std_band            = (upper - lower) / 3.0
        diverging_threshold = lower - (std_mult_used * std_band)

        if precio_actual > upper:
            curve_status = "ahead"
        elif precio_actual >= lower:
            curve_status = "on_track"
        elif precio_actual >= diverging_threshold:
            curve_status = "lagging"
        else:
            curve_status = "diverging"
    else:
        usar_bandas = False
        if tracking_ratio >= AHEAD_THRESHOLD:
            curve_status = "ahead"
        elif tracking_ratio >= LAGGING_THRESHOLD:
            curve_status = "on_track"
        elif tracking_ratio >= threshold_used:
            curve_status = "lagging"
        else:
            curve_status = "diverging"

    # ─── PnL actual vs precio de entrada REAL ─────────────
    # Usa entry_price de positions_meta (precio real pagado).
    # Fallback a price_now de la curva si no está disponible.
    ret_vs_entrada     = None
    ret_esperado_total = None

    entry_price_real = 0.0
    try:
        from positions_meta import get_all
        meta             = get_all()
        entry_price_real = float(meta.get(ticker, {}).get("entry_price", 0) or 0)
    except Exception:
        pass

    if curve:
        price_now_orig = entry_price_real if entry_price_real > 0 else float(curve.get("price_now", 0))
        path           = curve.get("price_path", [])
        if price_now_orig > 0:
            ret_vs_entrada = round((precio_actual / price_now_orig - 1) * 100, 2)
            if path:
                ret_esperado_total = round((path[-1] / price_now_orig - 1) * 100, 2)

    # ─── Lógica de sugerencia — considera PnL actual ──────
    #
    # PROBLEMA ANTERIOR (v2.x):
    #   diverging → CERRAR siempre, aunque PnL sea +18%
    #
    # LÓGICA v3.0/v3.1:
    #   diverging + PnL negativo          → CERRAR (proteger capital)
    #   diverging + PnL < TRAILING_PNL    → CERRAR (ganancia pequeña, riesgo reversión)
    #   diverging + PnL >= TRAILING_PNL   → TRAILING (proteger ganancia, no cortar)
    #   lagging                           → MANTENER (esperar recuperación)
    #   on_track / ahead                  → MANTENER
    #
    # NOTA v3.1: la curva_futura (slope, peak, valle) se agrega al resultado
    # como contexto informativo para que el PM refine la decisión.
    # El tracker NO cambia su sugerencia basándose en curva_futura.
    # ──────────────────────────────────────────────────────

    pnl = ret_vs_entrada if ret_vs_entrada is not None else 0.0

    if curve_status == "diverging":
        if pnl >= TRAILING_PNL_THRESHOLD * 100:
            sugerencia = "TRAILING"
            razon_cierre = (
                f"diverging pero PnL={pnl:.1f}% >= {TRAILING_PNL_THRESHOLD*100:.0f}% → "
                f"trailing stop, no cerrar directo"
            )
        elif pnl > 0:
            sugerencia   = "TRAILING"
            razon_cierre = f"diverging + PnL={pnl:.1f}% positivo → trailing, no cerrar"

        elif dia_actual <= MIN_HOLD_DAYS_BEFORE_CURVE_EXIT and pnl > -SAME_DAY_STOP_LOSS_OVERRIDE_PCT:
            # [AUD-P9] Piso de días mínimos: no cerrar por curva el mismo
            # día de apertura, salvo que la pérdida ya sea severa
            # (>= SAME_DAY_STOP_LOSS_OVERRIDE_PCT).
            sugerencia   = "MANTENER"
            razon_cierre = (
                f"diverging + PnL={pnl:.1f}% negativo, pero día {dia_actual} "
                f"<= {MIN_HOLD_DAYS_BEFORE_CURVE_EXIT} (piso mínimo) → "
                f"esperar antes de cerrar por curva"
            )
        else:
            sugerencia   = "CERRAR"
            razon_cierre = (
                f"diverging + PnL={pnl:.1f}% negativo → cerrar para limitar pérdida"
            )
    elif curve_status == "lagging":
        sugerencia   = "MANTENER"
        razon_cierre = f"lagging pero dentro del cono → esperar recuperación"
    else:
        sugerencia   = "MANTENER"
        razon_cierre = f"curve_status={curve_status} → sin acción"

    logger.info(
        f"📊 {ticker} día {dia_actual} | "
        f"actual={precio_actual:.2f} esperado={precio_esperado:.2f} "
        f"ratio={tracking_ratio:.3f} PnL={pnl:.1f}% → {curve_status} → {sugerencia}"
        + (f" [bandas]" if usar_bandas else " [fallback]")
        + f" [yahoo]"
        + (" [⚠️ noticia bajista: stop ajustado]" if has_bearish_news else "")
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
        "pnl_actual_pct":         round(pnl, 2),
        "ret_esperado_total_pct": ret_esperado_total,
        "sugerencia":             sugerencia,       # MANTENER / CERRAR / TRAILING
        "razon":                  razon_cierre,
        "uso_bandas":             usar_bandas,
        "fuente_precio":          "yahoo",
        "news_bearish_alert":     has_bearish_news,   # [N2] trazabilidad
    }

    if usar_bandas:
        result["upper_band_hoy"]      = round(upper, 4)
        result["lower_band_hoy"]      = round(lower, 4)
        result["diverging_threshold"] = round(diverging_threshold, 4)

    if curve and curve.get("analysis"):
        best_exit = curve["analysis"].get("best_exit_day")
        if best_exit:
            result["best_exit_day"]    = best_exit
            result["dias_para_salida"] = max(0, best_exit - dia_actual)

    # ── v3.1: contexto de curva futura desde dia_actual ─────
    # Informativo para el PM. NO modifica sugerencia del tracker.
    # Campos clave para el PM:
    #   slope_futura="baja" + sugerencia="CERRAR" + pnl>0
    #     → PM puede confirmar cierre tomando ganancia antes de caída
    #   slope_futura="sube" + sugerencia="CERRAR" + pnl>0
    #     → PM puede reconsiderar y mantener (modelo erró hoy, destino alcista)
    #   en_zona_valle=True
    #     → PM podría considerar no cerrar o incluso promediar a la baja
    curva_futura = _analizar_curva_futura(curve, dia_actual, precio_actual) if curve else None
    if curva_futura:
        result["curva_futura"] = curva_futura

    return result


# =========================================================
# API PRINCIPAL — llamada desde el orchestrator
# =========================================================

def evaluar_timing_entrada(tickers: List[str]) -> Dict[str, Dict]:
    """
    Recibe lista de tickers que el PM ya decidió comprar.
    Evalúa timing intraday para cada uno.
    Retorna dict {ticker: señal} con sugerencia ENTRAR/ESPERAR.
    La decisión final sigue siendo del orchestrator/CG.
    """
    if not tickers:
        return {}

    ahora     = datetime.now(timezone.utc)
    fecha_str = ahora.date().isoformat()
    hora_str  = ahora.strftime("%H:%M")

    logger.info(f"📡 Tracker entrada v3.1 | {fecha_str} {hora_str} UTC | tickers={tickers}")

    resultado: Dict[str, Dict] = {}

    for ticker in tickers:
        try:
            señal = _evaluate_entry_timing(ticker.upper())
            if señal:
                resultado[ticker.upper()] = {**señal, "evaluado_hora": hora_str}
        except Exception as e:
            logger.error(f"❌ Error entry timing {ticker}: {e}")

    # Guardar snapshot
    _guardar_snapshot(fecha_str, hora_str, "entradas", resultado)

    entrar  = [t for t, s in resultado.items() if s.get("sugerencia") == "ENTRAR"]
    esperar = [t for t, s in resultado.items() if s.get("sugerencia") == "ESPERAR"]
    logger.info(f"📡 Timing entrada | ENTRAR={entrar} | ESPERAR={esperar}")

    return resultado


def evaluar_posiciones_abiertas(tickers: List[str]) -> Dict[str, Dict]:
    """
    Recibe lista de tickers con posiciones abiertas (del PM/portfolio).
    Evalúa si el momento intraday sugiere cerrar, mantener o trailing.
    Retorna dict {ticker: señal} con sugerencia MANTENER/CERRAR/TRAILING.
    La decisión final sigue siendo del orchestrator.
    """
    if not tickers:
        return {}

    ahora     = datetime.now(timezone.utc)
    fecha_str = ahora.date().isoformat()
    hora_str  = ahora.strftime("%H:%M")

    logger.info(f"📊 Tracker posiciones v3.1 | {fecha_str} {hora_str} UTC | tickers={tickers}")

    posiciones = _get_open_positions_with_date(tickers_filter=tickers)
    resultado: Dict[str, Dict] = {}

    # [N2] Cargar UNA vez para todas las posiciones de este ciclo,
    # no una vez por ticker.
    bearish_news_tickers = _load_bearish_news_tickers()
    if bearish_news_tickers:
        logger.info(f"⚠️ Tickers con noticia bajista hoy: {sorted(bearish_news_tickers)}")

    for pos in posiciones:
        try:
            señal = _evaluate_open_position(
                pos["ticker"], pos["dia_actual"], pos["entry_date"],
                bearish_news_tickers=bearish_news_tickers,
            )
            if señal:
                resultado[pos["ticker"]] = {**señal, "evaluado_hora": hora_str}
        except Exception as e:
            logger.error(f"❌ Error monitor {pos['ticker']}: {e}")

    # Guardar snapshot
    _guardar_snapshot(fecha_str, hora_str, "posiciones", resultado)

    cerrar   = [t for t, s in resultado.items() if s.get("sugerencia") == "CERRAR"]
    trailing = [t for t, s in resultado.items() if s.get("sugerencia") == "TRAILING"]
    mantener = [t for t, s in resultado.items() if s.get("sugerencia") == "MANTENER"]
    logger.info(
        f"📊 Monitor posiciones | "
        f"CERRAR={cerrar} | TRAILING={trailing} | MANTENER={mantener}"
    )

    return resultado


# =========================================================
# SNAPSHOT — persiste evaluaciones del día
# =========================================================

def _guardar_snapshot(fecha_str: str, hora_str: str, tipo: str, data: Dict) -> None:
    snapshot_path = INTRADAY_DIR / f"{fecha_str}.json"
    snapshot      = _load_json(snapshot_path)
    if hora_str not in snapshot:
        snapshot[hora_str] = {}
    snapshot[hora_str][tipo]        = data
    snapshot[hora_str]["timestamp"] = datetime.now(timezone.utc).isoformat()
    _save_json(snapshot_path, snapshot)
    logger.info(
        f"💾 Snapshot | {tipo}={len(data)} tickers | {fecha_str} {hora_str}"
    )


# =========================================================
# API LEGACY — compatibilidad con monitor-close router
# =========================================================

def get_entry_signals_today() -> Dict[str, Dict]:
    """
    Compatibilidad legacy. Retorna las últimas señales de entrada del día.
    NOTA: en v3.x el orchestrator llama evaluar_timing_entrada() directamente.
    """
    fecha_str     = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = INTRADAY_DIR / f"{fecha_str}.json"
    snapshot      = _load_json(snapshot_path)
    if not snapshot:
        return {}

    best_by_ticker: Dict[str, Dict] = {}
    for hora, data in snapshot.items():
        entradas = data.get("entradas", {})
        for ticker, señal in entradas.items():
            score = señal.get("entry_score", 0)
            if ticker not in best_by_ticker or score > best_by_ticker[ticker]["entry_score"]:
                best_by_ticker[ticker] = {**señal, "mejor_hora": hora}
    return best_by_ticker


def get_position_status_today() -> Dict[str, Dict]:
    """
    Compatibilidad legacy. Retorna el último estado de posiciones del día.
    NOTA: en v3.x el orchestrator llama evaluar_posiciones_abiertas() directamente.
    """
    fecha_str     = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = INTRADAY_DIR / f"{fecha_str}.json"
    snapshot      = _load_json(snapshot_path)
    if not snapshot:
        return {}

    latest: Dict[str, Dict] = {}
    for hora in sorted(snapshot.keys()):
        posiciones = snapshot[hora].get("posiciones", {})
        for ticker, señal in posiciones.items():
            latest[ticker] = {**señal, "ultima_hora": hora}
    return latest

# =========================================================
# intraday_tracker.py — INTRADAY ENTRY OPTIMIZER v1.0
# =========================================================
# Corre cada 30 minutos durante horario de mercado.
# Propósito: mejorar el timing de COMPRA comparando
# precio actual vs price_curve esperada del pipeline.
#
# Guarda señales en /data/intraday/{fecha}.json
# El TradingOrchestrator las lee en el run de las 16:30.
#
# Lógica de entrada:
#   - precio_actual <= precio_curva_dia1 * 0.995  → debajo de lo esperado (buena entrada)
#   - momentum_30m > 0                             → precio subiendo
#   - ambas condiciones → entry_score alto
# =========================================================

import os
import json
import logging
from datetime import datetime, timedelta, timezone
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

# Umbral de entrada — precio actual debe estar <= X% sobre curva esperada
ENTRY_THRESHOLD  = float(os.getenv("INTRADAY_ENTRY_THRESHOLD", "1.005"))
# Mínimo alpha score para considerar una señal COMPRA
MIN_ALPHA_COMPRA = float(os.getenv("INTRADAY_MIN_ALPHA", "0.65"))

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
    """
    Obtiene barras de 30 minutos de las últimas lookback_minutes.
    Retorna lista de closes [más antiguo → más reciente].
    """
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
    """
    Score de momentum entre -1 y 1.
    Compara última barra vs promedio de las anteriores.
    """
    if not closes or len(closes) < 3:
        return 0.0
    last    = closes[-1]
    prev    = float(np.mean(closes[-4:-1]))  # promedio últimas 3 barras
    if prev <= 0:
        return 0.0
    change  = (last - prev) / prev
    return float(np.clip(change * 100, -1.0, 1.0))  # normalizado -1 a 1


# =========================================================
# CURVA ESPERADA
# =========================================================

def _get_expected_price_day1(ticker: str) -> Optional[float]:
    """
    Lee la predicción más reciente y retorna el precio esperado
    para el día 1 (price_curve.price_path[0]).
    """
    ticker_dir = PRED_DIR / ticker
    if not ticker_dir.exists():
        return None

    candidates = sorted(ticker_dir.glob("*.json"))
    if not candidates:
        return None

    pred = _load_json(candidates[-1])
    curve = pred.get("price_curve", {}).get("price_path", [])
    if not curve:
        return None

    return float(curve[0])


# =========================================================
# SEÑALES COMPRA PENDIENTES
# =========================================================

def _get_compra_candidates() -> List[str]:
    """
    Lee alpha_last.json y retorna tickers con recomendación COMPRA
    que superen el mínimo de alpha score.
    """
    alpha_data = _load_json(ALPHA_FILE)
    results    = alpha_data.get("results", {})
    candidates = []

    for ticker, data in results.items():
        if not isinstance(data, data.__class__) or not isinstance(data, dict):
            continue
        score = float(data.get("alpha_score", 0))
        if score < MIN_ALPHA_COMPRA:
            continue

        # Leer recomendación desde predicción más reciente
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
# EVALUAR TIMING PARA UN TICKER
# =========================================================

def _evaluate_entry_timing(ticker: str) -> Optional[Dict]:
    """
    Evalúa si el momento actual es bueno para entrar en un ticker.

    Retorna dict con:
      - ticker
      - precio_actual
      - precio_esperado_dia1
      - tracking_ratio: actual / esperado (< 1 = barato vs predicción)
      - momentum: -1 a 1
      - entry_score: 0 a 1
      - entrar_ahora: bool
      - razon: string explicativo
    """
    precio_actual   = _get_current_price(ticker)
    precio_esperado = _get_expected_price_day1(ticker)

    if not precio_actual or not precio_esperado:
        return None

    if precio_esperado <= 0:
        return None

    # Ratio: < 1 significa que está más barato que lo predicho → buena entrada
    tracking_ratio = precio_actual / precio_esperado

    # Momentum intradía
    bars     = _get_intraday_bars(ticker, lookback_minutes=90)
    momentum = _momentum_score(bars) if bars else 0.0

    # ── SCORING DE ENTRADA ─────────────────────────────────────
    # Componente precio: mejor cuanto más barato vs curva
    # tracking_ratio=0.99 → precio_score=0.8 (muy bueno)
    # tracking_ratio=1.00 → precio_score=0.5 (neutro)
    # tracking_ratio=1.01 → precio_score=0.2 (caro, esperar)
    precio_score = float(np.clip(0.5 + (1.0 - tracking_ratio) * 50, 0.0, 1.0))

    # Componente momentum: normalizado 0-1
    momentum_score = float(np.clip((momentum + 1.0) / 2.0, 0.0, 1.0))

    # Entry score combinado: precio pesa más (60%) que momentum (40%)
    entry_score = round(0.60 * precio_score + 0.40 * momentum_score, 3)

    # ── DECISIÓN ───────────────────────────────────────────────
    # Entrar si:
    #   1. precio <= esperado * umbral (no está caro)
    #   2. momentum positivo (está subiendo)
    #   3. entry_score >= 0.55
    precio_ok   = tracking_ratio <= ENTRY_THRESHOLD
    momentum_ok = momentum > 0
    score_ok    = entry_score >= 0.55

    entrar_ahora = precio_ok and momentum_ok and score_ok

    # Razón legible
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
        "precio_actual":        round(precio_actual, 4),
        "precio_esperado_dia1": round(precio_esperado, 4),
        "tracking_ratio":       round(tracking_ratio, 4),
        "momentum":             round(momentum, 4),
        "entry_score":          entry_score,
        "entrar_ahora":         entrar_ahora,
        "razon":                " | ".join(razones),
    }


# =========================================================
# RUN PRINCIPAL — llamado desde scheduler cada 30 min
# =========================================================

def run_intraday_tracker() -> Dict:
    """
    Evalúa todos los candidatos COMPRA y guarda señales.
    Retorna resumen de la ejecución.
    """
    ahora      = datetime.now(timezone.utc)
    fecha_str  = ahora.date().isoformat()
    hora_str   = ahora.strftime("%H:%M")

    logger.info(f"📡 Intraday tracker | {fecha_str} {hora_str} UTC")

    candidatos = _get_compra_candidates()
    if not candidatos:
        logger.info("ℹ️  Sin candidatos COMPRA activos")
        return {"hora": hora_str, "candidatos": 0, "senales": []}

    logger.info(f"🔍 Evaluando {len(candidatos)} candidatos: {candidatos}")

    senales    = []
    entrar_now = []

    for ticker in candidatos:
        try:
            resultado = _evaluate_entry_timing(ticker)
            if resultado:
                senales.append(resultado)
                if resultado["entrar_ahora"]:
                    entrar_now.append(ticker)
                    logger.info(
                        f"✅ ENTRADA OK {ticker} | "
                        f"ratio={resultado['tracking_ratio']} "
                        f"momentum={resultado['momentum']} "
                        f"score={resultado['entry_score']}"
                    )
                else:
                    logger.info(
                        f"⏳ ESPERAR {ticker} | {resultado['razon']}"
                    )
        except Exception as e:
            logger.error(f"❌ Error evaluando {ticker}: {e}")

    # Cargar snapshot del día y agregar esta hora
    snapshot_path = INTRADAY_DIR / f"{fecha_str}.json"
    snapshot      = _load_json(snapshot_path)

    snapshot[hora_str] = {
        "timestamp": ahora.isoformat(),
        "senales":   senales,
        "entrar_ahora": entrar_now,
    }

    _save_json(snapshot_path, snapshot)

    logger.info(
        f"💾 Snapshot guardado | candidatos={len(senales)} entrar={len(entrar_now)}"
    )

    return {
        "hora":         hora_str,
        "candidatos":   len(senales),
        "entrar_ahora": entrar_now,
        "senales":      senales,
    }


# =========================================================
# LEER SEÑAL PARA TRADING ORCHESTRATOR
# =========================================================

def get_entry_signals_today() -> Dict[str, Dict]:
    """
    Llamado por TradingOrchestrator en el run de las 16:30.
    Retorna el mejor momento del día para cada ticker.

    Formato: { "AAPL": { "entry_score": 0.82, "entrar": True, ... } }
    """
    fecha_str     = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = INTRADAY_DIR / f"{fecha_str}.json"
    snapshot      = _load_json(snapshot_path)

    if not snapshot:
        return {}

    # Por ticker: tomar el snapshot con mayor entry_score del día
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

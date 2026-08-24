"""
darwin_engine/executor_shadow_evaluator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHIVO NUEVO [AUD-D2] — resuelve el bug estructural #2 de la auditoría:
"los executor shadows nunca pueden competir — no hay pipeline de trades
reales para ellos".

PROBLEMA QUE RESUELVE:
  Solo el executor campeón opera con dinero real en Alpaca. Los genomas
  shadow (v2, v3, ...) nunca acumulan trades propios, por lo que
  `cand_trades` en arena.py._select_champion() permanece en 0
  indefinidamente y nunca alcanzan MIN_TRADES_TO_COMPETE — el campeón
  nunca puede ser reemplazado, sin importar cuán bueno sea un candidato.

DISEÑO (confirmado con el usuario, punto por punto):
  - Corre en el MISMO ciclo que trading_orchestrator.py — sin desfase
    temporal entre shadows y campeón, para que ninguno tenga ventaja
    por operar en un momento distinto del mercado.
  - Cada shadow simula su PROPIO portfolio, independiente del campeón
    y de los demás shadows — mide las consecuencias de las decisiones
    de ESE genoma, no las del campeón.
  - Capital ficticio: el mismo FIXED_CAPITAL que usa el sistema real.
  - Universo de apertura: alpha_map completo del día (mismos tickers
    con alpha_score que ve el campeón) — ningún shadow tiene un
    universo más favorable que otro.
  - Apertura: cada shadow decide con su PROPIO criterio, gobernado
    por los genes entry_confirmation del genoma (min_agreeing_h,
    min_horizon_for_entry) — genes que YA EXISTEN en ExecutorGenome
    desde su creación pero que ningún código usaba hasta este fix.
    Esto es intencional: la apertura debe derivarse de genes reales
    que Darwin puede mutar, no de un umbral inventado aparte para
    la simulación — de lo contrario la competencia no sería justa
    ni auditable.
  - Mantención/cierre: usa ExecutorGenome.evaluate(), la misma
    función que ya usa el sistema real para el campeón — ningún
    criterio distinto para shadows vs campeón en esta parte.
  - Precios: se leen desde el mismo alpha_map / price_now del ciclo
    (no requiere Alpaca ni ninguna cuenta adicional — es lectura de
    datos ya disponibles en el ciclo, nunca ejecución de órdenes).

PERSISTENCIA:
  /data/darwin/shadow_portfolios/{genome_id}/positions.json
      Portfolio simulado actual del shadow (lista de posiciones).
  /data/darwin/shadow_trades/{genome_id}/*.json
      Un archivo por trade simulado cerrado — mismo esquema de
      campos relevantes que darwin/trades/*.json para que
      get_resolved_trades() (o un equivalente) pueda calcular
      fitness de forma comparable.

AUDITABILIDAD:
  Cada decisión (abrir, mantener, cerrar) queda registrada con la
  razón exacta y los valores de los genes que la motivaron, para que
  cualquier promoción futura de un shadow a campeón sea trazable
  hasta las reglas específicas que la justificaron.
"""

import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, Any, List, Optional

from darwin_engine.executor_genome import ExecutorGenome, GENOME_DIR
from darwin_engine.trade_tracker import _read_h_signals

logger = logging.getLogger("executor_shadow_evaluator")

DATA_PATH            = Path(os.getenv("DATA_PATH", "/data"))
SHADOW_PORTFOLIO_DIR = DATA_PATH / "darwin" / "shadow_portfolios"
SHADOW_TRADES_DIR    = DATA_PATH / "darwin" / "shadow_trades"

FIXED_CAPITAL = float(os.getenv("FIXED_CAPITAL", "1000000"))

# Tamaño de posición simulada — mismo default que usa el sistema real
# para aperturas por alpha (ver trading_orchestrator.py ALPHA_INJECT)
DEFAULT_TARGET_PCT = 0.05


# ══════════════════════════════════════════════════════
# HELPERS DE DISCO
# ══════════════════════════════════════════════════════

def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)


def _positions_path(genome_id: str) -> Path:
    return SHADOW_PORTFOLIO_DIR / genome_id / "positions.json"


def _load_shadow_positions(genome_id: str) -> List[Dict]:
    data = _load_json(_positions_path(genome_id), default=[])
    return data if isinstance(data, list) else []


def _save_shadow_positions(genome_id: str, positions: List[Dict]) -> None:
    _save_json(_positions_path(genome_id), positions)


def _save_shadow_trade(genome_id: str, trade: Dict) -> None:
    trade_dir = SHADOW_TRADES_DIR / genome_id
    path = trade_dir / f"{trade['trade_id']}.json"
    _save_json(path, trade)


# ══════════════════════════════════════════════════════
# [AUD-D2] CRITERIO DE APERTURA GOBERNADO POR GENES
# ══════════════════════════════════════════════════════

def should_open(
    genome: ExecutorGenome,
    ticker: str,
    h_signals: Dict[str, Any],
    alpha_score: float,
) -> Optional[Dict[str, Any]]:
    """
    [AUD-D2] Decide si el genoma abriría una posición en `ticker`,
    usando ÚNICAMENTE los genes entry_confirmation que ya existen en
    ExecutorGenome (min_agreeing_h, min_horizon_for_entry) — genes
    presentes desde la creación del genoma pero que hasta este fix
    ningún código consultaba.

    No usa ningún umbral inventado para la simulación: la regla de
    apertura del shadow es la misma que sus genes describen, por lo
    tanto es tan auditable y mutable por Darwin como el resto del
    comportamiento del genoma.

    Retorna un dict con la decisión y el detalle de la evidencia que
    la sustenta, o None si el genoma no abriría.
    """
    entry_cfg = genome.data.get("entry_confirmation", {})
    min_agreeing_h        = int(entry_cfg.get("min_agreeing_h", 3))
    min_horizon_for_entry = int(entry_cfg.get("min_horizon_for_entry", 4))

    direction = "COMPRA" if alpha_score > 0 else "VENDE"

    agreeing = 0
    evidence = []
    for h_key, sig in h_signals.items():
        if h_key == "main" or not h_key.startswith("H"):
            continue
        try:
            h_num = int(h_key[1:])
        except ValueError:
            continue
        if h_num < min_horizon_for_entry:
            continue
        pred_ret = float(sig.get("pred_return", 0))
        confirms = (pred_ret > 0 if direction == "COMPRA" else pred_ret < 0)
        if confirms:
            agreeing += 1
            evidence.append({"h": h_key, "pred_return": pred_ret})

    if agreeing < min_agreeing_h:
        return None

    return {
        "ticker":      ticker,
        "direction":   direction,
        "agreeing_h":  agreeing,
        "min_required": min_agreeing_h,
        "evidence":    evidence,
        "alpha_score": alpha_score,
    }


# ══════════════════════════════════════════════════════
# CICLO DE UN SHADOW — evalúa cierres y aperturas
# ══════════════════════════════════════════════════════

def evaluate_shadow_cycle(
    genome: ExecutorGenome,
    alpha_map: Dict[str, Dict],
    price_map: Dict[str, float],
    entry_date_str: str = None,
) -> Dict[str, Any]:
    """
    Ejecuta un ciclo completo de evaluación para UN shadow:
      1. Evalúa cada posición abierta del portfolio propio del shadow
         con genome.evaluate() → HOLD o CLOSE.
      2. Para cierres, registra el trade simulado en
         darwin/shadow_trades/{genome_id}/ y lo quita del portfolio.
      3. Para tickers sin posición, evalúa apertura con should_open()
         usando el mismo alpha_map que ve el campeón ese día.
      4. Persiste el portfolio actualizado.

    No ejecuta ninguna orden real — todo el estado vive en
    darwin/shadow_portfolios/ y darwin/shadow_trades/.

    price_map: {ticker: precio_actual} — mismos precios que ya calculó
    el ciclo real (no se hace ninguna llamada adicional a Alpaca ni a
    ningún proveedor de datos aquí).
    """
    genome_id = genome.genome_id
    today = entry_date_str or datetime.now(timezone.utc).date().isoformat()
    positions = _load_shadow_positions(genome_id)

    closed_trades = []
    still_open    = []

    # ── PASO 1: evaluar posiciones abiertas (HOLD/CLOSE) ──────────
    for pos in positions:
        ticker = pos["ticker"]
        price_now = price_map.get(ticker.upper())
        if price_now is None or price_now <= 0:
            # Sin precio confiable este ciclo → mantener sin evaluar,
            # igual criterio conservador que usa el sistema real
            # ([F13] en trading_orchestrator.py): no inventar precio.
            still_open.append(pos)
            continue

        entry_price = float(pos["entry_price"])
        pnl_pct     = (price_now / entry_price - 1.0) * 100
        days_held   = (
            date.fromisoformat(today) - date.fromisoformat(pos["entry_date"])
        ).days

        h_signals   = _read_h_signals(ticker, date.fromisoformat(today))
        alpha_entry = alpha_map.get(ticker.upper(), {})
        alpha_score = float(alpha_entry.get("alpha_score", 0.0))
        h_hit_rates = pos.get("h_hit_rates_at_entry", {})

        decision = genome.evaluate(
            trade            = pos,
            current_pnl_pct  = pnl_pct,
            alpha_score      = alpha_score,
            h_signals        = h_signals,
            h_hit_rates      = h_hit_rates,
            days_held        = days_held,
        )

        if decision["action"] == "CLOSE":
            trade = deepcopy(pos)
            trade.update({
                "status":         "closed",
                "exit_price":     round(price_now, 4),
                "exit_date":      today,
                "reason_close":   decision["reason"],
                "days_held":      days_held,
                "pnl_real_pct":   round(pnl_pct, 4),
                "pnl_real_usd":   round(pnl_pct / 100 * entry_price * pos["shares"], 2),
                "decision_confidence": decision.get("confidence"),
            })
            _save_shadow_trade(genome_id, trade)
            closed_trades.append(trade)
        else:
            pos["last_evaluated"] = today
            pos["last_decision"]  = decision
            still_open.append(pos)

    # ── PASO 2: evaluar aperturas nuevas ──────────────────────────
    open_tickers = {p["ticker"].upper() for p in still_open}
    new_positions = []

    for ticker, alpha_entry in alpha_map.items():
        ticker_upper = ticker.upper()
        if ticker_upper in open_tickers:
            continue

        alpha_score = float(alpha_entry.get("alpha_score", 0.0))
        price_now   = price_map.get(ticker_upper)
        if not price_now or price_now <= 0:
            continue

        h_signals = _read_h_signals(ticker_upper, date.fromisoformat(today))
        if not h_signals:
            continue

        open_signal = should_open(genome, ticker_upper, h_signals, alpha_score)
        if open_signal is None:
            continue

        shares = int((FIXED_CAPITAL * DEFAULT_TARGET_PCT) // price_now)
        if shares <= 0:
            continue

        new_trade_id = f"{ticker_upper}_{today}_{genome_id}"
        new_pos = {
            "trade_id":              new_trade_id,
            "ticker":                ticker_upper,
            "entry_price":           round(price_now, 4),
            "entry_date":            today,
            "shares":                shares,
            "h_signals_at_entry":    {"main": {"recommendation": open_signal["direction"]}},
            "open_evidence":         open_signal,
            "alpha_score_entry":     round(alpha_score, 4),
        }
        new_positions.append(new_pos)

    still_open.extend(new_positions)
    _save_shadow_positions(genome_id, still_open)

    logger.info(
        f"🌑 Shadow eval | {genome_id} | "
        f"cerrados={len(closed_trades)} abiertos_nuevos={len(new_positions)} "
        f"portfolio_actual={len(still_open)}"
    )

    return {
        "genome_id":       genome_id,
        "closed":          len(closed_trades),
        "opened":          len(new_positions),
        "portfolio_size":  len(still_open),
    }


# ══════════════════════════════════════════════════════
# API PÚBLICA: correr todos los shadows en el ciclo actual
# ══════════════════════════════════════════════════════

def run_all_shadows(
    shadow_genomes: List[ExecutorGenome],
    alpha_map: Dict[str, Dict],
    price_map: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Ejecuta evaluate_shadow_cycle() para cada shadow. Debe llamarse en
    el MISMO ciclo que trading_orchestrator.py procesa al campeón real,
    con el mismo alpha_map y price_map de ese ciclo — nunca en un cron
    separado ni con datos de un momento distinto, para que ningún
    shadow tenga ventaja o desventaja temporal frente a otro o frente
    al campeón.
    """
    results = []
    for genome in shadow_genomes:
        try:
            result = evaluate_shadow_cycle(genome, alpha_map, price_map)
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Error evaluando shadow {genome.genome_id}: {e}")
            results.append({"genome_id": genome.genome_id, "error": str(e)})
    return results


# ══════════════════════════════════════════════════════
# LECTURA DE TRADES SIMULADOS RESUELTOS DE UN SHADOW
# (equivalente a get_resolved_trades() de trade_tracker.py,
#  pero apuntando a shadow_trades/ en vez de darwin/trades/)
# ══════════════════════════════════════════════════════

def get_shadow_resolved_trades(genome_id: str, last_n: int = 200) -> List[Dict]:
    trade_dir = SHADOW_TRADES_DIR / genome_id
    if not trade_dir.exists():
        return []

    trades = []
    for path in trade_dir.glob("*.json"):
        trade = _load_json(path, default=None)
        if trade and trade.get("status") == "closed":
            trades.append(trade)

    trades.sort(key=lambda t: t.get("exit_date", ""), reverse=True)
    return trades[:last_n]

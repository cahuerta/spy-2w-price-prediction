"""
darwin_engine/pnl_fitness.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHIVO NUEVO — darwin_engine/pnl_fitness.py

Calcula el fitness real de cada genome combinando:
  1. trade_tracker   → atribución de trades a genomas
  2. Alpaca orders   → PnL real ejecutado
  3. evaluator       → si el H acertó en su horizonte teórico

Fitness final = Sharpe ponderado - penalización por oportunidad perdida

Usado por arena.py para decidir qué genome promueve a producción.
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("pnl_fitness")

DATA_PATH   = Path(os.getenv("DATA_PATH", "/data"))
TRACKER_DIR = DATA_PATH / "darwin" / "trades"
EVAL_DIR    = DATA_PATH / "evaluations"
FITNESS_DIR = DATA_PATH / "darwin" / "fitness"

# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def _load_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)


def _sharpe(returns: List[float], annualize: int = 252) -> float:
    """Sharpe ratio anualizado. Retorna 0 si no hay suficientes datos."""
    if len(returns) < 5:
        return 0.0
    arr = np.array(returns, dtype=float)
    std = arr.std()
    if std < 1e-9:
        return 0.0
    return float(arr.mean() / std * np.sqrt(annualize))


def _max_drawdown(returns: List[float]) -> float:
    """Max drawdown desde retornos porcentuales."""
    if not returns:
        return 0.0
    equity = np.cumprod(1 + np.array(returns) / 100)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    return float(abs(dd.min()))


# ══════════════════════════════════════════════════════
# FUENTE 1: TRADES DEL TRACKER
# ══════════════════════════════════════════════════════

def _load_resolved_trades(genome_id: str, genome_type: str) -> List[Dict]:
    """
    Carga trades resueltos atribuidos a un genome específico.
    genome_type: "executor" | "predictor"
    """
    if not TRACKER_DIR.exists():
        return []

    field = f"{genome_type}_genome_id"
    trades = []

    for path in TRACKER_DIR.glob("*.json"):
        t = _load_json(path)
        if t.get("status") == "resolved" and t.get(field) == genome_id:
            trades.append(t)

    trades.sort(key=lambda x: x.get("resolved_at", ""), reverse=True)
    return trades


# ══════════════════════════════════════════════════════
# FUENTE 2: ALPACA REAL (sin broker async — lee desde tracker)
# ══════════════════════════════════════════════════════

def _enrich_with_alpaca_pnl(trades: List[Dict]) -> List[Dict]:
    """
    Si el trade tiene pnl_real_pct calculado por el tracker, lo usa.
    Si no, intenta calcularlo desde entry/exit price disponibles.
    """
    enriched = []
    for t in trades:
        pnl = t.get("pnl_real_pct")
        if pnl is None:
            ep = float(t.get("entry_price") or 0)
            xp = float(t.get("exit_price") or 0)
            if ep > 0 and xp > 0:
                rec = t.get("h_signals_at_entry", {}).get("main", {}).get("recommendation", "COMPRA")
                pnl = (xp - ep) / ep * 100 if rec != "VENDE" else (ep - xp) / ep * 100
                t = dict(t)
                t["pnl_real_pct"] = round(pnl, 4)
        if pnl is not None:
            enriched.append(t)

    return enriched


# ══════════════════════════════════════════════════════
# FUENTE 3: EVALUACIONES H (acierto en horizonte teórico)
# ══════════════════════════════════════════════════════

def _get_h_hit_rates() -> Dict[str, float]:
    """
    Lee las evaluaciones existentes y calcula hit rate por H.
    Retorna {"H1": 0.41, "H2": 0.47, ..., "H10": 0.44}
    """
    hit_counts  = defaultdict(int)
    eval_counts = defaultdict(int)

    if not EVAL_DIR.exists():
        return {}

    for ticker_dir in EVAL_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue
        for eval_file in ticker_dir.glob("*.json"):
            ev = _load_json(eval_file)
            models = ev.get("models_diagnostics", {})
            for h_key, h_data in models.items():
                if not isinstance(h_data, dict):
                    continue
                eval_counts[h_key] += 1
                if h_data.get("hit_sign"):
                    hit_counts[h_key] += 1

    result = {}
    for h in [f"H{i}" for i in range(1, 11)]:
        total = eval_counts.get(h, 0)
        if total >= 20:
            result[h] = round(hit_counts[h] / total, 4)

    return result


# ══════════════════════════════════════════════════════
# FITNESS DEL EXECUTOR GENOME
# ══════════════════════════════════════════════════════

def compute_executor_fitness(genome_id: str) -> Dict:
    """
    Fitness del executor = qué tan bien decide cuándo entrar/salir.

    Mide:
    - Sharpe sobre retornos reales
    - Max drawdown
    - Penalización por oportunidad perdida (cerró antes y dejó dinero)
    - Premio por cortar pérdidas (cerró antes y evitó caída mayor)
    """
    trades = _load_resolved_trades(genome_id, "executor")
    trades = _enrich_with_alpaca_pnl(trades)

    if len(trades) < 5:
        return {
            "genome_id":    genome_id,
            "genome_type":  "executor",
            "fitness":      0.0,
            "status":       "insufficient_data",
            "n_trades":     len(trades),
            "min_required": 5,
        }

    returns          = [float(t["pnl_real_pct"]) for t in trades]
    oportunidades    = [float(t.get("oportunidad_pct") or 0) for t in trades]
    cerro_antes      = [t.get("closed_before_horizon", False) for t in trades]

    sharpe   = _sharpe(returns)
    max_dd   = _max_drawdown(returns)
    win_rate = float(np.mean([r > 0 for r in returns]))
    avg_pnl  = float(np.mean(returns))

    # Penalización: cerró antes Y dejó dinero sobre la mesa
    oport_perdida = [
        abs(op) for op, antes in zip(oportunidades, cerro_antes)
        if antes and op > 0  # op > 0 = teórico mejor que real
    ]
    penalizacion = float(np.mean(oport_perdida)) if oport_perdida else 0.0

    # Premio: cerró antes Y evitó pérdida mayor
    oport_ganada = [
        abs(op) for op, antes in zip(oportunidades, cerro_antes)
        if antes and op < 0  # op < 0 = real mejor que teórico
    ]
    premio = float(np.mean(oport_ganada)) if oport_ganada else 0.0

    # Fitness final
    fitness = sharpe - (max_dd * 2) - (penalizacion * 0.5) + (premio * 0.3)

    result = {
        "genome_id":         genome_id,
        "genome_type":       "executor",
        "fitness":           round(fitness, 4),
        "status":            "ok",
        "n_trades":          len(trades),
        "sharpe":            round(sharpe, 4),
        "max_drawdown":      round(max_dd, 4),
        "win_rate":          round(win_rate, 4),
        "avg_pnl_pct":       round(avg_pnl, 4),
        "penalizacion_oport": round(penalizacion, 4),
        "premio_timing":     round(premio, 4),
        "computed_at":       datetime.now(timezone.utc).isoformat(),
    }

    # Guardar en disco
    path = FITNESS_DIR / f"{genome_id}_executor.json"
    _save_json(path, result)

    logger.info(
        f"🏋️ Executor fitness | {genome_id} | "
        f"fitness={fitness:.4f} sharpe={sharpe:.4f} "
        f"dd={max_dd:.4f} win={win_rate:.2%} n={len(trades)}"
    )
    return result


# ══════════════════════════════════════════════════════
# FITNESS DEL PREDICTOR GENOME
# ══════════════════════════════════════════════════════

def compute_predictor_fitness(genome_id: str) -> Dict:
    """
    Fitness del predictor = qué tan bien predice la dirección correcta.

    Mide:
    - Hit rate en horizonte teórico (¿acertó la dirección al vencimiento?)
    - Confianza calibrada (alta confianza + acierto = mejor fitness)
    - Penalización por alta confianza + error (el peor caso)
    """
    trades = _load_resolved_trades(genome_id, "predictor")
    trades = [t for t in trades if t.get("predictor_was_right") is not None]

    # Complementar con evaluaciones del evaluator existente
    h_hit_rates = _get_h_hit_rates()

    if len(trades) < 5 and not h_hit_rates:
        return {
            "genome_id":    genome_id,
            "genome_type":  "predictor",
            "fitness":      0.0,
            "status":       "insufficient_data",
            "n_trades":     len(trades),
        }

    # Fitness base desde trades atribuidos
    predictor_hits   = []
    confidence_scores = []

    for t in trades:
        was_right  = t.get("predictor_was_right", False)
        confidence = float(
            t.get("h_signals_at_entry", {})
             .get("main", {})
             .get("confidence", 0.5)
        )
        predictor_hits.append(1.0 if was_right else 0.0)
        confidence_scores.append(confidence)

    if predictor_hits:
        hit_rate  = float(np.mean(predictor_hits))
        # Calibración: penaliza alta confianza con error
        calibration_errors = [
            abs(conf - hit) * conf  # error ponderado por confianza
            for conf, hit in zip(confidence_scores, predictor_hits)
        ]
        calibration_penalty = float(np.mean(calibration_errors))
    else:
        # Sin trades propios → usar hit rates del evaluator como proxy
        if h_hit_rates:
            hit_rate            = float(np.mean(list(h_hit_rates.values())))
            calibration_penalty = 0.1  # penalización neutral sin datos propios
        else:
            hit_rate            = 0.5
            calibration_penalty = 0.1

    # Bonus por H con mejor hit rate
    best_h_bonus = 0.0
    if h_hit_rates:
        best_h_rate  = max(h_hit_rates.values())
        best_h_bonus = max(0, best_h_rate - 0.50) * 2  # bonus proporcional al exceso sobre 50%

    fitness = (hit_rate - 0.50) * 4 + best_h_bonus - calibration_penalty

    result = {
        "genome_id":            genome_id,
        "genome_type":          "predictor",
        "fitness":              round(fitness, 4),
        "status":               "ok",
        "n_trades":             len(trades),
        "hit_rate":             round(hit_rate, 4),
        "calibration_penalty":  round(calibration_penalty, 4),
        "best_h_bonus":         round(best_h_bonus, 4),
        "h_hit_rates":          h_hit_rates,
        "computed_at":          datetime.now(timezone.utc).isoformat(),
    }

    path = FITNESS_DIR / f"{genome_id}_predictor.json"
    _save_json(path, result)

    logger.info(
        f"🧠 Predictor fitness | {genome_id} | "
        f"fitness={fitness:.4f} hit_rate={hit_rate:.2%} "
        f"calib_penalty={calibration_penalty:.4f} n={len(trades)}"
    )
    return result


# ══════════════════════════════════════════════════════
# FITNESS COMBINADO — para el arena
# ══════════════════════════════════════════════════════

def compute_combined_fitness(
    predictor_genome_id: str,
    executor_genome_id: str,
) -> Dict:
    """
    Fitness combinado predictor + executor.
    El arena usa esto para decidir qué par promover.

    Un buen sistema necesita AMBOS:
    - Predictor que acierte la dirección
    - Executor que sepa cuándo entrar y salir
    """
    pred_fit = compute_predictor_fitness(predictor_genome_id)
    exec_fit = compute_executor_fitness(executor_genome_id)

    p_fitness = pred_fit.get("fitness", 0.0)
    e_fitness = exec_fit.get("fitness", 0.0)

    # Ponderación: el executor tiene más impacto en el PnL real
    combined = (p_fitness * 0.40) + (e_fitness * 0.60)

    # Penalización si uno de los dos es muy malo
    # (no sirve tener buen predictor con mal executor ni viceversa)
    if p_fitness < -0.5 or e_fitness < -0.5:
        combined *= 0.7
        logger.warning(
            f"⚠️ Penalización combinada | "
            f"pred={p_fitness:.3f} exec={e_fitness:.3f}"
        )

    result = {
        "predictor_genome_id": predictor_genome_id,
        "executor_genome_id":  executor_genome_id,
        "combined_fitness":    round(combined, 4),
        "predictor_fitness":   p_fitness,
        "executor_fitness":    e_fitness,
        "status": (
            "insufficient_data"
            if pred_fit["status"] == "insufficient_data"
            and exec_fit["status"] == "insufficient_data"
            else "ok"
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    path = FITNESS_DIR / f"combined_{predictor_genome_id}_{executor_genome_id}.json"
    _save_json(path, result)

    logger.info(
        f"⚡ Combined fitness | "
        f"pred={predictor_genome_id} exec={executor_genome_id} | "
        f"combined={combined:.4f} "
        f"(pred={p_fitness:.4f} exec={e_fitness:.4f})"
    )
    return result


# ══════════════════════════════════════════════════════
# RANKING DE TODOS LOS GENOMAS CONOCIDOS
# ══════════════════════════════════════════════════════

def rank_all_genomes() -> List[Dict]:
    """
    Lee todos los fitness guardados y los rankea.
    Usado por arena.py para seleccionar los mejores.
    """
    if not FITNESS_DIR.exists():
        return []

    scores = []
    seen   = set()

    for path in FITNESS_DIR.glob("combined_*.json"):
        data = _load_json(path)
        if not data:
            continue
        key = (data.get("predictor_genome_id"), data.get("executor_genome_id"))
        if key in seen:
            continue
        seen.add(key)
        scores.append(data)

    scores.sort(key=lambda x: x.get("combined_fitness", -999), reverse=True)
    return scores


def get_current_champion() -> Optional[Dict]:
    """Retorna el genome con mayor fitness combinado."""
    ranking = rank_all_genomes()
    return ranking[0] if ranking else None


def get_fitness_summary() -> Dict:
    """Resumen del estado actual de todos los genomas. Para el dashboard."""
    ranking = rank_all_genomes()

    if not ranking:
        return {
            "total_genomes": 0,
            "champion": None,
            "status": "no_data",
        }

    champion = ranking[0]
    return {
        "total_genomes":    len(ranking),
        "champion": {
            "predictor_id":   champion.get("predictor_genome_id"),
            "executor_id":    champion.get("executor_genome_id"),
            "fitness":        champion.get("combined_fitness"),
            "computed_at":    champion.get("computed_at"),
        },
        "top_3": [
            {
                "predictor_id": r.get("predictor_genome_id"),
                "executor_id":  r.get("executor_genome_id"),
                "fitness":      r.get("combined_fitness"),
            }
            for r in ranking[:3]
        ],
        "status": "ok",
    }

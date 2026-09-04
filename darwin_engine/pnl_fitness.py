"""
darwin_engine/pnl_fitness.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIX v1.1:
  [F1] _max_drawdown: filtra nan/inf antes de calcular — trades con
       exit_price=0 (bug broker pre-v3.3) generaban pnl_real_pct=nan
       que se propagaba a nan en fitness, bloqueando la selección
       de campeón indefinidamente (nan >= umbral → siempre False).
  [F2] compute_executor_fitness: filtra retornos inválidos (nan/inf/None)
       antes de calcular sharpe y drawdown. Si quedan menos de 5 trades
       válidos retorna insufficient_data en vez de nan.
  [F3] compute_combined_fitness: usa math.isnan() para detectar nan
       en e_fitness/p_fitness antes de la penalización — numpy evalúa
       nan < -0.5 como False, dejando pasar el nan al combined.
  [F4] _select_champion en arena.py: idem, protección contra nan en
       champion_fitness e improvement.

FIX v1.2 [AUD-P6] (auditoría 2026-08-24):
  [F5] _enrich_with_alpaca_pnl invertía el signo del PnL (fallback,
       cuando pnl_real_pct no venía calculado por el tracker) tratando
       recommendation=="VENDE" como si fuera una posición corta. El
       sistema nunca abre cortos (confirmado en trading_orchestrator.py
       — solo compra `shares` positivos), así que el cálculo debe ser
       siempre el de una posición larga, sin excepción. Mismo fix
       aplicado en darwin_engine/trade_tracker.py (fuente canónica de
       pnl_real_pct/pnl_teorico_pct).

FIX v1.3 (auditoría 2026-08-27, Problema 1):
  [F6] _load_resolved_trades(genome_type="executor") solo leía
       TRACKER_DIR (trades REALES). Únicamente el campeón opera con
       dinero real, así que para todo mutante (v2, v3, v6_x, v7_x)
       esto siempre devolvía [] → fitness=0.0 idéntico para todos,
       antes de llegar siquiera al test estadístico — Darwin comparaba
       ruido contra ruido. Ya el conteo de elegibilidad en arena.py
       (_select_champion, [AUD-D2]) usa
       executor_shadow_evaluator.get_shadow_resolved_trades() para
       decidir si un shadow puede competir, pero el VALOR de fitness
       en sí seguía sin ese fallback.
       Fix: si TRACKER_DIR no tiene trades reales para el genome_id
       (genome_type="executor"), se cae automáticamente a
       get_shadow_resolved_trades() — mismo criterio, mismo dato que
       ya usa arena.py, ahora también para el número que decide quién
       gana. El campeón nunca cae a esta rama porque siempre tiene
       trades reales.

FIX v1.4 (auditoría 2026-08-28, Problema 1 / Bug B):
  [F7] compute_executor_fitness() restaba penalizacion*0.5 y sumaba
       premio*0.3 al fitness — ambos derivados de oportunidad_pct y
       closed_before_horizon, campos que SOLO existen para trades
       reales (los llena trade_tracker.resolve_pending_trades() días
       después de cerrado el trade, consultando el precio en la fecha
       de horizonte original). Los trades de los shadows nunca pasan
       por ese proceso — dan 0 siempre. Resultado: el campeón se
       restaba en promedio ~1.33 puntos que ningún shadow paga jamás,
       sin ser una ventaja real del shadow, solo un hueco de datos que
       distorsionaba la comparación entre "quién decide mejor cuándo
       entrar/salir".
       Fix: penalizacion_oport y premio_timing se siguen calculando y
       guardando en el resultado (visibles para auditar "¿dejó plata
       sobre la mesa?" cuando SÍ hay datos), pero salen de la fórmula
       de fitness competitivo: fitness = sharpe - (max_dd * 2). Así
       campeón y shadows se comparan con exactamente los mismos
       ingredientes.

FIX v1.5 [F8] (2026-09-01, Problema 1 — bug de signo en compute_combined_fitness):
  [F8] compute_combined_fitness() combinaba predictor (40%) + executor (60%)
       y aplicaba `combined *= 0.7` como "penalización" cuando alguno de los
       dos fitness era muy malo (< -0.5). Si `combined` ya era negativo,
       multiplicar por 0.7 lo acerca a cero — es decir, MEJORA el score de
       un genoma malo en vez de empeorarlo. Esto corrompía la selección de
       campeón en todo el motor evolutivo.
       Fix: en vez de parchar el multiplicador, se simplifica la decisión.
       `combined_fitness` pasa a ser directamente el fitness del executor
       (`e_fitness`) — quien decide mejor cuándo entrar/salir gana, sin
       combinarse con el predictor ni aplicar penalización alguna. El
       fitness del predictor se sigue calculando y guardando (columna
       `predictor_fitness`, visible para diagnóstico/dashboard), pero deja
       de participar en la decisión de quién es campeón.
"""

import json
import logging
import math
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


def _is_valid_return(v) -> bool:
    """[F1][F2] True solo si el valor es un float finito y no None."""
    if v is None:
        return False
    try:
        f = float(v)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


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
    """
    [F1] Max drawdown desde retornos porcentuales.
    Filtra nan/inf antes de calcular — antes un solo nan en el array
    propagaba nan a toda la cadena de fitness.
    """
    if not returns:
        return 0.0
    # [F1] Filtrar valores inválidos
    clean = [r for r in returns if _is_valid_return(r)]
    if not clean:
        return 0.0
    equity = np.cumprod(1 + np.array(clean, dtype=float) / 100)
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

    [F6] Para "executor": si no hay trades REALES para este genome_id
    (TRACKER_DIR solo lo alimenta el campeón — ningún mutante opera
    dinero real), se cae a los trades simulados que el shadow
    evaluator ya acumula en /data/darwin/shadow_trades/{genome_id}/.
    Mismo dato que arena.py usa para decidir elegibilidad — ahora
    también sirve para calcular el fitness en sí. El campeón siempre
    tiene trades reales, así que nunca entra a esta rama.
    """
    field  = f"{genome_type}_genome_id"
    trades = []

    if TRACKER_DIR.exists():
        for path in TRACKER_DIR.glob("*.json"):
            t = _load_json(path)
            if t.get("status") == "resolved" and t.get(field) == genome_id:
                trades.append(t)

    if not trades and genome_type == "executor":
        try:
            from darwin_engine.executor_shadow_evaluator import get_shadow_resolved_trades
            trades = get_shadow_resolved_trades(genome_id)
        except Exception as e:
            logger.warning(f"_load_resolved_trades: fallback shadow no disponible para {genome_id}: {e}")

    trades.sort(key=lambda x: x.get("resolved_at") or x.get("exit_date", ""), reverse=True)
    return trades


# ══════════════════════════════════════════════════════
# FUENTE 2: ALPACA REAL (sin broker async — lee desde tracker)
# ══════════════════════════════════════════════════════

def _enrich_with_alpaca_pnl(trades: List[Dict]) -> List[Dict]:
    """
    Si el trade tiene pnl_real_pct calculado por el tracker, lo usa.
    Si no, intenta calcularlo desde entry/exit price disponibles.

    [F5] Siempre como posición larga — el sistema no opera en corto.
    Antes esta rama invertía el signo si recommendation=="VENDE",
    tratando el trade como si fuera un corto que nunca existió.
    """
    enriched = []
    for t in trades:
        pnl = t.get("pnl_real_pct")
        if pnl is None:
            ep = float(t.get("entry_price") or 0)
            xp = float(t.get("exit_price") or 0)
            if ep > 0 and xp > 0:
                pnl = (xp - ep) / ep * 100
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

    [F2] Filtra retornos inválidos (nan/inf/None) antes de calcular.
    """
    trades = _load_resolved_trades(genome_id, "executor")
    trades = _enrich_with_alpaca_pnl(trades)

    # [F2] Filtrar trades con pnl inválido (exit_price=0 → pnl=-100% o nan)
    valid_trades = [t for t in trades if _is_valid_return(t.get("pnl_real_pct"))]
    invalid_count = len(trades) - len(valid_trades)
    if invalid_count > 0:
        logger.warning(
            f"⚠️ {genome_id}: {invalid_count} trades con PnL inválido descartados "
            f"(probablemente exit_price=0 — fix broker v3.3)"
        )

    if len(valid_trades) < 5:
        return {
            "genome_id":      genome_id,
            "genome_type":    "executor",
            "fitness":        0.0,
            "status":         "insufficient_data",
            "n_trades":       len(valid_trades),
            "n_invalid":      invalid_count,
            "min_required":   5,
        }

    returns       = [float(t["pnl_real_pct"]) for t in valid_trades]
    oportunidades = [float(t.get("oportunidad_pct") or 0) for t in valid_trades]
    cerro_antes   = [t.get("closed_before_horizon", False) for t in valid_trades]

    sharpe   = _sharpe(returns)
    max_dd   = _max_drawdown(returns)   # [F1] ya filtra nan internamente
    win_rate = float(np.mean([r > 0 for r in returns]))
    avg_pnl  = float(np.mean(returns))

    # Penalización: cerró antes Y dejó dinero sobre la mesa
    oport_perdida = [
        abs(op) for op, antes in zip(oportunidades, cerro_antes)
        if antes and op > 0
    ]
    penalizacion = float(np.mean(oport_perdida)) if oport_perdida else 0.0

    # Premio: cerró antes Y evitó pérdida mayor
    oport_ganada = [
        abs(op) for op, antes in zip(oportunidades, cerro_antes)
        if antes and op < 0
    ]
    premio = float(np.mean(oport_ganada)) if oport_ganada else 0.0

    # FIX (auditoría 2026-08-28, Problema 1 / Bug B):
    # penalizacion/premio dependen de oportunidad_pct + closed_before_horizon,
    # que solo existen para trades REALES (los llena resolve_pending_trades()
    # días después de cerrado el trade). Los trades de los shadows nunca
    # pasan por ese proceso, así que para ellos este término siempre da 0 —
    # el campeón se restaba puntos que ningún shadow paga jamás, sin que
    # fuera una ventaja real, solo un hueco de datos.
    #
    # Se sigue calculando y guardando (penalizacion_oport / premio_timing
    # abajo) para poder auditar "¿dejó plata sobre la mesa?" cuando SÍ hay
    # datos — pero ya no forma parte del fitness competitivo, que debe
    # poder compararse en igualdad de condiciones entre campeón y shadows.
    fitness = sharpe - (max_dd * 2)

    result = {
        "genome_id":          genome_id,
        "genome_type":        "executor",
        "fitness":            round(fitness, 4),
        "status":             "ok",
        "n_trades":           len(valid_trades),
        "n_invalid":          invalid_count,
        "sharpe":             round(sharpe, 4),
        "max_drawdown":       round(max_dd, 4),
        "win_rate":           round(win_rate, 4),
        "avg_pnl_pct":        round(avg_pnl, 4),
        "penalizacion_oport": round(penalizacion, 4),
        "premio_timing":      round(premio, 4),
        "computed_at":        datetime.now(timezone.utc).isoformat(),
    }

    path = FITNESS_DIR / f"{genome_id}_executor.json"
    _save_json(path, result)

    logger.info(
        f"🏋️ Executor fitness | {genome_id} | "
        f"fitness={fitness:.4f} sharpe={sharpe:.4f} "
        f"dd={max_dd:.4f} win={win_rate:.2%} n={len(valid_trades)} "
        f"(descartados={invalid_count})"
    )
    return result


# ══════════════════════════════════════════════════════
# FITNESS DEL PREDICTOR GENOME
# ══════════════════════════════════════════════════════

def compute_predictor_fitness(genome_id: str) -> Dict:
    """
    Fitness del predictor = qué tan bien predice la dirección correcta.
    """
    trades = _load_resolved_trades(genome_id, "predictor")
    trades = [t for t in trades if t.get("predictor_was_right") is not None]

    h_hit_rates = _get_h_hit_rates()

    if len(trades) < 5 and not h_hit_rates:
        return {
            "genome_id":    genome_id,
            "genome_type":  "predictor",
            "fitness":      0.0,
            "status":       "insufficient_data",
            "n_trades":     len(trades),
        }

    predictor_hits    = []
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
        hit_rate = float(np.mean(predictor_hits))
        calibration_errors = [
            abs(conf - hit) * conf
            for conf, hit in zip(confidence_scores, predictor_hits)
        ]
        calibration_penalty = float(np.mean(calibration_errors))
    else:
        if h_hit_rates:
            hit_rate            = float(np.mean(list(h_hit_rates.values())))
            calibration_penalty = 0.1
        else:
            hit_rate            = 0.5
            calibration_penalty = 0.1

    best_h_bonus = 0.0
    if h_hit_rates:
        best_h_rate  = max(h_hit_rates.values())
        best_h_bonus = max(0, best_h_rate - 0.50) * 2

    fitness = (hit_rate - 0.50) * 4 + best_h_bonus - calibration_penalty

    result = {
        "genome_id":           genome_id,
        "genome_type":         "predictor",
        "fitness":             round(fitness, 4),
        "status":              "ok",
        "n_trades":            len(trades),
        "hit_rate":            round(hit_rate, 4),
        "calibration_penalty": round(calibration_penalty, 4),
        "best_h_bonus":        round(best_h_bonus, 4),
        "h_hit_rates":         h_hit_rates,
        "computed_at":         datetime.now(timezone.utc).isoformat(),
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
    [F8] combined_fitness = fitness del executor (e_fitness), sin combinar
    con el predictor y sin el multiplicador de "penalización" que tenía
    el signo invertido (ver FIX v1.5 en el encabezado del archivo).
    El fitness del predictor se sigue calculando y guardando (visible
    para diagnóstico/dashboard), pero ya no participa en la decisión.

    [F3] Usa math.isnan() para detectar nan en e_fitness/p_fitness.
    numpy evalúa nan < -0.5 como False, dejando pasar el nan al combined.
    """
    pred_fit = compute_predictor_fitness(predictor_genome_id)
    exec_fit = compute_executor_fitness(executor_genome_id)

    p_fitness = pred_fit.get("fitness", 0.0)
    e_fitness = exec_fit.get("fitness", 0.0)

    # [F3] Reemplazar nan por 0.0 explícitamente antes de operar
    if not _is_valid_return(p_fitness):
        logger.warning(f"⚠️ p_fitness nan/inf para {predictor_genome_id} → usando 0.0")
        p_fitness = 0.0
    if not _is_valid_return(e_fitness):
        logger.warning(f"⚠️ e_fitness nan/inf para {executor_genome_id} → usando 0.0")
        e_fitness = 0.0

    # [F8] combined = fitness real del executor, y nada más.
    combined = e_fitness

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
        f"combined={combined:.4f} (= fitness del executor) "
        f"(pred_fitness={p_fitness:.4f}, informativo, no usado)"
    )
    return result


# ══════════════════════════════════════════════════════
# RANKING DE TODOS LOS GENOMAS CONOCIDOS
# ══════════════════════════════════════════════════════
def rank_all_genomes() -> List[Dict]:
    """
    Lee todos los fitness guardados y los rankea.
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
        "total_genomes": len(ranking),
        "champion": {
            "predictor_id": champion.get("predictor_genome_id"),
            "executor_id":  champion.get("executor_genome_id"),
            "fitness":      champion.get("combined_fitness"),
            "computed_at":  champion.get("computed_at"),
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
      

"""
darwin_engine/mutator.py
━━━━━━━━━━━━━━━━━━━━━━━━
ARCHIVO NUEVO — darwin_engine/mutator.py

Genera nuevas variantes de ExecutorGenome mediante:
  1. Mutación   → modifica un parámetro del genoma
  2. Crossover  → combina dos genomas ganadores
  3. Reset      → genera un genoma aleatorio dentro de límites seguros

El mutador tiene sesgo inteligente:
  - Si COMPRA hit rate < 45% → muta parámetros de entrada/confianza
  - Si oportunidad perdida alta → muta días de mantención
  - Si drawdown alto → muta loss_cut y profit_lock

FIX [AUD-D1] (bug Darwin #1 — colisión de genome_id):
  generate_next_generation() producía hasta 5 hijos por ciclo, todos
  derivados del mismo padre (best). _next_genome_id() calcula el ID
  únicamente a partir del genome_id del padre — así que los 5 hijos
  recibían el MISMO ID (ej. "executor_v2"). Como ExecutorGenome.save()
  escribe siempre a GENOME_DIR / f"{gid}.json" y arena.py guarda cada
  hijo en un bucle (`for child in new_generation: child.save()`) sin
  manejar colisión, solo el último hijo en guardarse sobrevivía en
  disco — de 5 estrategias de mutación/crossover distintas, el espacio
  de búsqueda real explorado era una fracción del diseñado.

  Mismo patrón de fix ya aplicado y en producción en
  darwin_engine/predictor_mutator.py ([MC1]): justo antes de retornar,
  se reasignan IDs únicos por posición
  (ej. "executor_v2_1".."executor_v2_5"). Confirmado que
  ExecutorGenome.load_champion() usa glob("executor_*.json"), que
  sigue matcheando este formato de ID sin cambios adicionales.

FIX (auditoría 2026-08-27, Problema 2):
  Mutación de "early_exit.loss_cut_pct" era puramente aleatoria y
  simétrica dentro de (-5.0, -1.0), sin memoria de resultados. Los
  mutantes convergieron todos a -1.0 (stop más ajustado permitido) y
  llevan generaciones perdiendo el test estadístico frente al campeón
  (-2.5) sin que el mutador lo detecte ni corrija el rumbo.

  Fix autocontenido en este archivo (sin tocar arena.py): se persiste
  un pequeño estado en GENOME_DIR/_mutation_direction_state.json con
  el mejor fitness visto y cuántas generaciones van sin mejora. Si el
  fitness del mejor genoma no mejora durante STAGNATION_THRESHOLD
  generaciones seguidas, se invierte el sesgo direccional para el
  grupo "risk": en vez de moverse simétricamente, la mutación de
  loss_cut_pct empuja explícitamente en la dirección contraria a la
  ya explorada (ej. si veníamos ajustando el stop, ahora se ensancha).
  Esto no depende del test de Mann-Whitney de arena.py — usa
  estancamiento de fitness como proxy, disponible aquí mismo vía
  ranked_genomes[0].fitness.
"""

import json
import logging
import os
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from darwin_engine.executor_genome import ExecutorGenome, GENOME_DIR

logger = logging.getLogger("mutator")

# ══════════════════════════════════════════════════════
# LÍMITES DE MUTACIÓN — rangos seguros para cada gen
# ══════════════════════════════════════════════════════

MUTATION_LIMITS = {
    "min_hold_days.high":   (3,  14),
    "min_hold_days.medium": (2,   8),
    "min_hold_days.low":    (1,   4),

    "confidence_thresholds.high":   (0.65, 0.90),
    "confidence_thresholds.medium": (0.45, 0.70),

    "early_exit.profit_lock_pct": (1.5,  8.0),
    "early_exit.loss_cut_pct":   (-5.0, -1.0),

    "kill_shield.min_confirmations":   (1, 4),
    "kill_shield.min_h_horizon":       (3, 8),
    "kill_shield.min_hit_rate":        (0.48, 0.65),
    "kill_shield.override_kill_below": (-0.80, -0.30),

    "entry_confirmation.min_agreeing_h":    (2, 5),
    "entry_confirmation.min_horizon_for_entry": (2, 6),

    # Pesos H — cada uno entre 0.02 y 0.25, suma debe ser ~1.0
    "h_weights.H1":  (0.02, 0.10),
    "h_weights.H2":  (0.02, 0.12),
    "h_weights.H3":  (0.03, 0.14),
    "h_weights.H4":  (0.05, 0.16),
    "h_weights.H5":  (0.06, 0.18),
    "h_weights.H6":  (0.08, 0.25),
    "h_weights.H7":  (0.06, 0.22),
    "h_weights.H8":  (0.05, 0.20),
    "h_weights.H9":  (0.04, 0.18),
    "h_weights.H10": (0.03, 0.16),
}

# Magnitud del cambio por mutación (fracción del rango)
MUTATION_STRENGTH = 0.20

# [Problema 2] Estado de sesgo direccional para mutaciones de riesgo
DIRECTION_STATE_FILE = GENOME_DIR / "_mutation_direction_state.json"
STAGNATION_THRESHOLD = 5  # generaciones sin mejora antes de invertir el sesgo
DIRECTIONAL_GENES = {"early_exit.loss_cut_pct"}  # genes con sesgo direccional activo


def _load_direction_state() -> Dict:
    if DIRECTION_STATE_FILE.exists():
        try:
            return json.loads(DIRECTION_STATE_FILE.read_text())
        except Exception:
            pass
    return {"best_fitness_seen": None, "stagnant_generations": 0, "risk_direction": "widen"}


def _save_direction_state(state: Dict) -> None:
    try:
        DIRECTION_STATE_FILE.write_text(json.dumps(state))
    except Exception as e:
        logger.warning(f"No se pudo guardar _mutation_direction_state.json: {e}")


def _update_direction_state(best_fitness: Optional[float]) -> str:
    """
    Actualiza el estado de estancamiento y devuelve la dirección de
    sesgo vigente para el grupo "risk" ("widen" = stop más ancho,
    "tighten" = stop más ajustado).
    """
    state = _load_direction_state()

    if best_fitness is not None:
        prev = state["best_fitness_seen"]
        if prev is not None and best_fitness <= prev:
            state["stagnant_generations"] += 1
        else:
            state["stagnant_generations"] = 0
        state["best_fitness_seen"] = max(best_fitness, prev) if prev is not None else best_fitness

    if state["stagnant_generations"] >= STAGNATION_THRESHOLD:
        state["risk_direction"] = "tighten" if state["risk_direction"] == "widen" else "widen"
        state["stagnant_generations"] = 0
        logger.info(
            f"🔄 Sin mejora de fitness en {STAGNATION_THRESHOLD} generaciones — "
            f"invirtiendo sesgo de riesgo a '{state['risk_direction']}'"
        )

    _save_direction_state(state)
    return state["risk_direction"]


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _get_nested(d: Dict, key_path: str):
    """Accede a dict anidado con path 'a.b.c'."""
    keys = key_path.split(".")
    val  = d
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return None
    return val


def _set_nested(d: Dict, key_path: str, value) -> Dict:
    """Setea valor en dict anidado con path 'a.b.c'."""
    keys   = key_path.split(".")
    target = d
    for k in keys[:-1]:
        if k not in target:
            target[k] = {}
        target = target[k]
    target[keys[-1]] = value
    return d


def _normalize_h_weights(genome_dict: Dict) -> Dict:
    """Asegura que los pesos H sumen 1.0."""
    weights = genome_dict.get("h_weights", {})
    total   = sum(weights.values())
    if total > 0:
        genome_dict["h_weights"] = {
            k: round(v / total, 4) for k, v in weights.items()
        }
    return genome_dict


def _next_genome_id(base_id: str) -> str:
    """Genera el próximo ID de genome: executor_v1 → executor_v2."""
    parts = base_id.rsplit("_v", 1)
    if len(parts) == 2:
        try:
            version = int(parts[1]) + 1
            return f"{parts[0]}_v{version}"
        except ValueError:
            pass
    # Buscar el número más alto en disco
    if GENOME_DIR.exists():
        existing = [
            p.stem for p in GENOME_DIR.glob("executor_v*.json")
        ]
        versions = []
        for name in existing:
            try:
                versions.append(int(name.split("_v")[-1]))
            except ValueError:
                pass
        if versions:
            return f"executor_v{max(versions) + 1}"
    return f"{base_id}_mut"


# ══════════════════════════════════════════════════════
# MUTACIÓN SIMPLE
# ══════════════════════════════════════════════════════

def mutate(
    genome: ExecutorGenome,
    n_mutations: int = 1,
    bias: Optional[str] = None,
    risk_direction: Optional[str] = None,
) -> ExecutorGenome:
    """
    Genera una variante mutada del genome.

    Args:
        genome:      genome padre
        n_mutations: cuántos parámetros cambiar (1-3)
        bias:        sesgo opcional:
                     "compra"     → muta parámetros de entrada/confianza
                     "hold"       → muta días de mantención
                     "risk"       → muta profit_lock y loss_cut
                     "weights"    → muta pesos H
                     None         → mutación aleatoria
        risk_direction: para genes en DIRECTIONAL_GENES (ej. loss_cut_pct),
                     fuerza la dirección del delta en vez de simétrica:
                     "widen"   → stop más ancho (más negativo)
                     "tighten" → stop más ajustado (menos negativo)
                     None      → delta simétrico de siempre

    Returns:
        Nuevo ExecutorGenome (el padre no se modifica)
    """
    new_data  = deepcopy(genome.to_dict())
    new_id    = _next_genome_id(genome.genome_id)
    mutations = []

    # Seleccionar genes a mutar según sesgo
    candidate_genes = _select_genes_by_bias(bias)

    # Asegurar al menos n_mutations sin repetir
    selected = random.sample(
        candidate_genes,
        min(n_mutations, len(candidate_genes))
    )

    for gene_path in selected:
        limits  = MUTATION_LIMITS.get(gene_path)
        if not limits:
            continue

        lo, hi      = limits
        current_val = _get_nested(new_data, gene_path)

        if current_val is None:
            continue

        # Calcular rango de mutación
        mutation_range = (hi - lo) * MUTATION_STRENGTH

        if risk_direction and gene_path in DIRECTIONAL_GENES:
            # Delta de un solo signo: exploramos a propósito la
            # dirección contraria a la que llevamos generaciones
            # explorando sin resultado.
            magnitude = random.uniform(0, mutation_range)
            delta = -magnitude if risk_direction == "widen" else magnitude
        else:
            delta = random.uniform(-mutation_range, mutation_range)

        # Aplicar mutación
        if isinstance(current_val, int):
            new_val = int(round(_clamp(current_val + delta, lo, hi)))
        else:
            new_val = round(_clamp(float(current_val) + delta, lo, hi), 4)

        _set_nested(new_data, gene_path, new_val)
        mutations.append({
            "gene":    gene_path,
            "before":  current_val,
            "after":   new_val,
            "delta":   round(delta, 4),
        })

    # Normalizar pesos H si se mutaron
    if any("h_weights" in m["gene"] for m in mutations):
        new_data = _normalize_h_weights(new_data)

    # Metadata
    new_data["genome_id"]  = new_id
    new_data["generation"] = genome.generation + 1
    new_data["parent_ids"] = [genome.genome_id]
    new_data["fitness"]    = None
    new_data["created_at"] = datetime.now(timezone.utc).isoformat()
    new_data["mutations"]  = mutations
    new_data["n_trades_eval"] = 0

    child = ExecutorGenome.from_dict(new_data)

    logger.info(
        f"🧬 Mutación | {genome.genome_id} → {new_id} | "
        f"genes={[m['gene'] for m in mutations]}"
    )
    for m in mutations:
        logger.info(f"   {m['gene']}: {m['before']} → {m['after']} (Δ{m['delta']:+.4f})")

    return child


# ══════════════════════════════════════════════════════
# CROSSOVER
# ══════════════════════════════════════════════════════

def crossover(
    parent_a: ExecutorGenome,
    parent_b: ExecutorGenome,
    strategy: str = "best_gene",
) -> ExecutorGenome:
    """
    Combina dos genomas ganadores para crear un hijo.

    Strategies:
      "best_gene"  → por cada gen, hereda del padre con mejor fitness
      "random_mix" → hereda aleatoriamente de cada padre
      "weighted"   → hereda proporcional al fitness de cada padre
    """
    new_data = deepcopy(parent_a.to_dict())
    new_id   = _next_genome_id(
        f"executor_v{max(parent_a.generation, parent_b.generation)}"
    )

    fit_a = parent_a.fitness or 0.0
    fit_b = parent_b.fitness or 0.0

    for gene_path in MUTATION_LIMITS.keys():
        val_a = _get_nested(parent_a.to_dict(), gene_path)
        val_b = _get_nested(parent_b.to_dict(), gene_path)

        if val_a is None or val_b is None:
            continue

        if strategy == "best_gene":
            # Hereda del padre con mejor fitness
            inherit_from_b = fit_b > fit_a
        elif strategy == "random_mix":
            inherit_from_b = random.random() > 0.5
        elif strategy == "weighted":
            # Probabilidad proporcional al fitness relativo
            total = abs(fit_a) + abs(fit_b)
            p_b   = abs(fit_b) / total if total > 0 else 0.5
            inherit_from_b = random.random() < p_b
        else:
            inherit_from_b = random.random() > 0.5

        chosen = val_b if inherit_from_b else val_a
        _set_nested(new_data, gene_path, chosen)

    # Normalizar H weights
    new_data = _normalize_h_weights(new_data)

    # Metadata
    new_data["genome_id"]  = new_id
    new_data["generation"] = max(parent_a.generation, parent_b.generation) + 1
    new_data["parent_ids"] = [parent_a.genome_id, parent_b.genome_id]
    new_data["fitness"]    = None
    new_data["created_at"] = datetime.now(timezone.utc).isoformat()
    new_data["crossover_strategy"] = strategy
    new_data["n_trades_eval"] = 0

    child = ExecutorGenome.from_dict(new_data)

    logger.info(
        f"🔀 Crossover [{strategy}] | "
        f"{parent_a.genome_id} (fit={fit_a:.3f}) × "
        f"{parent_b.genome_id} (fit={fit_b:.3f}) → {new_id}"
    )
    return child


# ══════════════════════════════════════════════════════
# GENERACIÓN NUEVA COMPLETA
# ══════════════════════════════════════════════════════

def generate_next_generation(
    ranked_genomes: List[ExecutorGenome],
    generation_size: int = 5,
    fitness_context: Optional[Dict] = None,
) -> List[ExecutorGenome]:
    """
    Genera la próxima generación de genomas.

    Estrategia:
      - Top 1 → 2 mutaciones con sesgo inteligente
      - Top 2 → crossover "best_gene"
      - Top 2 → crossover "weighted"
      - Top 1 → mutación aleatoria (exploración)
      - Top 1 → mutación fuerte (exploración agresiva)

    Args:
        ranked_genomes:  lista de genomas ordenados por fitness (mejor primero)
        generation_size: cuántos hijos generar
        fitness_context: métricas actuales para sesgo inteligente
                         {"compra_hit_rate": 0.31, "avg_oportunidad": 2.1, ...}

    Returns:
        Lista de nuevos genomas (no guardados aún)

    [AUD-D1] Antes de retornar, se reasignan genome_id únicos por
    posición (ej. "executor_v2_1".."executor_v2_5"). Todos los hijos
    parten del mismo padre (best), así que _next_genome_id() calculaba
    el mismo ID para todos — cada child.save() en arena.py sobrescribía
    al anterior y solo 1 de N sobrevivía en disco. Mismo patrón de fix
    ya aplicado en predictor_mutator.py ([MC1]).
    """
    if not ranked_genomes:
        logger.warning("⚠️ Sin genomas para evolucionar — generando desde default")
        default = ExecutorGenome.load_or_default()
        children = [mutate(default, n_mutations=1) for _ in range(generation_size)]
        next_gen = default.generation + 1
        for i, child in enumerate(children):
            child.data["genome_id"] = f"executor_v{next_gen}_{i + 1}"
        return children

    best  = ranked_genomes[0]
    second = ranked_genomes[1] if len(ranked_genomes) > 1 else best

    # Determinar sesgo según métricas actuales
    bias = _determine_bias(fitness_context or {})

    # [Problema 2] Actualizar/consultar sesgo direccional de riesgo
    # según estancamiento de fitness entre ciclos
    risk_direction = _update_direction_state(best.fitness)
    mutate_risk_direction = risk_direction if bias == "risk" else None

    children = []

    # 1. Mutación sesgada del mejor
    children.append(mutate(best, n_mutations=1, bias=bias, risk_direction=mutate_risk_direction))

    # 2. Mutación sesgada del mejor (segundo gen diferente)
    children.append(mutate(best, n_mutations=2, bias=bias, risk_direction=mutate_risk_direction))

    # 3. Crossover best_gene
    if second.genome_id != best.genome_id:
        children.append(crossover(best, second, strategy="best_gene"))

        # 4. Crossover weighted
        children.append(crossover(best, second, strategy="weighted"))
    else:
        children.append(mutate(best, n_mutations=1, bias="weights"))
        children.append(mutate(best, n_mutations=2, bias="risk", risk_direction=risk_direction))

    # 5. Mutación de exploración (sin sesgo, más fuerte)
    children.append(_mutate_strong(best))

    children = children[:generation_size]

    # [AUD-D1] IDs únicos por posición — evita colisión al guardar.
    # Sin esto, todos los hijos comparten genome_id (derivan del mismo
    # "best") y arena.py los pisa entre sí al hacer child.save() en bucle.
    next_gen = best.generation + 1
    for i, child in enumerate(children):
        child.data["genome_id"] = f"executor_v{next_gen}_{i + 1}"

    logger.info(
        f"🌱 Nueva generación | {len(children)} hijos "
        f"[{', '.join(c.genome_id for c in children)}] | "
        f"padre={best.genome_id} fit={best.fitness} | sesgo={bias}"
    )
    return children


# ══════════════════════════════════════════════════════
# HELPERS INTERNOS
# ══════════════════════════════════════════════════════

def _select_genes_by_bias(bias: Optional[str]) -> List[str]:
    """Retorna lista de genes a mutar según el sesgo."""
    all_genes = list(MUTATION_LIMITS.keys())

    groups = {
        "compra": [
            "confidence_thresholds.high",
            "confidence_thresholds.medium",
            "entry_confirmation.min_agreeing_h",
            "entry_confirmation.min_horizon_for_entry",
        ],
        "hold": [
            "min_hold_days.high",
            "min_hold_days.medium",
            "min_hold_days.low",
            "kill_shield.min_confirmations",
            "kill_shield.min_h_horizon",
            "kill_shield.min_hit_rate",
        ],
        "risk": [
            "early_exit.profit_lock_pct",
            "early_exit.loss_cut_pct",
            "kill_shield.override_kill_below",
        ],
        "weights": [k for k in all_genes if k.startswith("h_weights")],
    }

    if bias and bias in groups:
        # 70% genes del grupo sesgado + 30% aleatorios
        biased  = groups[bias]
        random_pool = [g for g in all_genes if g not in biased]
        n_random    = max(1, len(biased) // 3)
        return biased + random.sample(random_pool, min(n_random, len(random_pool)))

    return all_genes


def _determine_bias(context: Dict) -> Optional[str]:
    """
    Determina el sesgo de mutación basado en las métricas actuales.
    Ataca el problema más urgente primero.
    """
    compra_hit = context.get("compra_hit_rate", 0.5)
    avg_oport  = context.get("avg_oportunidad_perdida", 0.0)
    max_dd     = context.get("max_drawdown", 0.0)

    if compra_hit < 0.40:
        logger.info(f"🎯 Sesgo: COMPRA (hit_rate={compra_hit:.2%})")
        return "compra"
    elif avg_oport > 2.0:
        logger.info(f"🎯 Sesgo: HOLD (oportunidad_perdida={avg_oport:.2f}%)")
        return "hold"
    elif max_dd > 0.15:
        logger.info(f"🎯 Sesgo: RISK (drawdown={max_dd:.2%})")
        return "risk"
    else:
        return None  # mutación aleatoria


def _mutate_strong(genome: ExecutorGenome) -> ExecutorGenome:
    """Mutación fuerte: 3 genes, mayor magnitud. Para exploración."""
    global MUTATION_STRENGTH
    original = MUTATION_STRENGTH
    MUTATION_STRENGTH = 0.40  # doble de fuerza
    child = mutate(genome, n_mutations=3, bias=None)
    MUTATION_STRENGTH = original
    return child


# ══════════════════════════════════════════════════════
# GENERACIÓN DE GENOMA ALEATORIO (reset)
# ══════════════════════════════════════════════════════

def random_genome(genome_id: str = None) -> ExecutorGenome:
    """
    Genera un genome completamente aleatorio dentro de límites seguros.
    Útil para exploración cuando el fitness actual es muy bajo.
    """
    from darwin_engine.executor_genome import DEFAULT_GENOME
    new_data = deepcopy(DEFAULT_GENOME)

    for gene_path, (lo, hi) in MUTATION_LIMITS.items():
        current = _get_nested(new_data, gene_path)
        if current is None:
            continue
        if isinstance(current, int):
            new_val = random.randint(int(lo), int(hi))
        else:
            new_val = round(random.uniform(lo, hi), 4)
        _set_nested(new_data, gene_path, new_val)

    new_data = _normalize_h_weights(new_data)

    gid = genome_id or _next_genome_id("executor_v0")
    new_data["genome_id"]  = gid
    new_data["generation"] = 1
    new_data["parent_ids"] = ["random"]
    new_data["fitness"]    = None
    new_data["created_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(f"🎲 Genome aleatorio generado: {gid}")
    return ExecutorGenome.from_dict(new_data)

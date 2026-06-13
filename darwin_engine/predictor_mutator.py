"""
darwin_engine/predictor_mutator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Muta y cruza PredictorGenomes para evolucionar H1-H10.

Operaciones:
  1. mutate_features      → agrega/quita/reemplaza features
  2. mutate_params        → cambia alpha_ridge, max_pca, clip_ret
  3. mutate_decay         → cambia sample_weight_decay [SW1]
  4. crossover            → combina features de dos H distintos
  5. generate_children    → genera nueva generación completa

Sesgo inteligente:
  - hit_rate < 48%              → muta features agresivamente
  - hit_rate 48-52%             → muta parámetros del modelo
  - hit_rate > 52%              → exploración conservadora
  - bias_score >= 0.70 [SW2]   → muta decay prioritariamente
    (predictor alcista en mercado bajista → dar más peso a datos recientes)
"""

import logging
import random
from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from darwin_engine.predictor_genome import (
    PredictorGenome,
    FEATURE_POOL,
    BASE_FEATURES_BY_HORIZON,
    DECAY_LIMITS,
    BIAS_SCORE_THRESHOLD,
)

logger = logging.getLogger("predictor_mutator")

# ══════════════════════════════════════════════════════
# LÍMITES DE PARÁMETROS
# ══════════════════════════════════════════════════════

PARAM_LIMITS = {
    "alpha_ridge": (0.1,  5.0),
    "max_pca":     (4,    20),
    "clip_ret":    (0.03, 0.25),
}

MIN_FEATURES = 4
MAX_FEATURES = 14


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def _next_genome_id(parent_id: str) -> str:
    parts = parent_id.rsplit("_v", 1)
    if len(parts) == 2:
        try:
            return f"{parts[0]}_v{int(parts[1]) + 1}"
        except ValueError:
            pass
    return f"{parent_id}_mut"


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _all_features() -> List[str]:
    return list(FEATURE_POOL.keys())


def _base_genome_data(genome: PredictorGenome, mutation_label: str) -> Dict:
    """Helper para crear la base de un genoma hijo."""
    new_data = deepcopy(genome.to_dict())
    new_data["genome_id"]     = _next_genome_id(genome.genome_id)
    new_data["generation"]    = genome.generation + 1
    new_data["parent_ids"]    = [genome.genome_id]
    new_data["fitness"]       = None
    new_data["hit_rate"]      = None
    new_data["bias_score"]    = None  # [SW2] se recalcula tras evaluación
    new_data["created_at"]    = datetime.now(timezone.utc).isoformat()
    new_data["n_evaluations"] = 0
    new_data["mutations"]     = [mutation_label]
    return new_data


# ══════════════════════════════════════════════════════
# MUTACIÓN DE FEATURES
# ══════════════════════════════════════════════════════

def mutate_features(genome: PredictorGenome, strength: str = "normal") -> PredictorGenome:
    """
    Muta las features del genoma.

    strength:
      "conservative" → cambia 1 feature
      "normal"       → cambia 1-2 features
      "aggressive"   → cambia 2-3 features
    """
    new_data      = deepcopy(genome.to_dict())
    current_feats = list(new_data["features"])
    all_feats     = _all_features()
    available     = [f for f in all_feats if f not in current_feats]

    n_changes = {
        "conservative": 1,
        "normal":       random.randint(1, 2),
        "aggressive":   random.randint(2, 3),
    }
    n = n_changes.get(strength, 1)

    mutations = []
    for _ in range(n):
        op = random.choice(["add", "remove", "replace"])

        if op == "add" and len(current_feats) < MAX_FEATURES and available:
            feat = random.choice(available)
            current_feats.append(feat)
            available.remove(feat)
            mutations.append(f"ADD:{feat}")

        elif op == "remove" and len(current_feats) > MIN_FEATURES:
            feat = random.choice(current_feats)
            current_feats.remove(feat)
            available.append(feat)
            mutations.append(f"REM:{feat}")

        elif op == "replace" and available:
            old_feat = random.choice(current_feats)
            new_feat = random.choice(available)
            idx = current_feats.index(old_feat)
            current_feats[idx] = new_feat
            available.remove(new_feat)
            available.append(old_feat)
            mutations.append(f"REP:{old_feat}→{new_feat}")

    new_data["features"]      = current_feats
    new_data["genome_id"]     = _next_genome_id(genome.genome_id)
    new_data["generation"]    = genome.generation + 1
    new_data["parent_ids"]    = [genome.genome_id]
    new_data["fitness"]       = None
    new_data["hit_rate"]      = None
    new_data["bias_score"]    = None
    new_data["created_at"]    = datetime.now(timezone.utc).isoformat()
    new_data["mutations"]     = mutations
    new_data["n_evaluations"] = 0

    child = PredictorGenome.from_dict(new_data)
    logger.info(
        f"🧬 Feature mutation H{genome.horizon} | "
        f"{genome.genome_id} → {child.genome_id} | {mutations}"
    )
    return child


# ══════════════════════════════════════════════════════
# MUTACIÓN DE PARÁMETROS
# ══════════════════════════════════════════════════════

def mutate_params(genome: PredictorGenome) -> PredictorGenome:
    """
    Muta los parámetros del modelo (alpha_ridge, max_pca, clip_ret).
    No toca sample_weight_decay — eso lo maneja mutate_decay [SW1].
    """
    new_data   = deepcopy(genome.to_dict())
    params     = new_data["model_params"]
    param_name = random.choice(list(PARAM_LIMITS.keys()))
    lo, hi     = PARAM_LIMITS[param_name]

    current = params.get(param_name, (lo + hi) / 2)
    delta   = (hi - lo) * 0.15 * random.choice([-1, 1])

    if isinstance(current, int):
        new_val = int(_clamp(round(current + delta), lo, hi))
    else:
        new_val = round(_clamp(current + delta, lo, hi), 3)

    params[param_name]       = new_val
    new_data["model_params"] = params
    new_data["genome_id"]    = _next_genome_id(genome.genome_id)
    new_data["generation"]   = genome.generation + 1
    new_data["parent_ids"]   = [genome.genome_id]
    new_data["fitness"]      = None
    new_data["hit_rate"]     = None
    new_data["bias_score"]   = None
    new_data["created_at"]   = datetime.now(timezone.utc).isoformat()
    new_data["mutations"]    = [f"PARAM:{param_name}:{current}→{new_val}"]
    new_data["n_evaluations"] = 0

    child = PredictorGenome.from_dict(new_data)
    logger.info(
        f"⚙️ Param mutation H{genome.horizon} | "
        f"{genome.genome_id} → {child.genome_id} | "
        f"{param_name}: {current} → {new_val}"
    )
    return child


# ══════════════════════════════════════════════════════
# [SW1] MUTACIÓN DE SAMPLE_WEIGHT_DECAY
# ══════════════════════════════════════════════════════

def mutate_decay(
    genome: PredictorGenome,
    direction: str = "auto",
) -> PredictorGenome:
    """
    [SW1] Muta sample_weight_decay del genoma.

    Solo aplica a H1-H9 — H10 tiene walk-forward propio.

    direction:
      "up"   → subir decay (más peso a datos recientes)
               Usado cuando bias_score alto → modelo demasiado alcista
      "down" → bajar decay (volver a ponderación uniforme)
               Usado cuando bias_score bajó → ya no hay sesgo temporal
      "auto" → decide según bias_score del genome

    Lógica de mutación:
      - Si decay actual es 0.0 y subimos → saltar directo a 0.5
        (primer paso significativo, evitar cambios insignificantes)
      - Si decay > 0 → incrementos de 0.25 hacia arriba o abajo
      - Clamped entre DECAY_LIMITS = (0.0, 3.0)

    Ejemplo de impacto:
      decay=0.0 → todos los días pesan igual (Ridge estándar)
      decay=0.5 → últimos datos ~1.6x más peso que los primeros
      decay=1.5 → últimos datos ~4.5x más peso
      decay=3.0 → últimos datos ~20x más peso (muy agresivo)
    """
    # H10 no usa decay — no mutar
    if genome.horizon == 10:
        logger.warning(f"⚠️ mutate_decay ignorado para H10 (usa walk-forward)")
        return genome

    new_data = deepcopy(genome.to_dict())
    params   = new_data.get("model_params", {})
    lo, hi   = DECAY_LIMITS

    current_decay = float(params.get("sample_weight_decay", 0.0))

    # Decidir dirección automáticamente
    if direction == "auto":
        bs = genome.bias_score
        if bs is not None and bs >= BIAS_SCORE_THRESHOLD:
            direction = "up"
        else:
            direction = "down"

    # Calcular nuevo decay
    if direction == "up":
        if current_decay == 0.0:
            new_decay = 0.5   # primer salto significativo
        else:
            new_decay = round(current_decay + 0.25, 2)
    else:
        new_decay = round(max(0.0, current_decay - 0.25), 2)

    new_decay = round(_clamp(new_decay, lo, hi), 2)
    params["sample_weight_decay"] = new_decay
    new_data["model_params"]      = params
    new_data["genome_id"]         = _next_genome_id(genome.genome_id)
    new_data["generation"]        = genome.generation + 1
    new_data["parent_ids"]        = [genome.genome_id]
    new_data["fitness"]           = None
    new_data["hit_rate"]          = None
    new_data["bias_score"]        = None
    new_data["created_at"]        = datetime.now(timezone.utc).isoformat()
    new_data["mutations"]         = [
        f"DECAY:{current_decay}→{new_decay} ({direction})"
    ]
    new_data["n_evaluations"] = 0

    child = PredictorGenome.from_dict(new_data)
    logger.info(
        f"⏱️ Decay mutation H{genome.horizon} | "
        f"{genome.genome_id} → {child.genome_id} | "
        f"decay: {current_decay} → {new_decay} ({direction}) | "
        f"bias_score={genome.bias_score}"
    )
    return child


# ══════════════════════════════════════════════════════
# CROSSOVER ENTRE DOS H
# ══════════════════════════════════════════════════════

def crossover(
    parent_a: PredictorGenome,
    parent_b: PredictorGenome,
) -> PredictorGenome:
    """
    Combina features de dos genomas.
    El hijo hereda el horizonte del padre A.
    El decay se hereda del padre con mejor hit_rate.
    """
    feats_a = set(parent_a.features)
    feats_b = set(parent_b.features)

    common     = feats_a & feats_b
    only_a     = feats_a - feats_b
    only_b     = feats_b - feats_a
    inherited_a = {f for f in only_a if random.random() > 0.5}
    inherited_b = {f for f in only_b if random.random() > 0.5}

    child_features = list(common | inherited_a | inherited_b)

    if len(child_features) > MAX_FEATURES:
        child_features = random.sample(child_features, MAX_FEATURES)
    elif len(child_features) < MIN_FEATURES:
        available = [f for f in _all_features() if f not in child_features]
        child_features += random.sample(available, MIN_FEATURES - len(child_features))

    hit_a  = parent_a.hit_rate or 0.5
    hit_b  = parent_b.hit_rate or 0.5
    params = deepcopy(parent_a.model_params if hit_a >= hit_b else parent_b.model_params)

    new_data = deepcopy(parent_a.to_dict())
    new_data["features"]      = child_features
    new_data["model_params"]  = params
    new_data["genome_id"]     = _next_genome_id(
        f"H{parent_a.horizon}_v{max(parent_a.generation, parent_b.generation)}"
    )
    new_data["generation"]    = max(parent_a.generation, parent_b.generation) + 1
    new_data["parent_ids"]    = [parent_a.genome_id, parent_b.genome_id]
    new_data["fitness"]       = None
    new_data["hit_rate"]      = None
    new_data["bias_score"]    = None
    new_data["created_at"]    = datetime.now(timezone.utc).isoformat()
    new_data["n_evaluations"] = 0
    new_data["crossover"]     = {
        "parent_a":          parent_a.genome_id,
        "parent_b":          parent_b.genome_id,
        "common_features":   len(common),
        "inherited_from_a":  len(inherited_a),
        "inherited_from_b":  len(inherited_b),
    }

    child = PredictorGenome.from_dict(new_data)
    logger.info(
        f"🔀 Crossover H{parent_a.horizon} | "
        f"{parent_a.genome_id} × {parent_b.genome_id} → {child.genome_id} | "
        f"features={len(child_features)}"
    )
    return child


# ══════════════════════════════════════════════════════
# GENERACIÓN COMPLETA
# ══════════════════════════════════════════════════════

def generate_children(
    champion: PredictorGenome,
    shadow_genomes: List[PredictorGenome],
    n_children: int = 4,
) -> List[PredictorGenome]:
    """
    Genera la próxima generación para un H específico.

    Prioridad de mutación:
      1. [SW2] bias_score >= 0.70 → mutate_decay(up) prioritario
         El modelo tiene sesgo alcista sistemático en período bajista.
         Dar más peso a datos recientes para que aprenda el régimen actual.
      2. hit_rate < 48% → mutación agresiva de features
      3. hit_rate 48-52% → mutación de parámetros
      4. hit_rate > 52% → exploración conservadora + crossover

    Para H10: mutate_decay nunca se llama (usa walk-forward propio).
    """
    hit      = champion.hit_rate or 0.50
    bias     = champion.bias_score
    children = []

    # ── [SW2] PRIORIDAD: bias temporal detectado ──────────────────
    if bias is not None and bias >= BIAS_SCORE_THRESHOLD and champion.horizon < 10:
        logger.info(
            f"⚠️ H{champion.horizon} bias_score={bias:.2f} >= {BIAS_SCORE_THRESHOLD} "
            f"→ mutación de decay prioritaria"
        )
        # Hijo 1: subir decay (corrección principal)
        children.append(mutate_decay(champion, direction="up"))
        # Hijo 2: subir decay + mutar feature (exploración combinada)
        child_feat = mutate_features(champion, strength="normal")
        child_combo = mutate_decay(child_feat, direction="up")
        children.append(child_combo)
        # Hijo 3: mutar parámetros (alternativa ortogonal)
        children.append(mutate_params(champion))
        # Hijo 4: mutar features agresivamente (por si el problema es de señal)
        children.append(mutate_features(champion, strength="aggressive"))

        return children[:n_children]

    # ── ESTRATEGIA NORMAL BASADA EN HIT RATE ─────────────────────
    if hit < 0.48:
        logger.info(f"🔥 H{champion.horizon} hit={hit:.2%} → mutación agresiva")
        children.append(mutate_features(champion, strength="aggressive"))
        children.append(mutate_features(champion, strength="aggressive"))
        children.append(mutate_params(champion))
        children.append(mutate_features(champion, strength="normal"))

    elif hit < 0.52:
        logger.info(f"🟡 H{champion.horizon} hit={hit:.2%} → mutación moderada")
        children.append(mutate_params(champion))
        children.append(mutate_features(champion, strength="normal"))
        children.append(mutate_params(champion))
        children.append(mutate_features(champion, strength="conservative"))

    else:
        logger.info(f"✅ H{champion.horizon} hit={hit:.2%} → exploración conservadora")
        children.append(mutate_features(champion, strength="conservative"))
        children.append(mutate_params(champion))

        if shadow_genomes:
            best_shadow = max(shadow_genomes, key=lambda g: g.hit_rate or 0.0)
            if best_shadow.hit_rate and best_shadow.hit_rate > 0.48:
                children.append(crossover(champion, best_shadow))
            else:
                children.append(mutate_features(champion, strength="normal"))
        else:
            children.append(mutate_features(champion, strength="normal"))

        # [SW1] Si el decay está activo y el hit rate es bueno,
        # probar reducirlo gradualmente (quizás ya no es necesario)
        current_decay = champion.model_params.get("sample_weight_decay", 0.0)
        if current_decay > 0.0 and champion.horizon < 10:
            children.append(mutate_decay(champion, direction="down"))
        else:
            children.append(mutate_features(champion, strength="aggressive"))

    return children[:n_children]
      

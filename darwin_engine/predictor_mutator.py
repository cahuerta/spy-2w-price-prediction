"""
darwin_engine/predictor_mutator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHIVO NUEVO — darwin_engine/predictor_mutator.py

Muta y cruza PredictorGenomes para evolucionar H1-H10.

Operaciones:
  1. mutate_features   → agrega/quita/reemplaza features
  2. mutate_params     → cambia alpha_ridge, max_pca, clip_ret
  3. crossover         → combina features de dos H distintos
  4. generate_children → genera nueva generación completa

Sesgo inteligente:
  - hit_rate < 48% → muta features agresivamente
  - hit_rate 48-52% → muta parámetros del modelo
  - hit_rate > 52% → exploración conservadora
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
    new_data     = deepcopy(genome.to_dict())
    current_feats = list(new_data["features"])
    all_feats     = _all_features()
    available     = [f for f in all_feats if f not in current_feats]

    n_changes = {"conservative": 1, "normal": random.randint(1, 2), "aggressive": random.randint(2, 3)}
    n         = n_changes.get(strength, 1)

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

    new_data["features"]   = current_feats
    new_data["genome_id"]  = _next_genome_id(genome.genome_id)
    new_data["generation"] = genome.generation + 1
    new_data["parent_ids"] = [genome.genome_id]
    new_data["fitness"]    = None
    new_data["hit_rate"]   = None
    new_data["created_at"] = datetime.now(timezone.utc).isoformat()
    new_data["mutations"]  = mutations
    new_data["n_evaluations"] = 0

    child = PredictorGenome.from_dict(new_data)
    logger.info(
        f"🧬 Feature mutation H{genome.horizon} | "
        f"{genome.genome_id} → {child.genome_id} | "
        f"{mutations}"
    )
    return child


# ══════════════════════════════════════════════════════
# MUTACIÓN DE PARÁMETROS
# ══════════════════════════════════════════════════════

def mutate_params(genome: PredictorGenome) -> PredictorGenome:
    """
    Muta los parámetros del modelo (alpha_ridge, max_pca, clip_ret).
    """
    new_data   = deepcopy(genome.to_dict())
    params     = new_data["model_params"]
    param_name = random.choice(list(PARAM_LIMITS.keys()))
    lo, hi     = PARAM_LIMITS[param_name]

    current = params[param_name]
    delta   = (hi - lo) * 0.15 * random.choice([-1, 1])

    if isinstance(current, int):
        new_val = int(_clamp(round(current + delta), lo, hi))
    else:
        new_val = round(_clamp(current + delta, lo, hi), 3)

    params[param_name]      = new_val
    new_data["model_params"] = params
    new_data["genome_id"]   = _next_genome_id(genome.genome_id)
    new_data["generation"]  = genome.generation + 1
    new_data["parent_ids"]  = [genome.genome_id]
    new_data["fitness"]     = None
    new_data["hit_rate"]    = None
    new_data["created_at"]  = datetime.now(timezone.utc).isoformat()
    new_data["mutations"]   = [f"PARAM:{param_name}:{current}→{new_val}"]
    new_data["n_evaluations"] = 0

    child = PredictorGenome.from_dict(new_data)
    logger.info(
        f"⚙️ Param mutation H{genome.horizon} | "
        f"{genome.genome_id} → {child.genome_id} | "
        f"{param_name}: {current} → {new_val}"
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

    Útil para cruzar H6 (mejor hit rate) con H7 o H8
    para transferir features exitosas.
    """
    feats_a = set(parent_a.features)
    feats_b = set(parent_b.features)

    # Features comunes → siempre heredadas
    common = feats_a & feats_b

    # Features únicas → heredar aleatoriamente
    only_a = feats_a - feats_b
    only_b = feats_b - feats_a

    inherited_a = {f for f in only_a if random.random() > 0.5}
    inherited_b = {f for f in only_b if random.random() > 0.5}

    child_features = list(common | inherited_a | inherited_b)

    # Respetar límites
    if len(child_features) > MAX_FEATURES:
        child_features = random.sample(child_features, MAX_FEATURES)
    elif len(child_features) < MIN_FEATURES:
        available = [f for f in _all_features() if f not in child_features]
        child_features += random.sample(available, MIN_FEATURES - len(child_features))

    # Parámetros: hereda del padre con mejor hit rate
    hit_a = parent_a.hit_rate or 0.5
    hit_b = parent_b.hit_rate or 0.5
    params = deepcopy(parent_a.model_params if hit_a >= hit_b else parent_b.model_params)

    new_data = deepcopy(parent_a.to_dict())
    new_data["features"]    = child_features
    new_data["model_params"] = params
    new_data["genome_id"]   = _next_genome_id(
        f"H{parent_a.horizon}_v{max(parent_a.generation, parent_b.generation)}"
    )
    new_data["generation"]  = max(parent_a.generation, parent_b.generation) + 1
    new_data["parent_ids"]  = [parent_a.genome_id, parent_b.genome_id]
    new_data["fitness"]     = None
    new_data["hit_rate"]    = None
    new_data["created_at"]  = datetime.now(timezone.utc).isoformat()
    new_data["crossover"]   = {
        "parent_a": parent_a.genome_id,
        "parent_b": parent_b.genome_id,
        "common_features": len(common),
        "inherited_from_a": len(inherited_a),
        "inherited_from_b": len(inherited_b),
    }
    new_data["n_evaluations"] = 0

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

    Estrategia basada en hit rate actual:
      < 48%: mutación agresiva de features (el modelo no sirve)
      48-52%: mutación de parámetros (el modelo es mediocre)
      > 52%: exploración conservadora + crossover (el modelo es bueno)
    """
    hit = champion.hit_rate or 0.50
    children = []

    if hit < 0.48:
        # Mutación agresiva — el modelo necesita cambios grandes
        logger.info(f"🔥 H{champion.horizon} hit={hit:.2%} → mutación agresiva")
        children.append(mutate_features(champion, strength="aggressive"))
        children.append(mutate_features(champion, strength="aggressive"))
        children.append(mutate_params(champion))
        children.append(mutate_features(champion, strength="normal"))

    elif hit < 0.52:
        # Mutación moderada — ajustar parámetros
        logger.info(f"🟡 H{champion.horizon} hit={hit:.2%} → mutación moderada")
        children.append(mutate_params(champion))
        children.append(mutate_features(champion, strength="normal"))
        children.append(mutate_params(champion))
        children.append(mutate_features(champion, strength="conservative"))

    else:
        # Buen hit rate — explorar y cruzar
        logger.info(f"✅ H{champion.horizon} hit={hit:.2%} → exploración conservadora")
        children.append(mutate_features(champion, strength="conservative"))
        children.append(mutate_params(champion))

        # Crossover con el mejor shadow si existe
        if shadow_genomes:
            best_shadow = max(
                shadow_genomes,
                key=lambda g: g.hit_rate or 0.0
            )
            if best_shadow.hit_rate and best_shadow.hit_rate > 0.48:
                children.append(crossover(champion, best_shadow))
            else:
                children.append(mutate_features(champion, strength="normal"))
        else:
            children.append(mutate_features(champion, strength="normal"))

        children.append(mutate_features(champion, strength="aggressive"))

    return children[:n_children]

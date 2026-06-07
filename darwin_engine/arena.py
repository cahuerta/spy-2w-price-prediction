"""
darwin_engine/arena.py
━━━━━━━━━━━━━━━━━━━━━━
FIX v1.1:
  [F4] _select_champion: math.isfinite() explícito para
       champion_fitness e improvement — numpy evalúa
       nan >= umbral como False, dejando al campeón con
       fitness=nan inamovible indefinidamente.
       Ahora nan → 0.0 para comparación, permitiendo que
       candidatos con datos reales puedan desbancarlo.
"""

import json
import logging
import math
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from darwin_engine.executor_genome import ExecutorGenome, list_executor_genomes, GENOME_DIR
from darwin_engine.pnl_fitness import (
    compute_executor_fitness,
    compute_predictor_fitness,
    compute_combined_fitness,
    rank_all_genomes,
    get_fitness_summary,
)
from darwin_engine.mutator import generate_next_generation, random_genome
from darwin_engine.trade_tracker import get_resolved_trades, get_tracker_summary

logger = logging.getLogger("arena")

DATA_PATH     = Path(os.getenv("DATA_PATH", "/data"))
ARENA_DIR     = DATA_PATH / "darwin" / "arena"
CHAMPION_FILE = DATA_PATH / "darwin" / "champion.json"
HISTORY_FILE  = DATA_PATH / "darwin" / "evolution_history.json"

MIN_TRADES_TO_COMPETE = int(os.getenv("DARWIN_MIN_TRADES",   "10"))
MIN_IMPROVEMENT_PCT   = float(os.getenv("DARWIN_MIN_IMPROVE", "0.05"))
MAX_SHADOW_GENOMES    = int(os.getenv("DARWIN_MAX_SHADOW",    "8"))
GENERATION_SIZE       = int(os.getenv("DARWIN_GEN_SIZE",      "5"))
GITHUB_ENABLED        = os.getenv("GITHUB_TOKEN") is not None


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


def _append_history(entry: Dict) -> None:
    history = _load_json(HISTORY_FILE)
    events  = history.get("events", [])
    events.append(entry)
    history["events"]      = events[-200:]
    history["last_update"] = datetime.now(timezone.utc).isoformat()
    _save_json(HISTORY_FILE, history)


# ══════════════════════════════════════════════════════
# GITHUB: escribir campeón al repo
# ══════════════════════════════════════════════════════

def _write_champion_to_github(genome: ExecutorGenome) -> bool:
    if not GITHUB_ENABLED:
        logger.info("⚠️ GitHub no configurado — campeón solo guardado en disco")
        return False

    try:
        import base64
        import httpx

        token  = os.getenv("GITHUB_TOKEN")
        owner  = os.getenv("GITHUB_OWNER")
        repo   = os.getenv("GITHUB_REPO")
        branch = os.getenv("GITHUB_BRANCH", "main")

        if not all([token, owner, repo]):
            logger.warning("⚠️ GITHUB_OWNER o GITHUB_REPO no configurados")
            return False

        file_path = "darwin_engine/genomes/champion.json"
        url       = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        headers   = {
            "Authorization": f"Bearer {token}",
            "Accept":        "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        content = json.dumps(genome.to_dict(), indent=2, ensure_ascii=False)
        encoded = base64.b64encode(content.encode()).decode()

        sha = None
        r   = httpx.get(url, headers=headers, params={"ref": branch})
        if r.status_code == 200:
            sha = r.json().get("sha")

        payload = {
            "message":   f"darwin: promote {genome.genome_id} gen={genome.generation} fit={genome.fitness:.4f}",
            "content":   encoded,
            "branch":    branch,
            "committer": {"name": "Darwin Engine", "email": "darwin@quantenterprise.cl"},
        }
        if sha:
            payload["sha"] = sha

        resp = httpx.put(url, headers=headers, json=payload)
        resp.raise_for_status()

        logger.info(f"✅ Campeón escrito al repo: {genome.genome_id}")
        return True

    except Exception as e:
        logger.error(f"❌ Error escribiendo al repo: {e}")
        return False


# ══════════════════════════════════════════════════════
# CARGAR GENOMAS ACTIVOS
# ══════════════════════════════════════════════════════

def _load_all_active_genomes() -> Tuple[Optional[ExecutorGenome], List[ExecutorGenome]]:
    champion_data = _load_json(CHAMPION_FILE)
    champion      = ExecutorGenome.from_dict(champion_data) if champion_data else None

    if champion is None:
        logger.info("🆕 Sin campeón previo — cargando o creando default")
        champion = ExecutorGenome.load_or_default("executor_v1")

    shadow = []
    if GENOME_DIR.exists():
        for path in GENOME_DIR.glob("executor_v*.json"):
            try:
                data = json.loads(path.read_text())
                g    = ExecutorGenome.from_dict(data)
                if g.genome_id != champion.genome_id:
                    shadow.append(g)
            except Exception:
                continue

    logger.info(
        f"📦 Genomas cargados | champion={champion.genome_id} | "
        f"shadow={len(shadow)}"
    )
    return champion, shadow


# ══════════════════════════════════════════════════════
# EVALUAR TODOS LOS GENOMAS
# ══════════════════════════════════════════════════════

def _evaluate_all_genomes(
    champion: ExecutorGenome,
    shadow: List[ExecutorGenome],
    predictor_genome_id: str = "predictor_v1",
) -> Dict[str, Dict]:
    results = {}

    all_genomes = [champion] + shadow
    for genome in all_genomes:
        try:
            fitness_result = compute_combined_fitness(
                predictor_genome_id = predictor_genome_id,
                executor_genome_id  = genome.genome_id,
            )
            results[genome.genome_id] = fitness_result

            combined = fitness_result.get("combined_fitness", 0.0)
            # [F4] No persistir nan en el objeto genome
            genome.fitness = combined if (combined is not None and math.isfinite(combined)) else 0.0
            genome.save()

        except Exception as e:
            logger.error(f"❌ Error evaluando {genome.genome_id}: {e}")
            results[genome.genome_id] = {
                "combined_fitness": 0.0,
                "status": f"error: {e}",
            }

    return results


# ══════════════════════════════════════════════════════
# SELECCIÓN: ¿PROMOVER NUEVO CAMPEÓN?  [F4]
# ══════════════════════════════════════════════════════

def _select_champion(
    current_champion: ExecutorGenome,
    candidates: List[ExecutorGenome],
    fitness_results: Dict[str, Dict],
) -> Tuple[ExecutorGenome, bool]:
    """
    [F4] math.isfinite() explícito — numpy evalúa nan >= umbral
    como False, dejando al campeón inamovible cuando fitness=nan.
    Ahora nan → 0.0 para comparación.
    """
    raw_champion_fitness = fitness_results.get(
        current_champion.genome_id, {}
    ).get("combined_fitness", 0.0)

    # [F4] nan/inf → 0.0
    champion_fitness = (
        raw_champion_fitness
        if (raw_champion_fitness is not None and math.isfinite(raw_champion_fitness))
        else 0.0
    )
    if raw_champion_fitness != champion_fitness:
        logger.warning(
            f"⚠️ {current_champion.genome_id} fitness={raw_champion_fitness} → "
            f"tratado como {champion_fitness} para comparación"
        )

    champion_trades = len(
        get_resolved_trades(executor_genome_id=current_champion.genome_id)
    )

    best_candidate = None
    best_fitness   = float("-inf")

    for candidate in candidates:
        if candidate.genome_id == current_champion.genome_id:
            continue

        raw_cand_fitness = fitness_results.get(
            candidate.genome_id, {}
        ).get("combined_fitness", 0.0)

        # [F4] Candidato con nan no puede ganar
        if raw_cand_fitness is None or not math.isfinite(raw_cand_fitness):
            logger.warning(
                f"⚠️ {candidate.genome_id} fitness={raw_cand_fitness} → saltado"
            )
            continue

        cand_trades = len(
            get_resolved_trades(executor_genome_id=candidate.genome_id)
        )

        if cand_trades < MIN_TRADES_TO_COMPETE:
            logger.info(
                f"⏳ {candidate.genome_id} insuficientes trades "
                f"({cand_trades}/{MIN_TRADES_TO_COMPETE})"
            )
            continue

        if raw_cand_fitness > best_fitness:
            best_fitness   = raw_cand_fitness
            best_candidate = candidate

    if best_candidate is None:
        logger.info(f"👑 Campeón actual se mantiene: {current_champion.genome_id}")
        return current_champion, False

    if champion_trades < MIN_TRADES_TO_COMPETE:
        logger.info(
            f"🔄 Promoviendo {best_candidate.genome_id} "
            f"(campeón sin datos suficientes)"
        )
        return best_candidate, True

    improvement = best_fitness - champion_fitness
    if improvement >= MIN_IMPROVEMENT_PCT:
        logger.info(
            f"🏆 NUEVO CAMPEÓN: {best_candidate.genome_id} | "
            f"fitness={best_fitness:.4f} vs actual={champion_fitness:.4f} "
            f"(mejora={improvement:+.4f})"
        )
        return best_candidate, True
    else:
        logger.info(
            f"👑 Campeón se mantiene: {current_champion.genome_id} | "
            f"mejor candidato={best_candidate.genome_id} "
            f"mejora={improvement:+.4f} < mínimo={MIN_IMPROVEMENT_PCT}"
        )
        return current_champion, False


# ══════════════════════════════════════════════════════
# CONTEXTO DE FITNESS PARA SESGO INTELIGENTE
# ══════════════════════════════════════════════════════

def _build_fitness_context() -> Dict:
    context = {}

    try:
        eval_dir     = DATA_PATH / "evaluations"
        compra_hits  = 0
        compra_total = 0

        if eval_dir.exists():
            for ticker_dir in list(eval_dir.iterdir())[:50]:
                if not ticker_dir.is_dir():
                    continue
                for ef in list(ticker_dir.glob("*.json"))[-5:]:
                    ev  = _load_json(ef)
                    rec = ev.get("recommendation", "")
                    if rec == "COMPRA":
                        compra_total += 1
                        if ev.get("decision_correct"):
                            compra_hits += 1

        if compra_total > 0:
            context["compra_hit_rate"] = compra_hits / compra_total
    except Exception as e:
        logger.debug(f"fitness_context compra: {e}")

    try:
        resolved = get_resolved_trades(last_n=50)
        oportunidades = [
            float(t.get("oportunidad_pct") or 0)
            for t in resolved
            if t.get("closed_before_horizon") and t.get("oportunidad_pct") is not None
        ]
        if oportunidades:
            context["avg_oportunidad_perdida"] = float(np.mean([
                o for o in oportunidades if o > 0
            ] or [0]))
    except Exception as e:
        logger.debug(f"fitness_context oportunidad: {e}")

    try:
        resolved = get_resolved_trades(last_n=100)
        returns  = [
            float(t.get("pnl_real_pct") or 0)
            for t in resolved
            if t.get("pnl_real_pct") is not None and math.isfinite(float(t.get("pnl_real_pct") or 0))
        ]
        if returns:
            equity = np.cumprod(1 + np.array(returns) / 100)
            peak   = np.maximum.accumulate(equity)
            dd     = (equity - peak) / peak
            context["max_drawdown"] = float(abs(dd.min()))
    except Exception as e:
        logger.debug(f"fitness_context drawdown: {e}")

    logger.info(f"📊 Fitness context: {context}")
    return context


# ══════════════════════════════════════════════════════
# CICLO PRINCIPAL DEL ARENA
# ══════════════════════════════════════════════════════

def run_evolution_cycle(
    predictor_genome_id: str = "predictor_v1",
    dry_run: bool = False,
) -> Dict:
    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("🧬 DARWIN ENGINE — CICLO EVOLUTIVO")
    logger.info(f"   {start_time.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 60)

    # Paso 1: Cargar genomas
    champion, shadow = _load_all_active_genomes()
    tracker_summary  = get_tracker_summary()

    logger.info(
        f"📦 Tracker | total={tracker_summary['total']} "
        f"open={tracker_summary['open']} "
        f"resueltos={tracker_summary['resolved']}"
    )

    # Paso 2: Evaluar fitness
    all_genomes     = [champion] + shadow
    fitness_results = _evaluate_all_genomes(champion, shadow, predictor_genome_id)

    ranked = sorted(
        all_genomes,
        key=lambda g: (
            fitness_results.get(g.genome_id, {}).get("combined_fitness") or float("-inf")
        ),
        reverse=True,
    )

    logger.info("🏆 Ranking actual:")
    for i, g in enumerate(ranked[:5]):
        fit    = fitness_results.get(g.genome_id, {}).get("combined_fitness", 0)
        trades = len(get_resolved_trades(executor_genome_id=g.genome_id))
        logger.info(f"   {i+1}. {g.genome_id} | fitness={fit:.4f} | trades={trades}")

    # Paso 3: Seleccionar campeón
    new_champion, was_promoted = _select_champion(champion, shadow, fitness_results)

    # Paso 4: Promover si corresponde
    github_written = False
    if was_promoted and not dry_run:
        _save_json(CHAMPION_FILE, new_champion.to_dict())
        logger.info(f"👑 Nuevo campeón guardado: {new_champion.genome_id}")
        github_written = _write_champion_to_github(new_champion)

    # Paso 5: Generar nueva generación
    fitness_context = _build_fitness_context()
    new_generation  = generate_next_generation(
        ranked_genomes  = ranked,
        generation_size = GENERATION_SIZE,
        fitness_context = fitness_context,
    )

    # Paso 6: Guardar nueva generación en shadow
    if not dry_run:
        _prune_shadow_genomes(shadow, fitness_results)
        for child in new_generation:
            child.save()
            logger.info(
                f"🌱 Shadow guardado: {child.genome_id} "
                f"(gen={child.generation} padre={child.data.get('parent_ids')})"
            )

    # Paso 7: Registrar historial
    cycle_duration = (datetime.now(timezone.utc) - start_time).total_seconds()

    cycle_summary = {
        "timestamp":         start_time.isoformat(),
        "duration_sec":      round(cycle_duration, 1),
        "champion_before":   champion.genome_id,
        "champion_after":    new_champion.genome_id,
        "was_promoted":      was_promoted,
        "github_written":    github_written,
        "dry_run":           dry_run,
        "genomes_evaluated": len(all_genomes),
        "new_generation":    [g.genome_id for g in new_generation],
        "fitness_context":   fitness_context,
        "tracker_summary":   tracker_summary,
        "top_3_fitness": [
            {
                "genome_id": g.genome_id,
                "fitness":   fitness_results.get(g.genome_id, {}).get("combined_fitness", 0),
            }
            for g in ranked[:3]
        ],
    }

    if not dry_run:
        _append_history(cycle_summary)

    logger.info("=" * 60)
    logger.info(
        f"✅ CICLO COMPLETADO | {cycle_duration:.1f}s | "
        f"campeón={'NUEVO: ' + new_champion.genome_id if was_promoted else new_champion.genome_id} | "
        f"nueva_gen={len(new_generation)} genomas"
    )
    logger.info("=" * 60)

    return cycle_summary


# ══════════════════════════════════════════════════════
# LIMPIEZA DE SHADOW GENOMAS
# ══════════════════════════════════════════════════════

def _prune_shadow_genomes(
    shadow: List[ExecutorGenome],
    fitness_results: Dict,
) -> None:
    if len(shadow) <= MAX_SHADOW_GENOMES:
        return

    ranked_shadow = sorted(
        shadow,
        key=lambda g: (
            fitness_results.get(g.genome_id, {}).get("combined_fitness") or float("-inf")
        ),
        reverse=True,
    )

    to_archive  = ranked_shadow[MAX_SHADOW_GENOMES:]
    archive_dir = GENOME_DIR / "archived"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for genome in to_archive:
        src = GENOME_DIR / f"{genome.genome_id}.json"
        dst = archive_dir / f"{genome.genome_id}.json"
        if src.exists():
            src.rename(dst)
            logger.info(
                f"📦 Archivado: {genome.genome_id} | "
                f"fitness={fitness_results.get(genome.genome_id, {}).get('combined_fitness', 0):.4f}"
            )


# ══════════════════════════════════════════════════════
# ESTADO DEL ARENA — para dashboard
# ══════════════════════════════════════════════════════

def get_arena_status() -> Dict:
    champion_data = _load_json(CHAMPION_FILE)
    history       = _load_json(HISTORY_FILE)
    tracker_sum   = get_tracker_summary()
    fitness_sum   = get_fitness_summary()

    events     = history.get("events", [])
    promotions = [e for e in events if e.get("was_promoted")]

    return {
        "champion": {
            "genome_id":  champion_data.get("genome_id", "executor_v1"),
            "generation": champion_data.get("generation", 1),
            "fitness":    champion_data.get("fitness"),
            "created_at": champion_data.get("created_at"),
        },
        "evolution": {
            "total_cycles":     len(events),
            "total_promotions": len(promotions),
            "last_cycle":       events[-1].get("timestamp") if events else None,
            "last_promotion":   promotions[-1].get("timestamp") if promotions else None,
        },
        "tracker": tracker_sum,
        "fitness": fitness_sum,
        "config": {
            "min_trades_to_compete": MIN_TRADES_TO_COMPETE,
            "min_improvement_pct":   MIN_IMPROVEMENT_PCT,
            "max_shadow_genomes":    MAX_SHADOW_GENOMES,
            "generation_size":       GENERATION_SIZE,
            "github_enabled":        GITHUB_ENABLED,
        },
    }


def get_champion() -> Optional[ExecutorGenome]:
    """Retorna el genome campeón actual. Para dashboard y status."""
    champion_data = _load_json(CHAMPION_FILE)
    if champion_data:
        return ExecutorGenome.from_dict(champion_data)
    return ExecutorGenome.load_or_default("executor_v1")


# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        logger.info("🔍 DRY RUN — no se promoverá ni escribirá al repo")
    result = run_evolution_cycle(dry_run=dry_run)
    print(json.dumps(result, indent=2, default=str))

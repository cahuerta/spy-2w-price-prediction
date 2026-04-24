"""
darwin_engine/predictor_arena.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHIVO NUEVO — darwin_engine/predictor_arena.py

Ciclo evolutivo para los predictores H1-H10.

Ciclo semanal (viernes, después del executor arena):
  1. Leer hit rates reales del evaluator para H1-H10
  2. Actualizar fitness de cada genome campeón
  3. Comparar campeón vs shadow de cada H
  4. Promover si hay mejor hit rate con suficientes evaluaciones
  5. Generar nueva generación shadow
  6. Escribir campeones al repo vía GitHub API

Los predictores H leen su genoma activo automáticamente.
Sin genoma → usan parámetros originales (retrocompatible).
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from darwin_engine.predictor_genome import (
    PredictorGenome,
    load_active_genome,
    update_genome_hit_rate,
    initialize_all_genomes,
    GENOME_BASE,
)
from darwin_engine.predictor_mutator import generate_children

logger = logging.getLogger("predictor_arena")

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
EVAL_DIR  = DATA_PATH / "evaluations"

# Configuración
MIN_EVALS_TO_COMPETE   = int(os.getenv("PRED_MIN_EVALS",    "30"))
MIN_HIT_IMPROVEMENT    = float(os.getenv("PRED_MIN_IMPROVE", "0.015"))  # 1.5% mejora mínima
MAX_SHADOW_PER_H       = int(os.getenv("PRED_MAX_SHADOW",    "4"))
GITHUB_ENABLED         = os.getenv("GITHUB_TOKEN") is not None


# ══════════════════════════════════════════════════════
# LEER HIT RATES REALES DEL EVALUATOR
# ══════════════════════════════════════════════════════

def _read_hit_rates_from_evaluator() -> Dict[int, Tuple[float, int]]:
    """
    Lee las evaluaciones existentes y calcula hit rate por H.
    Retorna {horizon: (hit_rate, n_evaluaciones)}
    """
    hit_counts   = defaultdict(int)
    eval_counts  = defaultdict(int)

    if not EVAL_DIR.exists():
        logger.warning("⚠️ Sin directorio de evaluaciones")
        return {}

    # Muestrear evaluaciones recientes (últimas 60 días por ticker)
    for ticker_dir in EVAL_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue
        files = sorted(ticker_dir.glob("*.json"))[-60:]
        for f in files:
            try:
                ev   = json.loads(f.read_text())
                diag = ev.get("models_diagnostics") or {}
                for h in range(1, 11):
                    key  = f"H{h}"
                    data = diag.get(key)
                    if not isinstance(data, dict):
                        continue
                    hit_sign = data.get("hit_sign")
                    if hit_sign is None:
                        continue
                    eval_counts[h] += 1
                    if hit_sign:
                        hit_counts[h] += 1
            except Exception:
                continue

    result = {}
    for h in range(1, 11):
        total = eval_counts.get(h, 0)
        if total >= 10:
            result[h] = (round(hit_counts[h] / total, 4), total)

    logger.info(f"📊 Hit rates leídos: {result}")
    return result


# ══════════════════════════════════════════════════════
# EVALUAR SHADOW GENOMES
# ══════════════════════════════════════════════════════

def _evaluate_shadow_hit_rates(horizon: int) -> List[Dict]:
    """
    Lee hit rates de los genomas shadow de un H.
    Los shadow genomas guardan su propio historial de evaluaciones
    en predictor_genomes/H{n}/shadow/evals/{genome_id}.json
    """
    shadow_eval_dir = GENOME_BASE / f"H{horizon}" / "shadow" / "evals"
    if not shadow_eval_dir.exists():
        return []

    results = []
    for path in shadow_eval_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            results.append(data)
        except Exception:
            continue

    return results


# ══════════════════════════════════════════════════════
# ESCRIBIR CAMPEÓN AL REPO
# ══════════════════════════════════════════════════════

def _write_genome_to_github(genome: PredictorGenome) -> bool:
    if not GITHUB_ENABLED:
        return False
    try:
        import base64
        import httpx

        token  = os.getenv("GITHUB_TOKEN")
        owner  = os.getenv("GITHUB_OWNER")
        repo   = os.getenv("GITHUB_REPO")
        branch = os.getenv("GITHUB_BRANCH", "main")

        if not all([token, owner, repo]):
            return False

        file_path = f"predictor_genomes/H{genome.horizon}/champion.json"
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
            "message":   f"darwin: H{genome.horizon} promote {genome.genome_id} hit={genome.hit_rate:.3f}",
            "content":   encoded,
            "branch":    branch,
            "committer": {"name": "Darwin Engine", "email": "darwin@quantenterprise.cl"},
        }
        if sha:
            payload["sha"] = sha

        resp = httpx.put(url, headers=headers, json=payload)
        resp.raise_for_status()
        logger.info(f"✅ H{genome.horizon} champion escrito al repo: {genome.genome_id}")
        return True

    except Exception as e:
        logger.error(f"❌ GitHub write H{genome.horizon}: {e}")
        return False


# ══════════════════════════════════════════════════════
# CICLO EVOLUTIVO POR H
# ══════════════════════════════════════════════════════

def _run_h_cycle(
    horizon: int,
    hit_rate: float,
    n_evals: int,
    dry_run: bool = False,
) -> Dict:
    """Ciclo evolutivo para un H específico."""

    champion = PredictorGenome.load_champion(horizon)
    shadow   = PredictorGenome.load_shadow_genomes(horizon)

    # Actualizar hit rate del campeón
    champion.hit_rate = hit_rate
    champion.data["n_evaluations"] = n_evals

    # ¿Hay shadow con mejor hit rate?
    was_promoted  = False
    github_written = False

    shadow_evals = _evaluate_shadow_hit_rates(horizon)
    best_shadow  = None
    best_shadow_hit = 0.0

    for se in shadow_evals:
        shadow_hit   = float(se.get("hit_rate", 0))
        shadow_n     = int(se.get("n_evaluations", 0))
        shadow_id    = se.get("genome_id")

        if shadow_n < MIN_EVALS_TO_COMPETE:
            continue
        if shadow_hit > best_shadow_hit:
            best_shadow_hit = shadow_hit
            best_shadow     = se

    if (
        best_shadow is not None
        and best_shadow_hit - hit_rate >= MIN_HIT_IMPROVEMENT
        and not dry_run
    ):
        # Promover shadow a campeón
        new_champion_data = None
        for path in (GENOME_BASE / f"H{horizon}" / "shadow").glob("*.json"):
            try:
                d = json.loads(path.read_text())
                if d.get("genome_id") == best_shadow.get("genome_id"):
                    new_champion_data = d
                    break
            except Exception:
                continue

        if new_champion_data:
            new_champion          = PredictorGenome.from_dict(new_champion_data)
            new_champion.hit_rate = best_shadow_hit
            new_champion.save_as_champion()
            champion       = new_champion
            was_promoted   = True
            github_written = _write_genome_to_github(new_champion)
            logger.info(
                f"🏆 H{horizon} NUEVO CAMPEÓN: {new_champion.genome_id} | "
                f"hit={best_shadow_hit:.2%} vs anterior={hit_rate:.2%}"
            )
    else:
        if not dry_run:
            champion.save_as_champion()
            if was_promoted or hit_rate > (champion.hit_rate or 0):
                github_written = _write_genome_to_github(champion)

    # Generar nueva generación shadow
    new_children = []
    if not dry_run:
        children = generate_children(champion, shadow, n_children=MAX_SHADOW_PER_H)
        for child in children:
            child.save_as_shadow()
            new_children.append(child.genome_id)

        # Limpiar shadow viejos
        _prune_shadow(horizon)

    return {
        "horizon":       horizon,
        "hit_rate":      hit_rate,
        "n_evals":       n_evals,
        "champion":      champion.genome_id,
        "was_promoted":  was_promoted,
        "github_written": github_written,
        "new_shadow":    new_children,
    }


# ══════════════════════════════════════════════════════
# LIMPIEZA SHADOW
# ══════════════════════════════════════════════════════

def _prune_shadow(horizon: int) -> None:
    """Mantiene solo los MAX_SHADOW_PER_H shadow más recientes."""
    shadow_dir = GENOME_BASE / f"H{horizon}" / "shadow"
    if not shadow_dir.exists():
        return

    files = sorted(shadow_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    excess = len(files) - MAX_SHADOW_PER_H
    if excess > 0:
        for f in files[:excess]:
            try:
                # Archivar en vez de borrar
                archive = shadow_dir / "archived" / f.name
                archive.parent.mkdir(exist_ok=True)
                f.rename(archive)
            except Exception:
                pass


# ══════════════════════════════════════════════════════
# CICLO PRINCIPAL — TODOS LOS H
# ══════════════════════════════════════════════════════

def run_predictor_evolution(dry_run: bool = False) -> Dict:
    """
    Ciclo evolutivo completo para H1-H10.
    Llamar desde arena.py o scheduler.py (viernes post-market).
    """
    start = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("🧬 PREDICTOR ARENA — CICLO EVOLUTIVO H1-H10")
    logger.info(f"   {start.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 60)

    # Asegurar que existen todos los genomas
    initialize_all_genomes()

    # Leer hit rates reales
    hit_rates = _read_hit_rates_from_evaluator()

    if not hit_rates:
        logger.warning("⚠️ Sin hit rates disponibles — ciclo abortado")
        return {"status": "no_data", "timestamp": start.isoformat()}

    results    = {}
    promotions = 0

    for h in range(1, 11):
        if h not in hit_rates:
            logger.info(f"⏭ H{h}: sin suficientes evaluaciones — saltando")
            continue

        hit_rate, n_evals = hit_rates[h]
        logger.info(f"🔬 H{h}: hit_rate={hit_rate:.2%} | n_evals={n_evals}")

        try:
            result      = _run_h_cycle(h, hit_rate, n_evals, dry_run)
            results[f"H{h}"] = result
            if result.get("was_promoted"):
                promotions += 1
        except Exception as e:
            logger.error(f"❌ H{h} ciclo falló: {e}")
            results[f"H{h}"] = {"error": str(e)}

    duration = (datetime.now(timezone.utc) - start).total_seconds()

    # Ranking final
    logger.info("\n📊 RANKING H1-H10:")
    ranked = sorted(
        [(h, hit_rates[h][0]) for h in hit_rates],
        key=lambda x: x[1], reverse=True
    )
    for h, hr in ranked:
        champion = results.get(f"H{h}", {}).get("champion", "?")
        promoted = "🏆 PROMOVIDO" if results.get(f"H{h}", {}).get("was_promoted") else ""
        logger.info(f"   H{h}: {hr:.2%} | {champion} {promoted}")

    summary = {
        "status":      "ok",
        "timestamp":   start.isoformat(),
        "duration_sec": round(duration, 1),
        "dry_run":     dry_run,
        "h_evaluated": len(results),
        "promotions":  promotions,
        "results":     results,
        "ranking":     [{"horizon": h, "hit_rate": hr} for h, hr in ranked],
    }

    logger.info(
        f"\n✅ PREDICTOR ARENA COMPLETADO | {duration:.1f}s | "
        f"{len(results)} H evaluados | {promotions} promovidos"
    )
    return summary


# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    result  = run_predictor_evolution(dry_run=dry_run)
    print(json.dumps(result, indent=2, default=str))

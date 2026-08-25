"""
darwin_engine/predictor_arena.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ciclo evolutivo para los predictores H1-H10.

Ciclo semanal (viernes, después del executor arena):
  1. Leer hit rates reales del evaluator para H1-H10
  2. Actualizar fitness de cada genome campeón
  3. Comparar campeón vs shadow de cada H
  4. Promover si hay mejor hit rate con suficientes evaluaciones
  5. Generar nueva generación shadow
  6. Escribir campeones al repo vía GitHub API

FIXES:
  [SW2] Cálculo de bias_score agregado en _run_h_cycle().
        Para cada H, calcula el bias_score promedio sobre todos
        los tickers del universo y lo escribe al genome campeón.
        Darwin (generate_children) usa ese score para decidir
        si mutar sample_weight_decay.

  [AR1] _run_h_cycle: se captura previous_hit_rate ANTES de
        sobreescribir champion.hit_rate con el valor nuevo. Antes,
        la comparación `hit_rate > champion.hit_rate` en la rama
        "sin promoción" siempre era hit_rate > hit_rate (False),
        porque champion.hit_rate ya había sido pisado más arriba.
        Esto significaba que el respaldo a GitHub nunca se disparaba
        cuando el campeón mejoraba orgánicamente sin que un shadow
        lo superara. Ahora se compara contra el hit_rate real anterior.

  [AR2] (en _evaluate_shadow_hit_rates) excluye champion_baseline.json.

  [AR3] (auditoría 2026-08-25, Calibración punto 3) _run_h_cycle:
        al promover un shadow a campeón, el nuevo campeón heredaba
        `hit_rate = best_shadow_hit` (validado con shadow_n >=
        MIN_EVALS_TO_COMPETE evaluaciones simuladas) pero su propio
        campo `n_evaluations` (contador de evaluaciones REALES del
        evaluator en producción) seguía en 0 — un genoma recién
        mutado nunca operó en producción todavía. El sistema
        terminaba mostrando un hit_rate "de fábrica" como si fuera
        un resultado validado en vivo (confirmado en
        predictor_genomes/H{2,4,5,10}/champion.json: hit_rate con
        valor pero n_evaluations=0).
        Fix: si el genoma recién promovido tiene n_evaluations=0,
        se fuerza hit_rate=None explícitamente en vez de heredar el
        valor del shadow — el campeón queda "sin validar" hasta que
        el evaluator en producción le genere evaluaciones reales
        (momento en que _read_hit_rates_from_evaluator() lo
        alimentará con datos genuinos en el próximo ciclo).
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
    update_genome_bias_score,
    calc_bias_score_from_evals,
    initialize_all_genomes,
    GENOME_BASE,
    BIAS_SCORE_THRESHOLD,
)
from darwin_engine.predictor_mutator import generate_children

logger = logging.getLogger("predictor_arena")

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
EVAL_DIR  = DATA_PATH / "evaluations"

# Configuración
MIN_EVALS_TO_COMPETE = int(os.getenv("PRED_MIN_EVALS",    "30"))
MIN_HIT_IMPROVEMENT  = float(os.getenv("PRED_MIN_IMPROVE", "0.015"))
MAX_SHADOW_PER_H     = int(os.getenv("PRED_MAX_SHADOW",    "4"))
GITHUB_ENABLED       = os.getenv("GITHUB_TOKEN") is not None


# ══════════════════════════════════════════════════════
# LEER HIT RATES REALES DEL EVALUATOR
# ══════════════════════════════════════════════════════

def _read_hit_rates_from_evaluator() -> Dict[int, Tuple[float, int]]:
    """
    Lee las evaluaciones existentes y calcula hit rate por H.
    Retorna {horizon: (hit_rate, n_evaluaciones)}
    """
    hit_counts  = defaultdict(int)
    eval_counts = defaultdict(int)

    if not EVAL_DIR.exists():
        logger.warning("⚠️ Sin directorio de evaluaciones")
        return {}

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
# [SW2] CALCULAR BIAS SCORE PARA UN HORIZONTE
# ══════════════════════════════════════════════════════

def _calc_universe_bias_score(horizon: int) -> Optional[float]:
    """
    [SW2] Calcula el bias_score promedio de todos los tickers
    del universo para un horizonte dado.

    bias_score por ticker = predicciones_positivas_que_fallaron
                            / total_predicciones_positivas

    Si la mayoría de tickers tienen bias alto en este horizonte,
    significa que el predictor H{horizon} tiene sesgo alcista
    sistemático en el régimen actual.

    Retorna None si no hay suficientes datos.
    """
    if not EVAL_DIR.exists():
        return None

    scores = []
    for ticker_dir in EVAL_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        bs = calc_bias_score_from_evals(ticker, horizon, min_evals=8)
        if bs is not None:
            scores.append(bs)

    if len(scores) < 5:  # necesitamos al menos 5 tickers para ser confiables
        return None

    universe_bias = float(np.mean(scores))
    logger.info(
        f"🎯 H{horizon} bias_score universo: {universe_bias:.2f} "
        f"({len(scores)} tickers) | "
        f"{'⚠️ BIAS ALTO' if universe_bias >= BIAS_SCORE_THRESHOLD else '✅ OK'}"
    )
    return round(universe_bias, 4)


# ══════════════════════════════════════════════════════
# EVALUAR SHADOW GENOMES
# ══════════════════════════════════════════════════════

def _evaluate_shadow_hit_rates(horizon: int) -> List[Dict]:
    """
    [AR2] Excluye champion_baseline.json — es un placeholder que
    intraday_evaluator.py escribe con el hit_rate del CAMPEÓN, no de
    un shadow real. Antes de que predictor_shadow_evaluator.py existiera,
    este archivo era el único dato en shadow/evals/, y como su
    genome_id ("H{h}_champion_baseline") nunca coincide con ningún
    shadow real al buscar el match para promoción, no rompía nada
    directamente — pero si su hit_rate superaba al del campeón por
    ruido, sí podía inflar best_shadow_hit sin que hubiera ningún
    shadow real detrás. Ahora se filtra explícitamente.
    """
    shadow_eval_dir = GENOME_BASE / f"H{horizon}" / "shadow" / "evals"
    if not shadow_eval_dir.exists():
        return []
    results = []
    for path in shadow_eval_dir.glob("*.json"):
        if path.stem == "champion_baseline":
            continue
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
            "Authorization":        f"Bearer {token}",
            "Accept":               "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        content = json.dumps(genome.to_dict(), indent=2, ensure_ascii=False)
        encoded = base64.b64encode(content.encode()).decode()

        sha = None
        r   = httpx.get(url, headers=headers, params={"ref": branch})
        if r.status_code == 200:
            sha = r.json().get("sha")

        payload = {
            "message":   f"darwin: H{genome.horizon} promote {genome.genome_id} hit={genome.hit_rate if genome.hit_rate is not None else 'None'}",
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

    # [AR1] Guardar hit_rate anterior ANTES de sobreescribir, para
    # poder comparar si hubo mejora real y decidir si respaldar a GitHub
    previous_hit_rate = champion.hit_rate

    # Actualizar hit rate del campeón
    champion.hit_rate = hit_rate
    champion.data["n_evaluations"] = n_evals

    # [SW2] Calcular y escribir bias_score al genome
    # Solo para H1-H9 — H10 no usa sample_weight_decay
    bias_score = None
    if horizon < 10:
        bias_score = _calc_universe_bias_score(horizon)
        if bias_score is not None:
            champion.bias_score = bias_score
            champion.data["bias_score_updated_at"] = datetime.now(timezone.utc).isoformat()
            if bias_score >= BIAS_SCORE_THRESHOLD:
                logger.warning(
                    f"⚠️ H{horizon} bias_score={bias_score:.2f} >= {BIAS_SCORE_THRESHOLD} "
                    f"→ Darwin activará mutación de decay"
                )

    # ¿Hay shadow con mejor hit rate?
    was_promoted   = False
    github_written = False

    shadow_evals    = _evaluate_shadow_hit_rates(horizon)
    best_shadow     = None
    best_shadow_hit = 0.0

    for se in shadow_evals:
        shadow_hit = float(se.get("hit_rate", 0))
        shadow_n   = int(se.get("n_evaluations", 0))
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
            new_champion = PredictorGenome.from_dict(new_champion_data)

            # [AR3] El genoma recién promovido puede no tener
            # evaluaciones REALES en producción todavía (n_evaluations
            # heredado de la mutación, típicamente 0). Su hit_rate
            # validado en shadow (best_shadow_hit) no debe presentarse
            # como si fuera un resultado en vivo — se fuerza a None
            # hasta que el evaluator real lo alimente con datos
            # genuinos en un ciclo posterior.
            inherited_n_evals = int(new_champion_data.get("n_evaluations", 0) or 0)
            if inherited_n_evals == 0:
                logger.warning(
                    f"⚠️ H{horizon} nuevo campeón {new_champion.genome_id} promovido "
                    f"con n_evaluations=0 (hit_rate shadow={best_shadow_hit:.2%} no "
                    f"validado en producción) → hit_rate forzado a None hasta "
                    f"evaluación real"
                )
                new_champion.hit_rate = None
            else:
                new_champion.hit_rate = best_shadow_hit
            new_champion.data["n_evaluations"] = inherited_n_evals

            # [SW2] Preservar bias_score en el nuevo campeón
            if bias_score is not None:
                new_champion.bias_score = bias_score
            new_champion.save_as_champion()
            champion       = new_champion
            was_promoted   = True
            github_written = _write_genome_to_github(new_champion)
            logger.info(
                f"🏆 H{horizon} NUEVO CAMPEÓN: {new_champion.genome_id} | "
                f"hit_rate={new_champion.hit_rate if new_champion.hit_rate is not None else 'sin validar'} "
                f"vs anterior={hit_rate:.2%}"
            )
    else:
        if not dry_run:
            champion.save_as_champion()
            # [AR1] Comparar contra el hit_rate REAL anterior, no contra
            # champion.hit_rate (que ya fue sobreescrito arriba con el
            # valor nuevo y siempre daría False en la comparación vieja)
            if hit_rate > (previous_hit_rate or 0):
                github_written = _write_genome_to_github(champion)

    # Generar nueva generación shadow
    # generate_children usará bias_score para decidir si mutar decay [SW2]
    new_children = []
    if not dry_run:
        children = generate_children(champion, shadow, n_children=MAX_SHADOW_PER_H)
        for child in children:
            child.save_as_shadow()
            new_children.append(child.genome_id)

        _prune_shadow(horizon)

    return {
        "horizon":        horizon,
        "hit_rate":       hit_rate,
        "n_evals":        n_evals,
        "bias_score":     bias_score,   # [SW2] trazabilidad
        "champion":       champion.genome_id,
        "was_promoted":   was_promoted,
        "github_written": github_written,
        "new_shadow":     new_children,
        "decay_active":   champion.model_params.get("sample_weight_decay", 0.0) > 0,
    }


# ══════════════════════════════════════════════════════
# LIMPIEZA SHADOW
# ══════════════════════════════════════════════════════

def _prune_shadow(horizon: int) -> None:
    shadow_dir = GENOME_BASE / f"H{horizon}" / "shadow"
    if not shadow_dir.exists():
        return
    files  = sorted(shadow_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    excess = len(files) - MAX_SHADOW_PER_H
    if excess > 0:
        for f in files[:excess]:
            try:
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

    initialize_all_genomes()

    hit_rates = _read_hit_rates_from_evaluator()

    if not hit_rates:
        logger.warning("⚠️ Sin hit rates disponibles — ciclo abortado")
        return {"status": "no_data", "timestamp": start.isoformat()}

    results    = {}
    promotions = 0
    bias_alerts = []

    for h in range(1, 11):
        if h not in hit_rates:
            logger.info(f"⏭ H{h}: sin suficientes evaluaciones — saltando")
            continue

        hit_rate, n_evals = hit_rates[h]
        logger.info(f"🔬 H{h}: hit_rate={hit_rate:.2%} | n_evals={n_evals}")

        try:
            result = _run_h_cycle(h, hit_rate, n_evals, dry_run)
            results[f"H{h}"] = result
            if result.get("was_promoted"):
                promotions += 1
            # [SW2] Registrar alertas de bias
            bs = result.get("bias_score")
            if bs is not None and bs >= BIAS_SCORE_THRESHOLD:
                bias_alerts.append(f"H{h} bias={bs:.2f}")
        except Exception as e:
            logger.error(f"❌ H{h} ciclo falló: {e}")
            results[f"H{h}"] = {"error": str(e)}

    duration = (datetime.now(timezone.utc) - start).total_seconds()

    logger.info("\n📊 RANKING H1-H10:")
    ranked = sorted(
        [(h, hit_rates[h][0]) for h in hit_rates],
        key=lambda x: x[1], reverse=True
    )
    for h, hr in ranked:
        r        = results.get(f"H{h}", {})
        champion = r.get("champion", "?")
        promoted = "🏆 PROMOVIDO" if r.get("was_promoted") else ""
        bias_tag = f"⚠️ bias={r.get('bias_score', 0):.2f}" if r.get("bias_score", 0) >= BIAS_SCORE_THRESHOLD else ""
        decay    = "⏱️ decay_ON" if r.get("decay_active") else ""
        logger.info(f"   H{h}: {hr:.2%} | {champion} {promoted} {bias_tag} {decay}")

    if bias_alerts:
        logger.warning(f"\n⚠️ BIAS DETECTADO: {', '.join(bias_alerts)} → decay mutado")

    summary = {
        "status":       "ok",
        "timestamp":    start.isoformat(),
        "duration_sec": round(duration, 1),
        "dry_run":      dry_run,
        "h_evaluated":  len(results),
        "promotions":   promotions,
        "bias_alerts":  bias_alerts,   # [SW2]
        "results":      results,
        "ranking":      [{"horizon": h, "hit_rate": hr} for h, hr in ranked],
    }

    logger.info(
        f"\n✅ PREDICTOR ARENA COMPLETADO | {duration:.1f}s | "
        f"{len(results)} H evaluados | {promotions} promovidos | "
        f"{len(bias_alerts)} alertas bias"
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

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

  [AR4] (2026-09-15) run_predictor_evolution — bloque de ranking:
        `r.get("bias_score", 0)` solo usa el default 0 cuando la
        CLAVE "bias_score" no existe en el dict — pero para H10 la
        clave sí existe, con valor None explícito (bias_score solo
        se calcula para H1-H9, ver _run_h_cycle: "if horizon < 10").
        Como la clave está presente, .get() devolvía None, no 0, y
        `None >= BIAS_SCORE_THRESHOLD` lanzaba TypeError
        ("'>=' not supported between instances of 'NoneType' and
        'float'"), interrumpiendo TODO el ciclo evolutivo justo antes
        de terminar de imprimir el ranking — afectaba cada corrida de
        /internal/darwin/run, no solo cuando H10 estaba en juego.
        Fix: leer bias_val = r.get("bias_score") (sin default) y
        chequear `bias_val is not None` ANTES de comparar con >=,
        mismo patrón ya usado correctamente en la recolección de
        bias_alerts un poco más arriba en este mismo archivo.

  [AR5] (auditoría 2026-09-19, Problema 3) _run_h_cycle — promoción:
        `inherited_n_evals` se leía de `new_champion_data.get(
        "n_evaluations", 0)` — el archivo del GENOMA en sí
        (shadow/{id}.json), que predictor_mutator.py SIEMPRE
        inicializa en 0 al crear un hijo, sin importar cuánta
        evidencia real haya acumulado después. El fix [AR3] (27-ago)
        tenía la intención correcta (no mostrar hit_rate sin validar)
        pero apuntaba a la fuente equivocada — por eso TODO campeón
        promovido terminaba con hit_rate=None, incluso los que sí
        tenían >= MIN_EVALS_TO_COMPETE evaluaciones reales.
        El dato correcto ya estaba disponible en `best_shadow` (viene
        de shadow/evals/{id}.json vía _evaluate_shadow_hit_rates), que
        es justamente el mismo diccionario ya usado más arriba para
        filtrar candidatos por `shadow_n >= MIN_EVALS_TO_COMPETE`.
        Fix: usar `best_shadow.get("n_evaluations", 0)` en vez de
        `new_champion_data.get(...)` — mismo dato, fuente correcta.

  [AR6] (auditoría 2026-09-19, Problema 2) _prune_shadow():
        Archivaba por antigüedad de archivo (`st_mtime`) apenas se
        superaba MAX_SHADOW_PER_H=4, sin mirar cuánta evidencia
        (n_evaluations) había acumulado cada shadow. Los horizontes
        largos (H7-H10) tardan semanas en madurar las 30 evaluaciones
        de MIN_EVALS_TO_COMPETE — el shadow era archivado antes de
        alcanzar ese umbral, dejando a H3/H7/H8/H9/H10 sin evolucionar
        desde 2026-07-07 (>2 meses). Mismo bug que ya se había
        corregido en el archivo hermano de executors (arena.py::
        _prune_shadow_genomes, fix AUD-D1, 2026-09-05) pero nunca se
        portó aquí.
        Fix: mismo patrón de elitismo — solo se archivan shadows que
        AÚN NO llegan a MIN_EVALS_TO_COMPETE evaluaciones (leídas de
        shadow/evals/{id}.json). Un shadow con evidencia suficiente
        para competir ya no se archiva por antigüedad, aunque eso
        signifique superar temporalmente el tope de MAX_SHADOW_PER_H.
"""

import json
import logging
import math
import os
import time
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
# [AR7][2026-09-21] NOTA: se evaluó agregar acá un candado compartido
# con predictor_shadow_evaluator.py (LOCK_FILE) para coordinar el
# acceso a champion.json durante el swap temporal que ese archivo
# hacía. Se descartó: la solución de fondo fue eliminar el swap por
# completo (ver load_active_genome(override_genome=...) en
# predictor_genome.py y _run_shadow_batch() en
# predictor_shadow_evaluator.py) — predictor_shadow_evaluator.py ya
# no toca champion.json en ningún momento, así que no hay nada que
# coordinar con un candado. Dos procesos que nunca escriben el mismo
# archivo no pueden pisarse.

logger = logging.getLogger("predictor_arena")

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
EVAL_DIR  = DATA_PATH / "evaluations"

# Configuración
MIN_EVALS_TO_COMPETE = int(os.getenv("PRED_MIN_EVALS",    "30"))
MIN_HIT_IMPROVEMENT  = float(os.getenv("PRED_MIN_IMPROVE", "0.015"))
MAX_SHADOW_PER_H     = int(os.getenv("PRED_MAX_SHADOW",    "4"))
GITHUB_ENABLED       = os.getenv("GITHUB_TOKEN") is not None

# [AR8][2026-09-21] Umbral z para el test de significancia de dos
# proporciones — z=1.645 equivale aprox. a p<0.05 a una cola.
STAT_TEST_Z_THRESHOLD = float(os.getenv("PRED_STAT_TEST_Z", "1.645"))


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
# [AR8] TEST DE SIGNIFICANCIA — dos proporciones
# ══════════════════════════════════════════════════════

def _is_hit_rate_significantly_better(
    champ_hit: float, champ_n: int,
    shadow_hit: float, shadow_n: int,
    z: float = STAT_TEST_Z_THRESHOLD,
) -> bool:
    """
    [AR8][2026-09-21, auditoría Problema 1] ANTES: la promoción solo
    exigía `shadow_hit - hit_rate >= MIN_HIT_IMPROVEMENT (0.015)` — un
    delta absoluto sobre un hit_rate puntual, sin considerar el tamaño
    de muestra de cada lado. Evidencia real de que esto ya causó daño:
      - H1 (17-sep): promovido con hit_rate=0.6098 sobre apenas 41
        evaluaciones — con n=41, el error estándar de una proporción
        ~0.55 es de ~7.8 puntos porcentuales; un salto de ese tamaño
        es indistinguible del ruido estadístico.
      - H6 (18-sep): el campeón vigente (49.03% sobre 7.294 evals)
        fue reemplazado por un shadow con 46.77% sobre 881 evals —
        objetivamente PEOR (ver también [AR7] — probable condición de
        carrera con el swap de predictor_shadow_evaluator.py).
    A diferencia de arena.py (executors), acá NO existen los retornos
    individuales de cada evaluación — solo el hit_rate agregado y el
    conteo (n_evaluations). Con esos dos números el bootstrap de
    arena.py no aplica; la herramienta correcta para "¿esta diferencia
    de proporciones es real o es ruido?" es un test de dos
    proporciones (aproximación normal), el mismo principio, adaptado
    al tipo de dato disponible.

    Retorna True solo si la mejora es estadísticamente significativa
    (z >= STAT_TEST_Z_THRESHOLD, ~p<0.05 a una cola) Y ambos lados
    tienen al menos MIN_EVALS_TO_COMPETE muestras.
    """
    if champ_n < MIN_EVALS_TO_COMPETE or shadow_n < MIN_EVALS_TO_COMPETE:
        return False

    p_pool = (champ_hit * champ_n + shadow_hit * shadow_n) / (champ_n + shadow_n)
    se     = math.sqrt(p_pool * (1 - p_pool) * (1 / champ_n + 1 / shadow_n))

    if se == 0:
        return False

    return (shadow_hit - champ_hit) / se >= z


# ══════════════════════════════════════════════════════
# [AR6] EVIDENCIA DE UN SHADOW — para elitismo en _prune_shadow
# ══════════════════════════════════════════════════════

def _load_shadow_n_evals(horizon: int, genome_filename: str) -> int:
    """
    [AR6] Lee n_evaluations desde shadow/evals/{mismo nombre de
    archivo que el genoma}.json — misma fuente que usa
    _evaluate_shadow_hit_rates()/_run_h_cycle() para decidir
    elegibilidad de promoción. Retorna 0 si no existe el archivo de
    evaluación (shadow recién creado, sin evidencia todavía).
    """
    eval_path = GENOME_BASE / f"H{horizon}" / "shadow" / "evals" / genome_filename
    if not eval_path.exists():
        return 0
    try:
        data = json.loads(eval_path.read_text())
        return int(data.get("n_evaluations", 0) or 0)
    except Exception:
        return 0


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
    best_shadow_n   = 0

    for se in shadow_evals:
        shadow_hit = float(se.get("hit_rate", 0))
        shadow_n   = int(se.get("n_evaluations", 0))
        if shadow_n < MIN_EVALS_TO_COMPETE:
            continue
        if shadow_hit > best_shadow_hit:
            best_shadow_hit = shadow_hit
            best_shadow     = se
            best_shadow_n   = shadow_n

    # [AR8][2026-09-21] ANTES: solo `best_shadow_hit - hit_rate >=
    # MIN_HIT_IMPROVEMENT` — un delta absoluto sin considerar tamaño
    # de muestra, que ya promovió al menos un campeón peor (H6,
    # 18-sep) y otro basado en ruido puro (H1, 17-sep, n=41). Ahora
    # exige además significancia estadística real vía
    # _is_hit_rate_significantly_better() (ver docstring de esa
    # función para la evidencia completa).
    is_promotable = (
        best_shadow is not None
        and best_shadow_hit - hit_rate >= MIN_HIT_IMPROVEMENT
        and _is_hit_rate_significantly_better(hit_rate, n_evals, best_shadow_hit, best_shadow_n)
    )

    if not is_promotable and best_shadow is not None and best_shadow_hit - hit_rate >= MIN_HIT_IMPROVEMENT:
        logger.info(
            f"⏳ H{horizon}: {best_shadow.get('genome_id')} tiene mejor hit_rate "
            f"({best_shadow_hit:.2%} vs {hit_rate:.2%}) pero no alcanzó significancia "
            f"estadística (z>={STAT_TEST_Z_THRESHOLD}, n_champ={n_evals}, "
            f"n_shadow={best_shadow_n}) → no se promueve"
        )

    if is_promotable and not dry_run:
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

            # [AR5][2026-09-19] El campo n_evaluations que SÍ importa es
            # el de la evaluación real (best_shadow, viene de
            # shadow/evals/{id}.json — mismo dato ya usado arriba para
            # filtrar por MIN_EVALS_TO_COMPETE), NO el del archivo del
            # genoma (new_champion_data), que predictor_mutator.py
            # siempre inicializa en 0 al crear el hijo. Leer de la
            # fuente equivocada hacía que TODO campeón promovido
            # quedara con hit_rate=None, incluso los que sí tenían
            # evidencia real suficiente (best_shadow_hit ya validado
            # con shadow_n >= MIN_EVALS_TO_COMPETE más arriba).
            real_n_evals = int(best_shadow.get("n_evaluations", 0) or 0)
            if real_n_evals < MIN_EVALS_TO_COMPETE:
                # No debería ocurrir (best_shadow ya pasó ese filtro
                # arriba), pero se protege igual por si el archivo de
                # evals cambió entre la selección y este punto.
                logger.warning(
                    f"⚠️ H{horizon} nuevo campeón {new_champion.genome_id} promovido "
                    f"con n_evaluations={real_n_evals} (< {MIN_EVALS_TO_COMPETE}) → "
                    f"hit_rate forzado a None hasta evaluación real"
                )
                new_champion.hit_rate = None
            else:
                new_champion.hit_rate = best_shadow_hit
            new_champion.data["n_evaluations"] = real_n_evals

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
                f"(n_evaluations={real_n_evals}) vs anterior={hit_rate:.2%}"
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
    """
    [AR6][2026-09-19] Elitismo: antes archivaba por antigüedad de
    archivo apenas se superaba MAX_SHADOW_PER_H, sin mirar evidencia
    acumulada — los horizontes largos (H7-H10) tardan semanas en
    juntar las MIN_EVALS_TO_COMPETE evaluaciones necesarias para
    competir, y el shadow era archivado antes de llegar a ese umbral.
    Mismo bug ya corregido en arena.py::_prune_shadow_genomes
    (executors, fix AUD-D1, 2026-09-05), portado aquí ahora.

    Un shadow con evidencia suficiente (n_evaluations >=
    MIN_EVALS_TO_COMPETE) YA NO se archiva por antigüedad — solo se
    poda entre los que aún no llegan a ese umbral, empezando por el
    más antiguo. Esto puede dejar temporalmente más de
    MAX_SHADOW_PER_H genomas activos si varios ya tienen evidencia
    suficiente — es intencional, protege candidatos con evidencia real
    de ser destruidos antes de poder competir.
    """
    shadow_dir = GENOME_BASE / f"H{horizon}" / "shadow"
    if not shadow_dir.exists():
        return

    files  = sorted(shadow_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    excess = len(files) - MAX_SHADOW_PER_H
    if excess <= 0:
        return

    # [AR6] Solo los que AÚN NO tienen evidencia suficiente son
    # candidatos a poda — ordenados por antigüedad (más viejo primero).
    candidates = [
        f for f in files
        if _load_shadow_n_evals(horizon, f.name) < MIN_EVALS_TO_COMPETE
    ]

    to_archive = candidates[:excess]
    for f in to_archive:
        try:
            archive = shadow_dir / "archived" / f.name
            archive.parent.mkdir(exist_ok=True)
            f.rename(archive)
        except Exception:
            pass

    if len(to_archive) < excess:
        logger.info(
            f"🛡️ H{horizon} elitismo: {len(files) - len(to_archive)} genomas "
            f"protegidos de poda (por encima del límite {MAX_SHADOW_PER_H}) — "
            f"tienen evidencia suficiente para competir."
        )


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
        # [AR4][2026-09-15] r.get("bias_score", 0) solo usa el default 0
        # cuando la CLAVE no existe — pero para H10 la clave sí existe
        # con valor None explícito (bias_score solo se calcula para
        # H1-H9). Eso hacía que bias_val fuera None, no 0, y
        # None >= BIAS_SCORE_THRESHOLD explotaba con TypeError,
        # interrumpiendo todo el ciclo evolutivo antes de terminar.
        bias_val = r.get("bias_score")
        bias_tag = (
            f"⚠️ bias={bias_val:.2f}"
            if bias_val is not None and bias_val >= BIAS_SCORE_THRESHOLD
            else ""
        )
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
                                  

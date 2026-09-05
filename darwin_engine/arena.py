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

FIX v1.2 [AUD-D2]:
  [D2] _select_champion() contaba cand_trades con
       get_resolved_trades(executor_genome_id=candidate.genome_id),
       que solo lee darwin/trades/*.json — el pipeline de trades
       REALES que únicamente alimenta el campeón (los shadows nunca
       operan con dinero real). Resultado: cand_trades era siempre 0
       para todo shadow, nunca alcanzaban MIN_TRADES_TO_COMPETE, y el
       campeón nunca podía ser reemplazado sin importar cuán bueno
       fuera un candidato.

       Ahora, para genomas shadow, cand_trades se cuenta con
       get_shadow_resolved_trades() (darwin_engine.executor_shadow_evaluator),
       que lee darwin/shadow_trades/{genome_id}/ — el historial de
       decisiones simuladas que cada shadow acumula evaluando el
       MISMO ciclo que el campeón (mismo momento, mismo alpha_map,
       portfolio propio independiente). El campeón sigue midiéndose
       exclusivamente con sus trades reales — sin cambios ahí.

FIX v1.3 [AUD-P2] (auditoría 2026-08-26, Problema 2):
  [P2] MIN_IMPROVEMENT_PCT=0.05 era un delta ABSOLUTO fijo sobre un
       fitness que en 19 generaciones osciló entre -0.60 y +0.38, con
       3 generaciones devolviendo NaN — un umbral fijo así es
       estadísticamente inalcanzable dado ese nivel de ruido: un
       candidato podía tener mejor fitness real y aun así nunca
       superar 0.05 de diferencia por pura varianza, sin que eso
       dijera nada sobre si la diferencia era genuina o ruido.

       Fix: se reemplaza el delta absoluto por un test de
       significancia estadística (Mann-Whitney U, no paramétrico —
       no asume distribución normal de los retornos) entre los
       retornos reales del campeón (pnl_real_pct de sus trades) y los
       retornos simulados del candidato (pnl_real_pct de sus shadow
       trades). Solo se promueve si hay al menos 10 muestras de cada
       lado, el test da p < 0.10 (el candidato es significativamente
       mejor, no solo mejor "por casualidad"), y su fitness combinado
       sigue siendo mayor. Si scipy no está disponible o hay menos de
       10 muestras de un lado, cae de vuelta al criterio anterior
       (delta absoluto MIN_IMPROVEMENT_PCT) — no bloquea el ciclo.

FIX v1.1 (auditoría 2026-08-28, Problema 1 / Bug C):
  [F8] Mann-Whitney U pregunta "¿los valores de un grupo tienden a
       ser más grandes, en general?" — compara la distribución
       completa (esencialmente la mediana), no el promedio. Los
       retornos reales tienen cola larga (muchas pérdidas chicas,
       pocas ganancias grandes), así que la mediana puede ser ~0
       aunque el promedio sea claramente positivo. Confirmado en
       producción: executor_v4 con mediana=0.0% pero media=+1.22%
       daba p=0.649 (sin significancia) pese a tener +1.11 puntos de
       fitness sobre el campeón — el test bloqueaba una promoción
       legítima.
       Fix: _is_significantly_better() ahora usa un bootstrap sobre
       la diferencia de MEDIAS (remuestreo con reemplazo de cada
       lado, p-value = fracción de remuestreos donde el candidato no
       queda por encima). Compara promedios directamente, sin asumir
       normalidad ni simetría, y ya no depende de scipy.

FIX v1.4 [AUD-U1] (auditoría 2026-08-29, Problema 2 — causa raíz):
  [U1] _load_alpha_map() construía el universo de apertura de los
       shadows leyendo alpha_last.json COMPLETO, sin aplicar el
       filtro ALPACA_UNSUPPORTED (tickers/sufijos europeos y
       canadienses no ejecutables) que sí aplica register_open() en
       trade_tracker.py antes de que un trade real pueda existir.

       Resultado: los shadows podían "operar" activos que el campeón
       jamás podría tocar en la realidad — compitiendo sobre
       distribuciones de oportunidades distintas, no las mismas
       condiciones. Esto inflaba artificialmente la varianza del
       lado candidato (confirmado en producción: std=7.21% shadow
       vs std=4.23% campeón, casi el doble) y le restaba poder al
       test de significancia de _is_significantly_better(), que
       nunca lograba detectar mejoras reales (p≈0.245 con datos que
       deberían haber dado una promoción).

       Fix: _load_alpha_map() ahora excluye los mismos tickers y
       sufijos no soportados que trade_tracker.py, usando la MISMA
       constante ALPACA_UNSUPPORTED (importada, no duplicada) — el
       universo de apertura de los shadows queda así idéntico al que
       realmente puede operar el campeón.

FIX v1.5 [AUD-D1] (auditoría 2026-09-05, Problema 1):
  [D1] _prune_shadow_genomes() archivaba (de forma IRREVERSIBLE,
       src.rename() fuera de GENOME_DIR) a cualquier genoma shadow
       que no quedara en el top-MAX_SHADOW_GENOMES por
       combined_fitness DEL DÍA. No distinguía "genoma malo" de
       "mejor histórico con un mal día puntual de fitness" — una vez
       archivado, _load_all_active_genomes() solo lee GENOME_DIR
       (glob), así que el genoma desaparecía para siempre: nunca más
       acumulaba shadow trades ni podía alcanzar MIN_TRADES_TO_COMPETE
       para ser comparado formalmente contra el campeón vía
       _is_significantly_better().
       Confirmado en producción: executor_v6_4 archivado con
       executor_fitness=+1.0659 (combined_fitness=+0.6475), ~12x
       mejor que el campeón vigente executor_v1 (-0.1112). v2, v3,
       v6 en la misma situación.
       Fix: elitismo — nunca se archiva al mejor del ranking actual
       ni a ningún genoma que ya acumuló >= MIN_TRADES_TO_COMPETE
       shadow trades (es decir, ya tiene evidencia suficiente para
       competir formalmente; archivarlo tira esa evidencia a la
       basura). Solo se archivan genomas sin evidencia suficiente Y
       fuera del top-MAX_SHADOW_GENOMES — los "hijos" nuevos de cada
       generación que aún no demostraron nada.
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
from darwin_engine.trade_tracker import get_resolved_trades, get_tracker_summary, ALPACA_UNSUPPORTED
from darwin_engine.executor_shadow_evaluator import get_shadow_resolved_trades, run_all_shadows

# [AUD-P2→F8] scipy ya no es necesario para el test de significancia
# (ver _is_significantly_better, ahora bootstrap puro con numpy). Se
# deja la detección por si algún otro punto del código lo usa.
try:
    from scipy import stats as _scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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

# [AUD-P2] Mínimo de muestras por lado para poder correr el test
# estadístico, y umbral de significancia (p-value).
STAT_TEST_MIN_SAMPLES = int(os.getenv("DARWIN_STAT_MIN_SAMPLES", "10"))
STAT_TEST_P_VALUE     = float(os.getenv("DARWIN_STAT_P_VALUE",   "0.10"))

# [F8] Nº de remuestreos del bootstrap sobre la diferencia de medias.
STAT_TEST_BOOTSTRAP_ITERS = int(os.getenv("DARWIN_BOOTSTRAP_ITERS", "5000"))


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
# [AUD-P2] TEST DE SIGNIFICANCIA ESTADÍSTICA
# ══════════════════════════════════════════════════════

def _extract_returns(trades: List[Dict]) -> List[float]:
    """Extrae pnl_real_pct válidos (finitos) de una lista de trades."""
    out = []
    for t in trades:
        v = t.get("pnl_real_pct")
        if v is None:
            continue
        try:
            f = float(v)
            if math.isfinite(f):
                out.append(f)
        except (TypeError, ValueError):
            continue
    return out


def _is_significantly_better(
    champion_genome_id: str,
    candidate_genome_id: str,
) -> Optional[bool]:
    """
    [F8] FIX (auditoría 2026-08-28, Problema 1 / Bug C):
    Antes usaba Mann-Whitney U, que pregunta "¿los valores de un grupo
    tienden a ser más grandes, en general?" — compara la distribución
    completa (esencialmente la mediana), no el promedio.

    Los retornos reales tienen cola larga: muchas pérdidas chicas y
    pocas ganancias grandes. Con ese patrón la MEDIANA puede ser ~0
    aunque el PROMEDIO (lo que de verdad importa para la plata, y lo
    que ya usa el fitness vía Sharpe) sea claramente positivo.
    Confirmado en producción: executor_v4 con mediana=0.0% pero
    media=+1.22% le daba p=0.649 en Mann-Whitney (sin significancia)
    pese a tener +1.11 puntos de fitness sobre el campeón — el test
    bloqueaba una promoción legítima.

    Ahora: bootstrap de la diferencia de MEDIAS entre candidato y
    campeón. Remuestrea con reemplazo ambos grupos por separado
    STAT_TEST_BOOTSTRAP_ITERS veces, arma la distribución de
    (media_candidato - media_campeón), y el p-value es la fracción de
    remuestreos donde esa diferencia es <= 0 (candidato no mejor).
    No asume distribución normal ni simetría — funciona igual de bien
    con colas largas, porque compara promedios directamente en vez de
    orden/mediana. No depende de scipy.

    Retorna:
      True  → el candidato es significativamente mejor en promedio
              (p < STAT_TEST_P_VALUE)
      False → no hay evidencia suficiente de que sea mejor
      None  → muestra insuficiente de un lado — el llamador debe usar
              el criterio de respaldo (delta absoluto).
    """
    champ_returns = _extract_returns(get_resolved_trades(executor_genome_id=champion_genome_id))
    cand_returns  = _extract_returns(get_shadow_resolved_trades(candidate_genome_id))

    if len(champ_returns) < STAT_TEST_MIN_SAMPLES or len(cand_returns) < STAT_TEST_MIN_SAMPLES:
        return None

    try:
        champ_arr = np.asarray(champ_returns, dtype=float)
        cand_arr  = np.asarray(cand_returns, dtype=float)

        rng = np.random.default_rng()
        boot_champ_means = rng.choice(champ_arr, size=(STAT_TEST_BOOTSTRAP_ITERS, len(champ_arr)), replace=True).mean(axis=1)
        boot_cand_means  = rng.choice(cand_arr,  size=(STAT_TEST_BOOTSTRAP_ITERS, len(cand_arr)),  replace=True).mean(axis=1)

        diffs   = boot_cand_means - boot_champ_means
        p_value = float(np.mean(diffs <= 0))

        return bool(p_value < STAT_TEST_P_VALUE)
    except Exception as e:
        logger.warning(f"⚠️ bootstrap de medias falló ({champion_genome_id} vs {candidate_genome_id}): {e}")
        return None

# ══════════════════════════════════════════════════════
# SELECCIÓN: ¿PROMOVER NUEVO CAMPEÓN?  [F4][AUD-D2][AUD-P2]
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

    [AUD-D2] cand_trades para genomas SHADOW ahora se cuenta con
    get_shadow_resolved_trades() (trades simulados que cada shadow
    acumula evaluando el mismo ciclo que el campeón), no con
    get_resolved_trades() — que solo ve el pipeline de trades reales
    y siempre da 0 para shadows, dejándolos incapaces de competir
    sin importar su fitness. El campeón sigue contándose con sus
    trades reales, sin cambios.

    [AUD-P2] La decisión final de promoción ya no es un delta
    absoluto fijo (MIN_IMPROVEMENT_PCT) sobre un fitness ruidoso —
    primero se intenta un test de significancia estadística
    (Mann-Whitney U) entre los retornos reales del campeón y los
    simulados del mejor candidato. Si el test no se puede correr
    (pocas muestras o scipy ausente), se cae de vuelta al criterio
    anterior de delta absoluto, sin bloquear el ciclo.
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

    # El campeón se mide SIEMPRE con sus trades reales — sin cambios.
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

        # [AUD-D2] Shadows se miden con su historial simulado propio —
        # ya no con get_resolved_trades(), que para ellos siempre es 0.
        cand_trades = len(
            get_shadow_resolved_trades(candidate.genome_id)
        )

        if cand_trades < MIN_TRADES_TO_COMPETE:
            logger.info(
                f"⏳ {candidate.genome_id} insuficientes trades simulados "
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

    # [AUD-P2] Intentar primero el test de significancia estadística.
    stat_result = _is_significantly_better(
        current_champion.genome_id, best_candidate.genome_id
    )

    if stat_result is not None:
        if stat_result and best_fitness > champion_fitness:
            logger.info(
                f"🏆 NUEVO CAMPEÓN (test estadístico): {best_candidate.genome_id} | "
                f"fitness={best_fitness:.4f} vs actual={champion_fitness:.4f} | "
                f"p<{STAT_TEST_P_VALUE} (bootstrap de medias, n>={STAT_TEST_MIN_SAMPLES}/lado)"
            )
            return best_candidate, True
        else:
            logger.info(
                f"👑 Campeón se mantiene (test estadístico): {current_champion.genome_id} | "
                f"candidato {best_candidate.genome_id} no alcanzó significancia "
                f"(p>={STAT_TEST_P_VALUE}) o fitness no mejoró"
            )
            return current_champion, False

    # [AUD-P2] Respaldo: sin scipy o sin muestra suficiente de un lado
    # → criterio anterior (delta absoluto), sin bloquear el ciclo.
    logger.info(
        f"ℹ️ Test estadístico no disponible ({current_champion.genome_id} vs "
        f"{best_candidate.genome_id}) — usando criterio de respaldo (delta absoluto)"
    )
    improvement = best_fitness - champion_fitness
    if improvement >= MIN_IMPROVEMENT_PCT:
        logger.info(
            f"🏆 NUEVO CAMPEÓN (delta absoluto): {best_candidate.genome_id} | "
            f"fitness={best_fitness:.4f} vs actual={champion_fitness:.4f} "
            f"(mejora={improvement:+.4f})"
        )
        return best_candidate, True
    else:
        logger.info(
            f"👑 Campeón se mantiene (delta absoluto): {current_champion.genome_id} | "
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
# [AUD-D2] CONSTRUCCIÓN DE alpha_map / price_map PARA SHADOWS
# ══════════════════════════════════════════════════════

ALPHA_FILE = DATA_PATH / "alpha_last.json"
PRED_DIR   = DATA_PATH / "predictions"

# [AUD-U1] Mismos sufijos no ejecutables en Alpaca que ya usa
# trade_tracker.py (register_open, etc.) — se repite aquí en vez de
# importarse porque en trade_tracker.py está definida localmente
# dentro de cada función, no como constante de módulo. Si se mueve
# a una constante compartida ahí, actualizar también acá.
_ALPACA_UNSUPPORTED_SUFFIXES = (".PA", ".DE", ".SW", ".TO", ".MC", ".AS", ".MI", ".BR")


def _is_alpaca_unsupported(ticker: str) -> bool:
    t = ticker.upper()
    return t in ALPACA_UNSUPPORTED or any(t.endswith(s) for s in _ALPACA_UNSUPPORTED_SUFFIXES)


def _load_alpha_map() -> Dict[str, Dict]:
    """
    Mismo alpha_last.json que ya consume trading_orchestrator.py y
    trade_tracker.py — ningún dato nuevo, ninguna llamada adicional
    a un proveedor externo.

    [AUD-U1] Excluye tickers ALPACA_UNSUPPORTED / sufijos europeos y
    canadienses — el mismo filtro que register_open() aplica antes
    de que un trade real pueda existir. Sin esto, los shadows abrían
    posiciones simuladas en activos que el campeón nunca podría
    operar en la realidad, compitiendo sobre universos distintos.
    """
    try:
        if not ALPHA_FILE.exists():
            logger.warning("⚠️ alpha_last.json no encontrado — shadows sin universo este ciclo")
            return {}
        data = json.loads(ALPHA_FILE.read_text())
        raw  = {
            t.upper(): d
            for t, d in data.get("results", {}).items()
            if isinstance(d, dict)
        }
        filtered = {
            t: d for t, d in raw.items()
            if not _is_alpaca_unsupported(t)
        }
        skipped = len(raw) - len(filtered)
        if skipped:
            logger.info(
                f"⏭ _load_alpha_map — {skipped} ticker(s) no soportados en Alpaca "
                f"excluidos del universo de shadows (mismas condiciones que el campeón)"
            )
        return filtered
    except Exception as e:
        logger.warning(f"⚠️ _load_alpha_map falló: {e}")
        return {}


def _load_price_map(tickers: List[str]) -> Dict[str, float]:
    """
    [AUD-D2] price_now por ticker leído desde el mismo archivo de
    predicción diaria que ya lee trade_tracker._read_h_signals()
    (prediction.price_now) — no es una fuente nueva, es la misma
    que ya persiste el pipeline de predicción cada día.
    """
    price_map = {}
    for ticker in tickers:
        try:
            ticker_dir = PRED_DIR / ticker.upper()
            if not ticker_dir.exists():
                continue
            candidates = sorted(ticker_dir.glob("*.json"))
            if not candidates:
                continue
            data  = json.loads(candidates[-1].read_text(encoding="utf-8"))
            price = float(data.get("prediction", {}).get("price_now", 0))
            if price > 0:
                price_map[ticker.upper()] = price
        except Exception as e:
            logger.debug(f"_load_price_map {ticker}: {e}")
            continue
    return price_map


def _run_shadow_evaluation(shadow: List[ExecutorGenome]) -> None:
    """
    [AUD-D2] Cierra el circuito: corre la evaluación de todos los
    executor shadows EN EL MISMO CICLO del arena (que corre junto al
    ciclo diario, sin desfase respecto al momento en que se generó
    alpha_last.json / las predicciones del día) — no depende de que
    trading_orchestrator.py llame a nada. Usa exclusivamente datos ya
    persistidos en disco por el pipeline existente.

    Si no hay shadows o no hay alpha_map, no hace nada (no bloquea el
    resto del ciclo del arena).
    """
    if not shadow:
        return

    alpha_map = _load_alpha_map()
    if not alpha_map:
        logger.warning("⚠️ Sin alpha_map — shadows no evaluados este ciclo")
        return

    price_map = _load_price_map(list(alpha_map.keys()))

    logger.info(
        f"🌑 Evaluando {len(shadow)} executor shadows | "
        f"universo={len(alpha_map)} tickers | precios_disponibles={len(price_map)}"
    )
    results = run_all_shadows(shadow, alpha_map, price_map)
    for r in results:
        if "error" in r:
            logger.error(f"❌ Shadow {r['genome_id']}: {r['error']}")


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

    # [AUD-D2] Paso 1.5: correr la evaluación de shadows ANTES de
    # calcular fitness — así el fitness de este ciclo ya refleja
    # cualquier trade simulado que se cierre en este mismo paso.
    if not dry_run:
        _run_shadow_evaluation(shadow)
    else:
        logger.info("🔍 DRY RUN — evaluación de shadows omitida")

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
        trades = (
            len(get_resolved_trades(executor_genome_id=g.genome_id))
            if g.genome_id == champion.genome_id
            else len(get_shadow_resolved_trades(g.genome_id))
        )
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
    """
    [AUD-D1] Poda con elitismo: nunca archiva al mejor del ranking
    actual ni a ningún genoma que ya acumuló evidencia suficiente
    (>= MIN_TRADES_TO_COMPETE shadow trades) para competir formalmente
    contra el campeón. Solo archiva "hijos" nuevos sin evidencia aún
    y fuera del top-MAX_SHADOW_GENOMES.
    """
    if len(shadow) <= MAX_SHADOW_GENOMES:
        return

    ranked_shadow = sorted(
        shadow,
        key=lambda g: (
            fitness_results.get(g.genome_id, {}).get("combined_fitness") or float("-inf")
        ),
        reverse=True,
    )

    # [AUD-D1] Elitismo: proteger al mejor histórico vivo (aunque hoy
    # tenga un mal día de fitness) y a cualquier candidato que ya
    # acumuló suficientes shadow trades para ser evaluado formalmente.
    protected_ids = {ranked_shadow[0].genome_id}
    for g in ranked_shadow:
        if len(get_shadow_resolved_trades(g.genome_id)) >= MIN_TRADES_TO_COMPETE:
            protected_ids.add(g.genome_id)

    to_archive = [
        g for g in ranked_shadow[MAX_SHADOW_GENOMES:]
        if g.genome_id not in protected_ids
    ]

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

    if len(protected_ids) > MAX_SHADOW_GENOMES:
        logger.info(
            f"🛡️ Elitismo: {len(protected_ids)} genomas protegidos de poda "
            f"(por encima del límite {MAX_SHADOW_GENOMES}) — mejor histórico "
            f"y/o candidatos con evidencia suficiente para competir."
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
            "stat_test_enabled":     True,     # [F8] bootstrap de medias, no requiere scipy
            "stat_test_method":      "bootstrap_mean_diff",
            "stat_test_min_samples": STAT_TEST_MIN_SAMPLES,
            "stat_test_p_value":     STAT_TEST_P_VALUE,
            "stat_test_bootstrap_iters": STAT_TEST_BOOTSTRAP_ITERS,
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

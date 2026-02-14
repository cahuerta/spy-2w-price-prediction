# =========================================================
# screener.py — ENTERPRISE SCREENER (FAST + STRUCTURAL)
# =========================================================
# ✔ Screener real (rápido)
# ✔ Filtro liviano inicial
# ✔ Engine completo SOLO sobre shortlist
# ✔ Ranking siempre visible
# ✔ Contrato JSON intacto
# =========================================================

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from screener_data_layer import fetch_multiple_symbols
from screener_engine import compute_score

try:
    from screener_ia_module import enrich_screener_candidates_batch
    IA_AVAILABLE = True
except Exception:
    IA_AVAILABLE = False


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("screener")


# =========================================================
# ⚡ FAST PRE-FILTER (ULTRA LIVIANO)
# =========================================================

def fast_filter(item: Dict[str, Any]) -> bool:
    """
    Filtro rápido:
    - Tendencia simple 20d positiva
    - Volumen promedio suficiente
    """

    closes = item["closes"]
    volumes = item["volumes"]

    if len(closes) < 20:
        return False

    # Retorno 20 días
    ret_20 = (closes[-1] / closes[-20]) - 1

    # Volumen promedio
    avg_dollar_volume = sum(closes[-20:] * volumes[-20:]) / 20

    return (
        ret_20 > 0.02 and                 # +2% en 20d
        avg_dollar_volume > 20_000_000    # liquidez mínima básica
    )


# =========================================================
# FILTRO ESTRUCTURAL FINAL (PASAN A MODEL)
# =========================================================

def is_structural_candidate(x: Dict[str, Any]) -> bool:
    return (
        x["sharpe_ratio"] > 0.8 and
        x["trend_3m_pct"] > 3 and
        x["avg_dollar_volume"] > 50_000_000 and
        x["max_drawdown_pct"] > -25
    )


# =========================================================
# CORE ASYNC
# =========================================================

async def run_screener_async() -> Dict[str, Any]:

    logger.info("🔍 Running FAST screener...")

    raw_data = fetch_multiple_symbols()

    if not raw_data:
        logger.warning("⚠️ No data returned from data layer")

    # =====================================================
    # 1️⃣ FAST FILTER (reduce universo)
    # =====================================================
    pre_filtered = [x for x in raw_data if fast_filter(x)]

    logger.info(f"Universe raw: {len(raw_data)}")
    logger.info(f"After fast filter: {len(pre_filtered)}")

    # Limitar shortlist para engine pesado
    shortlist = pre_filtered[:60]  # máximo 60

    evaluated: List[Dict[str, Any]] = []

    # =====================================================
    # 2️⃣ ENGINE ENTERPRISE SOLO SOBRE SHORTLIST
    # =====================================================
    for item in shortlist:

        score_data = compute_score(
            closes=item["closes"],
            volumes=item["volumes"],
        )

        if not score_data:
            continue

        evaluated.append({
            "ticker": item["symbol"],
            **score_data
        })

    # =====================================================
    # 3️⃣ RANKING GLOBAL
    # =====================================================
    evaluated.sort(key=lambda x: x["score"], reverse=True)

    ranking_top20 = evaluated[:20]

    # =====================================================
    # 4️⃣ IA SOLO SOBRE TOP20
    # =====================================================
    if IA_AVAILABLE and ranking_top20:
        try:
            ranking_top20 = await enrich_screener_candidates_batch(ranking_top20)
        except Exception as e:
            logger.warning(f"IA failed: {e}")

    # =====================================================
    # 5️⃣ CANDIDATOS REALES
    # =====================================================
    candidates = [
        x for x in evaluated
        if is_structural_candidate(x)
    ]

    # =====================================================
    # 6️⃣ CONTRATO JSON (NO CAMBIA)
    # =====================================================
    result = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_evaluated": len(evaluated),
        "candidates_strict": candidates,
        "top20_global": ranking_top20,
    }

    logger.info(f"Evaluated (engine): {len(evaluated)}")
    logger.info(f"Candidates: {len(candidates)}")

    return result


# =========================================================
# WRAPPER SYNC
# =========================================================

def run_screener() -> Dict[str, Any]:
    return asyncio.run(run_screener_async())


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    result = run_screener()
    print(json.dumps(result, indent=2))

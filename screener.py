# =========================================================
# screener.py — ENTERPRISE COORDINADOR (CONTRATO INTACTO)
# =========================================================
# ✔ NO cambia estructura JSON
# ✔ Ranking siempre visible
# ✔ Candidatos estructurales reales
# ✔ IA solo sobre Top20
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
# FILTRO ESTRUCTURAL (NO ROMPE JSON)
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

    logger.info("🔍 Running screener...")

    raw_data = fetch_multiple_symbols()

    evaluated: List[Dict[str, Any]] = []

    for item in raw_data:

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

    # =============================
    # ORDEN GLOBAL
    # =============================
    evaluated.sort(key=lambda x: x["score"], reverse=True)

    top20 = evaluated[:20]  # 🔹 siempre visible

    # =============================
    # IA SOLO SOBRE TOP20
    # =============================
    if IA_AVAILABLE and top20:
        try:
            top20 = await enrich_screener_candidates_batch(top20)
        except Exception as e:
            logger.warning(f"IA failed: {e}")

    # =============================
    # CANDIDATOS ESTRICTOS
    # =============================
    candidates = [
        x for x in evaluated
        if is_structural_candidate(x)
    ]

    # =============================
    # 🔒 CONTRATO ORIGINAL (NO CAMBIA)
    # =============================
    result = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_evaluated": len(evaluated),
        "candidates_strict": candidates,
        "top20_global": top20,
    }

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

# =========================================================
# screener.py — COORDINADOR PURO (ROBUSTO)
# =========================================================
# ✔ No universo
# ✔ No env vars
# ✔ No contratos nuevos
# ✔ Solo coordina Layer → Engine → IA
# ✔ Logs reales de diagnóstico
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
# CORE ASYNC — COORDINADOR REAL
# =========================================================

async def run_screener_async() -> Dict[str, Any]:

    logger.info("🚀 Screener started")

    # 1️⃣ DATA LAYER
    raw_data = fetch_multiple_symbols()

    if not raw_data:
        logger.warning("⚠️ No data returned from data layer")
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "n_universe": 0,
            "n_valid_data": 0,
            "n_evaluated": 0,
            "candidates_strict": [],
            "top20_global": [],
        }

    n_universe = len(raw_data)
    logger.info(f"📊 Raw symbols received: {n_universe}")

    # Filtrar objetos válidos
    valid_data = [item for item in raw_data if item]
    n_valid_data = len(valid_data)

    logger.info(f"✅ Valid data objects: {n_valid_data}")

    if n_valid_data == 0:
        logger.warning("❌ All symbols failed data validation")
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "n_universe": n_universe,
            "n_valid_data": 0,
            "n_evaluated": 0,
            "candidates_strict": [],
            "top20_global": [],
        }

    # 2️⃣ ENGINE
    evaluated: List[Dict[str, Any]] = []

    for item in valid_data:

        try:
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

        except Exception as e:
            logger.warning(f"⚠️ Engine failed for {item.get('symbol')}: {e}")
            continue

    n_evaluated = len(evaluated)
    logger.info(f"🏁 Evaluated tickers: {n_evaluated}")

    if n_evaluated == 0:
        logger.warning("❌ All symbols filtered out by engine")

    # 3️⃣ Orden global
    evaluated.sort(key=lambda x: x["score"], reverse=True)

    # 4️⃣ IA opcional
    if IA_AVAILABLE and evaluated:
        try:
            logger.info("🤖 Running IA enrichment...")
            evaluated = await enrich_screener_candidates_batch(evaluated)
        except Exception as e:
            logger.warning(f"⚠️ IA failed: {e}")

    # 5️⃣ Resultado final
    result = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_universe": n_universe,
        "n_valid_data": n_valid_data,
        "n_evaluated": n_evaluated,
        "candidates_strict": evaluated,
        "top20_global": evaluated[:20],
    }

    logger.info("✅ Screener finished")

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

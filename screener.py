# =========================================================
# screener.py — COORDINADOR PURO
# =========================================================
# ✔ No universo
# ✔ No env vars
# ✔ No contratos nuevos
# ✔ Solo coordina Layer → Engine → IA
# =========================================================

import json
import logging
import asyncio
from pathlib import Path
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

    # 🔹 El layer decide qué símbolos buscar.
    # Aquí NO definimos universo.
    raw_data = fetch_multiple_symbols()

    evaluated: List[Dict[str, Any]] = []

    for item in raw_data:
        if not item:
            continue

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

    # Orden global
    evaluated.sort(key=lambda x: x["score"], reverse=True)

    # IA opcional
    if IA_AVAILABLE and evaluated:
        try:
            evaluated = await enrich_screener_candidates_batch(evaluated)
        except Exception as e:
            logger.warning(f"IA failed: {e}")

    result = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_evaluated": len(evaluated),
        "candidates_strict": evaluated,
        "top20_global": evaluated[:20],
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

# =========================================================
# screener.py — COORDINADOR CANÓNICO FINAL
# =========================================================
# ✔ No cambia contratos
# ✔ Usa screener_data_layer
# ✔ Usa compute_score existente
# ✔ Usa screener_ia_module si está disponible
# ✔ MISMO JSON de salida
# =========================================================

import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from screener_data_layer import fetch_multiple_symbols
from screener_data_layer import compute_score

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
# CONFIG (NO CAMBIAR NOMBRES)
# =========================================================

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
OUTPUT_FILE = DATA_PATH / "screener_candidates.json"

TOP_GLOBAL = int(os.getenv("SCREENER_TOP_GLOBAL", "20"))
MIN_SCORE = float(os.getenv("SCREENER_MIN_SCORE", "0.55"))
MIN_DOLLAR_VOLUME = float(os.getenv("SCREENER_MIN_DOLLAR_VOL", "50000000"))
MIN_VOL = float(os.getenv("SCREENER_MIN_VOL", "0.015"))
MAX_VOL = float(os.getenv("SCREENER_MAX_VOL", "0.06"))


# =========================================================
# CORE ASYNC — FIRMA ORIGINAL (NO CAMBIAR)
# =========================================================

async def run_screener_async() -> Dict[str, Any]:

    # Universo definido externamente
    symbols = os.getenv("SCREENER_UNIVERSE")
    if not symbols:
        raise RuntimeError("SCREENER_UNIVERSE not defined")

    symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    # ============================================
    # DATA LAYER
    # ============================================

    raw_data = fetch_multiple_symbols(symbols)

    evaluated = []

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

    # ============================================
    # TOP GLOBAL (SIN FILTRO)
    # ============================================

    top_global = sorted(
        evaluated,
        key=lambda x: x["score"],
        reverse=True
    )[:TOP_GLOBAL]

    # ============================================
    # FILTRO ESTRICTO
    # ============================================

    candidates_strict = [
        r for r in evaluated
        if (
            r["score"] >= MIN_SCORE and
            r["avg_dollar_volume"] >= MIN_DOLLAR_VOLUME and
            MIN_VOL <= r["volatility"] <= MAX_VOL
        )
    ]

    candidates_strict.sort(key=lambda x: x["score"], reverse=True)

    # ============================================
    # IA OPCIONAL
    # ============================================

    if IA_AVAILABLE and candidates_strict:
        try:
            candidates_strict = await enrich_screener_candidates_batch(
                candidates_strict
            )
        except Exception as e:
            logger.warning(f"IA failed: {e}")

    # ============================================
    # OUTPUT CONTRACTUAL
    # ============================================

    result = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_universe": len(symbols),
        "n_evaluated": len(evaluated),
        "n_candidates_strict": len(candidates_strict),
        "candidates_strict": candidates_strict,
        "top20_global": top_global,
    }

    return result


# =========================================================
# WRAPPER SYNC — PIPELINE CRON
# =========================================================

def run_screener() -> Dict[str, Any]:

    result = asyncio.run(run_screener_async())

    DATA_PATH.mkdir(parents=True, exist_ok=True)

    tmp = OUTPUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(OUTPUT_FILE)

    logger.info(f"Screener saved → {OUTPUT_FILE}")

    return result


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    run_screener()

# =========================================================
# screener.py — MARKET SCREENER (CRON MODE) ✅ SAFE
# =========================================================
# ✔ No rompe contratos
# ✔ Firma compatible con pipeline_router
# ✔ Guarda mismo JSON
# ✔ IA opcional
# ✔ Quant first, IA second
# =========================================================

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from screener_data_layer import fetch_multiple_symbols
from screener_engine import compute_score
from screener_ia_module import enrich_screener_candidates_batch

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("screener")

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = Path(os.getenv("DATA_PATH", "./data"))
OUTPUT_FILE = DATA_PATH / "screener.json"  # ⚠ MISMO NOMBRE

TOP_GLOBAL = 20

DEFAULT_UNIVERSE = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA",
    "AMD","NFLX","SPY","QQQ","IWM","JNJ","KO","PG",
    "MCD","XOM","CVX","BA","JPM","GS","BAC"
]

MIN_SCORE = float(os.getenv("SCREENER_MIN_SCORE", "0.55"))
MIN_DOLLAR_VOLUME = float(os.getenv("SCREENER_MIN_DOLLAR_VOL", "50000000"))
MIN_VOL = float(os.getenv("SCREENER_MIN_VOL", "0.015"))
MAX_VOL = float(os.getenv("SCREENER_MAX_VOL", "0.06"))

# =========================================================
# CORE ASYNC — NO CAMBIA FIRMA
# =========================================================
async def run_screener_async() -> Dict[str, Any]:

    symbols = DEFAULT_UNIVERSE
    logger.info(f"Universe size: {len(symbols)}")

    raw_data = fetch_multiple_symbols(symbols)

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

    # =============================
    # TOP GLOBAL (SIEMPRE)
    # =============================
    top20_global = sorted(
        evaluated,
        key=lambda x: x["score"],
        reverse=True
    )[:TOP_GLOBAL]

    # =============================
    # FILTRO ESTRICTO
    # =============================
    candidates_strict = [
        r for r in evaluated
        if (
            r["score"] >= MIN_SCORE and
            r["avg_dollar_volume"] >= MIN_DOLLAR_VOLUME and
            MIN_VOL <= r["volatility"] <= MAX_VOL
        )
    ]

    candidates_strict.sort(key=lambda x: x["score"], reverse=True)

    # =============================
    # IA SOLO A CANDIDATOS
    # =============================
    if candidates_strict:
        try:
            candidates_strict = await enrich_screener_candidates_batch(candidates_strict)
        except Exception as e:
            logger.warning(f"IA enrichment failed: {e}")

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_universe": len(symbols),
        "n_evaluated": len(evaluated),
        "n_candidates_strict": len(candidates_strict),
        "candidates_strict": candidates_strict,
        "top20_global": top20_global,
    }

    return output


# =========================================================
# WRAPPER SYNC — NO CAMBIA
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

# =========================================================
# screener.py — MARKET SCREENER (CRON COORDINATOR)
# =========================================================
# ✔ Determinista
# ✔ Candidatos estrictos (puede ser 0)
# ✔ Top 20 global SIEMPRE presentes
# ✔ NO rompe contratos existentes
# ✔ Guarda screener_candidates.json
# =========================================================

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# =========================================================
# IMPORT CAPAS (NO CAMBIAR)
# =========================================================
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
# CONFIG (CONTRATO ESTABLE)
# =========================================================
DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
OUTPUT_FILE = DATA_PATH / "screener_candidates.json"

LOOKBACK_DAYS = int(os.getenv("SCREENER_LOOKBACK", "90"))
MIN_DOLLAR_VOLUME = float(os.getenv("SCREENER_MIN_DOLLAR_VOL", "50_000_000"))
MIN_VOLATILITY = float(os.getenv("SCREENER_MIN_VOL", "0.015"))
MAX_VOLATILITY = float(os.getenv("SCREENER_MAX_VOL", "0.06"))
MIN_SCORE = float(os.getenv("SCREENER_MIN_SCORE", "0.55"))

TOP_GLOBAL = 20

# =========================================================
# UNIVERSO BASE (NO ROMPER)
# =========================================================
DEFAULT_UNIVERSE = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA",
    "AMD","NFLX","SPY","QQQ","IWM","JNJ","KO","PG",
    "MCD","XOM","CVX","BA","JPM","GS","BAC"
]

# =========================================================
# CORE ASYNC
# =========================================================
async def run_screener_async(symbols: List[str]) -> Dict[str, Any]:

    logger.info(f"Universe size: {len(symbols)}")

    raw_data = fetch_multiple_symbols(symbols)

    evaluated: List[Dict[str, Any]] = []

    # =====================================================
    # Evaluación cuantitativa
    # =====================================================
    for item in raw_data:
        if not item:
            continue

        score_data = compute_score(
            closes=item["closes"],
            volumes=item["volumes"],
        )

        if not score_data:
            continue

        result = {
            "ticker": item["symbol"],
            **score_data
        }

        evaluated.append(result)

    # =====================================================
    # Top 20 global (SIEMPRE)
    # =====================================================
    top20_global = sorted(
        evaluated,
        key=lambda x: x["score"],
        reverse=True
    )[:TOP_GLOBAL]

    # =====================================================
    # Candidatos estrictos (puede ser 0)
    # =====================================================
    candidates_strict = [
        r for r in evaluated
        if (
            r["avg_dollar_volume"] >= MIN_DOLLAR_VOLUME and
            MIN_VOLATILITY <= r["volatility"] <= MAX_VOLATILITY and
            r["score"] >= MIN_SCORE
        )
    ]

    candidates_strict.sort(key=lambda x: x["score"], reverse=True)

    # =====================================================
    # IA SOLO SOBRE candidatos estrictos
    # =====================================================
    if candidates_strict:
        try:
            candidates_strict = await enrich_screener_candidates_batch(
                candidates_strict
            )
        except Exception as e:
            logger.warning(f"IA enrichment failed: {e}")

    # =====================================================
    # OUTPUT (CONTRATO ESTABLE)
    # =====================================================
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
# WRAPPER CRON (NO CAMBIAR NOMBRE ARCHIVO)
# =========================================================
def run_screener(symbols: Optional[List[str]] = None) -> Dict[str, Any]:

    symbols = symbols or DEFAULT_UNIVERSE

    result = asyncio.run(run_screener_async(symbols))

    DATA_PATH.mkdir(parents=True, exist_ok=True)

    tmp = OUTPUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(OUTPUT_FILE)

    logger.info(f"Saved screener output → {OUTPUT_FILE}")

    return result


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    result = run_screener()

    print("\nStrict Candidates:")
    print(json.dumps(result["candidates_strict"][:5], indent=2))

    print("\nTop 20 Global:")
    print(json.dumps(result["top20_global"][:5], indent=2))

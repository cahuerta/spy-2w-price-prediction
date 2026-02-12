# =========================================================
# screener.py — COORDINADOR CRON CANÓNICO
# =========================================================
# ✔ NO ejecuta trades
# ✔ NO modifica tickers.json
# ✔ Respeta contratos existentes
# ✔ No cambia firmas
# ✔ No agrega fallback
# ✔ Guarda screener.json (contrato frontend)
# =========================================================

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# ==========================
# CAPAS (SIN CAMBIAR NOMBRES)
# ==========================
from screener_data_layer import fetch_multiple_symbols
from screener_engine import compute_score
from screener_ia_module import enrich_screener_candidates_batch

# ==========================
# LOGGING
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("screener")

# ==========================
# CONFIG
# ==========================
DATA_PATH = Path(os.getenv("DATA_PATH", "./data"))
OUTPUT_FILE = DATA_PATH / "screener.json"   # 🔒 NO CAMBIAR

TOP_GLOBAL = 20

# Universo lo define el pipeline o usa default
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
# CORE ASYNC — FIRMA NO CAMBIA
# =========================================================
async def run_screener_async(symbols: List[str]) -> Dict[str, Any]:

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

    # ==========================
    # TOP GLOBAL (SIEMPRE)
    # ==========================
    top_global = sorted(
        evaluated,
        key=lambda x: x["score"],
        reverse=True
    )[:TOP_GLOBAL]

    # ==========================
    # CANDIDATOS ESTRICTOS
    # ==========================
    candidates = [
        r for r in evaluated
        if (
            r["avg_dollar_volume"] >= MIN_DOLLAR_VOLUME and
            MIN_VOL <= r["volatility"] <= MAX_VOL and
            r["score"] >= MIN_SCORE
        )
    ]

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # ==========================
    # IA (OPCIONAL)
    # ==========================
    if candidates:
        try:
            candidates = await enrich_screener_candidates_batch(candidates)
        except Exception as e:
            logger.warning(f"IA enrichment failed: {e}")

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "n_universe": len(symbols),
        "n_evaluated": len(evaluated),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "top_global": top_global
    }


# =========================================================
# WRAPPER CRON — NO CAMBIA FIRMA
# =========================================================
def run_screener(symbols: List[str] = None) -> Dict[str, Any]:

    symbols = symbols or DEFAULT_UNIVERSE

    result = asyncio.run(run_screener_async(symbols))

    DATA_PATH.mkdir(parents=True, exist_ok=True)

    tmp = OUTPUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(OUTPUT_FILE)

    logger.info(f"Screener guardado → {OUTPUT_FILE}")

    return result


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    run_screener()

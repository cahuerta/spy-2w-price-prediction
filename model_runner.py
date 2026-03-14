# =========================================================
# model_runner.py — MODEL EXECUTION LAYER
# =========================================================
# ✔ Ejecuta model.py (que llama MasterOrchestrator)
# ✔ NO graba en disco — el orquestador es el dueño
# ✔ Filtra SKIP y ERROR antes de gastar tiempo en el pipeline
# ✔ Atomic write eliminado de acá — solo el orquestador escribe
#
# FIXES vMR-2:
#   [MR1] Respetar status=SKIP de model.py — no correr orquestador
#   [MR2] Respetar status=ERROR — loggear y continuar
#   [MR3] NO guardar resultado — orquestador ya grabó
#   [MR4] Conteo separado: ok / failed / skipped
# =========================================================

import os
import json
import logging
from pathlib import Path
from datetime import datetime

from model import run_model

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
PREDICTIONS_PATH = DATA_PATH / "predictions"
TICKERS_FILE = DATA_PATH / "tickers.json"
MODEL_HORIZON = int(os.getenv("MODEL_HORIZON_DAYS", "10"))

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s"
)
logger = logging.getLogger("model_runner")

logger.info(f"[MODEL_RUNNER] DATA_PATH = {DATA_PATH.resolve()}")
logger.info(f"[MODEL_RUNNER] DATA_PATH exists = {DATA_PATH.exists()}")
logger.info(f"[MODEL_RUNNER] PREDICTIONS_PATH = {PREDICTIONS_PATH.resolve()}")


# =========================================================
# HELPERS
# =========================================================
def load_tickers() -> list[str]:
    if not TICKERS_FILE.exists():
        raise RuntimeError(f"tickers.json no existe: {TICKERS_FILE}")

    data = json.loads(TICKERS_FILE.read_text())

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "tickers" in data:
        return data["tickers"]

    raise RuntimeError("Formato inválido en tickers.json")


# =========================================================
# CORE — [MR3] NO graba, el orquestador ya lo hizo
# =========================================================
def run_model_for_ticker(ticker: str) -> dict:
    logger.info(f"📈 Ejecutando modelo para {ticker}")

    result = run_model(ticker=ticker, horizon=MODEL_HORIZON)

    # [MR1] Filtrar SKIP — datos insuficientes, no correr orquestador
    if result.get("status") == "SKIP":
        logger.info(f"⏭️  [{ticker}] SKIP: {result.get('message', '')}")
        return result

    # [MR2] Filtrar ERROR — algo falló en el orquestador
    if result.get("status") == "ERROR":
        logger.warning(f"⚠️  [{ticker}] ERROR: {result.get('message', '')}")
        return result

    # [MR3] Resultado válido — el orquestador YA grabó en disco
    # NO llamar save_json aquí
    logger.info(f"✅ [{ticker}] OK")
    return result


# =========================================================
# BATCH
# =========================================================
def run_all_models():
    tickers = load_tickers()
    logger.info(f"🚀 Ejecutando modelo para {len(tickers)} tickers")

    # [MR4] Conteo separado para visibilidad real en logs
    ok, failed, skipped = 0, 0, 0

    for t in tickers:
        try:
            result = run_model_for_ticker(t)

            status = result.get("status")
            if status == "SKIP":
                skipped += 1
            elif status == "ERROR":
                failed += 1
            else:
                ok += 1

        except Exception as e:
            failed += 1
            logger.error(f"❌ {t} error inesperado: {e}")

    logger.info(
        f"🏁 MODEL RUN FINALIZADO | OK={ok} | FAIL={failed} | SKIP={skipped}"
    )


# =========================================================
# CLI / CRON ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    run_all_models()

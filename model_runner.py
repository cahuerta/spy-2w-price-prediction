# =========================================================
# model_runner.py — MODEL EXECUTION & PERSISTENCE LAYER
# =========================================================
# ✔ Ejecuta model.py
# ✔ Guarda predicciones en /data/predictions/{ticker}/
# ✔ Formato 100% compatible con dashboard.py
# ✔ Atomic write (seguro en cron)
# ✔ NO decide | NO evalúa | NO señales
# =========================================================

import os
import json
import logging
from pathlib import Path
from datetime import datetime

from model import run_model   # 🔑 TU MOTOR REAL

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
PREDICTIONS_PATH = DATA_PATH / "predictions"
TICKERS_FILE = DATA_PATH / "tickers.json"

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s"
)
logger = logging.getLogger("model_runner")

# =========================================================
# 🔍 AUDITORÍA DE RUTAS (FIX 1)
# =========================================================
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


def save_json_atomic(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# =========================================================
# CORE
# =========================================================
def run_model_for_ticker(ticker: str) -> dict:
    logger.info(f"📈 Ejecutando modelo para {ticker}")

    result = run_model(ticker=ticker)

    today = datetime.utcnow().date().isoformat()

    # =====================================================
    # 📁 Asegurar directorio del ticker (FIX 2)
    # =====================================================
    ticker_dir = PREDICTIONS_PATH / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    out_file = ticker_dir / f"{today}.json"

    save_json_atomic(out_file, result)

    # =====================================================
    # ✅ Verificación post-escritura (FIX 3)
    # =====================================================
    if not out_file.exists():
        raise RuntimeError(
            f"[MODEL_RUNNER] Archivo NO existe tras grabar: {out_file}"
        )

    logger.info(f"💾 Guardado: {out_file.resolve()}")
    return result


def run_all_models():
    tickers = load_tickers()
    logger.info(f"🚀 Ejecutando modelo para {len(tickers)} tickers")

    ok, failed = 0, 0

    for t in tickers:
        try:
            run_model_for_ticker(t)
            ok += 1
        except Exception as e:
            failed += 1
            logger.error(f"❌ {t} falló: {e}")

    logger.info(
        f"🏁 MODEL RUN FINALIZADO | OK={ok} | FAIL={failed}"
    )


# =========================================================
# CLI / CRON ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    run_all_models()

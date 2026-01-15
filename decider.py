# =========================================================
# decider.py — SCREENER → TICKERS MEMORY DECIDER (AUTO-ADD) v2.2
# =========================================================
# ✅ Lee screener_candidates.json DESDE DISCO
# ✅ AGREGA automáticamente a tickers.json
# ✅ CONCURRENCIA: Atomic write + backup
# ❌ NUNCA borra tickers
# ❌ NO decide trades | NO toca señales ni broker
# =========================================================

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# =========================
# Configuración
# =========================
DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))

SCREENER_FILE = DATA_PATH / "screener_candidates.json"
TICKERS_FILE = DATA_PATH / "tickers.json"

# Criterios conservadores
MIN_SCORE = 0.75
ALLOWED_QUALITIES = {"STRONG", "GOOD"}

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s"
)
logger = logging.getLogger("decider")

# =========================
# Helpers CONCURRENCIA
# =========================
def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Archivo requerido no existe: {path}")
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.error(f"❌ JSON corrupto: {path}")
        raise


def save_json_atomic(path: Path, data: Dict[str, Any]):
    if path.exists():
        backup = path.with_suffix(".backup")
        backup.write_text(path.read_text())
        logger.debug(f"📦 Backup creado: {backup}")

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def normalize_tickers(data: Any) -> List[str]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "tickers" in data and isinstance(data["tickers"], list):
        return data["tickers"]
    raise RuntimeError("Formato inválido en tickers.json")

# =========================
# Core logic
# =========================
def run_decider() -> Dict[str, Any]:
    # -------------------------
    # Leer screener DESDE DISCO
    # -------------------------
    screener = load_json(SCREENER_FILE)
    candidates = screener.get("candidates", [])

    if not isinstance(candidates, list):
        raise RuntimeError("Formato inválido en screener_candidates.json")

    # -------------------------
    # Cargar tickers.json existente
    # -------------------------
    if TICKERS_FILE.exists():
        tickers_data = load_json(TICKERS_FILE)
        tickers_list = normalize_tickers(tickers_data)
        logger.info(f"📂 Cargados {len(tickers_list)} tickers existentes")
    else:
        logger.warning("⚠️ tickers.json no existe, se creará")
        tickers_list = []

    tickers_set = set(tickers_list)
    added: List[str] = []

    # -------------------------
    # Evaluar candidatos (MISMA LÓGICA)
    # -------------------------
    for c in candidates:
        try:
            ticker = c.get("ticker")
            score = float(c.get("score", 0))
            quality = c.get("quality")

            if not ticker:
                continue
            if quality not in ALLOWED_QUALITIES:
                continue
            if score < MIN_SCORE:
                continue

            if ticker not in tickers_set:
                tickers_set.add(ticker)
                added.append(ticker)
                logger.info(f"➕ AGREGADO: {ticker} ({quality}, score={score:.3f})")

        except Exception as e:
            logger.debug(f"Skip candidato: {e}")

    # -------------------------
    # Guardar ATÓMICAMENTE
    # -------------------------
    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "source": "screener_decider_v2.2",
        "total_tickers": len(tickers_set),
        "added_today": len(added),
        "tickers": sorted(tickers_set),
    }

    save_json_atomic(TICKERS_FILE, output)

    logger.info(
        f"✅ Decider v2.2 | nuevos={len(added)} | total={len(tickers_set)}"
    )

    return {
        "ok": True,
        "added": added,
        "total": len(tickers_set),
        "file": str(TICKERS_FILE),
    }

# =========================
# CLI (debug manual)
# =========================
if __name__ == "__main__":
    try:
        result = run_decider()
        print("\n🧠 TICKERS NUEVOS AGREGADOS:")
        if result["added"]:
            for t in result["added"]:
                print(f"➕ {t}")
        else:
            print("ℹ️  Ningún nuevo candidato calificado")

        print(f"\n📦 Total tickers en memoria: {result['total']}")
        print(f"💾 Archivo: {result['file']}")

    except Exception as e:
        logger.error(f"❌ Decider falló: {e}")
        exit(1)

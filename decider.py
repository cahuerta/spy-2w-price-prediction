# =========================================================
# decider.py — SCREENER → TICKERS MEMORY DECIDER v3.0
# =========================================================
# ✅ Lee screener_candidates.json
# ✅ Usa candidates_strict (contrato real)
# ✅ Soporta emojis en quality
# ✅ Guarda tickers.json como LISTA PURA
# ✅ Limpia estructura inválida automáticamente
# ❌ Nunca borra tickers válidos
# =========================================================

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# =========================
# CONFIG
# =========================
DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))

SCREENER_FILE = DATA_PATH / "screener_candidates.json"
TICKERS_FILE = DATA_PATH / "tickers.json"

MIN_SCORE = 0.75

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s"
)
logger = logging.getLogger("decider_v3")

# =========================================================
# HELPERS
# =========================================================

def load_json(path: Path):
    if not path.exists():
        raise RuntimeError(f"Archivo requerido no existe: {path}")
    return json.loads(path.read_text())


def save_json_atomic(path: Path, data: List[str]):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def normalize_tickers(data: Any) -> List[str]:
    """
    Garantiza que tickers.json sea lista limpia.
    Elimina basura estructural.
    """
    if isinstance(data, list):
        return [str(x) for x in data if isinstance(x, str)]

    if isinstance(data, dict):
        # si viene formato viejo con metadata
        if "tickers" in data and isinstance(data["tickers"], list):
            return [str(x) for x in data["tickers"] if isinstance(x, str)]

    logger.warning("⚠️ tickers.json corrupto → se reinicializa limpio")
    return []


def quality_is_valid(q: str) -> bool:
    if not q:
        return False
    return "STRONG" in q or "INSTITUTIONAL" in q


# =========================================================
# CORE
# =========================================================

def run_decider() -> Dict[str, Any]:

    logger.info("🧠 Decider v3.0 iniciado")

    # -------------------------
    # Leer screener
    # -------------------------
    screener = load_json(SCREENER_FILE)

    candidates = screener.get("candidates_strict", [])

    if not isinstance(candidates, list):
        raise RuntimeError("Formato inválido en screener_candidates.json")

    # -------------------------
    # Cargar tickers actuales
    # -------------------------
    if TICKERS_FILE.exists():
        raw = load_json(TICKERS_FILE)
        tickers_list = normalize_tickers(raw)
        logger.info(f"📂 {len(tickers_list)} tickers cargados")
    else:
        tickers_list = []
        logger.info("📦 tickers.json no existía → se creará")

    tickers_set = set(tickers_list)
    added = []

    # -------------------------
    # Evaluar candidatos
    # -------------------------
    for c in candidates:
        try:
            ticker = c.get("ticker")
            score = float(c.get("score", 0))
            quality = c.get("quality", "")

            if not ticker:
                continue
            if score < MIN_SCORE:
                continue
            if not quality_is_valid(quality):
                continue

            if ticker not in tickers_set:
                tickers_set.add(ticker)
                added.append(ticker)
                logger.info(f"➕ AGREGADO: {ticker} | score={score:.3f}")

        except Exception as e:
            logger.debug(f"Skip candidato: {e}")

    # -------------------------
    # Guardar lista limpia
    # -------------------------
    final_list = sorted(tickers_set)

    save_json_atomic(TICKERS_FILE, final_list)

    logger.info(
        f"✅ Decider finalizado | nuevos={len(added)} | total={len(final_list)}"
    )

    return {
        "ok": True,
        "added": added,
        "total": len(final_list),
        "file": str(TICKERS_FILE),
        "timestamp": datetime.utcnow().isoformat(),
    }


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    try:
        result = run_decider()
        print("\n🧠 NUEVOS TICKERS:")
        if result["added"]:
            for t in result["added"]:
                print(f"➕ {t}")
        else:
            print("ℹ️ Ningún nuevo ticker agregado")

        print(f"\n📦 Total tickers: {result['total']}")
        print(f"💾 Archivo: {result['file']}")

    except Exception as e:
        logger.error(f"❌ Decider falló: {e}")
        exit(1)

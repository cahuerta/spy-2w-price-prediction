# =========================================================
# decision_log.py — SYSTEM DECISION LOGGER v1.6 (FINAL)
# =========================================================
# ✅ Append-only JSONL
# ✅ Thread-safe
# ✅ OS-level atomic append
# ✅ Rotación segura
# =========================================================

import json
from pathlib import Path
from datetime import datetime
from threading import Lock
from typing import Dict, Any

DATA_PATH = Path("/data")
LOG_FILE = DATA_PATH / "decision_log.jsonl"

MAX_SIZE_MB = 50
MAX_LINES = 100_000

_lock = Lock()

def _needs_rotation() -> bool:
    if not LOG_FILE.exists():
        return False

    stat = LOG_FILE.stat()
    if stat.st_size > MAX_SIZE_MB * 1024 * 1024:
        return True

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for i, _ in enumerate(f, 1):
                if i > MAX_LINES:
                    return True
    except Exception:
        return False

    return False

def _rotate_log():
    old = LOG_FILE.with_suffix(".old")
    old.unlink(missing_ok=True)
    LOG_FILE.rename(old)

def log_decision(event: Dict[str, Any]):
    """
    Registra una decisión del sistema.
    Append-only, seguro y simple.
    """
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        **event,
    }

    line = json.dumps(record, ensure_ascii=False) + "\n"

    with _lock:
        if _needs_rotation():
            _rotate_log()

        # OS-level atomic append
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)

# =========================
# Test CLI
# =========================
if __name__ == "__main__":
    log_decision({"test": "system_ready", "pid": 12345})
    print(f"✅ Log escrito: {LOG_FILE}")

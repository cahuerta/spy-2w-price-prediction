# =========================================================
# pipeline_daily.py — PIPELINE TRIGGER (CRON)
# =========================================================
# ✔ NO ejecuta lógica
# ✔ NO importa módulos del sistema
# ✔ NO escribe a disco
# ✔ SOLO despierta a Main vía HTTP
# =========================================================

import os
import sys
import requests
from datetime import datetime

# =========================
# CONFIG
# =========================
MAIN_URL = os.getenv(
    "MAIN_PIPELINE_URL",
    "https://spy-2w-price-prediction.onrender.com/internal/pipeline/run"
)

PIPELINE_KEY = os.getenv("PIPELINE_KEY")

if not PIPELINE_KEY:
    print("❌ PIPELINE_KEY no definida")
    sys.exit(1)

# =========================
# TRIGGER
# =========================
def main():
    ts = datetime.utcnow().isoformat()
    print("=" * 60)
    print(f"🚀 PIPELINE TRIGGER START | {ts}")
    print("=" * 60)

    try:
        res = requests.post(
            MAIN_URL,
            headers={
                "X-PIPELINE-KEY": PIPELINE_KEY,
                "Content-Type": "application/json",
            },
            timeout=60 * 30,  # hasta 30 min
        )

        if res.status_code != 200:
            print("❌ Pipeline HTTP error")
            print(res.status_code, res.text)
            sys.exit(1)

        print("✅ Pipeline triggered OK")
        print(res.json())

    except Exception as e:
        print("❌ Pipeline trigger failed")
        print(str(e))
        sys.exit(1)

    print("=" * 60)
    print(f"🏁 PIPELINE TRIGGER END | {datetime.utcnow().isoformat()}")
    print("=" * 60)


# =========================
# ENTRYPOINT (CRON)
# =========================
if __name__ == "__main__":
    main()

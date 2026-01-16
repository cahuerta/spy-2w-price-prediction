# =========================================================
# pipeline_daily.py — ORQUESTADOR DIARIO (FINAL REAL)
# =========================================================
# Flujo:
# 1) Screener
# 2) Backend: decider + evaluación + broker (paper)
#
# ✔ Un solo cron
# ✔ Backend es la única fuente de verdad
# =========================================================

from datetime import datetime
import os
import traceback
import requests

from screener import run_screener

BACKEND_URL = os.getenv("BACKEND_URL")
PIPELINE_KEY = os.getenv("PIPELINE_KEY")

HEADERS = {"X-PIPELINE-KEY": PIPELINE_KEY}


def post(path: str, payload=None):
    r = requests.post(
        f"{BACKEND_URL}{path}",
        json=payload,
        headers=HEADERS,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def main():
    start_ts = datetime.utcnow().isoformat()
    print("=" * 60)
    print(f"🚀 PIPELINE DAILY START | {start_ts}")
    print("=" * 60)

    try:
        # -------------------------
        # 1) SCREENER
        # -------------------------
        print("🔍 [1/2] Running SCREENER...")
        screener_out = run_screener()
        print(f"✅ Screener OK | candidates={screener_out.get('n_candidates')}")

        post("/internal/screener/result", screener_out)
        print("📤 Screener enviado al backend")

        # -------------------------
        # 2) DAILY SYSTEM RUN
        # -------------------------
        print("🧠 [2/2] Running DAILY SYSTEM...")
        out = post("/internal/system/daily-run")

        print(
            f"✅ Daily run OK | decisions={len(out.get('decisions', []))} | "
            f"executed={len(out.get('executed', []))}"
        )

    except Exception as e:
        print("❌ PIPELINE FAILED")
        print(str(e))
        traceback.print_exc()
        return

    end_ts = datetime.utcnow().isoformat()
    print("=" * 60)
    print(f"🏁 PIPELINE DAILY END | {end_ts}")
    print("=" * 60)


if __name__ == "__main__":
    main()

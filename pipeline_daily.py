# =========================================================
# pipeline_daily.py — ORQUESTADOR DIARIO DEL SISTEMA (FINAL)
# =========================================================
# Ejecuta TODO el pipeline en orden:
# 1) Screener
# 2) Decider (vía backend)
# 3) Save All (model / predictions)
# 4) Evaluator / Signals
# 5) PositionManager (decisiones)
#
# ✔ UN SOLO CRON JOB
# ✔ Post-market
# ✔ NO ejecuta trades
# ✔ Broker queda en DRY / manual
# =========================================================

from datetime import datetime, timezone
import os
import sys
import traceback
import requests

from screener import run_screener

# =========================================================
# CONFIG
# =========================================================
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://spy-2w-price-prediction.onrender.com"
)

PIPELINE_KEY = os.getenv("PIPELINE_KEY", "")

TIMEOUT = 120  # segundos por etapa

# =========================================================
# HELPERS
# =========================================================
def now_utc():
    return datetime.now(timezone.utc).isoformat()

def call_backend(method: str, path: str, payload: dict | None = None):
    url = f"{BACKEND_URL}{path}"
    headers = {
        "X-PIPELINE-KEY": PIPELINE_KEY,
        "Content-Type": "application/json",
    }

    r = requests.request(
        method=method,
        url=url,
        json=payload,
        headers=headers,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()

# =========================================================
# PIPELINE
# =========================================================
def main():
    print("=" * 70)
    print(f"🚀 DAILY PIPELINE START | {now_utc()}")
    print("=" * 70)

    # -----------------------------------------------------
    # 1) SCREENER (LOCAL)
    # -----------------------------------------------------
    try:
        print("\n🔍 [1/5] Running SCREENER...")
        screener_out = run_screener()

        n = screener_out.get("n_candidates", 0)
        print(f"✅ Screener OK | candidates={n}")

    except Exception as e:
        print("❌ SCREENER FAILED")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------
    # 2) PUSH SCREENER → BACKEND (DECIDER AUTO)
    # -----------------------------------------------------
    try:
        print("\n📤 [2/5] Sending screener to backend (DECIDER)...")
        res = call_backend(
            "POST",
            "/internal/screener/result",
            screener_out,
        )

        added = len(res.get("decider", {}).get("added", []))
        total = res.get("decider", {}).get("total")

        print(f"🧠 Decider OK | added={added} | total={total}")

    except Exception as e:
        print("❌ DECIDER FAILED")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------
    # 3) SAVE ALL (MODELS / PREDICTIONS)
    # -----------------------------------------------------
    try:
        print("\n🧮 [3/5] Running SAVE-ALL (models / predictions)...")
        res = call_backend("POST", "/internal/run/save_all")
        print(f"✅ Save-all OK | tickers={res.get('tickers')}")

    except Exception as e:
        print("❌ SAVE-ALL FAILED")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------
    # 4) EVALUATOR / SIGNALS
    # -----------------------------------------------------
    try:
        print("\n📊 [4/5] Running EVALUATOR / SIGNALS...")
        res = call_backend("POST", "/internal/run/evaluator")
        print(f"✅ Evaluator OK | signals={res.get('count')}")

    except Exception as e:
        print("❌ EVALUATOR FAILED")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------
    # 5) POSITION MANAGER (DECISIONS ONLY)
    # -----------------------------------------------------
    try:
        print("\n🧠 [5/5] Running POSITION MANAGER...")
        res = call_backend("POST", "/internal/run/position_manager")

        print(
            f"✅ PM OK | actions={res.get('actions')} | "
            f"health={res.get('health')}"
        )

    except Exception as e:
        print("❌ POSITION MANAGER FAILED")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------
    # END
    # -----------------------------------------------------
    print("=" * 70)
    print(f"🏁 DAILY PIPELINE END | {now_utc()}")
    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()

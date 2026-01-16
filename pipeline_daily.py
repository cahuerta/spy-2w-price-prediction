# =========================================================
# pipeline_daily.py — ORQUESTADOR DIARIO DEL SISTEMA (FINAL)
# =========================================================
# Flujo completo diario:
# 1) Screener
# 2) Persistir screener
# 3) Decider (memoria de tickers)
# 4) Save All (prices + signals)
# 5) Evaluador de portafolio (PositionManager)
#
# ✔ Un solo cron job
# ✔ NO ejecuta trades (todavía)
# ✔ Broker queda listo para PAPER
# =========================================================

from datetime import datetime
import os
import traceback
import requests

from screener import run_screener
from decider import run_decider

BACKEND_URL = os.getenv("BACKEND_URL")
PIPELINE_KEY = os.getenv("PIPELINE_KEY")

HEADERS = {"X-PIPELINE-KEY": PIPELINE_KEY}


# ---------------------------------------------------------
# Helpers backend
# ---------------------------------------------------------
def post(path: str, payload: dict | None = None):
    url = f"{BACKEND_URL}{path}"
    r = requests.post(url, json=payload, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    start_ts = datetime.utcnow().isoformat()
    print("=" * 60)
    print(f"🚀 PIPELINE DAILY START | {start_ts}")
    print("=" * 60)

    try:
        # -------------------------
        # 1) SCREENER
        # -------------------------
        print("🔍 [1/5] Running SCREENER...")
        screener_out = run_screener()
        print(f"✅ Screener OK | candidates={screener_out.get('n_candidates')}")

        post("/internal/screener/result", screener_out)
        print("📤 Screener persistido")

        # -------------------------
        # 2) DECIDER
        # -------------------------
        print("🧠 [2/5] Running DECIDER...")
        decider_out = run_decider()
        print(
            f"✅ Decider OK | added={len(decider_out.get('added', []))} | "
            f"total={decider_out.get('total')}"
        )

        # -------------------------
        # 3) SAVE ALL (prices + signals)
        # -------------------------
        print("💾 [3/5] Saving ALL market data...")
        post("/internal/save_all")
        print("✅ Save all OK")

        # -------------------------
        # 4) EVALUADOR PORTAFOLIO
        # -------------------------
        print("📊 [4/5] Evaluating portfolio...")
        eval_out = post("/internal/portfolio/evaluate")
        print(
            f"✅ Portfolio evaluated | action={eval_out.get('next_action')}"
        )

        # -------------------------
        # 5) (FUTURO) BROKER
        # -------------------------
        print("🤖 [5/5] Broker: NO EXECUTION (paper disabled)")
        print("ℹ️  Broker listo pero desactivado por diseño")

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

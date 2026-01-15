# =========================================================
# pipeline_daily.py — ORQUESTADOR DIARIO DEL SISTEMA
# =========================================================
# Ejecuta en orden:
# 1) Screener (observa mercado + contexto fundamental)
# 2) Decider (agrega tickers nuevos a tickers.json)
#
# ✔ Un solo punto de entrada
# ✔ Diseñado para cron job (Render / Linux)
# ✔ NO ejecuta trades
# ✔ NO toca signals, model, broker
# =========================================================

from datetime import datetime
import traceback

# Importaciones del pipeline
from screener import run_screener
from decider import run_decider
import os
import requests


def main():
    start_ts = datetime.utcnow().isoformat()
    print("=" * 60)
    print(f"🚀 PIPELINE START | {start_ts}")
    print("=" * 60)
 # -------------------------
    # 1) HELPERS
    # -------------------------
    BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://spy-2w-price-prediction.onrender.com"
)

PIPELINE_KEY = os.getenv("PIPELINE_KEY", "")


def push_screener_to_backend(result: dict):
    r = requests.post(
        f"{BACKEND_URL}/internal/screener/result",
        json=result,
        timeout=30,
        headers={
            "X-PIPELINE-KEY": PIPELINE_KEY
        },
    )
    r.raise_for_status()

    # -------------------------
    # 1) SCREENER
    # -------------------------
    try:
        print("🔍 [1/2] Running SCREENER...")
        screener_out = run_screener()

n_candidates = screener_out.get("n_candidates")
print(f"✅ Screener OK | candidates={n_candidates}")

push_screener_to_backend(screener_out)
print("📤 Screener enviado al backend")


    except Exception as e:
        print("❌ Screener FAILED")
        print(str(e))
        traceback.print_exc()
        # Cortamos: decider no tiene sentido sin screener
        return

    # -------------------------
    # 2) DECIDER
    # -------------------------
    try:
        print("🧠 [2/2] Running DECIDER...")
        decider_out = run_decider()

        added = decider_out.get("added", [])
        total = decider_out.get("total")

        print(f"✅ Decider OK | added_today={len(added)} | total_tickers={total}")

        if added:
            print("➕ Nuevos tickers agregados:")
            for t in added:
                print(f"   • {t}")
        else:
            print("ℹ️  Ningún ticker nuevo agregado hoy")

    except Exception as e:
        print("❌ Decider FAILED")
        print(str(e))
        traceback.print_exc()
        return

    # -------------------------
    # FIN
    # -------------------------
    end_ts = datetime.utcnow().isoformat()
    print("=" * 60)
    print(f"🏁 PIPELINE END | {end_ts}")
    print("=" * 60)


# =========================================================
# Entry point (cron / CLI)
# =========================================================
if __name__ == "__main__":
    main()

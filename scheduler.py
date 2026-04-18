# scheduler.py
# Reemplaza el cron externo (pipeline_daily.py).
# Corre como thread daemon dentro del proceso FastAPI.
#
# Responsabilidades:
#   - Disparar el pipeline una vez al día a la hora configurada
#   - Loggear estado cada hora (alive check)

import os
import time
import threading
import requests
from datetime import datetime, date
import pytz

CHILE_TZ       = pytz.timezone("America/Santiago")
PIPELINE_URL   = os.getenv(
    "MAIN_PIPELINE_URL",
    "https://spy-2w-price-prediction.onrender.com/internal/pipeline/run"
)
PIPELINE_KEY   = os.getenv("PIPELINE_KEY", "")
HORA_EJECUCION = int(os.getenv("SCHEDULER_HORA", "8"))  # 08:00 hora Chile por defecto


def _trigger_pipeline():
    ts = datetime.now(CHILE_TZ).isoformat()
    print(f"🚀 Disparando pipeline | {ts}")
    try:
        res = requests.post(
            PIPELINE_URL,
            headers={
                "X-PIPELINE-KEY": PIPELINE_KEY,
                "Content-Type":   "application/json",
            },
            timeout=60 * 30
        )
        if res.status_code == 200:
            print(f"✅ Pipeline OK | {res.json()}")
        else:
            print(f"❌ Pipeline HTTP {res.status_code} | {res.text}")
    except Exception as e:
        print(f"❌ Pipeline trigger failed: {e}")


def _loop():
    print("🕐 Quant Scheduler iniciado")
    ultimo_run:    date | None = None
    ultimo_log_h:  int | None  = None

    while True:
        ahora = datetime.now(CHILE_TZ)
        hoy   = ahora.date()

        # Pipeline una vez al día
        if ahora.hour == HORA_EJECUCION and ahora.minute < 5 and ultimo_run != hoy:
            _trigger_pipeline()
            ultimo_run = hoy

        # Alive log cada hora
        if ahora.minute < 5 and ultimo_log_h != ahora.hour:
            print(f"💓 Scheduler alive | {ahora.strftime('%Y-%m-%d %H:%M')} Chile")
            ultimo_log_h = ahora.hour

        time.sleep(60)


def start_scheduler():
    if not PIPELINE_KEY:
        print("⚠️  PIPELINE_KEY no definida — scheduler no iniciado")
        return
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    print("🚀 Quant Scheduler iniciado")

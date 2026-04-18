# scheduler.py
# Reemplaza el cron externo (pipeline_daily.py).
# Corre como thread daemon dentro del proceso FastAPI.
#
# Responsabilidades:
#   - Disparar pipeline 30 min después de apertura NY (11:00 Chile)
#   - Disparar pipeline 30 min antes de cierre NY (16:30 Chile)
#   - Solo lunes a viernes (días hábiles)
#   - Loggear estado cada hora (alive check)
#
# Horario mercado US en hora Chile (verano UTC-3):
#   Apertura  09:30 ET = 10:30 Chile → pipeline 11:00 Chile
#   Cierre    16:00 ET = 17:00 Chile → pipeline 16:30 Chile

import os
import time
import threading
import requests
from datetime import datetime, date
import pytz

CHILE_TZ     = pytz.timezone("America/Santiago")
PIPELINE_URL = os.getenv(
    "MAIN_PIPELINE_URL",
    "https://spy-2w-price-prediction.onrender.com/internal/pipeline/run"
)
PIPELINE_KEY = os.getenv("PIPELINE_KEY", "")

# Horas de ejecución hora Chile
HORA_APERTURA = 11    # 30 min después de apertura
HORA_CIERRE   = 16    # 30 min antes de cierre
MIN_CIERRE    = 30


def _trigger_pipeline(motivo: str):
    ts = datetime.now(CHILE_TZ).isoformat()
    print(f"🚀 Disparando pipeline [{motivo}] | {ts}")
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
            print(f"✅ Pipeline OK [{motivo}] | {res.json()}")
        else:
            print(f"❌ Pipeline HTTP {res.status_code} [{motivo}] | {res.text}")
    except Exception as e:
        print(f"❌ Pipeline trigger failed [{motivo}]: {e}")


def _es_dia_habil(ahora: datetime) -> bool:
    return ahora.weekday() < 5  # 0=lunes … 4=viernes


def _loop():
    print("🕐 Quant Scheduler iniciado")

    ultimo_apertura: date | None = None
    ultimo_cierre:   date | None = None
    ultimo_log_h:    int  | None = None

    while True:
        ahora = datetime.now(CHILE_TZ)
        hoy   = ahora.date()

        if _es_dia_habil(ahora):

            # — Ejecución apertura: 11:00 Chile
            if (ahora.hour == HORA_APERTURA
                    and ahora.minute < 5
                    and ultimo_apertura != hoy):
                _trigger_pipeline("apertura")
                ultimo_apertura = hoy

            # — Ejecución cierre: 16:30 Chile
            if (ahora.hour == HORA_CIERRE
                    and ahora.minute >= MIN_CIERRE
                    and ahora.minute < MIN_CIERRE + 5
                    and ultimo_cierre != hoy):
                _trigger_pipeline("cierre")
                ultimo_cierre = hoy

        # — Alive log cada hora
        if ahora.minute < 5 and ultimo_log_h != ahora.hour:
            dia = "hábil" if _es_dia_habil(ahora) else "fin de semana"
            print(f"💓 Scheduler alive | {ahora.strftime('%Y-%m-%d %H:%M')} Chile | {dia}")
            ultimo_log_h = ahora.hour

        time.sleep(60)


def start_scheduler():
    if not PIPELINE_KEY:
        print("⚠️  PIPELINE_KEY no definida — scheduler no iniciado")
        return
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    print("🚀 Quant Scheduler iniciado")
    

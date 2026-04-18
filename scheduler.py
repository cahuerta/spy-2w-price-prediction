# scheduler.py
# Corre como thread daemon dentro del proceso FastAPI.
#
# Responsabilidades:
#   - Disparar pipeline cada 30 min en horario de mercado
#   - El pipeline tiene smart skip — pasos 1-4 solo corren
#     la primera vez del día. Pasos 5-10 + tracker siempre.
#   - Solo lunes a viernes
#   - Alive log cada hora
#
# Horario mercado US en hora Chile (verano UTC-3):
#   Apertura  09:30 ET = 10:30 Chile → primera ejecución 11:00
#   Cierre    16:00 ET = 17:00 Chile → última ejecución 16:30

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

HORA_INICIO = 11
HORA_FIN    = 16
MIN_FIN     = 30


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
    return ahora.weekday() < 5


def _en_horario_mercado(ahora: datetime) -> bool:
    if ahora.hour > HORA_INICIO and ahora.hour < HORA_FIN:
        return True
    if ahora.hour == HORA_INICIO:
        return True
    if ahora.hour == HORA_FIN and ahora.minute < MIN_FIN:
        return True
    return False


def _loop():
    print("🕐 Quant Scheduler iniciado")

    ultimo_trigger: int | None = None
    ultimo_log_h:   int | None = None

    while True:
        ahora = datetime.now(CHILE_TZ)

        if _es_dia_habil(ahora) and _en_horario_mercado(ahora):
            slot        = 0 if ahora.minute < 30 else 30
            trigger_key = ahora.hour * 100 + slot

            if ultimo_trigger != trigger_key:
                motivo = f"{ahora.hour:02d}:{slot:02d}"
                _trigger_pipeline(motivo)
                ultimo_trigger = trigger_key

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
    

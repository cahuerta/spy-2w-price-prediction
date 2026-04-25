# scheduler.py
# Corre como thread daemon dentro del proceso FastAPI.
#
# Responsabilidades:
#   - Disparar pipeline UNA vez al día a las 11:30 Chile
#     (mercado US abierto desde 10:30, precios disponibles)
#   - Solo lunes a viernes
#   - Alive log cada hora
#   - Darwin Engine: resolver trades diario post-market (17:05 Chile)
#   - Darwin Engine: ciclo evolutivo viernes post-market (18:00 Chile)
#
# Horario mercado US en hora Chile (verano UTC-3):
#   Apertura  09:30 ET = 10:30 Chile
#   Cierre    16:00 ET = 17:00 Chile
#
# ¿Por qué una vez al día?
#   - Suscripción Alpaca no permite datos intraday SIP
#   - Horizontes H1-H10 son diarios, no intradía
#   - Evita el loop buy/sell que ocurría cada 30 min
#   - Una predicción diaria con datos EOD es suficiente y correcta

import os
import time
import threading
import requests
from datetime import datetime
import pytz

CHILE_TZ     = pytz.timezone("America/Santiago")
PIPELINE_URL = os.getenv(
    "MAIN_PIPELINE_URL",
    "https://spy-2w-price-prediction.onrender.com/internal/pipeline/run"
)
PIPELINE_KEY = os.getenv("PIPELINE_KEY", "")

# ── Hora de ejecución diaria ──────────────────────────
HORA_PIPELINE  = int(os.getenv("PIPELINE_HOUR", "11"))
MIN_PIPELINE   = int(os.getenv("PIPELINE_MIN",  "30"))


# ══════════════════════════════════════════════════════
# PIPELINE TRIGGER
# ══════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════
# DARWIN ENGINE TRIGGERS
# ══════════════════════════════════════════════════════

def _trigger_darwin_resolve(motivo: str):
    """
    Resuelve trades cuyo horizonte ya venció.
    Corre diario post-market.
    """
    print(f"🧬 Darwin resolve_pending_trades [{motivo}]")
    try:
        from darwin_engine.trade_tracker import resolve_pending_trades
        resolved = resolve_pending_trades()
        print(f"✅ Darwin trades resueltos: {len(resolved)}")
        for t in resolved[:5]:
            print(
                f"   {t.get('ticker')} | "
                f"PnL real={t.get('pnl_real_pct', 0):+.2f}% | "
                f"PnL teórico={t.get('pnl_teorico_pct', 0):+.2f}% | "
                f"Oportunidad={t.get('oportunidad_pct', 0):+.2f}%"
            )
    except Exception as e:
        print(f"❌ Darwin resolve error: {e}")


def _trigger_darwin_evolution(motivo: str):
    """
    Ciclo evolutivo completo viernes post-market:
      1. Executor arena
      2. Predictor arena H1-H10
    """
    print(f"🧬 Darwin evolution cycle [{motivo}]")

    try:
        from darwin_engine.arena import run_evolution_cycle
        result = run_evolution_cycle()
        print(
            f"✅ Executor ciclo | "
            f"campeón={result.get('champion_after')} | "
            f"promovido={result.get('was_promoted')} | "
            f"nueva_gen={len(result.get('new_generation', []))}"
        )
    except Exception as e:
        print(f"❌ Executor arena error: {e}")

    try:
        from darwin_engine.predictor_arena import run_predictor_evolution
        pred_result = run_predictor_evolution()
        print(
            f"✅ Predictor ciclo | "
            f"H evaluados={pred_result.get('h_evaluated')} | "
            f"promovidos={pred_result.get('promotions')}"
        )
        for r in pred_result.get("ranking", []):
            print(f"   H{r['horizon']}: {r['hit_rate']:.2%}")
    except Exception as e:
        print(f"❌ Predictor arena error: {e}")


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def _es_dia_habil(ahora: datetime) -> bool:
    return ahora.weekday() < 5


# ══════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════

def _loop():
    print(f"🕐 Quant Scheduler iniciado | pipeline={HORA_PIPELINE:02d}:{MIN_PIPELINE:02d} Chile")

    pipeline_hoy:         str | None = None
    ultimo_log_h:         int | None = None
    darwin_resolve_hoy:   str | None = None
    darwin_evolution_hoy: str | None = None

    while True:
        ahora     = datetime.now(CHILE_TZ)
        fecha_hoy = ahora.strftime("%Y-%m-%d")

        # ── Pipeline: UNA vez al día a las 11:30 ──────
        if (
            _es_dia_habil(ahora)
            and ahora.hour == HORA_PIPELINE
            and ahora.minute >= MIN_PIPELINE
            and ahora.minute < MIN_PIPELINE + 5   # ventana de 5 min
            and pipeline_hoy != fecha_hoy
        ):
            _trigger_pipeline(f"diario_{HORA_PIPELINE:02d}:{MIN_PIPELINE:02d}")
            pipeline_hoy = fecha_hoy

        # ── Darwin: resolver trades (17:05 Chile) ─────
        if (
            _es_dia_habil(ahora)
            and ahora.hour == 17
            and 5 <= ahora.minute < 15
            and darwin_resolve_hoy != fecha_hoy
        ):
            _trigger_darwin_resolve("post_market_17:05")
            darwin_resolve_hoy = fecha_hoy

        # ── Darwin: ciclo evolutivo (viernes 18:00) ───
        if (
            ahora.weekday() == 4
            and ahora.hour == 18
            and ahora.minute < 10
            and darwin_evolution_hoy != fecha_hoy
        ):
            _trigger_darwin_evolution("viernes_18:00")
            darwin_evolution_hoy = fecha_hoy

        # ── Alive log cada hora ───────────────────────
        if ahora.minute < 5 and ultimo_log_h != ahora.hour:
            dia = "hábil" if _es_dia_habil(ahora) else "fin de semana"
            print(
                f"💓 Scheduler alive | "
                f"{ahora.strftime('%Y-%m-%d %H:%M')} Chile | {dia} | "
                f"pipeline_hoy={'✅' if pipeline_hoy == fecha_hoy else '⏳'}"
            )
            ultimo_log_h = ahora.hour

        time.sleep(60)


# ══════════════════════════════════════════════════════
# START
# ══════════════════════════════════════════════════════

def start_scheduler():
    if not PIPELINE_KEY:
        print("⚠️  PIPELINE_KEY no definida — scheduler no iniciado")
        return
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    print("🚀 Quant Scheduler iniciado")
    

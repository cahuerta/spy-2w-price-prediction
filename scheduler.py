# scheduler.py
# Corre como thread daemon dentro del proceso FastAPI.
#
# DOS FRECUENCIAS:
#
# 1. PIPELINE COMPLETO — una vez al día 11:30 Chile
#    → Predicciones H1-H10 con datos Alpaca EOD
#    → Decisiones de trading (OPEN/CLOSE)
#    → Ejecuta órdenes en Alpaca
#
# 2. MONITOREO HORARIO — cada hora 12:00-16:00 Chile
#    → Precios Yahoo Finance (15 min delay, gratis)
#    → Actualiza intraday_tracker
#    → Si posición en diverging → cierra (protección)
#    → NO abre nuevas posiciones
#
# Darwin Engine:
#    → resolve trades: diario 17:05 Chile
#    → ciclo evolutivo: viernes 18:00 Chile

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

HORA_PIPELINE = int(os.getenv("PIPELINE_HOUR", "11"))
MIN_PIPELINE  = int(os.getenv("PIPELINE_MIN",  "30"))

# Ventana de monitoreo horario
MONITOR_HORA_INICIO = int(os.getenv("MONITOR_HORA_INICIO", "12"))
MONITOR_HORA_FIN    = int(os.getenv("MONITOR_HORA_FIN",    "16"))


# ══════════════════════════════════════════════════════
# PIPELINE COMPLETO (una vez al día)
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
# MONITOREO HORARIO (Yahoo Finance)
# ══════════════════════════════════════════════════════

def _trigger_monitor(motivo: str):
    """
    Monitoreo horario de posiciones usando Yahoo Finance.
    Solo evalúa posiciones abiertas — NO ejecuta nuevas compras.
    Si detecta diverging → cierra vía endpoint de trading.
    """
    print(f"📡 Monitor horario [{motivo}]")
    try:
        from intraday_tracker import run_intraday_tracker
        result = run_intraday_tracker()

        n_posiciones = len(result.get("monitor_posiciones", []))
        n_candidatos = result.get("candidatos", 0)
        print(
            f"✅ Monitor OK | "
            f"posiciones={n_posiciones} | "
            f"candidatos={n_candidatos}"
        )

        # Si hay posiciones en diverging → disparar cierre defensivo
        diverging = [
            p for p in result.get("monitor_posiciones", [])
            if p.get("curve_status") == "diverging"
        ]
        if diverging:
            tickers = [p["ticker"] for p in diverging]
            print(f"⚠️ DIVERGING detectado: {tickers} → disparando cierre defensivo")
            _trigger_defensive_close(tickers)

    except Exception as e:
        print(f"❌ Monitor error: {e}")


def _trigger_defensive_close(tickers: list):
    """
    Dispara cierre de posiciones divergentes.
    Llama al endpoint de trading con flag solo_cierres=True.
    """
    try:
        res = requests.post(
            PIPELINE_URL.replace("/pipeline/run", "/trading/monitor-close"),
            headers={
                "X-PIPELINE-KEY": PIPELINE_KEY,
                "Content-Type":   "application/json",
            },
            json={"tickers": tickers, "reason": "intraday_diverging"},
            timeout=60,
        )
        if res.status_code == 200:
            print(f"✅ Cierre defensivo ejecutado: {tickers}")
        else:
            print(f"⚠️ Cierre defensivo HTTP {res.status_code}: {res.text}")
    except Exception as e:
        print(f"❌ Cierre defensivo error: {e}")


# ══════════════════════════════════════════════════════
# DARWIN ENGINE TRIGGERS
# ══════════════════════════════════════════════════════

def _trigger_darwin_resolve(motivo: str):
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


def _en_horario_monitor(ahora: datetime) -> bool:
    return MONITOR_HORA_INICIO <= ahora.hour <= MONITOR_HORA_FIN


# ══════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════

def _loop():
    print(
        f"🕐 Quant Scheduler iniciado | "
        f"pipeline={HORA_PIPELINE:02d}:{MIN_PIPELINE:02d} Chile | "
        f"monitor={MONITOR_HORA_INICIO:02d}:00-{MONITOR_HORA_FIN:02d}:00 Chile"
    )

    pipeline_hoy:         str | None = None
    ultimo_log_h:         int | None = None
    monitor_ultima_hora:  int | None = None
    darwin_resolve_hoy:   str | None = None
    darwin_evolution_hoy: str | None = None

    while True:
        ahora     = datetime.now(CHILE_TZ)
        fecha_hoy = ahora.strftime("%Y-%m-%d")

        # ── PIPELINE: una vez al día 11:30 ────────────
        if (
            _es_dia_habil(ahora)
            and ahora.hour == HORA_PIPELINE
            and MIN_PIPELINE <= ahora.minute < MIN_PIPELINE + 5
            and pipeline_hoy != fecha_hoy
        ):
            _trigger_pipeline(f"diario_{HORA_PIPELINE:02d}:{MIN_PIPELINE:02d}")
            pipeline_hoy = fecha_hoy

        # ── MONITOR HORARIO: cada hora 12:00-16:00 ────
        # Solo si el pipeline ya corrió hoy (hay predicciones frescas)
        if (
            _es_dia_habil(ahora)
            and _en_horario_monitor(ahora)
            and ahora.minute < 5          # primera oportunidad de cada hora
            and monitor_ultima_hora != ahora.hour
            and pipeline_hoy == fecha_hoy  # pipeline ya corrió hoy
        ):
            _trigger_monitor(f"{ahora.hour:02d}:00")
            monitor_ultima_hora = ahora.hour

        # ── Darwin: resolver trades 17:05 ─────────────
        if (
            _es_dia_habil(ahora)
            and ahora.hour == 17
            and 5 <= ahora.minute < 15
            and darwin_resolve_hoy != fecha_hoy
        ):
            _trigger_darwin_resolve("post_market_17:05")
            darwin_resolve_hoy = fecha_hoy

        # ── Darwin: ciclo evolutivo viernes 18:00 ─────
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
                f"pipeline={'✅' if pipeline_hoy == fecha_hoy else '⏳'}"
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
    

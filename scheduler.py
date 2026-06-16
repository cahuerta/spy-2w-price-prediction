# scheduler.py — v2.0
# =========================================================
# Corre como thread daemon dentro del proceso FastAPI.
#
# Responsabilidades:
#   - Pipeline APERTURA: 11:30 Chile — predice + abre posiciones
#   - Pipeline CIERRE:   15:30 Chile — solo cierra, no abre
#   - Monitor horario:   12:00-15:00 Yahoo Finance (posiciones)
#   - Solo lunes a viernes
#   - Alive log cada hora
#   - Darwin Engine: resolver trades diario post-market (17:05 Chile)
#   - Darwin Engine: ciclo evolutivo viernes post-market (18:00 Chile)
#
# Horario mercado US en hora Chile (verano UTC-3):
#   Apertura  09:30 ET = 10:30 Chile
#   Cierre    16:00 ET = 17:00 Chile
#
# EJECUCIONES DIARIAS:
#   11:30 → APERTURA  — predicción + alpha + trading (abre Y cierra)
#   15:30 → CIERRE    — solo cierra posiciones divergentes
#   12:00-15:00 → Monitor horario (cada hora) — vigila posiciones abiertas
#   17:05 → Darwin resolve trades
#   18:00 viernes → Darwin evolución
#
# v2.0 — Monitor horario actualizado para intraday_tracker v3.0:
#   - run_intraday_tracker() eliminado — usaba universo propio
#   - Ahora usa evaluar_posiciones_abiertas(tickers) con lista real
#   - Solo sugiere cierre cuando sugerencia="CERRAR"
#   - TRAILING → no cierra, deja correr posiciones ganadoras
# =========================================================

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

# ── Horarios ──────────────────────────────────────────
HORA_APERTURA  = int(os.getenv("PIPELINE_HOUR",        "11"))
MIN_APERTURA   = int(os.getenv("PIPELINE_MIN",         "30"))
HORA_CIERRE    = int(os.getenv("PIPELINE_CIERRE_HOUR", "15"))
MIN_CIERRE     = int(os.getenv("PIPELINE_CIERRE_MIN",  "30"))

MONITOR_HORA_INICIO = int(os.getenv("MONITOR_HORA_INICIO", "12"))
MONITOR_HORA_FIN    = int(os.getenv("MONITOR_HORA_FIN",    "15"))


# ══════════════════════════════════════════════════════
# PIPELINE TRIGGERS
# ══════════════════════════════════════════════════════

def _trigger_pipeline(motivo: str, close_only: bool = False):
    ts   = datetime.now(CHILE_TZ).isoformat()
    tipo = "🔴 CIERRE" if close_only else "🟢 APERTURA"
    print(f"{'='*55}")
    print(f"{tipo} [{motivo}] | {ts}")
    print(f"{'='*55}")
    try:
        res = requests.post(
            PIPELINE_URL,
            headers={
                "X-PIPELINE-KEY": PIPELINE_KEY,
                "Content-Type":   "application/json",
            },
            json={"close_only": close_only},
            timeout=60 * 30,
        )
        if res.status_code == 200:
            print(f"✅ Pipeline {tipo} OK | {res.json()}")
        else:
            print(f"❌ Pipeline {tipo} HTTP {res.status_code} | {res.text}")
    except Exception as e:
        print(f"❌ Pipeline {tipo} trigger failed: {e}")


# ══════════════════════════════════════════════════════
# MONITOR HORARIO (Yahoo Finance)
# v2.0 — usa evaluar_posiciones_abiertas() del tracker v3.0
#   El tracker recibe la lista real de posiciones abiertas.
#   Solo dispara cierre defensivo cuando sugerencia="CERRAR".
#   TRAILING → no cierra, deja correr posiciones ganadoras.
# ══════════════════════════════════════════════════════

def _trigger_monitor(motivo: str):
    print(f"📡 Monitor horario [{motivo}]")
    try:
        from intraday_tracker import evaluar_posiciones_abiertas
        from positions_meta import get_all

        # Obtener tickers de posiciones abiertas reales
        meta    = get_all()
        tickers = list(meta.keys())

        if not tickers:
            print("✅ Monitor OK | sin posiciones abiertas")
            return

        result = evaluar_posiciones_abiertas(tickers)
        n_pos  = len(result)
        print(f"✅ Monitor OK | posiciones={n_pos}")

        # Clasificar sugerencias
        cerrar   = [t for t, s in result.items() if s.get("sugerencia") == "CERRAR"]
        trailing = [t for t, s in result.items() if s.get("sugerencia") == "TRAILING"]
        mantener = [t for t, s in result.items() if s.get("sugerencia") == "MANTENER"]

        if mantener:
            print(f"✅ MANTENER: {mantener}")

        if trailing:
            # Posiciones ganadoras divergentes — no cerrar, proteger con trailing
            for t in trailing:
                pnl = result[t].get("pnl_actual_pct", 0)
                print(f"🛡 TRAILING {t} | PnL={pnl:.1f}% → no cerrar, dejar correr")

        if cerrar:
            print(f"⚠️  CERRAR: {cerrar} → disparando cierre defensivo")
            _trigger_defensive_close(cerrar)

    except Exception as e:
        print(f"❌ Monitor error: {e}")


def _trigger_defensive_close(tickers: list):
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
            print(f"⚠️  Cierre defensivo HTTP {res.status_code}: {res.text}")
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
        f"🕐 Quant Scheduler iniciado\n"
        f"   🟢 APERTURA: {HORA_APERTURA:02d}:{MIN_APERTURA:02d} Chile\n"
        f"   🔴 CIERRE:   {HORA_CIERRE:02d}:{MIN_CIERRE:02d} Chile\n"
        f"   📡 MONITOR:  {MONITOR_HORA_INICIO:02d}:00-{MONITOR_HORA_FIN:02d}:00 Chile (cada hora)"
    )

    apertura_hoy:         str | None = None
    cierre_hoy:           str | None = None
    monitor_ultima_hora:  int | None = None
    ultimo_log_h:         int | None = None
    darwin_resolve_hoy:   str | None = None
    darwin_evolution_hoy: str | None = None

    while True:
        ahora     = datetime.now(CHILE_TZ)
        fecha_hoy = ahora.strftime("%Y-%m-%d")

        # ── 🟢 APERTURA: 11:30 — predice + abre + cierra ──
        if (
            _es_dia_habil(ahora)
            and ahora.hour   == HORA_APERTURA
            and MIN_APERTURA <= ahora.minute < MIN_APERTURA + 5
            and apertura_hoy != fecha_hoy
        ):
            _trigger_pipeline(
                f"APERTURA_{HORA_APERTURA:02d}:{MIN_APERTURA:02d}",
                close_only=False,
            )
            apertura_hoy = fecha_hoy

        # ── 🔴 CIERRE: 15:30 — solo cierra posiciones ─────
        if (
            _es_dia_habil(ahora)
            and ahora.hour  == HORA_CIERRE
            and MIN_CIERRE  <= ahora.minute < MIN_CIERRE + 5
            and cierre_hoy  != fecha_hoy
            and apertura_hoy == fecha_hoy
        ):
            _trigger_pipeline(
                f"CIERRE_{HORA_CIERRE:02d}:{MIN_CIERRE:02d}",
                close_only=True,
            )
            cierre_hoy = fecha_hoy

        # ── 📡 MONITOR HORARIO: 12:00-15:00 cada hora ─────
        # Corre cada hora entre apertura y cierre.
        # Evalúa posiciones abiertas y cierra solo las que
        # el tracker sugiere CERRAR (no TRAILING ni MANTENER).
        if (
            _es_dia_habil(ahora)
            and _en_horario_monitor(ahora)
            and ahora.minute < 5
            and monitor_ultima_hora != ahora.hour
            and apertura_hoy == fecha_hoy
        ):
            _trigger_monitor(f"{ahora.hour:02d}:00")
            monitor_ultima_hora = ahora.hour

        # ── 🧬 Darwin: resolver trades 17:05 ──────────────
        if (
            _es_dia_habil(ahora)
            and ahora.hour == 17
            and 5 <= ahora.minute < 15
            and darwin_resolve_hoy != fecha_hoy
        ):
            _trigger_darwin_resolve("post_market_17:05")
            darwin_resolve_hoy = fecha_hoy

        # ── 🧬 Darwin: ciclo evolutivo viernes 18:00 ──────
        if (
            ahora.weekday() == 4
            and ahora.hour   == 18
            and ahora.minute < 10
            and darwin_evolution_hoy != fecha_hoy
        ):
            _trigger_darwin_evolution("viernes_18:00")
            darwin_evolution_hoy = fecha_hoy

        # ── 💓 Alive log cada hora ─────────────────────────
        if ahora.minute < 5 and ultimo_log_h != ahora.hour:
            dia = "hábil" if _es_dia_habil(ahora) else "fin de semana"
            print(
                f"💓 Scheduler alive | "
                f"{ahora.strftime('%Y-%m-%d %H:%M')} Chile | {dia} | "
                f"apertura={'✅' if apertura_hoy == fecha_hoy else '⏳'} | "
                f"cierre={'✅' if cierre_hoy == fecha_hoy else '⏳'}"
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

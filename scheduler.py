# scheduler.py — v2.3
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
#   - Darwin Engine: ciclo evolutivo DIARIO post-market (18:00 Chile) [v2.3]
#   - Darwin Engine: shadow evaluator nocturno diario (23:00 Chile) —
#     ahora incluye executor shadows, no solo predictor shadows [v2.3]
#   - Code Auditor Agent: auditoría de código diaria (01:00 Chile)
#
# Horario mercado US en hora Chile (verano UTC-3):
#   Apertura  09:30 ET = 10:30 Chile
#   Cierre    16:00 ET = 17:00 Chile
#
# EJECUCIONES DIARIAS:
#   01:00 → Code Auditor Agent — auditoría completa del repo
#   11:30 → APERTURA  — predicción + alpha + trading (abre Y cierra)
#   15:30 → CIERRE    — solo cierra posiciones divergentes
#   12:00-15:00 → Monitor horario (cada hora) — vigila posiciones abiertas
#   17:05 → Darwin resolve trades
#   18:00 → Darwin evolución (executor + predictor arena) — TODOS los
#           días hábiles desde v2.3 (antes: solo viernes)
#   23:00 → Darwin shadow evaluator — predictor Y executor desde v2.3
#           (antes: solo predictor)
#
# v2.2 — Agregado Code Auditor Agent:
#   - Corre a las 01:00 Chile, todos los días
#   - Lee repo completo de GitHub y detecta incoherencias
#   - Guarda reporte en /data/audits/
#
# v2.3 — FIX [AUD-D2-cableado] (auditoría 2026-08-25, Problema 3):
#   El sistema de shadow-trading del executor (executor_shadow_evaluator.py,
#   fix [AUD-D2] en arena.py) nunca generó un solo archivo en disco porque
#   _run_shadow_evaluation() solo se invocaba dentro de run_evolution_cycle(),
#   que corría una vez por semana (viernes). Con cadencia semanal, cada
#   shadow abre a lo sumo un trade por semana — juntar los 10 trades
#   cerrados que exige MIN_TRADES_TO_COMPETE podía tardar meses, dejando
#   al campeón (fitness real NEGATIVO, -1.8551) sin competencia posible.
#
#   Cambios:
#   [S1] _trigger_shadow_evaluator ahora también llama a
#        darwin_engine.arena._run_shadow_evaluation() para el executor,
#        no solo al shadow evaluator de predictores — corre TODOS los
#        días hábiles a las 23:00, no solo el viernes de promoción.
#   [S2] El ciclo de evolución/promoción (_trigger_darwin_evolution)
#        pasa de correr solo los viernes a correr TODOS los días
#        hábiles a las 18:00 — decisión del usuario: la cadencia
#        semanal demoró demasiado en producir un reemplazo de campeón,
#        se prueba con cadencia diaria.
# =========================================================

import os
import time
import threading
import subprocess
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

# v2.1 — Shadow evaluator nocturno
HORA_SHADOW_EVAL = int(os.getenv("SHADOW_EVAL_HOUR", "23"))
MIN_SHADOW_EVAL  = int(os.getenv("SHADOW_EVAL_MIN",  "0"))

# v2.2 — Code Auditor Agent
HORA_AUDITOR = int(os.getenv("AUDITOR_HOUR", "1"))
MIN_AUDITOR  = int(os.getenv("AUDITOR_MIN",  "0"))


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
# ══════════════════════════════════════════════════════

def _trigger_monitor(motivo: str):
    print(f"📡 Monitor horario [{motivo}]")
    try:
        from intraday_tracker import evaluar_posiciones_abiertas
        from positions_meta import get_all

        meta    = get_all()
        tickers = list(meta.keys())

        if not tickers:
            print("✅ Monitor OK | sin posiciones abiertas")
            return

        result = evaluar_posiciones_abiertas(tickers)
        n_pos  = len(result)
        print(f"✅ Monitor OK | posiciones={n_pos}")

        cerrar   = [t for t, s in result.items() if s.get("sugerencia") == "CERRAR"]
        trailing = [t for t, s in result.items() if s.get("sugerencia") == "TRAILING"]
        mantener = [t for t, s in result.items() if s.get("sugerencia") == "MANTENER"]

        if mantener:
            print(f"✅ MANTENER: {mantener}")

        if trailing:
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


def _trigger_shadow_evaluator(motivo: str):
    print(f"🌙 Darwin shadow evaluator [{motivo}]")

    # Shadow evaluator de PREDICTORES — sin cambios
    try:
        from darwin_engine.predictor_shadow_evaluator import run_shadow_evolution_cycle
        result = run_shadow_evolution_cycle()
        gen = result.get("generation", {})
        ev  = result.get("evaluation", {})
        print(
            f"✅ Shadow evaluator (predictor) | "
            f"generadas={gen.get('predictions_generated', 0)} | "
            f"evaluadas={ev.get('evaluated', 0)} | "
            f"pendientes={ev.get('pending', 0)}"
        )
        if gen.get("status") == "locked":
            print("⚠️  Shadow evaluator (predictor): lock activo, se saltó esta corrida")
    except Exception as e:
        print(f"❌ Shadow evaluator (predictor) error: {e}")

    # [S1] Shadow evaluator de EXECUTOR — fix auditoría 2026-08-25 (Problema 3).
    # Antes solo corría dentro de run_evolution_cycle() (viernes). Con esto
    # corre TODOS los días hábiles a las 23:00, para que los shadows del
    # executor acumulen trades cerrados a ritmo diario, no semanal.
    try:
        from darwin_engine.arena import _load_all_active_genomes, _run_shadow_evaluation
        _, shadow = _load_all_active_genomes()
        _run_shadow_evaluation(shadow)
        print(f"✅ Shadow evaluator (executor) | shadows evaluados={len(shadow)}")
    except Exception as e:
        print(f"❌ Shadow evaluator (executor) error: {e}")


# ══════════════════════════════════════════════════════
# CODE AUDITOR AGENT (v2.2)
# ══════════════════════════════════════════════════════

def _trigger_code_auditor(motivo: str):
    """
    v2.2 — Ejecuta el agente auditor de código.
    Lee el repo completo de GitHub y detecta incoherencias.
    Guarda reporte en /data/audits/.
    """
    print(f"🔍 Code Auditor Agent [{motivo}]")
    try:
        result = subprocess.run(
            ["python", "agents/code_auditor_agent.py"],
            timeout=3600,  # 1 hora máximo
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Code Auditor completado")
            print(result.stdout[-500:])  # últimas 500 chars del output
        else:
            print(f"❌ Code Auditor error: {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        print("❌ Code Auditor timeout (1 hora)")
    except Exception as e:
        print(f"❌ Code Auditor excepción: {e}")


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
        f"   🔍 AUDITOR:  {HORA_AUDITOR:02d}:{MIN_AUDITOR:02d} Chile (diario)\n"
        f"   🟢 APERTURA: {HORA_APERTURA:02d}:{MIN_APERTURA:02d} Chile\n"
        f"   🔴 CIERRE:   {HORA_CIERRE:02d}:{MIN_CIERRE:02d} Chile\n"
        f"   📡 MONITOR:  {MONITOR_HORA_INICIO:02d}:00-{MONITOR_HORA_FIN:02d}:00 Chile (cada hora)\n"
        f"   🧬 EVOLUCIÓN: 18:00 Chile (diario hábil desde v2.3, antes solo viernes)\n"
        f"   🌙 SHADOW EVAL: {HORA_SHADOW_EVAL:02d}:{MIN_SHADOW_EVAL:02d} Chile (predictor + executor desde v2.3)"
    )

    apertura_hoy:         str | None = None
    cierre_hoy:           str | None = None
    monitor_ultima_hora:  int | None = None
    ultimo_log_h:         int | None = None
    darwin_resolve_hoy:   str | None = None
    darwin_evolution_hoy: str | None = None
    shadow_eval_hoy:      str | None = None
    auditor_hoy:          str | None = None

    while True:
        ahora     = datetime.now(CHILE_TZ)
        fecha_hoy = ahora.strftime("%Y-%m-%d")

        # ── 🔍 Code Auditor: 01:00 diario ─────────────────
        if (
            ahora.hour      == HORA_AUDITOR
            and MIN_AUDITOR <= ahora.minute < MIN_AUDITOR + 10
            and auditor_hoy != fecha_hoy
        ):
            _trigger_code_auditor("diario_01:00")
            auditor_hoy = fecha_hoy

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

        # ── 🧬 Darwin: ciclo evolutivo — DIARIO desde v2.3 ─
        # [S2] Antes: solo viernes (ahora.weekday()==4). La cadencia
        # semanal demoró demasiado en reemplazar al campeón (fitness
        # real negativo desde el 24-abr) — se prueba con cadencia
        # diaria en todos los días hábiles.
        if (
            _es_dia_habil(ahora)
            and ahora.hour   == 18
            and ahora.minute < 10
            and darwin_evolution_hoy != fecha_hoy
        ):
            _trigger_darwin_evolution(f"diario_18:00")
            darwin_evolution_hoy = fecha_hoy

        # ── 🌙 Darwin: shadow evaluator nocturno (v2.1) ────
        # [S1] Desde v2.3 también evalúa shadows del executor, no solo
        # del predictor — ver _trigger_shadow_evaluator().
        if (
            _es_dia_habil(ahora)
            and ahora.hour       == HORA_SHADOW_EVAL
            and MIN_SHADOW_EVAL  <= ahora.minute < MIN_SHADOW_EVAL + 10
            and shadow_eval_hoy  != fecha_hoy
        ):
            _trigger_shadow_evaluator(f"nocturno_{HORA_SHADOW_EVAL:02d}:{MIN_SHADOW_EVAL:02d}")
            shadow_eval_hoy = fecha_hoy

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

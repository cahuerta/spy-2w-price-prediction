# scheduler.py — v2.7
# =========================================================
# Corre como thread daemon dentro del proceso FastAPI.
#
# Responsabilidades:
#   - Pipeline APERTURA: 11:30 Chile — predice + abre posiciones
#   - Pipeline CIERRE:   15:30 Chile — solo cierra, no abre
#   - Monitor horario:   12:00-15:00 Yahoo Finance (posiciones) —
#     desde v2.4, cada corrida trae noticias frescas ANTES de
#     evaluar posiciones (news_ranker.py, ver _trigger_monitor)
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
#   11:00 → Macro Factors — oro/petróleo/DXY/VIX/etc, antes de la
#           apertura, para que el régimen del día ya tenga los datos [v2.5]
#   11:30 → APERTURA  — predicción + alpha + trading (abre Y cierra)
#   15:30 → CIERRE    — solo cierra posiciones divergentes
#   12:00-15:00 → Monitor horario (cada hora): [v2.4] news_ranker.py
#                 primero, después vigila posiciones abiertas — así
#                 el ajuste de stop por noticia bajista (intraday_
#                 tracker.py::[N2]) usa datos del mismo ciclo, nunca
#                 de una corrida vieja.
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
#
# v2.6 — [F13] (2026-09-11) FIX CRASH OOM — parte 2.
#   El fix [F12] en pipeline_router.py aisló model_runner/evaluator/
#   alpha_engine en procesos hijos, pero el servicio siguió cayendo
#   por "Ran out of memory (used over 512MB)". Root cause adicional:
#   _trigger_shadow_evaluator (23:00, evalúa 12 genomas shadow del
#   executor contra ~500-700 tickers + shadow eval de predictores
#   H1-H10) y _trigger_darwin_evolution (18:00, executor arena +
#   predictor arena) corrían DIRECTO en el hilo del scheduler, dentro
#   del mismo proceso de FastAPI — sin aislar memoria, igual que los
#   pasos del pipeline antes del fix [F12].
#   Fix: ambas ahora corren en procesos hijos separados
#   (multiprocessing.Process, contexto "spawn"), mismo patrón que
#   pipeline_router.py. _trigger_code_auditor ya usaba subprocess.run()
#   y no se toca.
#
# v2.7 — [F14] (2026-09-11) FIX CRASH OOM — parte 3: concurrencia.
#   Con [F12]/[F13] cada paso pesado corre aislado, PERO si dos caen
#   encima (ej. apertura 11:30 se demora y el monitor de las 12:00 se
#   dispara igual) hay DOS procesos hijos pesados corriendo a la vez,
#   sumando RAM sobre el mismo límite de 512MB del contenedor —
#   aislar cada uno no evita que se acumulen si corren en paralelo.
#   Fix:
#   [F14a] _trigger_monitor ahora también corre en proceso hijo
#          separado (news_ranker + evaluar_posiciones_abiertas),
#          escribiendo su resultado a un JSON temporal que el proceso
#          padre lee después para decidir si dispara cierre defensivo
#          (esa llamada HTTP sí se queda en el proceso padre, es liviana).
#   [F14b] Guarda de concurrencia: antes de lanzar monitor, darwin
#          evolution o shadow evaluator, se revisa si el pipeline
#          principal (`pipeline_router._pipeline_running`) sigue
#          activo — de ser así, esa ejecución se salta en vez de sumar
#          otro proceso pesado encima. Como scheduler.py y
#          pipeline_router.py corren en el mismo proceso de Python,
#          esto es una simple lectura de atributo de módulo, sin
#          necesidad de archivos ni IPC adicional.
# =========================================================

import os
import json
import time
import threading
import subprocess
import multiprocessing
import requests
from pathlib import Path
from datetime import datetime
import pytz

# [F14b] Import del módulo (no de la variable) para leer siempre el
# valor actual de _pipeline_running — un `from ... import _pipeline_running`
# copiaría el valor solo al momento del import y quedaría desactualizado.
import pipeline_router

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))

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

# [v2.5] Macro Factors — datos macro (oro, petróleo, DXY, VIX, etc.)
# No cambian intradía de forma relevante, así que corre una vez al
# día, antes de la apertura (11:30), para que market_quant_context.py
# tenga /data/macro_context.json fresco cuando se evalúe el régimen.
HORA_MACRO_FACTORS = int(os.getenv("MACRO_FACTORS_HOUR", "11"))
MIN_MACRO_FACTORS  = int(os.getenv("MACRO_FACTORS_MIN",  "0"))

# [F13] Timeout máximo para los procesos hijos pesados del scheduler
DARWIN_SUBPROCESS_TIMEOUT_SEC = int(os.getenv("DARWIN_SUBPROCESS_TIMEOUT_SEC", str(20 * 60)))

# [F13] Contexto "spawn": el proceso hijo arranca limpio, sin heredar
# threads ni estado del proceso padre (FastAPI/Uvicorn + este mismo
# hilo del scheduler). Mismo patrón que pipeline_router.py [F12].
_mp_ctx = multiprocessing.get_context("spawn")


# ══════════════════════════════════════════════════════
# [F13] WORKERS DE PROCESO HIJO — funciones top-level para que
# multiprocessing con contexto "spawn" pueda picklearlas. Cada una
# importa su módulo pesado DENTRO de la función, así el import (y
# la memoria que carga) ocurre solo dentro del proceso hijo.
# ══════════════════════════════════════════════════════

def _mp_darwin_evolution():
    from darwin_engine.arena import run_evolution_cycle
    from darwin_engine.predictor_arena import run_predictor_evolution
    run_evolution_cycle()
    run_predictor_evolution()


def _mp_shadow_evaluator():
    from darwin_engine.predictor_shadow_evaluator import run_shadow_evolution_cycle
    from darwin_engine.arena import _load_all_active_genomes, _run_shadow_evaluation
    run_shadow_evolution_cycle()
    _, shadow = _load_all_active_genomes()
    _run_shadow_evaluation(shadow)


def _mp_monitor_worker(result_path: str):
    """
    [F14a] Corre en proceso hijo: trae noticias frescas y evalúa las
    posiciones abiertas. Como el proceso padre necesita saber qué
    tickers cerrar (para disparar el cierre defensivo por HTTP), el
    resultado se escribe a un JSON temporal en vez de perderse al
    terminar el proceso hijo.
    """
    result = {"cerrar": [], "trailing": [], "mantener": [], "n_posiciones": 0}

    try:
        from news_ranker import run_news_ranking
        news_result = run_news_ranking()
        result["news_bullish"] = len(news_result.get("bullish", []))
        result["news_bearish"] = len(news_result.get("bearish", []))
    except Exception as e:
        result["news_error"] = str(e)

    try:
        from intraday_tracker import evaluar_posiciones_abiertas
        from positions_meta import get_all

        meta    = get_all()
        tickers = list(meta.keys())
        result["n_posiciones"] = len(tickers)

        if tickers:
            eval_out = evaluar_posiciones_abiertas(tickers)
            for t, s in eval_out.items():
                sug = s.get("sugerencia")
                if sug == "CERRAR":
                    result["cerrar"].append(t)
                elif sug == "TRAILING":
                    result["trailing"].append([t, s.get("pnl_actual_pct", 0)])
                elif sug == "MANTENER":
                    result["mantener"].append(t)
    except Exception as e:
        result["error"] = str(e)

    Path(result_path).write_text(json.dumps(result))


def _run_in_subprocess_sync(target, step_name: str, args: tuple = (), timeout_sec: int = DARWIN_SUBPROCESS_TIMEOUT_SEC) -> bool:
    """
    [F13] Corre `target(*args)` en un proceso hijo separado y espera a
    que termine (bloqueante — el scheduler ya corre en su propio hilo
    daemon, así que bloquear acá no afecta a FastAPI). Retorna True si
    terminó OK (exitcode 0 y sin exceder el timeout).
    """
    proc = _mp_ctx.Process(target=target, args=args, name=step_name)
    proc.start()
    proc.join(timeout_sec)

    if proc.is_alive():
        print(f"⏱️ [F13] {step_name} no terminó en {timeout_sec}s — terminando proceso hijo")
        proc.terminate()
        proc.join(10)
        return False

    if proc.exitcode != 0:
        print(f"❌ [F13] {step_name} falló en proceso hijo (exitcode={proc.exitcode})")
        return False

    print(f"✅ [F13] {step_name} completado en proceso hijo (pid={proc.pid})")
    return True


def _pipeline_esta_corriendo() -> bool:
    """
    [F14b] Lee el flag global de pipeline_router.py — como scheduler.py
    y pipeline_router.py corren en el mismo proceso de Python, esto es
    una lectura directa de atributo de módulo, siempre actualizada.
    getattr con default False por si el nombre del flag cambia algún
    día en pipeline_router.py — evita que scheduler.py crashee por eso.
    """
    return bool(getattr(pipeline_router, "_pipeline_running", False))


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

    # [F14b] Si el pipeline principal sigue corriendo (ej. apertura
    # demorada), no sumar otro proceso pesado encima — se salta esta
    # corrida del monitor y se retoma en la próxima hora.
    if _pipeline_esta_corriendo():
        print("⏭️  [F14b] Monitor SKIP — pipeline principal todavía corriendo")
        return

    # [F14a] Corre en proceso hijo separado (news_ranker +
    # evaluar_posiciones_abiertas), escribiendo su resultado a un JSON
    # temporal que se lee acá para decidir el cierre defensivo.
    result_path = str(DATA_PATH / f"tmp_monitor_result_{os.getpid()}.json")
    ok = _run_in_subprocess_sync(
        _mp_monitor_worker, "monitor", args=(result_path,), timeout_sec=5 * 60
    )

    if not ok:
        print("❌ Monitor falló o excedió tiempo — sin cambios en posiciones")
        return

    try:
        result_file = Path(result_path)
        if not result_file.exists():
            print("❌ Monitor: proceso hijo terminó pero no dejó resultado")
            return

        result = json.loads(result_file.read_text())
        result_file.unlink(missing_ok=True)

        if result.get("news_error"):
            print(f"⚠️ News ranker falló (continuando con monitor igual): {result['news_error']}")
        else:
            print(
                f"📰 News ranker OK | bullish={result.get('news_bullish', 0)} "
                f"bearish={result.get('news_bearish', 0)}"
            )

        if result.get("error"):
            print(f"❌ Monitor error: {result['error']}")
            return

        if result.get("n_posiciones", 0) == 0:
            print("✅ Monitor OK | sin posiciones abiertas")
            return

        cerrar   = result.get("cerrar", [])
        trailing = result.get("trailing", [])
        mantener = result.get("mantener", [])
        print(f"✅ Monitor OK | posiciones={result.get('n_posiciones', 0)}")

        if mantener:
            print(f"✅ MANTENER: {mantener}")

        if trailing:
            for t, pnl in trailing:
                print(f"🛡 TRAILING {t} | PnL={pnl:.1f}% → no cerrar, dejar correr")

        if cerrar:
            print(f"⚠️  CERRAR: {cerrar} → disparando cierre defensivo")
            _trigger_defensive_close(cerrar)

    except Exception as e:
        print(f"❌ Monitor error post-proceso: {e}")


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

    # [F14b] Evitar sumar este proceso pesado si el pipeline principal
    # sigue corriendo a esta hora (poco común a las 18:00, pero cubre
    # el caso de una apertura muy demorada).
    if _pipeline_esta_corriendo():
        print("⏭️  [F14b] Darwin evolution SKIP — pipeline principal todavía corriendo")
        return

    # [F13] Corre en proceso hijo separado — executor arena + predictor
    # arena, con mutaciones y guardado de genomas, aislado del proceso
    # principal de FastAPI para liberar su memoria real al terminar.
    _run_in_subprocess_sync(_mp_darwin_evolution, "darwin_evolution")


def _trigger_shadow_evaluator(motivo: str):
    print(f"🌙 Darwin shadow evaluator [{motivo}]")

    # [F14b] Misma guarda de concurrencia — a las 23:00 es muy poco
    # probable que el pipeline siga activo, pero el chequeo es barato.
    if _pipeline_esta_corriendo():
        print("⏭️  [F14b] Shadow evaluator SKIP — pipeline principal todavía corriendo")
        return

    # [F13] Corre en proceso hijo separado — este era el candidato más
    # fuerte al OOM: 12 genomas shadow del executor evaluados contra
    # ~500-700 tickers, más el shadow eval de predictores H1-H10,
    # todo antes corría directo en el hilo del scheduler.
    _run_in_subprocess_sync(_mp_shadow_evaluator, "shadow_evaluator")


# ══════════════════════════════════════════════════════
# CODE AUDITOR AGENT (v2.2)
# ══════════════════════════════════════════════════════

def _trigger_macro_factors(motivo: str):
    """
    [v2.5] Ejecuta macro_factors.py — trae oro, petróleo, gas, cobre,
    plata, DXY, TNX, VIX, SPY vía Yahoo Finance y calcula los scores
    agregados (macro_stress_magnitude, macro_risk_off_score) que
    market_quant_context.py ya sabe leer opcionalmente.
    """
    print(f"🌍 Macro Factors [{motivo}]")
    try:
        from macro_factors import run_macro_factors
        result = run_macro_factors()
        print(
            f"✅ Macro Factors OK | {result.get('n_available')}/{result.get('n_configured')} factores | "
            f"stress={result.get('macro_stress_magnitude')} | "
            f"risk_off={result.get('macro_risk_off_score')}"
        )
    except Exception as e:
        print(f"⚠️ Macro Factors falló (continuando igual): {e}")


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
        f"   🌍 MACRO:    {HORA_MACRO_FACTORS:02d}:{MIN_MACRO_FACTORS:02d} Chile (diario)\n"
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
    macro_factors_hoy:    str | None = None

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

        # ── 🌍 MACRO FACTORS: 11:00 — antes de la apertura ────
        if (
            _es_dia_habil(ahora)
            and ahora.hour       == HORA_MACRO_FACTORS
            and MIN_MACRO_FACTORS <= ahora.minute < MIN_MACRO_FACTORS + 10
            and macro_factors_hoy != fecha_hoy
        ):
            _trigger_macro_factors(f"diario_{HORA_MACRO_FACTORS:02d}:{MIN_MACRO_FACTORS:02d}")
            macro_factors_hoy = fecha_hoy

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

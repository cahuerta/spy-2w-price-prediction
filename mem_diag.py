# =========================================================
# mem_diag.py — INSTRUMENTACIÓN REAL DE MEMORIA
# =========================================================
# [MEMDIAG][2026-09-22] Hasta ahora, cada bug de memoria se encontró
# leyendo código y razonando qué PODRÍA estar pasando — nunca
# midiendo qué pasa de verdad en el proceso mientras corre. Este
# módulo agrega esa medición real: cuánta memoria (RSS) usa el
# proceso ANTES y DESPUÉS de cada paso pesado, y las diferencias
# quedan en el log de Render — visibles sin necesitar herramientas
# externas de profiling.
#
# Usa resource.getrusage(RUSAGE_SELF).ru_maxrss — disponible siempre
# en Linux (sin instalar psutil ni nada nuevo). En Linux, ru_maxrss
# es el pico de memoria residente (RSS) del proceso hasta ese
# momento, en KB — se reporta en MB acá para que sea legible.
#
# IMPORTANTE: ru_maxrss es un PICO ACUMULADO dentro del proceso — en
# un proceso hijo recién nacido (spawn), arranca en ~0 y solo puede
# subir durante la vida de ESE proceso. Por eso las llamadas a
# log_mem() tienen más sentido DENTRO de cada proceso hijo aislado
# (que nace y muere por cada paso pesado) que en el proceso principal
# de FastAPI (que vive días, y ahí ru_maxrss solo acumularía el pico
# histórico completo, no el consumo de un paso puntual).
#
# USO:
#   from mem_diag import log_mem
#   log_mem("model_runner lote 3 — inicio")
#   ... trabajo pesado ...
#   log_mem("model_runner lote 3 — fin")
#
# Ambas líneas quedan en el log de Render con el mismo formato,
# fáciles de buscar y comparar: "[MEM] <tag> | rss=<N> MB"
# =========================================================

import logging
import resource

logger = logging.getLogger("mem_diag")


def get_rss_mb() -> float:
    """
    Memoria residente (RSS) actual del proceso, en MB.
    En Linux, ru_maxrss viene en KB — se convierte a MB acá.
    Si por algún motivo falla (no debería en Linux), retorna -1.0
    en vez de lanzar una excepción — un fallo de diagnóstico nunca
    debe tumbar el paso real que está midiendo.
    """
    try:
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / 1024, 1)
    except Exception:
        return -1.0


def log_mem(tag: str) -> float:
    """
    Loguea el RSS actual con una etiqueta identificable, y lo retorna
    por si el llamador quiere comparar antes/después él mismo (ej.
    calcular el delta exacto de un paso y decidir si loguear una
    alerta si supera cierto umbral).
    """
    rss_mb = get_rss_mb()
    logger.info(f"📊 [MEM] {tag} | rss={rss_mb} MB")
    print(f"📊 [MEM] {tag} | rss={rss_mb} MB")  # también a stdout — visible en logs de Render sin depender del nivel de logging configurado
    return rss_mb


def log_mem_delta(tag: str, rss_before_mb: float, warn_threshold_mb: float = 100.0) -> float:
    """
    Loguea el RSS actual junto con el delta respecto a una medición
    anterior (rss_before_mb, típicamente el valor que devolvió un
    log_mem() previo). Si el delta supera warn_threshold_mb, lo marca
    con ⚠️ para que sea fácil de encontrar en el log sin tener que
    calcular restas a mano.
    """
    rss_now = get_rss_mb()
    delta   = round(rss_now - rss_before_mb, 1) if rss_before_mb >= 0 and rss_now >= 0 else None

    if delta is not None and delta >= warn_threshold_mb:
        marker = f"⚠️ [MEM] {tag} | rss={rss_now} MB | delta=+{delta} MB (>= {warn_threshold_mb} MB)"
    else:
        marker = f"📊 [MEM] {tag} | rss={rss_now} MB | delta={delta if delta is not None else 'N/A'} MB"

    logger.info(marker)
    print(marker)
    return rss_now


# =========================================================
# [MEMDIAG][2026-09-22] SILENCIAR RUIDO — llamar al inicio de cada
# proceso hijo
# =========================================================
# El log de Render se llenaba con miles de líneas INFO por corrida
# (una por ticker, en varios pasos: "get_price_at_date X resuelto vía
# fallback", "Champion HX cargado", "Shadow eval | ..."), al punto de
# que un crash real quedaba enterrado entre ese ruido — imposible de
# ver a simple vista. La gran mayoría de ese volumen sale del módulo
# `logging` estándar de Python (estas líneas SIEMPRE llevan el
# formato "fecha [NIVEL] mensaje" — se ve claro en cualquier log
# real). Elevar el nivel del logger raíz a WARNING corta ese ruido de
# raíz, en una sola línea, sin tener que editar decenas de archivos
# que llaman logger.info(...) — los mensajes de error/advertencia
# reales (incluida cualquier pista de un crash) siguen pasando, y las
# líneas [MEM] de este módulo van por print() directo, nunca se
# silencian pase lo que pase.
#
# USO: llamar quiet_logs() como PRIMERA línea de cada función
# _mp_*worker — cada proceso hijo (spawn) arranca sin la
# configuración de logging del padre, así que hay que silenciarlo de
# nuevo en cada uno.
def quiet_logs(level: int = logging.WARNING) -> None:
    logging.getLogger().setLevel(level)
    for name in list(logging.root.manager.loggerDict.keys()):
        logging.getLogger(name).setLevel(level)

#!/usr/bin/env python3
# =========================================================
# analyze_screener_retroactive.py
# =========================================================
# [2026-09-23] Script de diagnóstico de una sola vez — NO modifica
# nada en producción, solo lee precios históricos y calcula.
#
# El screener nunca guardó su historial (screener_candidates.json se
# sobreescribe cada día, sin fecha en el nombre) — así que no se puede
# mirar "qué dijo el screener hace 2 meses" directamente. Pero SÍ se
# puede RECALCULAR retroactivamente: compute_score() es una función
# pura (recibe arrays de precios/volúmenes, no estado ni fecha actual),
# así que corriéndola sobre los precios disponibles HASTA hace 2 meses
# (sin ver nada posterior — sin fuga de información) reproduce
# exactamente lo que el screener real habría calculado ese día.
#
# Universo usado: sp500.json (549 tickers) — el pool real del que el
# screener rota su selección diaria, NO anchor_universe.json (que es
# solo una lista semilla de 10 tickers defensivos, un archivo
# distinto con otro propósito).
#
# Metodología por ticker:
#   1. Traer ~8 meses de historial (suficiente para el lookback de
#      3 meses que compute_score necesita, ANTES del corte de hace
#      2 meses, más los 2 meses posteriores para medir qué pasó).
#   2. Cortar la serie en la fecha de hace 2 meses — compute_score()
#      corre SOLO con datos hasta ese punto, igual que lo habría
#      visto el screener real ese día.
#   3. Calcular el retorno real desde esa fecha hasta hoy.
#   4. Agregar por etiqueta de calidad: ¿los tickers que el screener
#      retroactivo marcó STRONG/INSTITUTIONAL rindieron mejor que el
#      resto del universo, y mejor que el mercado (SPY) en el mismo
#      período?
#
# Reutiliza compute_score() de screener_engine.py y get_price_history()
# de data_provider.py — no duplica ninguna lógica de scoring ni de
# precios, para que el resultado sea fiel al screener real.
#
# USO (Shell de Render):
#   cd /app && python analyze_screener_retroactive.py
# =========================================================

import json
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, "/app")

import numpy as np
import pandas as pd

from data_provider import get_price_history
from screener_engine import compute_score

LOOKBACK_MONTHS_BEFORE_CUTOFF = 8   # historial total a traer por ticker
MONTHS_AGO_CUTOFF             = 2   # "hace 2 meses" — el punto de corte


def _load_universe() -> list:
    path = Path(__file__).resolve().parent / "sp500.json"
    if not path.exists():
        print(f"❌ No existe {path}")
        return []
    return json.loads(path.read_text())


def _closes_volumes_as_of(df: pd.DataFrame, cutoff: pd.Timestamp):
    """Recorta el DataFrame a solo datos <= cutoff — sin fuga de información."""
    past = df[df.index <= cutoff]
    if len(past) < 30:
        return None, None
    return past["Close"].values, past["Volume"].values


def analyze():
    universe = _load_universe()
    if not universe:
        return

    today  = pd.Timestamp(datetime.utcnow().date())
    cutoff = today - pd.DateOffset(months=MONTHS_AGO_CUTOFF)

    print(f"🔍 Universo real (sp500.json): {len(universe)} tickers")
    print(f"📅 Corte retroactivo: {cutoff.date()}  →  hoy: {today.date()}\n")

    # Benchmark: retorno de SPY en el mismo período, para tener con qué comparar
    spy_df = get_price_history("SPY", period="1y", interval="1d")
    spy_return = None
    if spy_df is not None and len(spy_df) > 0:
        spy_past = spy_df[spy_df.index <= cutoff]
        if len(spy_past) > 0:
            spy_price_then = float(spy_past["Close"].iloc[-1])
            spy_price_now  = float(spy_df["Close"].iloc[-1])
            spy_return     = (spy_price_now / spy_price_then - 1) * 100
    print(f"📈 SPY en el mismo período: {spy_return:+.2f}%\n" if spy_return is not None else "⚠️ No se pudo calcular retorno de SPY\n")

    by_quality = {}   # quality -> list of forward returns (%)
    skipped_no_data = 0
    processed = 0

    for i, ticker in enumerate(universe):
        try:
            df = get_price_history(ticker, period="1y", interval="1d")
            if df is None or len(df) == 0:
                skipped_no_data += 1
                continue

            df = df.sort_index()

            closes_then, volumes_then = _closes_volumes_as_of(df, cutoff)
            if closes_then is None:
                skipped_no_data += 1
                continue

            score_data = compute_score(closes=closes_then, volumes=volumes_then)
            if not score_data:
                skipped_no_data += 1
                continue

            quality = score_data.get("quality", "?")

            # Precio real en el momento del corte y hoy
            past = df[df.index <= cutoff]
            if len(past) == 0:
                skipped_no_data += 1
                continue

            price_then = float(past["Close"].iloc[-1])
            price_now  = float(df["Close"].iloc[-1])
            if price_then <= 0:
                skipped_no_data += 1
                continue

            fwd_return = (price_now / price_then - 1) * 100

            by_quality.setdefault(quality, []).append(fwd_return)
            processed += 1

        except Exception as e:
            skipped_no_data += 1

        if (i + 1) % 50 == 0:
            print(f"   ... {i + 1}/{len(universe)} procesados")

    print("\n" + "=" * 70)
    print("📊 RESULTADO — retorno real (hace 2 meses → hoy) por etiqueta de calidad")
    print("=" * 70)

    # Orden de mejor a peor calidad, mismo orden que screener_engine.py
    quality_order = ["🚀 INSTITUTIONAL", "✅ STRONG", "🟡 MODERATE", "❌ NOISE"]
    for q in quality_order:
        rets = by_quality.get(q)
        if not rets:
            print(f"\n{q}: sin tickers")
            continue
        arr = np.array(rets)
        hit_rate_positivo = float((arr > 0).mean() * 100)
        print(
            f"\n{q}  (n={len(arr)})\n"
            f"   Retorno promedio  : {arr.mean():+.2f}%\n"
            f"   Retorno mediana   : {np.median(arr):+.2f}%\n"
            f"   % con retorno >0  : {hit_rate_positivo:.1f}%"
        )
        if spy_return is not None:
            print(f"   vs SPY ({spy_return:+.2f}%): {'MEJOR' if arr.mean() > spy_return else 'PEOR'} que el mercado")

    # Cualquier etiqueta no contemplada arriba (por si acaso)
    for q, rets in by_quality.items():
        if q not in quality_order and rets:
            arr = np.array(rets)
            print(f"\n{q}  (n={len(arr)}) — retorno promedio: {arr.mean():+.2f}%")

    print(f"\n\nProcesados exitosamente: {processed}/{len(universe)}  |  saltados (sin datos suficientes): {skipped_no_data}")
    print("=" * 70)


if __name__ == "__main__":
    analyze()
          

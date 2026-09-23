#!/usr/bin/env python3
# =========================================================
# analyze_h10_raw_vs_adjusted.py
# =========================================================
# [2026-09-23] Script de diagnóstico de una sola vez — NO modifica
# nada en producción, solo lee /data/predictions y calcula dos hit
# rates distintos para H10 sobre los MISMOS días evaluados:
#
#   CRUDO    → models_diagnostics.H10.pred_price/pred_return
#              (la predicción de H10 tal cual salió del modelo,
#              guardada por master_orchestrator.py ANTES de aplicar
#              curve_adjust y el veto de consenso con H1-H9)
#
#   AJUSTADO → prediction.price_pred/ret_ens_pct
#              (lo que evaluator.py usa hoy para calcular "el hit
#              rate de H10" — YA mezclado con el ajuste de curva de
#              H1-H9 y el veto de signo)
#
# Hipótesis a probar: si H10 crudo tiene un hit rate notablemente
# mejor que el ajustado, significa que el modelo (el "rifle") apunta
# razonablemente bien, y es la capa de ajuste por consenso (el
# "viento") la que está degradando el resultado que le atribuimos a
# H10 — no el modelo en sí.
#
# Reutiliza get_price_at_date/nth_business_day/_calc_hit_sign de
# evaluator.py — no duplica lógica de precios ni de umbral de señal,
# para que la comparación sea contra el mismo criterio que ya usa el
# sistema real, no un criterio nuevo inventado para este análisis.
#
# USO (Shell de Render, dentro del contenedor):
#   cd /app && python analyze_h10_raw_vs_adjusted.py
# =========================================================

import json
import sys
from pathlib import Path
from datetime import datetime, date

sys.path.insert(0, "/app")

from evaluator import get_price_at_date, nth_business_day, _calc_hit_sign, DATA_PATH


def analyze():
    pred_root = Path(DATA_PATH) / "predictions"
    if not pred_root.exists():
        print(f"❌ No existe {pred_root}")
        return

    raw_hits, raw_total, raw_weak = 0, 0, 0
    adj_hits, adj_total, adj_weak = 0, 0, 0
    skipped_no_data  = 0
    skipped_no_price = 0
    examples_diff    = []  # casos donde el ajuste cambió el signo de la predicción

    ticker_dirs = sorted(p for p in pred_root.iterdir() if p.is_dir())
    print(f"🔍 Escaneando {len(ticker_dirs)} tickers en {pred_root} ...\n")

    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name

        for pred_file in sorted(ticker_dir.glob("*.json")):
            pred_date_str = pred_file.stem
            try:
                pred_date = datetime.strptime(pred_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            try:
                data = json.loads(pred_file.read_text())
            except Exception:
                continue

            prediction = data.get("prediction") or {}
            price_now  = prediction.get("price_now")
            if not price_now or float(price_now) <= 0:
                skipped_no_data += 1
                continue
            price_now = float(price_now)

            # AJUSTADO — lo que se reporta oficialmente hoy
            adj_price_pred = prediction.get("price_pred")
            adj_ret         = prediction.get("ret_ens_pct")

            # CRUDO — guardado aparte en el mismo archivo, nunca tocado
            # por curve_adjust ni por el veto de consenso
            h10_diag       = (data.get("models_diagnostics") or {}).get("H10") or {}
            raw_price_pred = h10_diag.get("pred_price")
            raw_ret        = h10_diag.get("pred_return")

            if adj_price_pred is None or raw_price_pred is None or adj_ret is None or raw_ret is None:
                skipped_no_data += 1
                continue

            # H10 = horizonte de 10 días hábiles
            target_date = nth_business_day(pred_date, 10)
            if target_date > date.today():
                continue  # aún no madura — no se puede evaluar todavía

            real_price = get_price_at_date(ticker, target_date)
            if not real_price:
                skipped_no_price += 1
                continue

            real_ret = (real_price / price_now - 1) * 100.0

            adj_hit, adj_is_weak = _calc_hit_sign(float(adj_ret), real_ret)
            raw_hit, raw_is_weak = _calc_hit_sign(float(raw_ret), real_ret)

            if adj_is_weak:
                adj_weak += 1
            elif adj_hit is not None:
                adj_total += 1
                adj_hits += int(adj_hit)

            if raw_is_weak:
                raw_weak += 1
            elif raw_hit is not None:
                raw_total += 1
                raw_hits += int(raw_hit)

            # Registrar los casos donde el ajuste literalmente invirtió
            # la dirección de la apuesta (crudo decía subir, ajustado
            # decía bajar, o viceversa) — evidencia directa del "viento"
            if (raw_ret > 0) != (adj_ret > 0) and abs(raw_ret) > 0.01 and abs(adj_ret) > 0.01:
                examples_diff.append({
                    "ticker":   ticker,
                    "fecha":    pred_date_str,
                    "raw_ret":  round(float(raw_ret), 3),
                    "adj_ret":  round(float(adj_ret), 3),
                    "real_ret": round(real_ret, 3),
                })

    print("=" * 70)
    print("📊 H10 — CRUDO vs AJUSTADO (mismos días evaluados en ambos casos)")
    print("=" * 70)

    print(f"\n🔹 CRUDO (models_diagnostics.H10 — antes del ajuste de consenso):")
    print(f"   n_evaluaciones = {raw_total}  |  señales débiles descartadas = {raw_weak}")
    if raw_total > 0:
        print(f"   hit_rate = {raw_hits / raw_total * 100:.2f}%  ({raw_hits}/{raw_total})")
    else:
        print("   (sin datos suficientes)")

    print(f"\n🔸 AJUSTADO (prediction.price_pred — lo que se reporta oficialmente hoy):")
    print(f"   n_evaluaciones = {adj_total}  |  señales débiles descartadas = {adj_weak}")
    if adj_total > 0:
        print(f"   hit_rate = {adj_hits / adj_total * 100:.2f}%  ({adj_hits}/{adj_total})")
    else:
        print("   (sin datos suficientes)")

    if raw_total > 0 and adj_total > 0:
        delta = (raw_hits / raw_total - adj_hits / adj_total) * 100
        print(f"\n📈 Diferencia (crudo - ajustado): {delta:+.2f} puntos porcentuales")

    print(f"\n⚠️ Casos donde el ajuste invirtió el signo de la predicción: {len(examples_diff)}")
    if examples_diff:
        print("   (primeros 10 ejemplos, más recientes primero)")
        for e in sorted(examples_diff, key=lambda x: x["fecha"], reverse=True)[:10]:
            print(
                f"   {e['ticker']:6s} {e['fecha']} | "
                f"crudo={e['raw_ret']:+7.2f}%  ajustado={e['adj_ret']:+7.2f}%  "
                f"real={e['real_ret']:+7.2f}%"
            )

    print(f"\nArchivos sin datos suficientes: {skipped_no_data}  |  sin precio real disponible: {skipped_no_price}")
    print("=" * 70)


if __name__ == "__main__":
    analyze()

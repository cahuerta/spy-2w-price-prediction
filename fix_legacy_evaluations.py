"""
fix_legacy_evaluations.py
=========================
Marca las evaluaciones con horizonte incorrecto (legacy_bad_horizon=True)
para que signals.py las excluya del cálculo de rolling_metrics.

Criterio: una evaluación es "bad_horizon" si:
  - gap entre prediction_date y evaluation_date != horizon_days real
  - O price_real es None (ticker no soportado en Alpaca)

NO borra ningún archivo — solo agrega el flag legacy_bad_horizon=True.
Esto preserva la trazabilidad histórica.

Ejecutar UNA sola vez desde el shell de Render:
  python3 fix_legacy_evaluations.py
"""

import json
import os
from pathlib import Path
from datetime import datetime

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
EVAL_ROOT = DATA_PATH / "evaluations"

ALPACA_UNSUPPORTED_SUFFIXES = (
    ".PA", ".DE", ".SW", ".TO", ".MC", ".AS", ".MI", ".BR",
    ".SN", ".SX", ".L", ".HK", ".T",
)


def _is_unsupported(ticker: str) -> bool:
    return any(ticker.upper().endswith(s) for s in ALPACA_UNSUPPORTED_SUFFIXES)


def fix_legacy_evaluations():
    total      = 0
    bad_horizon = 0
    no_price   = 0
    already_fixed = 0
    errors     = 0

    for ticker_dir in sorted(EVAL_ROOT.iterdir()):
        if not ticker_dir.is_dir():
            continue

        ticker = ticker_dir.name

        for f in sorted(ticker_dir.glob("*.json")):
            total += 1
            try:
                d = json.loads(f.read_text())

                # Ya procesado
                if d.get("legacy_bad_horizon") is not None:
                    already_fixed += 1
                    continue

                is_bad = False
                reason = []

                # Criterio 1: ticker no soportado en Alpaca
                if _is_unsupported(ticker) or d.get("price_real") is None:
                    is_bad = True
                    no_price += 1
                    reason.append("no_price_alpaca")

                # Criterio 2: gap incorrecto
                pred_date_str = d.get("prediction_date")
                eval_date_str = d.get("evaluation_date")
                h_days        = d.get("meta", {}).get("horizon_days", 10)

                if pred_date_str and eval_date_str and not is_bad:
                    from datetime import date
                    try:
                        pd_ = date.fromisoformat(pred_date_str)
                        ed_ = date.fromisoformat(eval_date_str)
                        gap = (ed_ - pd_).days
                        # Tolerancia: ±3 días del horizonte real
                        if abs(gap - h_days) > 3:
                            is_bad = True
                            bad_horizon += 1
                            reason.append(f"gap={gap}_vs_horizon={h_days}")
                    except Exception:
                        pass

                # Marcar y guardar
                d["legacy_bad_horizon"] = is_bad
                if reason:
                    d["legacy_bad_horizon_reason"] = " | ".join(reason)

                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False))
                tmp.replace(f)

                if is_bad:
                    print(f"  MARKED BAD | {ticker}/{f.name} | {' | '.join(reason)}")

            except Exception as e:
                errors += 1
                print(f"  ERROR {ticker}/{f.name}: {e}")

    print(f"\n{'='*60}")
    print(f"RESUMEN LIMPIEZA EVALUACIONES")
    print(f"{'='*60}")
    print(f"Total evaluaciones:     {total}")
    print(f"Ya procesadas:          {already_fixed}")
    print(f"Marcadas bad (no price): {no_price}")
    print(f"Marcadas bad (gap):      {bad_horizon}")
    print(f"Errores:                {errors}")
    print(f"{'='*60}")
    print(f"Evaluaciones limpias:   {total - already_fixed - no_price - bad_horizon}")
    print(f"\nAhora signals.py excluirá legacy_bad_horizon=True del rolling_metrics.")


if __name__ == "__main__":
    print("🔧 Iniciando limpieza de evaluaciones legacy...")
    fix_legacy_evaluations()
    print("✅ Limpieza completada.")

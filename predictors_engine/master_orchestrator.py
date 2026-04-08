import os
import json
import numpy as np
import pandas as pd
import sys
import importlib
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

DATA_DIR   = "predictions_data"
FINAL_DIR  = "/data/predictions"
EVAL_DIR   = "/data/evaluations"

# ======================================================
# FIXES v7.3 → v7.5:
#   [O1] extract_price_prediction — null-safe
#   [O2] extract_return_pct — null-safe
#   [O3] collect_results — normalized_ret None-safe
#   [O4] build_model_json — h10_json campos críticos validados
#   [O5] run() — mínimo 5 modelos
#   [O6] ÚNICO DUEÑO DE ESCRITURA — solo orquestador graba en disco
#   [O7] IN-PROCESS via importlib — sin subprocesos, RAM estable
#   [O8] build_model_json recalcula 'recommendation' después de ajustar
#        ret_ens_pct — antes quedaba desincronizado con el precio final.
#
# FIXES v7.5:
#   [O9]  BUG SIGNO: curve_adjust no puede invertir el signo de H10.
#         Si el ajuste invierte el signo → ret_final = 0, MANTÉN.
#   [O10] PONDERACIÓN DINÁMICA: lee /data/evaluations/{ticker}/ y
#         calcula hit_rate histórico por horizonte H1-H10.
#         Los horizontes con mejor hit_rate tienen más peso en el ensamble.
#         Si no hay historial → pesos iguales (comportamiento anterior).
# ======================================================


# ======================================================
# UTILIDADES
# ======================================================

def extract_price_prediction(data: Dict) -> Optional[float]:
    """[O1] Null-safe. Retorna None si no hay precio válido."""
    val = data.get("price_pred")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass

    pred = data.get("prediction")
    if isinstance(pred, dict):
        val = pred.get("price_pred")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass

    return None


def extract_return_pct(data: Dict) -> float:
    """[O2] Null-safe. Retorna 0.0 como fallback seguro."""
    for k in ["return_pct", "ret_ens_pct"]:
        val = data.get(k)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass

    pred = data.get("prediction")
    if isinstance(pred, dict):
        val = pred.get("ret_ens_pct")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass

    return 0.0


# ======================================================
# [O10] PONDERACIÓN DINÁMICA POR HORIZONTE
# ======================================================

def _load_horizon_weights(ticker: str) -> Dict[str, float]:
    """
    Lee /data/evaluations/{ticker}/*.json y calcula hit_rate por horizonte.
    Retorna pesos normalizados H1..H10.
    Si no hay historial suficiente → pesos iguales.
    Mínimo 10 evaluaciones por horizonte para confiar en el peso.
    """
    eval_path = Path(EVAL_DIR) / ticker
    if not eval_path.exists():
        return {}

    # Acumular hit_sign por horizonte
    hits: Dict[str, List[bool]] = {f"H{h}": [] for h in range(1, 11)}

    for f in sorted(eval_path.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            diag = data.get("models_diagnostics") or {}
            for h in range(1, 11):
                key = f"H{h}"
                if key in diag and diag[key].get("hit_sign") is not None:
                    hits[key].append(bool(diag[key]["hit_sign"]))
        except Exception:
            continue

    # Calcular hit_rate por horizonte — mínimo 10 muestras
    MIN_SAMPLES = 10
    rates: Dict[str, float] = {}
    for key, vals in hits.items():
        if len(vals) >= MIN_SAMPLES:
            rates[key] = float(np.mean(vals))

    if not rates:
        return {}  # Sin historial → pesos iguales

    # Normalizar: más peso a mejor hit_rate
    # Usar softmax para evitar pesos extremos
    keys   = list(rates.keys())
    values = np.array([rates[k] for k in keys])
    # Softmax con temperatura 2 para suavizar diferencias
    exp_v  = np.exp((values - values.mean()) * 2)
    norm   = exp_v / exp_v.sum()

    return {k: float(w) for k, w in zip(keys, norm)}


def _weighted_price_curve(results: List[Dict], weights: Dict[str, float], price_today: float) -> Optional[Dict]:
    """
    Construye curva de precios ponderada por hit_rate histórico.
    Solo H1-H9. Si no hay pesos → igual que antes (promedio simple).
    """
    filtered = [r for r in results if r["horizon"] < 10]
    if len(filtered) < 2:
        return None

    horizons = np.array([r["horizon"] for r in filtered])
    prices   = np.array([r["price_pred"] for r in filtered])

    # Aplicar pesos si existen
    if weights:
        w_array = np.array([
            weights.get(f"H{r['horizon']}", 1.0 / len(filtered))
            for r in filtered
        ])
        w_array = w_array / w_array.sum()  # renormalizar
        # Precio ponderado para cada horizonte
        weighted_prices = prices * w_array * len(filtered)
    else:
        weighted_prices = prices

    full_h       = np.arange(1, 10)
    interpolated = np.interp(full_h, horizons, weighted_prices)
    smoothed     = pd.Series(interpolated).rolling(3, center=True, min_periods=1).mean().values

    return {
        "days":       list(range(1, 10)),
        "price_path": smoothed.tolist(),
        "price_now":  price_today,
        }

   
# ======================================================
# ORCHESTRATOR
# ======================================================

class MasterOrchestrator:

    def __init__(self, ticker):
        self.ticker      = ticker.upper()
        self.results     = []
        self.price_today = None
        self.h10_json    = None

    # ======================================================
    # RECOLECTAR RESULTADOS
    # ======================================================

    def collect_results(self):
        curve = []

        for h in range(1, 11):
            possible_files = [
                f"{self.ticker}_H{h}.json",
                f"{self.ticker}_h{h}.json",
                f"{self.ticker}_H{h}_{h}d.json",
                f"{self.ticker}_H{h}_{h}d_v2.json",
                f"{self.ticker}_H{h}_{h}d_v3.json",
            ]

            for fname in possible_files:
                path = os.path.join(DATA_DIR, fname)
                if not os.path.exists(path):
                    continue

                try:
                    with open(path) as f:
                        data = json.load(f)

                    data["horizon"] = h

                    # [O1] null-safe
                    price = extract_price_prediction(data)
                    if price is None:
                        print(f"⚠️ H{h} sin price_pred válido — saltando")
                        continue

                    if self.price_today is None:
                        raw_price = (
                            data.get("price_now")
                            or data.get("price_today")
                            or (data.get("prediction") or {}).get("price_now")
                        )
                        if raw_price is None:
                            print(f"⚠️ H{h} sin price_today — saltando")
                            continue
                        self.price_today = float(raw_price)

                    data["price_pred"]     = price
                    # [O3] null-safe
                    data["normalized_ret"] = extract_return_pct(data)

                    curve.append(data)

                    if h == 10:
                        self.h10_json = data

                    print(f"   📊 H{h}: ${price:.2f}")
                    break

                except Exception as e:
                    print(f"⚠️ H{h} JSON error {e}")

        self.results = sorted(curve, key=lambda x: x["horizon"])
        print(f"📈 {len(self.results)}/10 modelos recolectados")
        return self.results

    # ======================================================
    # CURVA — ahora usa ponderación dinámica [O10]
    # ======================================================

    def build_price_curve(self, weights: Dict[str, float] = None) -> Optional[Dict]:
        return _weighted_price_curve(
            self.results,
            weights or {},
            self.price_today,
        )

    # ======================================================
    # ANALISIS FINAL
    # ======================================================

    def analyze_curve(self, curve) -> float:
        if curve is None:
            return 0.0
        curve_last = curve["price_path"][-1]
        return ((curve_last - self.price_today) / self.price_today) * 100

    # ======================================================
    # JSON FINAL
    # ======================================================

    def build_model_json(self, curve, weights: Dict[str, float] = None) -> Dict:
        if self.h10_json is None:
            raise RuntimeError("H10 missing")

        # [O4] Validar campos críticos
        pred = self.h10_json.get("prediction")
        if not pred or not isinstance(pred, dict):
            raise RuntimeError(f"[{self.ticker}] H10 prediction inválido o None")

        required_fields = ["price_pred", "price_now", "ret_ens_pct", "theta_dynamic_pct"]
        null_fields = [f for f in required_fields if pred.get(f) is None]
        if null_fields:
            raise RuntimeError(f"[{self.ticker}] H10 campos críticos null: {null_fields}")

        final_json    = self.h10_json.copy()
        h10_price     = extract_price_prediction(self.h10_json)
        curve_adjust  = self.analyze_curve(curve)

        # Retorno original de H10 — fuente de verdad del signo
        ret_original  = extract_return_pct(self.h10_json)
        sign_original = float(np.sign(ret_original)) if ret_original != 0 else 0.0

        consensus        = np.std([r["price_pred"] for r in self.results if r["horizon"] < 10])
        consensus_weight = 1 / (1 + consensus / self.price_today)
        adjust_factor    = 0.15 * consensus_weight

        final_price = h10_price * (1 + curve_adjust / 100 * adjust_factor)

        # Retorno ajustado (aritmético)
        ret_adjusted = ((final_price - self.price_today) / self.price_today) * 100

        # [O9] FIX SIGNO: si el ajuste invirtió el signo original → forzar MANTÉN
        sign_adjusted = float(np.sign(ret_adjusted)) if ret_adjusted != 0 else 0.0

        if sign_original != 0 and sign_adjusted != sign_original:
            print(
                f"⚠️ [{self.ticker}] curve_adjust invirtió signo "
                f"({ret_original:.3f}% → {ret_adjusted:.3f}%) → forzando ret=0, MANTÉN"
            )
            ret_final   = 0.0
            final_price = self.price_today  # precio sin cambio
        else:
            ret_final = round(ret_adjusted, 4)

        final_json["prediction"]["price_pred"]  = round(final_price, 2)
        final_json["prediction"]["ret_ens_pct"] = ret_final

        # [O8] Recalcular recommendation con ret_ens_pct ajustado
        theta_dyn = float(final_json["prediction"]["theta_dynamic_pct"])

        final_json["prediction"]["recommendation"] = (
            "COMPRA" if ret_final >= theta_dyn else
            "VENDE"  if ret_final <= -theta_dyn else
            "MANTÉN"
        )

        final_json["price_curve"]     = curve
        final_json["ensemble_models"] = len(self.results)

        # [O10] Incluir pesos usados en el diagnóstico
        final_json["horizon_weights"] = weights or {}

        final_json["models_diagnostics"] = {
            f"H{r['horizon']}": {
                "pred_price":  round(r["price_pred"], 4),
                "pred_return": round(r["normalized_ret"], 4),
                "horizon":     r["horizon"],
                "weight":      round(weights.get(f"H{r['horizon']}", 1.0 / len(self.results)), 4)
                               if weights else None,
            }
            for r in self.results
        }

        return final_json

    # ======================================================
    # RUN — V7.5 IN-PROCESS
    # ======================================================

    def run(self):
        os.makedirs(DATA_DIR, exist_ok=True)

        for i in range(1, 11):
            try:
                print(f"🚀 Ejecutando H{i}")
                mod  = importlib.import_module(f"predictors_engine.predictor_h{i}")
                func = getattr(mod, f"run_predictor_h{i}")
                result = func(self.ticker)

                if result is None:
                    print(f"⚠️ H{i} retornó None — datos insuficientes, saltando")
                    continue

                filename = f"{self.ticker}_H{i}.json"
                path = os.path.join(DATA_DIR, filename)
                with open(path, "w") as f:
                    json.dump(result, f, indent=2, default=str)

            except AttributeError:
                print(f"❌ H{i} FALLÓ → función run_predictor_h{i}() no encontrada")
            except Exception as e:
                print(f"❌ H{i} FALLÓ: {e}")
            finally:
                gc.collect()

        self.collect_results()

        # [O5] Mínimo 5 modelos
        if len(self.results) < 5:
            raise RuntimeError(
                f"Ensemble inválido → solo {len(self.results)} modelos válidos (mínimo 5)"
            )

        missing = [h for h in range(1, 11) if h not in [r["horizon"] for r in self.results]]
        if missing:
            print(f"⚠️ Horizontes faltantes: {missing}")

        # [O10] Cargar pesos históricos por horizonte
        weights = _load_horizon_weights(self.ticker)
        if weights:
            active = {k: round(v, 3) for k, v in weights.items()}
            print(f"⚖️  Pesos dinámicos cargados: {active}")
        else:
            print(f"⚖️  Sin historial suficiente — pesos iguales")

        curve = self.build_price_curve(weights)

        # [O4] RuntimeError si hay nulls → no graba nada
        final_json = self.build_model_json(curve, weights)

        # [O6] ÚNICA ESCRITURA — orquestador es el dueño del disco
        ticker_dir  = os.path.join(FINAL_DIR, self.ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        file_name   = datetime.utcnow().strftime("%Y-%m-%d") + ".json"
        output_file = os.path.join(ticker_dir, file_name)

        tmp_file = output_file + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(final_json, f, indent=2)
        os.replace(tmp_file, output_file)

        print(f"💾 ENSEMBLE guardado: {output_file}")
        return final_json


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "SPY").upper()
    print(f"🎯 MasterOrchestrator v7.5 → {ticker}")
    print("=" * 60)

    orch   = MasterOrchestrator(ticker)
    result = orch.run()

    print("\n🏆 RESULTADO FINAL:")
    print(json.dumps(result["prediction"], indent=2))
    print(f"\n🎯 RECOMENDACIÓN: {result['prediction']['recommendation']}")
    if result.get("horizon_weights"):
        print(f"\n⚖️  Pesos usados: {result['horizon_weights']}")
        

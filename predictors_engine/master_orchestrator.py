# ======================================================
# master_orchestrator_v4_model_compatible.py
# ======================================================
# Ejecuta H1-H10
# Construye JSON compatible con model.py
# Incluye:
#   - Curva H1-H10
#   - Análisis de consenso
#   - Acelerador de retorno esperado (segunda derivada)
# ======================================================

import os
import json
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path
import sys


PREDICTOR_DIR = "predictors_engine"
DATA_DIR = "predictions_data"
FINAL_OUTPUT_DIR = "final_reports"

for d in [DATA_DIR, FINAL_OUTPUT_DIR]:
    Path(d).mkdir(exist_ok=True)


class MasterOrchestrator:

    def __init__(self, ticker):

        self.ticker = ticker.upper()

        self.predictors = {h: f"predictor_h{h}.py" for h in range(1, 11)}

        self.results = []

    # ======================================================
    # RUN ALL PREDICTORS
    # ======================================================

    def run_all_predictors(self):

        print(f"\n🚀 EJECUTANDO SUITE H1-H10 → {self.ticker}")

        processes = []

        for h, script in self.predictors.items():

            script_path = os.path.join(PREDICTOR_DIR, script)

            if os.path.exists(script_path):

                p = subprocess.Popen(
                    ["python3", script_path, self.ticker],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                processes.append((h, p))

        for h, p in processes:

            try:
                p.wait(timeout=180)
                print(f"   ✅ H{h} terminado")

            except subprocess.TimeoutExpired:

                p.kill()
                print(f"   ⚠️ H{h} timeout")

    # ======================================================
    # COLLECT RESULTS
    # ======================================================

    def collect_results(self):

        curve = []

        for h in range(1, 11):

            file_path = os.path.join(DATA_DIR, f"{self.ticker}_h{h}.json")

            if os.path.exists(file_path):

                try:

                    with open(file_path) as f:

                        data = json.load(f)

                        data["horizon"] = h

                        curve.append(data)

                except Exception as e:

                    print(f"   ❌ Error leyendo H{h}: {e}")

        self.results = sorted(curve, key=lambda x: x["horizon"])

        print(f"\n📊 {len(self.results)}/10 predictores válidos recolectados.")

        return self.results

    # ======================================================
    # CURVE ANALYSIS + ACCELERATION
    # ======================================================

    def analyze_curve(self):

        if len(self.results) < 2:

            return {
                "recommendation": "MANTÉN",
                "weighted_return_pct": 0.0,
                "consensus_score": 0.0,
                "prediction_volatility": 0.0,
                "acceleration_score": 0.0,
                "acceleration_boost": 0.0,
                "models_used": len(self.results)
            }

        horizons = np.array([r["horizon"] for r in self.results])

        returns = np.array([r.get("expected_ret_pct", 0) for r in self.results])

        # ======================================================
        # Interpolación curva completa H1-H10
        # ======================================================

        full_h = np.arange(1, 11)

        repaired = np.interp(full_h, horizons, returns)

        # ======================================================
        # Weighted ensemble
        # ======================================================

        weights = np.linspace(1.0, 1.6, len(repaired))

        base_return = np.average(repaired, weights=weights)

        volatility = float(np.std(repaired))

        consensus = 1 / (1 + volatility / (np.mean(np.abs(repaired)) + 1e-6))

        # ======================================================
        # ACCELERATION ENGINE
        # ======================================================

        first_derivative = np.gradient(repaired)

        second_derivative = np.gradient(first_derivative)

        acceleration = float(np.mean(second_derivative))

        accel_boost = np.clip(acceleration * 4.0, -0.5, 0.5)

        weighted_return = base_return + accel_boost

        # ======================================================
        # Recommendation logic
        # ======================================================

        if weighted_return > 0.5:

            rec = "COMPRA"

        elif weighted_return < -0.5:

            rec = "VENDE"

        else:

            rec = "MANTÉN"

        return {

            "weighted_return_pct": round(float(weighted_return), 4),

            "prediction_volatility": round(volatility, 4),

            "consensus_score": round(consensus, 4),

            "acceleration_score": round(acceleration, 6),

            "acceleration_boost": round(accel_boost, 4),

            "recommendation": rec,

            "models_used": len(self.results)

        }

    # ======================================================
    # BUILD MODEL COMPATIBLE JSON
    # ======================================================

    def build_model_json(self, curve_analysis):

        if not self.results:

            raise RuntimeError("No hay resultados")

        # usamos H10 como base principal

        h10 = next((r for r in self.results if r["horizon"] == 10), self.results[-1])

        return {

            # ==========================================
            # CONTRATO model.py
            # ==========================================

            "meta": {

                "ticker": self.ticker,

                "horizon_days": 10,

                "pca_target": 25,

                "theta": 0.85,

                "k_neighbors": 30,

                "alpha": float(h10.get("alpha_used", 2.0)),

                "period": "max"

            },

            "historical": {

                "hit_rate_mean": None,

                "mae_mean": None,

                "rmse_mean": None,

                "pca_dims": 25,

                "n_features": 26,

                "n_windows": 0

            },

            "prediction": {

                "date_base": datetime.utcnow().date().isoformat(),

                "ret_global_pct": round(curve_analysis["weighted_return_pct"], 4),

                "ret_knn_pct": round(curve_analysis["weighted_return_pct"], 4),

                "ret_ens_pct": round(curve_analysis["weighted_return_pct"], 4),

                "price_now": float(h10.get("price_now", 0)),

                "price_pred": float(h10.get("price_pred", 0)),

                "recommendation": curve_analysis["recommendation"],

                "theta_dynamic_pct": 0.85,

                "pca_dims_effective": 25,

                "n_features": 26

            },

            # ==========================================
            # EXTRAS PARA APP / ANALYTICS
            # ==========================================

            "prediction_curve": self.results,

            "curve_analysis": curve_analysis,

            "engine_metadata": {

                "engine": "H1-H10_SUITE_MASTER",

                "version": "v4_acceleration_engine",

                "timestamp": datetime.utcnow().isoformat()

            }

        }


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"

    maestro = MasterOrchestrator(ticker)

    maestro.run_all_predictors()

    maestro.collect_results()

    analysis = maestro.analyze_curve()

    final_payload = maestro.build_model_json(analysis)

    output_path = os.path.join(FINAL_OUTPUT_DIR, f"{ticker}_prediction.json")

    with open(output_path, "w") as f:

        json.dump(final_payload, f, indent=2)

    print(f"\n🏆 JSON FINAL GENERADO → {output_path}")

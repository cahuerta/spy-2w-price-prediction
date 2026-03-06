import os
import json
import numpy as np
import pandas as pd
import sys
import subprocess

DATA_DIR = "predictions_data"


# ======================================================
# TRADUCTOR UNIVERSAL DE RETORNOS
# ======================================================

def get_return_from_data(data):

    keys = [
        "expected_ret_pct",
        "ret_pct",
        "return_8d_pct",
        "return_9d_pct",
        "return_6d_pct",
        "normalized_ret"
    ]

    for k in keys:
        if k in data:
            return float(data[k])

    return 0.0


# ======================================================
# ORCHESTRATOR V5.2 (PRODUCTION READY)
# ======================================================

class MasterOrchestrator:


    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.results = []
        
        # ======================================================
# EJECUCIÓN SECUENCIAL H1-H10
# ======================================================

    def run_all_predictors(self):

        print(f"🚀 EJECUTANDO SUITE H1-H10 → {self.ticker}")

        base_dir = os.path.dirname(__file__)

        for h in range(1, 11):

            script = os.path.join(base_dir, f"predictor_h{h}.py")

            if not os.path.exists(script):

                print(f"❌ predictor_h{h}.py no encontrado")
                continue

            try:

                subprocess.run(
                    [sys.executable, script, self.ticker],
                    check=True
                )

                print(f"   ✅ H{h} terminado")

            except subprocess.CalledProcessError as e:

                print(f"   ❌ H{h} falló: {e}")


# ======================================================
# RECOLECTOR DE RESULTADOS
# ======================================================

    def collect_results(self):

        curve = []

        for h in range(1, 11):

            possible_files = [

                f"{self.ticker}_h{h}.json",
                f"{self.ticker}_H{h}_{h}d_v2.json",
                f"{self.ticker}_H{h}.json",
                f"{self.ticker}_H{h}_{h}d.json",
                f"{self.ticker}_H{h}_{h}d_v3.json"

            ]

            for fname in possible_files:

                file_path = os.path.join(DATA_DIR, fname)

                if os.path.exists(file_path):

                    with open(file_path) as f:

                        data = json.load(f)

                        data["horizon"] = h

                        data["normalized_ret"] = get_return_from_data(data)

                        curve.append(data)

                    break

        self.results = sorted(curve, key=lambda x: x["horizon"])

        return self.results


# ======================================================
# ANALYSIS DEFAULT
# ======================================================

    def default_analysis(self):

        return {

            "weighted_return_pct": 0.0,
            "prediction_volatility": 0.0,
            "consensus_score": 0.0,
            "acceleration_score": 0.0,
            "acceleration_boost": 0.0,
            "recommendation": "MANTÉN",
            "models_used": 0

        }


# ======================================================
# ANÁLISIS DE CURVA (ORTOGONAL)
# ======================================================

    def analyze_curve(self):

        if len(self.results) < 3:
            return self.default_analysis()


        horizons = np.array([r["horizon"] for r in self.results])

        returns = np.array([r["normalized_ret"] for r in self.results], dtype=float)

        returns = np.nan_to_num(
            returns,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


# ======================================================
# INTERPOLACIÓN H1-H10
# ======================================================

        full_h = np.arange(1, 11)

        repaired = np.interp(full_h, horizons, returns)


# ======================================================
# SUAVIZADO ANTI RUIDO
# ======================================================

        smoothed = pd.Series(repaired).rolling(

            3,
            center=True,
            min_periods=1

        ).mean().values


# ======================================================
# SISTEMA ORTOGONAL (QR)
# ======================================================

        H = np.vstack([

            np.ones(10),
            np.linspace(-1, 1, 10),
            np.linspace(-1, 1, 10) ** 2

        ]).T


        Q, _ = np.linalg.qr(H)


        coeffs = Q.T @ smoothed


        reconstructed = (

            coeffs[0] * Q[:, 0] +
            coeffs[1] * Q[:, 1] +
            coeffs[2] * Q[:, 2]

        )


# ======================================================
# ACELERACIÓN
# ======================================================

        first_derivative = np.gradient(reconstructed)

        second_derivative = np.gradient(first_derivative)

        acceleration = float(np.mean(second_derivative))

        accel_boost = np.clip(acceleration * 2.0, -0.3, 0.3)


# ======================================================
# CONFIDENCE GATE
# ======================================================

        residual = float(

            np.sqrt(
                np.mean(
                    (smoothed - reconstructed) ** 2
                )
            )

        )


        volatility = float(np.std(reconstructed))


        if residual > np.std(smoothed) * 2:

            volatility *= 1.5


# ======================================================
# RETORNO PONDERADO
# ======================================================

        weighted_return = np.average(

            reconstructed,
            weights=np.linspace(0.5, 1.5, 10)

        ) + accel_boost


        weighted_return = float(
            np.clip(weighted_return, -20, 20)
        )


# ======================================================
# CONSENSO
# ======================================================

        consensus = float(1 / (1 + volatility))


# ======================================================
# RECOMENDACIÓN
# ======================================================

        recommendation = (

            "COMPRA"
            if weighted_return > 0.4
            else "VENDE"
            if weighted_return < -0.4
            else "MANTÉN"

        )


# ======================================================
# JSON FINAL
# ======================================================

        return {

            "weighted_return_pct": round(weighted_return, 4),

            "prediction_volatility": round(volatility, 4),

            "consensus_score": round(consensus, 4),

            "acceleration_score": round(acceleration, 6),

            "acceleration_boost": round(accel_boost, 4),

            "recommendation": recommendation,

            "models_used": len(self.results)

        }


# ======================================================
# RUN
# ======================================================

    def run(self):

        self.collect_results()

        return self.analyze_curve()


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":

    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"

    orch = MasterOrchestrator(ticker)

    result = orch.run()

    print(f"\n🏆 MASTER ORCHESTRATOR → {ticker}")

    print(json.dumps(result, indent=2))

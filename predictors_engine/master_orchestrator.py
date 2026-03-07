import os
import json
import numpy as np
import pandas as pd
import sys
import subprocess
from datetime import datetime

DATA_DIR = "predictions_data"

# ======================================================
# UTILIDADES
# ======================================================

def extract_price_prediction(data):
    for k in [
        "price_pred"
    ]:
        if k in data:
            return float(data[k])
    return None


def extract_return_pct(data):
    for k in [
        "return_pct"
    ]:
        if k in data:
            return float(data[k])
    return 0.0


# ======================================================
# ORCHESTRATOR
# ======================================================

class MasterOrchestrator:

    def __init__(self,ticker):

        self.ticker = ticker.upper()
        self.results = []
        self.price_now = None


# ======================================================
# EJECUCIÓN H1-H10
# ======================================================

    def run_all_predictors(self):

        print(f"🚀 EJECUTANDO SUITE H1-H10 → {self.ticker}")

        base_dir = os.path.dirname(__file__)

        for h in range(1,11):

            script = os.path.join(base_dir,f"predictor_h{h}.py")

            if not os.path.exists(script):

                print(f"⚠️ predictor_h{h}.py no encontrado")
                continue

            try:

                subprocess.run(
                    [sys.executable,script,self.ticker],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                print(f"   ✅ H{h} terminado")

            except subprocess.TimeoutExpired:

                print(f"   ⏰ H{h} timeout")

            except subprocess.CalledProcessError:

                print(f"   ❌ H{h} falló")

            except Exception as e:

                print(f"   ❌ H{h} error {e}")


# ======================================================
# RECOLECTAR RESULTADOS
# ======================================================

    def collect_results(self):

        curve = []

        for h in range(1,11):

            possible_files = [

                f"{self.ticker}_H{h}.json",
                f"{self.ticker}_h{h}.json",
                f"{self.ticker}_H{h}_{h}d.json",
                f"{self.ticker}_H{h}_{h}d_v2.json",
                f"{self.ticker}_H{h}_{h}d_v3.json"
            ]

            for fname in possible_files:

                file_path = os.path.join(DATA_DIR,fname)

                if os.path.exists(file_path):

                    try:

                        with open(file_path) as f:

                            data = json.load(f)

                        data["horizon"] = h

                        price = extract_price_prediction(data)

                        if price is None:
                            print(f"⚠️ H{h}: sin precio")
                            continue

                        data["price_pred"] = price

                        # detectar precio actual
                        if self.price_now is None:

                            self.price_now = (
                                data.get("price_now")
                                or data.get("price_today")
                            )

                            if self.price_now is None:
                                raise RuntimeError("price_now missing")

                            self.price_now = float(self.price_now)

                        data["normalized_ret"] = extract_return_pct(data)

                        curve.append(data)

                        print(f"   📊 H{h}: ${price:.2f}")

                        break

                    except Exception as e:

                        print(f"⚠️ H{h} JSON error {e}")

        self.results = sorted(curve,key=lambda x:x["horizon"])

        print(f"📈 {len(self.results)}/10 modelos recolectados")

        return self.results


# ======================================================
# CURVA DE PRECIOS
# ======================================================

    def build_price_curve(self):

        if len(self.results) < 2:
            return None

        horizons = np.array([r["horizon"] for r in self.results])
        prices = np.array([r["price_pred"] for r in self.results])

        full_h = np.arange(1,11)

        interpolated = np.interp(full_h,horizons,prices)

        smoothed = pd.Series(interpolated).rolling(
            3,
            center=True,
            min_periods=1
        ).mean().values

        H = np.vstack([
            np.ones(10),
            np.linspace(-1,1,10),
            np.linspace(-1,1,10)**2
        ]).T

        Q,_ = np.linalg.qr(H)

        coeffs = Q.T @ smoothed

        reconstructed = (
            coeffs[0]*Q[:,0] +
            coeffs[1]*Q[:,1] +
            coeffs[2]*Q[:,2]
        )

        velocity = np.gradient(reconstructed)

        acceleration = np.gradient(velocity)

        return {

            "days":list(range(1,11)),
            "price_path":reconstructed.tolist(),
            "velocity":velocity.tolist(),
            "acceleration":acceleration.tolist(),
            "residual_std":float(np.std(smoothed-reconstructed))
        }


# ======================================================
# DETECTAR CONFLICTO TEMPORAL
# ======================================================

    def detect_horizon_conflict(self):

        if len(self.results) < 6:
            return 1.0

        short = np.mean([
            r["price_pred"] for r in self.results if r["horizon"]<=3
        ])

        long = np.mean([
            r["price_pred"] for r in self.results if r["horizon"]>=8
        ])

        if (short-self.price_now)*(long-self.price_now) < 0:

            return 0.6

        return 1.0


# ======================================================
# ANÁLISIS FINAL
# ======================================================

    def analyze_curve(self,curve):

        if curve is None or self.price_now is None:

            return {

                "weighted_return_pct":0.0,
                "price_change_pct":0.0,
                "prediction_volatility":0.0,
                "consensus_score":0.0,
                "acceleration_score":0.0,
                "acceleration_boost":0.0,
                "recommendation":"MANTÉN",
                "models_used":len(self.results)
            }

        price_path = np.array(curve["price_path"])

        returns = np.diff(price_path)/price_path[:-1]

        volatility = float(np.std(returns))

        acceleration = float(np.mean(curve["acceleration"]))

        accel_boost = np.clip(acceleration*0.1,-0.3,0.3)

        price_final = price_path[-1]

        price_change_pct = ((price_final-self.price_now)/self.price_now)*100

        weighted_return = price_change_pct + accel_boost

        consensus = 1/(1+volatility*10)

        conflict_penalty = self.detect_horizon_conflict()

        consensus *= conflict_penalty

        consensus = max(0.0,min(1.0,consensus))

        signal_strength = weighted_return*consensus

        recommendation = (
            "COMPRA" if signal_strength>0.4
            else "VENDE" if signal_strength<-0.4
            else "MANTÉN"
        )

        return {

            "weighted_return_pct":round(weighted_return,4),
            "price_change_pct":round(price_change_pct,4),
            "prediction_volatility":round(volatility,4),
            "consensus_score":round(consensus,4),
            "acceleration_score":round(acceleration,6),
            "acceleration_boost":round(accel_boost,4),
            "recommendation":recommendation,
            "models_used":len(self.results)
        }


# ======================================================
# JSON FINAL (CONTRATO)
# ======================================================

    def build_model_json(self,analysis,curve):

        price_pred = curve["price_path"][-1] if curve else self.price_now

        model_json = {

            "meta":{
                "ticker":self.ticker,
                "horizon_days":10,
                "pca_target":None,
                "theta":None,
                "k_neighbors":None,
                "alpha":None,
                "period":"H1-H10_ENSEMBLE"
            },

            "historical":{
                "hit_rate_mean":None,
                "mae_mean":None,
                "rmse_mean":None,
                "pca_dims":None,
                "n_features":len(self.results),
                "n_windows":len(self.results)
            },

            "prediction":{

                "date_base":datetime.utcnow().strftime("%Y-%m-%d"),

                "ret_global_pct":round(analysis["weighted_return_pct"],4),
                "ret_knn_pct":round(analysis["price_change_pct"],4),
                "ret_ens_pct":round(analysis["weighted_return_pct"],4),

                "price_now":round(float(self.price_now),2),
                "price_pred":round(float(price_pred),2),

                "recommendation":analysis["recommendation"],

                "theta_dynamic_pct":round(analysis["consensus_score"],4),

                "pca_dims_effective":None,
                "n_features":len(self.results)
            }
        }

        model_json["price_curve"] = curve
        model_json["analysis"] = analysis
        model_json["models_used"] = analysis["models_used"]

        return model_json


# ======================================================
# RUN
# ======================================================

    def run(self):

        self.run_all_predictors()

        self.collect_results()

        curve = self.build_price_curve()

        analysis = self.analyze_curve(curve)

        final_json = self.build_model_json(analysis,curve)

        output_file = os.path.join(DATA_DIR,f"{self.ticker}_ENSEMBLE.json")

        os.makedirs(DATA_DIR,exist_ok=True)

        with open(output_file,"w") as f:

            json.dump(final_json,f,indent=2,default=str)

        print(f"💾 ENSEMBLE guardado: {output_file}")

        return final_json


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":

    ticker = (sys.argv[1] if len(sys.argv)>1 else "SPY").upper()

    print(f"🎯 MasterOrchestrator v6.1 → {ticker}")
    print("="*60)

    orch = MasterOrchestrator(ticker)

    result = orch.run()

    print("\n🏆 RESULTADO FINAL:")
    print(json.dumps(result["prediction"],indent=2))

    print(f"\n🎯 RECOMENDACIÓN: {result['prediction']['recommendation']}")

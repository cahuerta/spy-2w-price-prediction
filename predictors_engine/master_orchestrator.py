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

    for k in ["price_pred"]:

        if k in data:
            return float(data[k])

    if "prediction" in data:
        return float(data["prediction"]["price_pred"])

    return None


def extract_return_pct(data):

    for k in ["return_pct","ret_ens_pct"]:

        if k in data:
            return float(data[k])

    if "prediction" in data:
        return float(data["prediction"]["ret_ens_pct"])

    return 0.0


# ======================================================
# ORCHESTRATOR
# ======================================================

class MasterOrchestrator:

    def __init__(self,ticker):

        self.ticker = ticker.upper()

        self.results = []

        self.price_today = None

        self.h10_json = None


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

                path = os.path.join(DATA_DIR,fname)

                if os.path.exists(path):

                    try:

                        with open(path) as f:
                            data = json.load(f)

                        data["horizon"] = h

                        price = extract_price_prediction(data)

                        if price is None:
                            continue

                        if self.price_today is None:

                            self.price_now = (
                                data.get("price_now")
                                or data.get("price_today")
                                or data.get("prediction",{}).get("price_now")
                            )

                            if self.price_now is None:
                                raise RuntimeError("price_now missing")

                            self.price_now = float(self.price_now)

                        data["price_pred"] = price

                        data["normalized_ret"] = extract_return_pct(data)

                        curve.append(data)

                        if h == 10:

                            self.h10_json = data

                        print(f"   📊 H{h}: ${price:.2f}")

                        break

                    except Exception as e:

                        print(f"⚠️ H{h} JSON error {e}")

        self.results = sorted(curve,key=lambda x:x["horizon"])

        print(f"📈 {len(self.results)}/10 modelos recolectados")

        return self.results


# ======================================================
# CURVA (solo H1-H9)
# ======================================================

    def build_price_curve(self):

        filtered = [r for r in self.results if r["horizon"] < 10]

        if len(filtered) < 2:
            return None

        horizons = np.array([r["horizon"] for r in filtered])

        prices = np.array([r["price_pred"] for r in filtered])

        full_h = np.arange(1,10)

        interpolated = np.interp(full_h,horizons,prices)

        smoothed = pd.Series(interpolated).rolling(
            3,
            center=True,
            min_periods=1
        ).mean().values

        return {

            "days":list(range(1,10)),
            "price_path":smoothed.tolist()
        }


# ======================================================
# ANALISIS FINAL
# ======================================================

    def analyze_curve(self,curve):

        if curve is None:
            return 0

        curve_last = curve["price_path"][-1]

        curve_ret = ((curve_last-self.price_now)/self.price_now)*100

        return curve_ret


# ======================================================
# JSON FINAL
# ======================================================

    def build_model_json(self,curve):

        if self.h10_json is None:

            raise RuntimeError("H10 missing")

        final_json = self.h10_json.copy()

        h10_price = extract_price_prediction(self.h10_json)

        curve_adjust = self.analyze_curve(curve)

        # consenso entre modelos
        consensus = np.std([r["price_pred"] for r in self.results if r["horizon"] < 10])

        consensus_weight = 1/(1+consensus/self.price_today)

        # ajuste suave (H10 sigue mandando)
        adjust_factor = 0.15 * consensus_weight

        final_price = h10_price * (1 + curve_adjust/100 * adjust_factor)

        final_json["prediction"]["price_pred"] = round(final_price,2)

        final_json["prediction"]["ret_ens_pct"] = round(
            ((final_price-self.price_now)/self.price_now)*100,
            4
        )

        final_json["prediction"]["recommendation"] = (

            "COMPRA"
            if final_json["prediction"]["ret_ens_pct"] > 0.4
            else "VENDE"
            if final_json["prediction"]["ret_ens_pct"] < -0.4
            else "MANTÉN"

        )

        final_json["price_curve"] = curve

        final_json["ensemble_models"] = len(self.results)

        return final_json


# ======================================================
# RUN
# ======================================================

    def run(self):


        self.collect_results()

        curve = self.build_price_curve()

        final_json = self.build_model_json(curve)

        output_file = os.path.join(DATA_DIR,f"{self.ticker}_ENSEMBLE.json")

        os.makedirs(DATA_DIR,exist_ok=True)

        with open(output_file,"w") as f:

            json.dump(final_json,f,indent=2)

        print(f"💾 ENSEMBLE guardado: {output_file}")

        return final_json


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":

    ticker = (sys.argv[1] if len(sys.argv)>1 else "SPY").upper()

    print(f"🎯 MasterOrchestrator v7 → {ticker}")
    print("="*60)

    orch = MasterOrchestrator(ticker)

    result = orch.run()

    print("\n🏆 RESULTADO FINAL:")

    print(json.dumps(result["prediction"],indent=2))

    print(f"\n🎯 RECOMENDACIÓN: {result['prediction']['recommendation']}")

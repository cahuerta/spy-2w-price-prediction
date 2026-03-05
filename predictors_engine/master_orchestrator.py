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

        self.predictors = {
            1: "predictor_h1.py",
            2: "predictor_h2.py",
            3: "predictor_h3.py",
            4: "predictor_h4.py",
            5: "predictor_h5.py",
            6: "predictor_h6.py",
            7: "predictor_h7.py",
            8: "predictor_h8.py",
            9: "predictor_h9.py",
            10: "predictor_h10.py"
        }

        self.results = []

    # ======================================================
    # EJECUCIÓN DE PREDICTORES
    # ======================================================

    def run_all_predictors(self):

        print(f"🚀 MASTER ORQUESTADOR H1-H10 → {self.ticker}")
        print("=" * 60)

        processes = []

        for h, script_name in self.predictors.items():

            script_path = os.path.join(PREDICTOR_DIR, script_name)

            if os.path.exists(script_path):

                print(f"🔥 H{h:2d} → {script_name}")

                p = subprocess.Popen(
                    ["python3", script_path, self.ticker],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                processes.append((h, p))

            else:
                print(f"❌ H{h:2d} NO ENCONTRADO: {script_path}")

        completed = 0

        for h, p in processes:

            try:

                p.wait(timeout=180)

                print(f"   ✅ H{h:2d} COMPLETADO")

                completed += 1

            except subprocess.TimeoutExpired:

                p.kill()

                print(f"   ⚠️  H{h:2d} TIMEOUT")

        print(f"\n✅ {completed}/10 predictores ejecutados")

        return completed


    # ======================================================
    # RECOLECCIÓN DE RESULTADOS
    # ======================================================

    def collect_results(self):

        print("\n📊 RECOLECTANDO RESULTADOS H1-H10...")

        curve_data = []

        filename_patterns = {
            1: "H1_dia1_v2.json",
            2: "H2_48h_v2.json",
            3: "H3_72h_v2.json",
            4: "H4_96h_v2.json",
            5: "H5_semanal_v2.json",
            6: "H6_6d_v2.json",
            7: "H7_7d_v2.json",
            8: "H8_8d_v2.json",
            9: "H9_9d_v2.json",
            10: "H10_10d_v2.json"
        }

        for h in range(1, 11):

            pattern = filename_patterns[h]

            filepath = os.path.join(DATA_DIR, f"{self.ticker}_{pattern}")

            if os.path.exists(filepath):

                try:

                    with open(filepath, "r") as f:

                        data = json.load(f)

                        data["horizon"] = h
                        data["horizon_label"] = f"H{h}"

                        curve_data.append(data)

                        print(f"   ✅ H{h:2d} → {pattern}")

                except json.JSONDecodeError:

                    print(f"   ❌ H{h:2d} JSON corrupto")

            else:

                print(f"   ❌ H{h:2d} NO ENCONTRADO")

        self.results = sorted(curve_data, key=lambda x: x["horizon"])

        print(f"\n📈 {len(self.results)}/10 predictores válidos")

        return self.results


    # ======================================================
    # REPARACIÓN DE CURVA
    # ======================================================

    def repair_curve(self, horizons, returns):

        if len(horizons) < 2:
            return returns

        full_h = np.arange(1, 11)

        repaired = np.interp(full_h, horizons, returns)

        return repaired


    # ======================================================
    # ANÁLISIS MAESTRO
    # ======================================================

    def analyze_curve_and_recommend(self):

        if len(self.results) < 3:

            return {
                "recommendation": "🟡 INSUFICIENTES_PREDICTORES",
                "models_used": len(self.results),
                "status": "Necesita mínimo 3 modelos"
            }

        returns = []
        confidences = []
        horizons = []

        return_keys = [
            'return_72h_pct',
            'return_96h_pct',
            'return_5d_pct',
            'return_6d_pct',
            'return_7d_pct',
            'return_8d_pct',
            'return_9d_pct',
            'return_10d_pct',
            'expected_ret_pct'
        ]

        for r in self.results:

            horizons.append(r["horizon"])

            value = None

            for key in return_keys:
                if key in r:
                    value = r[key]
                    break

            if value is not None:
                returns.append(float(value))

            conf = r.get("confidence") or r.get("r2_train") or 0.5

            confidences.append(float(conf))

        if len(returns) == 0:

            return {"recommendation": "❌ SIN_RETORNOS_VÁLIDOS"}

        returns = np.array(self.repair_curve(horizons, returns))

        weights = np.linspace(0.8, 1.8, len(returns))

        conf_weights = np.array(confidences) / (np.max(confidences) + 1e-8)

        final_weights = weights * conf_weights

        weighted_avg = np.average(returns, weights=final_weights)

        volatility = np.std(returns)

        consensus = 1 / (1 + volatility / (np.mean(np.abs(returns)) + 1e-6))

        if volatility > 4.0:
            rec = "🟡 ESPERAR (Dispersión extrema)"

        elif weighted_avg > 1.5:
            rec = "🟢🟢 COMPRA FUERTE"

        elif weighted_avg > 0.4:
            rec = "🟢 COMPRA"

        elif weighted_avg < -1.5:
            rec = "🔴🔴 VENTA FUERTE"

        elif weighted_avg < -0.4:
            rec = "🔴 VENTA"

        else:
            rec = "⚪ NEUTRAL"

        return {

            "recommendation": rec,

            "weighted_return_pct": round(float(weighted_avg), 2),

            "consensus_score": round(float(consensus), 3),

            "prediction_volatility": round(float(volatility), 2),

            "models_used": len(self.results),

            "avg_confidence": round(float(np.mean(confidences)), 3),

            "strongest_signal": round(float(np.max(np.abs(returns))), 2),

            "status": "ANÁLISIS_COMPLETO"
        }


    # ======================================================
    # GUARDAR RESULTADO FINAL
    # ======================================================

    def save_final_json(self, analysis):

        final_payload = {

            "ticker": self.ticker,

            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "timestamp": datetime.now().isoformat(),

            "suite_version": "H1-H10_v2.2",

            "total_predictors": 10,

            "valid_predictors": len(self.results),

            "prediction_curve": self.results,

            "master_analysis": analysis

        }

        output_file = os.path.join(
            FINAL_OUTPUT_DIR,
            f"{self.ticker}_MASTER_H1H10_v2.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2, ensure_ascii=False)

        print(f"\n🏆 MASTER REPORT → {output_file}")

        return output_file


    # ======================================================
    # RESUMEN EJECUTIVO
    # ======================================================

    def print_executive_summary(self, analysis):

        print("\n" + "═" * 80)

        print(f"🏆 SUITE H1-H10 EXECUTIVE SUMMARY → {self.ticker}")

        print("═" * 80)

        print(f"📊 Modelos válidos:    {analysis['models_used']}/10")

        print(f"📈 Retorno ponderado:  {analysis['weighted_return_pct']:+6.2f}%")

        print(f"🤝 Consenso:           {analysis['consensus_score']:>6.3f}")

        print(f"📊 Volatilidad pred:   {analysis['prediction_volatility']:>5.2f}%")

        print(f"🔥 Señal más fuerte:   {analysis['strongest_signal']:+.2f}%")

        print(f"🎯 RECOMENDACIÓN:      {analysis['recommendation']}")

        print("═" * 80)



# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"

    maestro = MasterOrchestrator(ticker)

    print("🚀 INICIANDO SUITE H1-H10...\n")

    maestro.run_all_predictors()

    maestro.collect_results()

    analysis = maestro.analyze_curve_and_recommend()

    maestro.save_final_json(analysis)

    maestro.print_executive_summary(analysis)

    print(f"\n🎉 {ticker} H1-H10 MASTER COMPLETADO!")

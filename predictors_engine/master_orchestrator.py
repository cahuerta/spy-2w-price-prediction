import os
import json
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
import time

# ======================================================
# CONFIGURACIÓN DE RUTAS
# ======================================================
PREDICTOR_DIR = "predictors_engine" # Donde están tus .py
DATA_DIR = "predictions_data"       # Donde aterrizan los .json
FINAL_OUTPUT_DIR = "final_reports"  # Resultado final para el usuario

for d in [DATA_DIR, FINAL_OUTPUT_DIR]:
    if not os.path.exists(d): os.makedirs(d)

class MasterOrchestrator:
    def __init__(self, ticker):
        self.ticker = ticker
        self.horizons = range(1, 11)
        self.results = {}

    def run_all_predictors(self):
        """Lanza los 10 scripts en paralelo"""
        processes = []
        print(f"🚀 Iniciando motores para {self.ticker}...")
        
        for h in self.horizons:
            script = os.path.join(PREDICTOR_DIR, f"predictor_h{h}.py")
            if os.path.exists(script):
                # Ejecutamos cada predictor como un subproceso
                p = subprocess.Popen(["python", script, self.ticker])
                processes.append(p)
            else:
                print(f"⚠️ Alerta: No se encontró {script}")

        # Esperar a que todos terminen
        for p in processes:
            p.wait()
        print("✅ Todos los predictores han finalizado.")

    def collect_results(self):
        """Lee los archivos JSON generados"""
        curve_data = []
        for h in self.horizons:
            file_path = os.path.join(DATA_DIR, f"{self.ticker}_h{h}.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    curve_data.append(data)
        
        self.results = sorted(curve_data, key=lambda x: x["horizon"])
        return self.results

    def analyze_curve_and_recommend(self):
        """Lógica Maestra: Analiza la pendiente y la consistencia"""
        if not self.results: return "SIN_DATOS"

        returns = [r["expected_ret_pct"] for r in self.results]
        prices = [r["price_pred"] for r in self.results]
        current_price = self.results[0]["price_now"]
        
        # 1. Cálculo de Pendientes (Gradiente)
        short_term_avg = np.mean(returns[:3])  # H1-H3
        long_term_avg = np.mean(returns[7:])   # H8-H10
        full_slope = np.polyfit(range(10), returns, 1)[0]

        # 2. Lógica de Recomendación Inteligente
        # Si la mayoría de la curva es positiva y la pendiente es ascendente
        if short_term_avg > 0.5 and long_term_avg > 0:
            rec = "COMPRA FUERTE" if full_slope > 0 else "COMPRA"
        
        # Si el corto plazo sube pero el largo plazo se hunde (Divergencia)
        elif short_term_avg > 0.2 and long_term_avg < -0.5:
            rec = "MANTÉN (ALERTA DE GIRO)"
            
        # Si la curva es mayormente negativa
        elif short_term_avg < -0.5:
            rec = "VENDE" if long_term_avg < 0 else "VENTA TÉCNICA (REBOTE CERCANO)"
            
        else:
            rec = "MANTÉN"

        return {
            "recommendation": rec,
            "short_term_bias": round(short_term_avg, 2),
            "long_term_bias": round(long_term_avg, 2),
            "curve_slope": round(full_slope, 4),
            "expected_prices": prices
        }

    def save_final_json(self, analysis):
        """Graba el JSON maestro final tal cual lo pediste"""
        final_payload = {
            "ticker": self.ticker,
            "timestamp": datetime.now().isoformat(),
            "current_price": self.results[0]["price_now"],
            "prediction_curve": self.results,
            "master_analysis": analysis
        }
        
        output_file = os.path.join(FINAL_OUTPUT_DIR, f"{self.ticker}_MASTER.json")
        with open(output_file, "w") as f:
            json.dump(final_payload, f, indent=4)
        
        print(f"🏆 Archivo Maestro grabado con éxito: {output_file}")
        return final_payload

if __name__ == "__main__":
    ticker_to_process = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    
    maestro = MasterOrchestrator(ticker_to_process)
    maestro.run_all_predictors()
    maestro.collect_results()
    analysis = maestro.analyze_curve_and_recommend()
    maestro.save_final_json(analysis)
  

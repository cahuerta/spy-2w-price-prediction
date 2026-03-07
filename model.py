# ======================================================
# model.py — V6.4 (CON FILTRO DE CALIDAD - CAPA 1)
# ======================================================

import sys
import json
from datetime import datetime
from predictors_engine.master_orchestrator import MasterOrchestrator
from data_provider import get_price_history

def run_model(
    ticker="SPY",
    horizon=10,
    pca_target=50,
    theta=0.75,
    k_neighbors=20,
    alpha=0.5,
    period="max"
):
    """
    Ejecuta el orquestador con un filtro de calidad previo.
    """
    
    # ======================================================
    # CAPA 1: FILTRO DE CALIDAD (CORTA CABEZAS)
    # ======================================================
    raw_data = get_price_history(ticker=ticker, period="max", interval="1d")
    
    # Requisito mínimo: 400 días de cotización (aprox. 1.5 años de mercado)
    min_registros = 400 
    if raw_data is None or len(raw_data) < min_registros:
        return {
            "status": "SKIP",
            "message": f"Ticker {ticker} descartado: insuficiente historial ({len(raw_data) if raw_data is not None else 0} registros)",
            "meta": {"ticker": ticker}
        }
    
    # ======================================================
    # ORQUESTACIÓN
    # ======================================================
    try:
        maestro = MasterOrchestrator(ticker)
        result = maestro.run()
        return result
    except Exception as e:
        return {
            "status": "ERROR",
            "message": str(e),
            "ticker": ticker
        }

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    # Imprimimos el resultado como JSON limpio
    print(json.dumps(run_model(ticker), indent=2))

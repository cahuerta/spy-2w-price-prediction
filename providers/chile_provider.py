# =========================================================
# providers/chile_provider.py
# DATA PROVIDER — BOLSA DE SANTIAGO (OFFICIAL-LIKE)
# =========================================================

import logging
import requests
import pandas as pd
from datetime import datetime

logger = logging.getLogger("data_provider.chile")

def get_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Extrae datos de la Bolsa de Santiago.
    Ticker format: 'CHILE', 'COPEC', 'SQM-B'
    """
    logger.info(f"[CHILE] Extrayendo datos oficiales para {ticker}")
    
    # Limpiar ticker (quitar .SN si viene de Yahoo)
    symbol = ticker.replace(".SN", "").replace(".SCL", "")
    
    url = "https://www.bolsadesantiago.com/api/Series/getSeries"
    
    payload = {
        "Instrumento": symbol,
        "Periodo": "Anual", # Opcional: ajustar según lógica
        "Tipo": "Cierre"
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()

        if not data.get("ListaSeries"):
            raise ValueError(f"No hay datos para {symbol}")

        # Mapeo al formato que espera tu screener
        df = pd.DataFrame(data["ListaSeries"])
        
        # Ajuste de columnas (La Bolsa entrega: Fecha, Valor, etc.)
        df['Date'] = pd.to_datetime(df['FechaString'], dayfirst=True)
        df = df.rename(columns={'Ultimo': 'Close', 'Volumen': 'Volume'})
        
        # Asegurar columnas mínimas para tu engine
        df['Open'] = df['Close'] # La API de serie histórica a veces solo da cierre
        df['High'] = df['Close']
        df['Low'] = df['Close']
        
        df.set_index('Date', inplace=True)
        df = df.sort_index()

        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception as e:
        logger.error(f"[CHILE] Error en {symbol}: {e}")
        return pd.DataFrame() # Devolver vacío para que el screener lo ignore

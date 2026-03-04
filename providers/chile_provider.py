import logging
import requests
import pandas as pd
from providers import yahoo_provider # Reutilizamos tu provider existente

logger = logging.getLogger("data_provider.chile")

def get_price_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    # 1. Limpiar ticker para la Bolsa (ej: 'CHILE.SN' -> 'CHILE')
    symbol = ticker.split('.')[0].upper()
    
    # INTENTO 1: BOLSA DE SANTIAGO
    try:
        url = "https://www.bolsadesantiago.com/api/Series/getSeries"
        payload = {"Instrumento": symbol, "Periodo": "Cinco", "Tipo": "Cierre"}
        
        response = requests.post(url, json=payload, timeout=8)
        data = response.json()
        
        if data.get("ListaSeries"):
            df = pd.DataFrame(data["ListaSeries"])
            df['Date'] = pd.to_datetime(df['FechaString'], dayfirst=True)
            df = df.rename(columns={'Ultimo': 'Close', 'Valor': 'Close', 'Volumen': 'Volume'})
            
            # Asegurar columnas OHLC para compatibilidad
            for col in ['Open', 'High', 'Low']: df[col] = df['Close']
            
            df.set_index('Date', inplace=True)
            logger.info(f"✅ [CHILE] Datos obtenidos de Bolsa de Santiago para {symbol}")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
            
    except Exception as e:
        logger.warning(f"⚠️ [CHILE] Falló Bolsa de Santiago para {symbol}: {e}")

    # INTENTO 2: FALLBACK A YAHOO (Usando tu contrato actual)
    logger.info(f"🔄 [CHILE] Intentando Fallback a Yahoo para {ticker}")
    return yahoo_provider.get_price_history(ticker, period, interval)

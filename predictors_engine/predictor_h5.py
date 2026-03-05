import numpy as np
import pandas as pd
import os
import json
import sys
from datetime import datetime

# Importamos tu fachada de datos (Cache -> Chile -> Yahoo)
from data_provider import get_price_history

import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

# ======================================================
# CONFIGURACIÓN DE RUTAS DEL REPOSITORIO
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
if not os.path.exists(DATA_OUTPUT_DIR):
    os.makedirs(DATA_OUTPUT_DIR)

# --- PARÁMETROS ESPECÍFICOS H5 (Tendencia Semanal) ---
HORIZON = 5
ALPHA_H5 = 0.65     # Fuerte regularización para filtrar volatilidad intra-semanal
PCA_COMPONENTS = 25  # Máxima captura de componentes estructurales

def make_features_h5(df: pd.DataFrame):
    out = df.copy()
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"])
    
    # Lags: Para 1 semana, necesitamos ver el contexto de las últimas 5 semanas (25 días)
    for k in range(1, 26):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # Volatilidad de 20 días (Mes de trading)
    past = out["ret1"].shift(1)
    out["rv_20"] = past.rolling(20).std() 
    
    # RSI 14 (Estándar para detectar momentum semanal)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = delta.clip(upper=0).abs().rolling(14).mean()
    out["rsi_14"] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    
    # Media Móvil Exponencial (EMA) para suavizado de tendencia
    out["ema_20_dist"] = (out["Close"] - out["Close"].ewm(span=20).mean()) / out["Close"]
    
    return out

def run_predictor_h5(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")
    if raw is None or raw.empty:
        return None
    
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h5(df)
    
    # Columnas para H5
    feature_cols = ["range", "rv_20", "rsi_14", "ema_20_dist"] + [f"ret_lag_{k}" for k in range(1, 26)]
    target_col = "y_fwd"
    feat[target_col] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    
    clean = feat.dropna(subset=feature_cols + [target_col])
    if len(clean) < 180: return None

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(PCA_COMPONENTS, len(feature_cols)))),
        ("ridge", Ridge(alpha=ALPHA_H5))
    ])
    
    model.fit(clean[feature_cols].values, clean[target_col].values)
    
    last_row = feat.iloc[-1]
    X_last = last_row[feature_cols].values.reshape(1, -1)
    
    y_pred_log = model.predict(X_last)[0]
    price_now = float(last_row["Close"])
    price_pred = float(price_now * np.exp(y_pred_log))
    
    return {
        "ticker": ticker,
        "horizon": HORIZON,
        "price_now": round(price_now, 2),
        "price_pred": round(price_pred, 2),
        "expected_ret_pct": round(y_pred_log * 100, 4),
        "alpha_used": ALPHA_H5,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker_input = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    res = run_predictor_h5(ticker_input)
    
    if res:
        filename = f"{ticker_input}_h{HORIZON}.json"
        full_path = os.path.join(DATA_OUTPUT_DIR, filename)
        
        with open(full_path, "w") as f:
            json.dump(res, f, indent=2)
            
        print(f"✅ H5 FINALIZADO: {ticker_input} -> {full_path}")

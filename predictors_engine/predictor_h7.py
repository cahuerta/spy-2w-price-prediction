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

# --- UTILS PARA H7 ---
def hurst_simple(ts):
    """Cálculo simplificado de Hurst para inercia a 7 días"""
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# --- PARÁMETROS ESPECÍFICOS H7 (Persistencia) ---
HORIZON = 7
ALPHA_H7 = 1.0      # Regularización unitaria: muy conservador
PCA_COMPONENTS = 25 

def make_features_h7(df: pd.DataFrame):
    out = df.copy()
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"])
    
    # Lags: Expandimos a 30 días (un mes y medio de contexto)
    for k in range(1, 31):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # Volatilidad de 30 días
    past = out["ret1"].shift(1)
    out["rv_30"] = past.rolling(30).std() 
    
    # Hurst Exponent (Ventana de 60 días para detectar régimen)
    out["hurst_60"] = past.rolling(60).apply(hurst_simple, raw=True)
    
    # Distancia a la Media Móvil de 50 días (Fuerza Mayor)
    sma50 = out["Close"].rolling(window=50).mean()
    out["dist_sma50"] = (out["Close"] - sma50) / sma50
    
    return out

def run_predictor_h7(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")
    if raw is None or raw.empty:
        return None
    
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h7(df)
    
    # Columnas para H7
    feature_cols = ["range", "rv_30", "hurst_60", "dist_sma50"] + [f"ret_lag_{k}" for k in range(1, 31)]
    target_col = "y_fwd"
    feat[target_col] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    
    clean = feat.dropna(subset=feature_cols + [target_col])
    if len(clean) < 220: return None

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(PCA_COMPONENTS, len(feature_cols)))),
        ("ridge", Ridge(alpha=ALPHA_H7))
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
        "alpha_used": ALPHA_H7,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker_input = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    res = run_predictor_h7(ticker_input)
    
    if res:
        filename = f"{ticker_input}_h{HORIZON}.json"
        full_path = os.path.join(DATA_OUTPUT_DIR, filename)
        
        with open(full_path, "w") as f:
            json.dump(res, f, indent=2)
            
        print(f"✅ H7 FINALIZADO: {ticker_input} -> {full_path}")

import numpy as np
import pandas as pd
import os
import json
import sys
from datetime import datetime

# Importamos tu fachada de datos
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

# --- PARÁMETROS ESPECÍFICOS H2 (Transición Corto Plazo) ---
HORIZON = 2
ALPHA_H2 = 0.2      # Un poco más de estabilidad que H1 (0.1)
PCA_COMPONENTS = 15 # Aumentamos ligeramente la capacidad de captura

def make_features_h2(df: pd.DataFrame):
    out = df.copy()
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"])
    
    # Lags: Para 2 días, necesitamos una ventana de memoria un poco más amplia (15 días)
    for k in range(1, 16):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # Volatilidad de 10 días (ventana media-corta)
    past = out["ret1"].shift(1)
    out["rv_10"] = past.rolling(10).std() 
    
    # RSI de 9 periodos (más estable que el de 5 del H1)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(9).mean()
    loss = delta.clip(upper=0).abs().rolling(9).mean()
    out["rsi_9"] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    
    return out

def run_predictor_h2(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")
    if raw is None or raw.empty:
        return None
    
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h2(df)
    
    # Definición de columnas para H2
    feature_cols = ["range", "rv_10", "rsi_9"] + [f"ret_lag_{k}" for k in range(1, 16)]
    target_col = "y_fwd"
    feat[target_col] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    
    clean = feat.dropna(subset=feature_cols + [target_col])
    if len(clean) < 120: return None

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(PCA_COMPONENTS, len(feature_cols)))),
        ("ridge", Ridge(alpha=ALPHA_H2))
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
        "alpha_used": ALPHA_H2,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker_input = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    res = run_predictor_h2(ticker_input)
    
    if res:
        filename = f"{ticker_input}_h{HORIZON}.json"
        full_path = os.path.join(DATA_OUTPUT_DIR, filename)
        
        with open(full_path, "w") as f:
            json.dump(res, f, indent=2)
            
        print(f"✅ H2 FINALIZADO: {ticker_input} -> {full_path}")

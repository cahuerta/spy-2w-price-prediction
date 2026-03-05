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

# --- PARÁMETROS ESPECÍFICOS H9 (Ciclo de Capital) ---
HORIZON = 9
ALPHA_H9 = 1.5      # Alta presión regulatoria para estabilidad estructural
PCA_COMPONENTS = 25 

def make_features_h9(df: pd.DataFrame):
    out = df.copy()
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"])
    
    # Lags: Mantenemos la ventana de 30 días para estabilidad
    for k in range(1, 31):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # Volatilidad mensual (20 días)
    past = out["ret1"].shift(1)
    out["rv_20"] = past.rolling(20).std() 
    
    # MFI Simplificado (Money Flow Index)
    # Detecta si el volumen está respaldando el movimiento de precio
    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3
    money_flow = typical_price * out["Volume"]
    
    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    
    mfr = pd.Series(positive_flow).rolling(14).sum() / pd.Series(negative_flow).rolling(14).sum().replace(0, np.nan)
    out["mfi_14"] = 100 - (100 / (1 + mfr.values))
    
    # Distancia a la Banda Superior/Inferior de Bollinger (Estandarizada)
    sma20 = out["Close"].rolling(20).mean()
    std20 = out["Close"].rolling(20).std()
    out["bb_dist"] = (out["Close"] - sma20) / (2 * std20).replace(0, np.nan)
    
    return out

def run_predictor_h9(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")
    if raw is None or raw.empty:
        return None
    
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h9(df)
    
    # Columnas para H9
    feature_cols = ["range", "rv_20", "mfi_14", "bb_dist"] + [f"ret_lag_{k}" for k in range(1, 31)]
    target_col = "y_fwd"
    feat[target_col] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    
    clean = feat.dropna(subset=feature_cols + [target_col])
    if len(clean) < 250: return None

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(PCA_COMPONENTS, len(feature_cols)))),
        ("ridge", Ridge(alpha=ALPHA_H9))
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
        "alpha_used": ALPHA_H9,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker_input = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    res = run_predictor_h9(ticker_input)
    
    if res:
        filename = f"{ticker_input}_h{HORIZON}.json"
        full_path = os.path.join(DATA_OUTPUT_DIR, filename)
        
        with open(full_path, "w") as f:
            json.dump(res, f, indent=2)
            
        print(f"✅ H9 FINALIZADO: {ticker_input} -> {full_path}")
      

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

# --- PARÁMETROS ESPECÍFICOS H8 (Fuerza de Tendencia) ---
HORIZON = 8
ALPHA_H8 = 1.25     # Regularización robusta para proyecciones de 2 semanas
PCA_COMPONENTS = 25 

def make_features_h8(df: pd.DataFrame):
    out = df.copy()
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"])
    
    # Lags: Mantenemos la ventana de 30 días (1.5 meses de trading)
    for k in range(1, 31):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # Volatilidad mensual (20 días)
    past = out["ret1"].shift(1)
    out["rv_20"] = past.rolling(20).std() 
    
    # ADX Simplificado (Fuerza de tendencia)
    # Compara el rango actual con el movimiento direccional
    up_move = out["High"].diff()
    down_move = out["Low"].diff().abs()
    out["plus_dm"] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    out["minus_dm"] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    out["adx_14"] = (out["plus_dm"].rolling(14).mean() / 
                     out["range"].rolling(14).mean().replace(0, np.nan))
    
    # Pendiente de la EMA 50 (Dirección macro)
    ema50 = out["Close"].ewm(span=50).mean()
    out["ema50_slope"] = (ema50 - ema50.shift(5)) / ema50.shift(5)
    
    return out

def run_predictor_h8(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")
    if raw is None or raw.empty:
        return None
    
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h8(df)
    
    # Columnas para H8
    feature_cols = ["range", "rv_20", "adx_14", "ema50_slope"] + [f"ret_lag_{k}" for k in range(1, 31)]
    target_col = "y_fwd"
    feat[target_col] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    
    clean = feat.dropna(subset=feature_cols + [target_col])
    if len(clean) < 240: return None

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(PCA_COMPONENTS, len(feature_cols)))),
        ("ridge", Ridge(alpha=ALPHA_H8))
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
        "alpha_used": ALPHA_H8,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker_input = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    res = run_predictor_h8(ticker_input)
    
    if res:
        filename = f"{ticker_input}_h{HORIZON}.json"
        full_path = os.path.join(DATA_OUTPUT_DIR, filename)
        
        with open(full_path, "w") as f:
            json.dump(res, f, indent=2)
            
        print(f"✅ H8 FINALIZADO: {ticker_input} -> {full_path}")

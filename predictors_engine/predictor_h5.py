import numpy as np
import pandas as pd
import os
import json
import sys
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURACIÓN H5 - PREDICTOR SEMANAL (PRODUCCIÓN)
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 5
ALPHA_H5 = 0.65
PCA_COMPONENTS = 12 # Reducido para mayor generalización

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)

def make_features_h5(df: pd.DataFrame):
    """H5 v2.1 - Producción: Estructura y Tendencia"""
    out = df.copy()
    
    # BASE
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.35)
    
    # LAGS CLAVE (Memoria de swing)
    for k in [1, 2, 3, 5, 10, 20]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)
        
    # VOLATILIDAD SEMANAL (Normalizada)
    past = out["ret1"].shift(1)
    out["rv_20"] = past.rolling(20).std().fillna(method='bfill') * np.sqrt(252)
    
    # TENDENCIA Y ESTRUCTURA
    out["ema_20_dist"] = (out["Close"] - out["Close"].ewm(span=20).mean()) / out["Close"]
    # Pendiente (Slope) de 20 días
    out["slope_20"] = out["Close"].rolling(20).apply(lambda x: np.polyfit(range(20), x, 1)[0], raw=True)
    
    # RSI 14 (Estándar semanal)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = 100 - (100 / (1 + (gain / loss.replace(0, 1.0))))
    
    return out

def run_predictor_h5(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")
    if raw is None or len(raw) < 200: return None
        
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h5(df)
    
    feature_cols = [
        "range", "rv_20", "rsi_14", "ema_20_dist", "slope_20",
        "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5", "ret_lag_10", "ret_lag_20"
    ]
    
    # TARGET
    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    
    clean = feat.dropna(subset=feature_cols + ["y_fwd"])
    if len(clean) < 160: return None

    # PIPELINE
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(len(feature_cols), PCA_COMPONENTS))),
        ("ridge", Ridge(alpha=ALPHA_H5, random_state=42))
    ])
    
    model.fit(clean[feature_cols], clean["y_fwd"])
    
    # PREDICCIÓN
    last_feat = feat[feature_cols].iloc[-1:].fillna(method='ffill').fillna(method='bfill')
    y_pred_log = model.predict(last_feat)[0]
    
    # CLIP DE SEGURIDAD H5 (Máximo 15% semanal)
    y_pred_log = np.clip(y_pred_log, -0.15, 0.15)
    
    price_today = float(feat["Close"].iloc[-1])
    price_5d = price_today * np.exp(y_pred_log)
    
    # MÉTRICAS
    ridge_model = model.named_steps["ridge"]
    confidence = 1 / (1 + np.std(ridge_model.coef_))
    
    return {
        "ticker": ticker,
        "predictor": "H5_v2.1",
        "horizon_days": HORIZON,
        "price_today": round(price_today, 4),
        "price_target_5d": round(price_5d, 4),
        "return_5d_pct": round(y_pred_log * 100, 4),
        "confidence": round(confidence, 3),
        "timestamp": datetime.now().isoformat()
    }

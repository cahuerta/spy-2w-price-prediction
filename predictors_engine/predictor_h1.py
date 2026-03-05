import numpy as np
import pandas as pd
import os
import json
import sys
from datetime import datetime
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURACIÓN H1 - PREDICTOR DÍA 1 (PRODUCCIÓN)
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 1
ALPHA_H1 = 0.08
MAX_PCA = 8 # Límite superior deseado

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)

def make_features_h1(df: pd.DataFrame):
    """H1 PRODUCTION - Features validadas para Día 1"""
    out = df.copy()
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.25)
    out["ret_lag_1"] = out["ret1"].shift(1)
    out["ret_lag_2"] = out["ret1"].shift(2)
    out["ret_lag_5"] = out["ret1"].shift(5)
    
    past_returns = out["ret1"].shift(1)
    out["rv_short"] = past_returns.rolling(3).std().fillna(method='bfill')
    out["vol_ema"] = past_returns.ewm(span=5).std().fillna(method='bfill')
    out["mom_3d"] = past_returns.rolling(3).sum()
    out["mom_vol"] = out["mom_3d"] / out["rv_short"].replace(0, 0.01)
    
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(3).mean()
    loss = (-delta.clip(upper=0)).rolling(3).mean()
    rs = gain / loss.replace(0, 1.0)
    out["rsi_fast"] = 100 - (100 / (1 + rs))
    out["price_strength"] = out["ret1"] / out["rv_short"].replace(0, 0.01)
    out["trend_5"] = out["Close"].pct_change(5).fillna(0)
    
    return out

def run_predictor_h1(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")
    if raw is None or len(raw) < 150: return None
        
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(df) < 140: return None
        
    feat = make_features_h1(df)
    feature_cols = [
        "range", "ret_lag_1", "ret_lag_2", "ret_lag_5",
        "rv_short", "vol_ema", "mom_vol", "rsi_fast",
        "price_strength", "trend_5"
    ]
    
    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    clean = feat.dropna(subset=feature_cols + ["y_fwd"])
    if len(clean) < 130: return None
    
    # ======================================================
    # PCA DINÁMICO (Lógica de seguridad)
    # ======================================================
    n_samples = len(clean)
    n_features = len(feature_cols)
    n_pca_dynamic = min(MAX_PCA, n_features, n_samples - 1)
    n_pca_dynamic = max(1, n_pca_dynamic)
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca_dynamic, random_state=42)),
        ("ridge", Ridge(alpha=ALPHA_H1, random_state=42))
    ])
    
    model.fit(clean[feature_cols], clean["y_fwd"])
    
    last_features = feat[feature_cols].iloc[-1:].fillna(method='ffill').fillna(method='bfill')
    y_pred_log = model.predict(last_features)[0]
    
    price_today = float(feat["Close"].iloc[-1])
    price_tomorrow = price_today * np.exp(y_pred_log)
    ridge_model = model.named_steps["ridge"]
    confidence = 1 / (1 + np.std(ridge_model.coef_))
    r2_train = ridge_model.score(clean[feature_cols], clean["y_fwd"])
    
    return {
        "ticker": ticker,
        "predictor": "H1_v2.0_dynamic",
        "horizon_days": HORIZON,
        "date_today": datetime.now().strftime("%Y-%m-%d"),
        "price_today": round(price_today, 4),
        "price_tomorrow": round(price_tomorrow, 4),
        "return_pct": round(y_pred_log * 100, 4),
        "confidence": round(confidence, 3),
        "r2_train": round(r2_train, 4),
        "samples": n_samples,
        "model": {
            "alpha": ALPHA_H1,
            "pca_components": n_pca_dynamic, 
            "features_count": n_features
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    print(f"🚀 H1 PRODUCTION v2.0 (Dynamic PCA) → {ticker}")
    result = run_predictor_h1(ticker)
    
    if result:
        filename = f"{ticker}_H1_dia1_v2.json"
        path = os.path.join(DATA_OUTPUT_DIR, filename)
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n✅ H1 v2.0 - Finalizado con {result['model']['pca_components']} componentes.")
    else:
        print("❌ Datos insuficientes")
        

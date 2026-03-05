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
# CONFIGURACIÓN H3 - PREDICTOR 72 HORAS (PRODUCCIÓN ✓)
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 3
ALPHA_H3 = 0.35
PCA_COMPONENTS = 10

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)

def make_features_h3(df: pd.DataFrame):
    """H3 v2.1 - PRODUCTION READY (Correcciones finales)"""
    out = df.copy()
    
    # BASE SÓLIDA
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.25)
    
    # LAGS SELECTIVOS H3
    for k in [1, 2, 3, 5, 8]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)
    
    # VOLATILIDAD H3 - CORREGIDO
    past = out["ret1"].shift(1)
    out["rv_15"] = past.rolling(15).std().fillna(method='bfill').clip(1e-6) * np.sqrt(252/15)
    
    # MOMENTUM H3
    out["mom_5d"] = past.rolling(5).sum()
    out["mom_10d"] = past.rolling(10).sum()
    out["mom_ratio"] = (out["mom_10d"] / out["rv_15"].replace(0, 0.01)).clip(-6, 6)
    
    # RSI DUAL H3
    delta = out["Close"].diff()
    for period in [7, 14]:
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1.0)
        out[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    
    # MEAN REVERSION H3
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["dist_ma20"] = np.log(out["Close"] / out["ma_20"]).clip(-0.3, 0.3)
    
    # VOLUMEN Z-SCORE H3
    vol_mean = out["Volume"].rolling(20).mean()
    vol_std = out["Volume"].rolling(20).std().replace(0, 1)
    out["vol_z"] = ((out["Volume"] - vol_mean) / vol_std).clip(-3, 3)
    
    # TENDENCIA ESTRUCTURAL
    out["trend_20"] = out["Close"].pct_change(20).fillna(0)
    
    return out

def run_predictor_h3(ticker: str):
    """H3 v2.1 - Swing Trading 72h PRODUCTION"""
    
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")
    if raw is None or len(raw) < 180:
        return None
        
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(df) < 170:
        return None
        
    feat = make_features_h3(df)
    
    feature_cols = [
        "range", "rv_15", "mom_ratio", "dist_ma20", "vol_z",
        "mom_5d", "mom_10d", "rsi_7", "rsi_14",
        "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5", "ret_lag_8", "trend_20"
    ]
    
    # TARGET 72h
    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    
    # LIMPIEZA ESTRICTA
    clean = feat.dropna(subset=feature_cols + ["y_fwd"])
    if len(clean) < 150:
        return None
    
    # PIPELINE H3 OPTIMIZADO
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("ridge", Ridge(alpha=ALPHA_H3, random_state=42))
    ])
    
    model.fit(clean[feature_cols], clean["y_fwd"])
    
    # PREDICCIÓN ROBUSTA 72H
    last_features = feat[feature_cols].iloc[-1:].fillna(method='ffill').fillna(method='bfill')
    y_pred_log = model.predict(last_features)[0]
    
    # CLIP CONSERVADOR H3 (±7% realista para 72h)
    y_pred_log = np.clip(y_pred_log, -0.07, 0.07)
    
    price_today = float(feat["Close"].iloc[-1])
    price_72h = price_today * np.exp(y_pred_log)
    
    # MÉTRICAS COMPLETAS
    ridge_model = model.named_steps["ridge"]
    confidence = 1 / (1 + np.std(ridge_model.coef_))
    r2_train = ridge_model.score(clean[feature_cols], clean["y_fwd"])
    
    return {
        "ticker": ticker,
        "predictor": "H3_v2.1",  # Versión final
        "horizon_days": HORIZON,
        "date_today": datetime.now().strftime("%Y-%m-%d"),
        "price_today": round(price_today, 4),
        "price_72h": round(price_72h, 4),
        "return_72h_pct": round(y_pred_log * 100, 4),
        "confidence": round(confidence, 3),
        "r2_train": round(r2_train, 4),
        "samples": len(clean),
        "model": {
            "alpha": ALPHA_H3,
            "pca_components": PCA_COMPONENTS,
            "features": len(feature_cols)
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    
    print(f"🚀 H3 v2.1 Swing Predictor 72h → {ticker}")
    result = run_predictor_h3(ticker)
    
    if result:
        filename = f"{ticker}_H3_72h_v2.json"
        path = os.path.join(DATA_OUTPUT_DIR, filename)
        
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        print("
✅ H3 v2.1 - SWING 72H PRODUCTION")
        print(f"📊 {ticker}:           ${result['price_today']:>8.2f}")
        print(f"📊 Predicción 72h:     ${result['price_72h']:>8.2f}")
        print(f"📈 Retorno esperado:   {result['return_72h_pct']:>6.2f}%")
        print(f"🔥 Confianza:          {result['confidence']:>5.1f}")
        print(f"📈 R² entrenamiento:   {result['r2_train']:>6.3f}")
        print(f"📁 {path}")
    else:
        print("❌ Datos insuficientes para H3")

import numpy as np
import pandas as pd
import os
import json
import sys
from datetime import datetime

# Configuración de rutas
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURACIÓN H3 v2.2
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 3
ALPHA_H3 = 0.35  # Mayor regularización para mayor horizonte
MAX_PCA_COMPONENTS = 10

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)

# ======================================================
# FEATURE ENGINEERING
# ======================================================
def make_features_h3(df: pd.DataFrame):
    out = df.copy()

    # BASE
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.25)

    # LAGS
    for k in [1, 2, 3, 5, 8]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # VOLATILIDAD ANUALIZADA (15 días)
    past = out["ret1"].shift(1)
    out["rv_15"] = past.rolling(15).std().clip(1e-6) * np.sqrt(252) 

    # MOMENTUM
    out["mom_5d"] = past.rolling(5).sum()
    out["mom_10d"] = past.rolling(10).sum()
    out["mom_ratio"] = (out["mom_10d"] / out["rv_15"].replace(0, 0.01)).clip(-6, 6)

    # RSI DUAL (7, 14)
    delta = out["Close"].diff()
    for p in [7, 14]:
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / loss.replace(0, 1.0)
        out[f"rsi_{p}"] = 100 - (100 / (1 + rs))

    # MEAN REVERSION & VOLUMEN
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["dist_ma20"] = np.log(out["Close"] / out["ma_20"]).clip(-0.3, 0.3)
    
    vol_rolling = out["Volume"].rolling(20)
    out["vol_z"] = ((out["Volume"] - vol_rolling.mean()) / vol_rolling.std().replace(0, 1)).clip(-3, 3)

    # TENDENCIA
    out["trend_20"] = out["Close"].pct_change(20).fillna(0)

    return out

# ======================================================
# PREDICTOR H3
# ======================================================
def run_predictor_h3(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    if raw is None or len(raw) < 180:
        return None

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h3(df)

    feature_cols = [
        "range", "rv_15", "mom_ratio", "dist_ma20", "vol_z",
        "mom_5d", "mom_10d", "rsi_7", "rsi_14",
        "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5", "ret_lag_8", "trend_20"
    ]

    # TARGET: Log-return a T+3
    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])

    clean = feat.dropna(subset=feature_cols + ["y_fwd"])
    if len(clean) < 150:
        return None

    X_train = clean[feature_cols].values
    y_train = clean["y_fwd"].values

    # PCA Dinámico
    n_samples, n_features = X_train.shape
    dynamic_pca = max(1, min(MAX_PCA_COMPONENTS, n_features, n_samples - 1))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=dynamic_pca, random_state=42)),
        ("ridge", Ridge(alpha=ALPHA_H3))
    ])

    model.fit(X_train, y_train)

    # PREDICCIÓN
    last_features = feat[feature_cols].iloc[-1:]
    if last_features.isna().any().any():
        return None

    y_pred_log = float(model.predict(last_features)[0])
    y_pred_log = np.clip(y_pred_log, -0.07, 0.07) # Clip 72h

    price_today = float(feat["Close"].iloc[-1])
    price_72h = price_today * np.exp(y_pred_log)

    # MÉTRICAS
    r2_train = float(model.score(X_train, y_train))
    ridge_coefs = model.named_steps["ridge"].coef_
    confidence = 1 / (1 + np.std(ridge_coefs))

    return {
        "ticker": ticker,
        "predictor": "H3",
        "horizon_days": HORIZON,
        "date_today": datetime.now().strftime("%Y-%m-%d"),
        "price_today": round(price_today, 4),
        "price_pred": round(price_72h, 4),
        "return_pct": round(y_pred_log * 100, 4),
        "confidence": round(float(confidence), 3),
        "r2_train": round(r2_train, 4),
        "samples": len(clean),
        "model_info": {
            "alpha": ALPHA_H3,
            "pca_components": dynamic_pca
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    print(f"🚀 H3 Predictor 72h (v2.2) → {ticker}")

    result = run_predictor_h3(ticker)

    if result:
        filename = f"{ticker}_H3_72h_v2.json"
        path = os.path.join(DATA_OUTPUT_DIR, filename)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n✅ H3 PROCESADO CORRECTAMENTE")
        print(f"📊 {ticker}: ${result['price_today']:>8.2f} → Pred. 72h: ${result['price_72h']:>8.2f}")
        print(f"📈 Retorno: {result['return_72h_pct']:>6.2f}% | Confianza: {result['confidence']:>4.2f}")
        print(f"📁 Destino: {path}")
    else:
        print("❌ Error: Datos insuficientes.")

import numpy as np
import pandas as pd
import os
import json
import sys
from datetime import datetime

# ======================================================
# RUTAS
# ======================================================

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURACIÓN
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 7
ALPHA_H7 = 1.0
MAX_PCA_COMPONENTS = 12

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)

# ======================================================
# HURST EXPONENT (ROBUSTO)
# ======================================================

def get_hurst_exponent(ts, max_lag=20):

    ts = np.array(ts)

    lags = range(2, min(max_lag, len(ts)//4))

    if len(lags) < 2:
        return 0.5

    tau = [
        np.sqrt(np.std(ts[lag:] - ts[:-lag]) + 1e-9)
        for lag in lags
    ]

    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    return poly[0] * 2.0


# ======================================================
# FEATURE ENGINEERING
# ======================================================

def make_features_h7(df: pd.DataFrame):

    out = df.copy()

    # BASE
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0,0.35)

    # LAGS
    for k in [1,2,3,5,10,20,30]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    past = out["ret1"].shift(1)

    # VOLATILIDAD
    out["rv_30"] = past.rolling(30).std().clip(1e-6) * np.sqrt(252)

    # HURST
    out["hurst_60"] = (
        out["ret1"]
        .rolling(60)
        .apply(get_hurst_exponent, raw=True)
        .fillna(0.5)
        .clip(0.2,0.9)
    )

    # SMA50 DISTANCE
    close_past = out["Close"].shift(1)

    sma50 = close_past.rolling(50).mean()

    out["dist_sma50"] = np.log(close_past / sma50).clip(-0.3,0.3)

    # MONEY FLOW (ESTABLE)
    vol_log = np.log(out["Volume"].shift(1) + 1)

    out["money_flow"] = (vol_log * past).clip(-5,5)

    return out


# ======================================================
# PREDICTOR H7
# ======================================================

def run_predictor_h7(ticker: str):

    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    if raw is None or len(raw) < 260:
        return None

    df = raw[["Open","High","Low","Close","Volume"]].dropna()

    feat = make_features_h7(df)

    feature_cols = [
        "range",
        "rv_30",
        "hurst_60",
        "dist_sma50",
        "money_flow",
        "ret_lag_1",
        "ret_lag_3",
        "ret_lag_5",
        "ret_lag_10",
        "ret_lag_20",
        "ret_lag_30"
    ]

    # TARGET
    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])

    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 200:
        return None

    X = clean[feature_cols].values
    y = clean["y_fwd"].values

    # ======================================================
    # PCA DINÁMICO
    # ======================================================

    n_samples, n_features = X.shape

    dynamic_pca = max(
        1,
        min(MAX_PCA_COMPONENTS, n_features, n_samples-1)
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=dynamic_pca, random_state=42)),
        ("ridge", Ridge(alpha=ALPHA_H7))
    ])

    model.fit(X,y)

    # ======================================================
    # PREDICCIÓN
    # ======================================================

    last_features = feat[feature_cols].iloc[-1:]

    if last_features.isna().any().any():
        return None

    y_pred_log = float(model.predict(last_features)[0])

    y_pred_log = np.clip(y_pred_log, -0.15, 0.15)

    price_today = float(feat["Close"].iloc[-1])

    price_7d = price_today * np.exp(y_pred_log)

    # ======================================================
    # MÉTRICAS
    # ======================================================

    hurst_val = float(feat["hurst_60"].iloc[-1])

    regime = (
        "PERSISTENT"
        if hurst_val > 0.55
        else "MEAN-REV"
        if hurst_val < 0.45
        else "RANDOM"
    )

    ridge_model = model.named_steps["ridge"]

    confidence = 1 / (1 + np.std(ridge_model.coef_))

    return {

        "ticker": ticker,
        "predictor": "H7_v2.3_Persistence",

        "horizon_days": HORIZON,

        "price_today": round(price_today,4),
        "price_7d": round(price_7d,4),

        "return_7d_pct": round(y_pred_log*100,4),

        "hurst": round(hurst_val,3),
        "regime": regime,

        "confidence": round(float(confidence),3),

        "samples": len(clean),

        "model": {
            "alpha": ALPHA_H7,
            "pca_components": dynamic_pca
        },

        "timestamp": datetime.now().isoformat()
    }


# ======================================================
# EJECUCIÓN
# ======================================================

if __name__ == "__main__":

    ticker = sys.argv[1].upper() if len(sys.argv)>1 else "SPY"

    print(f"🚀 Iniciando H7 Persistencia Semanal → {ticker}")

    result = run_predictor_h7(ticker)

    if result:

        filename = f"{ticker}_H7_7d_v2.json"

        path = os.path.join(DATA_OUTPUT_DIR, filename)

        with open(path,"w") as f:
            json.dump(result,f,indent=2)

        print("\n✅ H7 ANALIZADO")
        print("----------------------------------------")
        print(f"📊 {ticker}: ${result['price_today']:,.2f}")
        print(f"🎯 7d target: ${result['price_7d']:,.2f}")
        print(f"📈 Retorno: {result['return_7d_pct']}%")
        print(f"📊 Hurst: {result['hurst']} ({result['regime']})")
        print(f"🔥 Confianza: {result['confidence']}")
        print("----------------------------------------")

    else:
        print("❌ Datos insuficientes.")

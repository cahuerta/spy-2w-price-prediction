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
# CONFIGURACIÓN H5
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 5
ALPHA_H5 = 0.65
MAX_PCA_COMPONENTS = 12

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)


# ======================================================
# FEATURE ENGINEERING
# ======================================================

def make_features_h5(df: pd.DataFrame):

    out = df.copy()

    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0,0.35)

    # LAGS
    for k in [1,2,3,5,10,20]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    past = out["ret1"].shift(1)

    # VOLATILIDAD
    out["rv_20"] = past.rolling(20).std().clip(1e-6) * np.sqrt(252)

    # EMA DISTANCE (SIN LEAKAGE)
    close_past = out["Close"].shift(1)
    ema20 = close_past.ewm(span=20).mean()

    out["ema_20_dist"] = ((close_past - ema20) / ema20).clip(-0.3,0.3)

    # SLOPE
    def calc_slope(x):
        if len(x) < 20:
            return 0.0
        coeff = np.polyfit(range(20), x, 1)[0]
        return coeff / x.mean()

    out["slope_20"] = close_past.rolling(20).apply(calc_slope, raw=True)

    # RSI
    delta = out["Close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0,1)

    out["rsi_14"] = 100 - (100/(1+rs))

    return out


# ======================================================
# PREDICTOR
# ======================================================

def run_predictor_h5(ticker:str):

    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    if raw is None or len(raw) < 200:
        return None

    df = raw[["Open","High","Low","Close","Volume"]].dropna()

    feat = make_features_h5(df)

    feature_cols = [
        "range",
        "rv_20",
        "rsi_14",
        "ema_20_dist",
        "slope_20",
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "ret_lag_5",
        "ret_lag_10",
        "ret_lag_20"
    ]

    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON)/feat["Close"])

    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 160:
        return None

    X = clean[feature_cols].values
    y = clean["y_fwd"].values

    # PCA DINÁMICO
    n_samples, n_features = X.shape

    dynamic_pca = max(1, min(MAX_PCA_COMPONENTS, n_features, n_samples-1))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=dynamic_pca, random_state=42)),
        ("ridge", Ridge(alpha=ALPHA_H5))
    ])

    model.fit(X,y)

    last_feat = feat[feature_cols].iloc[-1:]

    if last_feat.isna().any().any():
        return None

    y_pred_log = float(model.predict(last_feat)[0])

    y_pred_log = np.clip(y_pred_log, -0.12, 0.12)

    price_today = float(feat["Close"].iloc[-1])
    price_5d = price_today * np.exp(y_pred_log)

    ridge_model = model.named_steps["ridge"]

    confidence = 1/(1 + np.std(ridge_model.coef_))

    return {

        "ticker": ticker,
        "predictor": "H5",

        "horizon_days": HORIZON,

        "price_today": round(price_today,4),
        "price_pred": round(price_5d,4),

        "return_pct": round(y_pred_log*100,4),

        "confidence": round(float(confidence),3),

        "samples": len(clean),

        "model": {
            "alpha": ALPHA_H5,
            "pca_components": dynamic_pca
        },

        "timestamp": datetime.now().isoformat()
    }


# ======================================================
# EXECUCIÓN
# ======================================================

if __name__ == "__main__":

    ticker = sys.argv[1].upper() if len(sys.argv)>1 else "SPY"

    print(f"🚀 Iniciando H5 Semanal → {ticker}")

    result = run_predictor_h5(ticker)

    if result:

        filename = f"{ticker}_H5_5d_v2.json"
        path = os.path.join(DATA_OUTPUT_DIR, filename)

        with open(path,"w") as f:
            json.dump(result,f,indent=2)

        print("\n✅ H5 SEMANAL COMPLETADO")
        print("-----------------------------------")
        print(f"📊 {ticker}:      ${result['price_today']:,.2f}")
        print(f"🎯 Target 5d:     ${result['price_target_5d']:,.2f}")
        print(f"📈 Retorno Est:   {result['return_5d_pct']}%")
        print(f"🔥 Confianza:     {result['confidence']}")
        print("-----------------------------------")

    else:
        print("❌ Datos insuficientes para H5.")

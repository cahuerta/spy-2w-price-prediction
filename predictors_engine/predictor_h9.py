import numpy as np
import pandas as pd
import os
import json
import sys
from datetime import datetime

# ======================================================
# CONFIGURACIÓN DE ENTORNO
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
# CONFIGURACIÓN H9
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 9
ALPHA_H9 = 1.5
MAX_PCA_COMPONENTS = 12

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)


# ======================================================
# FEATURE ENGINEERING H9
# ======================================================

def make_features_h9(df: pd.DataFrame):

    out = df.copy()

    # BASE
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0,0.40)

    # LAGS
    for k in [1,3,5,10,20,30]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # ======================================================
    # MONEY FLOW INDEX (SIN LEAKAGE)
    # ======================================================

    tp = (out["High"] + out["Low"] + out["Close"]) / 3
    rmf = tp * out["Volume"]

    tp_shift = tp.shift(1)
    rmf_shift = rmf.shift(1)

    pos_flow = rmf_shift.where(tp_shift > tp_shift.shift(1),0)
    neg_flow = rmf_shift.where(tp_shift < tp_shift.shift(1),0)

    pos_sum = pos_flow.rolling(14).sum()
    neg_sum = neg_flow.rolling(14).sum()

    m_ratio = pos_sum / (neg_sum + 1e-9)

    out["mfi_14"] = (100 - (100/(1+m_ratio))).clip(5,95)

    # ======================================================
    # BOLLINGER DISTANCE
    # ======================================================

    close_past = out["Close"].shift(1)

    sma20 = close_past.rolling(20).mean()
    std20 = close_past.rolling(20).std().clip(1e-6)

    out["bb_dist"] = ((close_past - sma20) / (2*std20)).clip(-3,3)

    # ======================================================
    # CAPITAL FLOW PROXY
    # ======================================================

    out["capital_flow"] = ((out["mfi_14"] - 50)/50).clip(-1,1)

    return out


# ======================================================
# PREDICTOR H9
# ======================================================

def run_predictor_h9(ticker:str):

    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    if raw is None or len(raw) < 300:
        return None

    df = raw[["Open","High","Low","Close","Volume"]].dropna()

    feat = make_features_h9(df)

    feature_cols = [
        "range",
        "mfi_14",
        "bb_dist",
        "capital_flow",
        "ret_lag_1",
        "ret_lag_5",
        "ret_lag_10",
        "ret_lag_20",
        "ret_lag_30"
    ]

    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])

    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 250:
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
        ("ridge", Ridge(alpha=ALPHA_H9))
    ])

    model.fit(X,y)

    # ======================================================
    # PREDICCIÓN
    # ======================================================

    last_feat = feat[feature_cols].iloc[-1:]

    if last_feat.isna().any().any():
        return None

    y_pred_log = float(model.predict(last_feat)[0])

    y_pred_log = np.clip(y_pred_log,-0.18,0.18)

    price_today = float(feat["Close"].iloc[-1])
    price_9d = price_today * np.exp(y_pred_log)

    # ======================================================
    # MÉTRICAS
    # ======================================================

    ridge_model = model.named_steps["ridge"]

    confidence = 1/(1 + np.std(ridge_model.coef_))

    return {

        "ticker": ticker,
        "predictor": "H9_v2.3_CapitalCycle",

        "horizon_days": HORIZON,

        "price_today": round(price_today,4),
        "price_9d": round(price_9d,4),

        "return_9d_pct": round(y_pred_log*100,4),

        "mfi_14": round(float(feat["mfi_14"].iloc[-1]),1),
        "bb_position": round(float(feat["bb_dist"].iloc[-1]),2),

        "confidence": round(float(confidence),3),

        "samples": len(clean),

        "model":{
            "alpha": ALPHA_H9,
            "pca_components": dynamic_pca
        },

        "timestamp": datetime.now().isoformat()
    }


# ======================================================
# EJECUCIÓN
# ======================================================

if __name__ == "__main__":

    ticker = sys.argv[1].upper() if len(sys.argv)>1 else "SPY"

    print(f"🚀 Ejecutando H9 Capital Cycle (9d) → {ticker}")

    result = run_predictor_h9(ticker)

    if result:

        filename = f"{ticker}_H9_9d_v2.json"

        path = os.path.join(DATA_OUTPUT_DIR, filename)

        with open(path,"w") as f:
            json.dump(result,f,indent=2)

        print("\n✅ H9 COMPLETADO")
        print("----------------------------------------")
        print(f"📊 {ticker}: ${result['price_today']:,.2f}")
        print(f"🎯 9d target: ${result['price_9d']:,.2f}")
        print(f"📈 Retorno Est: {result['return_9d_pct']}%")
        print(f"💰 MFI: {result['mfi_14']} | BB Dist: {result['bb_position']}")
        print(f"🔥 Confianza: {result['confidence']}")
        print("----------------------------------------")

    else:
        print("❌ Datos insuficientes.")

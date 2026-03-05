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
# CONFIGURACIÓN H5 - PREDICTOR SEMANAL
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 5
ALPHA_H5 = 0.65
PCA_COMPONENTS = 12

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)


# ======================================================
# FEATURES
# ======================================================

def make_features_h5(df: pd.DataFrame):

    out = df.copy()

    # BASE
    out["ret1"] = np.log(out["Close"]).diff()

    out["range"] = np.log(
        out["High"] / out["Low"]
    ).clip(0, 0.35)

    # LAGS
    for k in [1, 2, 3, 5, 10, 20]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    past = out["ret1"].shift(1)

    # VOLATILIDAD 20d
    out["rv_20"] = (
        past.rolling(20)
        .std()
        .clip(1e-6)
    ) * np.sqrt(252/20)

    # EMA DISTANCE
    ema20 = out["Close"].ewm(span=20).mean()

    out["ema_20_dist"] = (
        (out["Close"] - ema20) /
        ema20
    ).clip(-0.3, 0.3)

    # SLOPE NORMALIZADO
    def calc_slope(x):

        if len(x) < 20:
            return 0.0

        coeff = np.polyfit(range(20), x[-20:], 1)[0]

        return coeff / x.mean()

    out["slope_20"] = (
        out["Close"]
        .rolling(20)
        .apply(calc_slope, raw=True)
    )

    # RSI
    delta = out["Close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, 1)

    out["rsi_14"] = 100 - (100 / (1 + rs))

    return out


# ======================================================
# PREDICTOR
# ======================================================

def run_predictor_h5(ticker: str):

    raw = get_price_history(
        ticker=ticker,
        period="2y",
        interval="1d"
    )

    if raw is None or len(raw) < 200:
        return None

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

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

    # TARGET
    feat["y_fwd"] = np.log(
        feat["Close"].shift(-HORIZON) /
        feat["Close"]
    )

    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 160:
        return None

    # PIPELINE
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(len(feature_cols), PCA_COMPONENTS))),
        ("ridge", Ridge(alpha=ALPHA_H5, random_state=42))
    ])

    model.fit(clean[feature_cols], clean["y_fwd"])

    # PREDICCIÓN
    last_feat = (
        feat[feature_cols]
        .iloc[-1:]
        .ffill()
        .bfill()
    )

    y_pred_log = model.predict(last_feat)[0]

    # CLIP REALISTA
    y_pred_log = np.clip(y_pred_log, -0.12, 0.12)

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


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"

    print(f"🚀 H5 Predictor Semanal → {ticker}")

    result = run_predictor_h5(ticker)

    if result:

        filename = f"{ticker}_H5_5d_v2.json"

        path = os.path.join(DATA_OUTPUT_DIR, filename)

        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print("\n✅ H5 v2.1 - SEMANAL")
        print(f"📊 {ticker}:      ${result['price_today']:>8.2f}")
        print(f"📊 Target 5d:     ${result['price_target_5d']:>8.2f}")
        print(f"📈 Retorno:       {result['return_5d_pct']:>6.2f}%")
        print(f"🔥 Confianza:     {result['confidence']:>5.1f}")
        print(f"📁 {path}")

    else:

        print("❌ Datos insuficientes H5")

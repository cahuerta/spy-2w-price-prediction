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
# CONFIGURACIÓN H6 - TENDENCIA MEDIO PLAZO
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 6
ALPHA_H6 = 0.8
PCA_COMPONENTS = 12

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)


# ======================================================
# FEATURE ENGINEERING H6
# ======================================================

def make_features_h6(df: pd.DataFrame):

    out = df.copy()

    # BASE
    out["ret1"] = np.log(out["Close"]).diff()

    out["range"] = np.log(
        out["High"] / out["Low"]
    ).clip(0, 0.35)

    # LAGS ESTRUCTURALES
    for k in [1, 2, 3, 5, 10, 20]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # VOLATILIDAD 25d
    past = out["ret1"].shift(1)

    out["rv_25"] = (
        past.rolling(25)
        .std()
        .clip(1e-6)
    ) * np.sqrt(252/25)

    # RSI
    delta = out["Close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rsi = 100 - (100 / (1 + (gain / loss.replace(0, 1))))

    # RSI Z SCORE
    out["rsi_z"] = (
        (rsi - rsi.rolling(60).mean()) /
        rsi.rolling(60).std().replace(0, 1)
    )

    out["rsi_z"] = out["rsi_z"].clip(-3, 3)

    # EMA SLOPE NORMALIZADO
    ema_20 = out["Close"].ewm(span=20).mean()

    out["ema_slope"] = (
        (ema_20 - ema_20.shift(5)) /
        ema_20.shift(5)
    )

    out["ema_slope"] = out["ema_slope"].clip(-0.05, 0.05)

    # TENDENCIA LINEAL 25d
    def calc_slope(x):

        if len(x) < 25:
            return 0.0

        coeffs = np.polyfit(range(25), x[-25:], 1)

        return coeffs[0]

    out["slope_25"] = (
        out["Close"]
        .rolling(25)
        .apply(calc_slope, raw=True)
        .fillna(0)
    )

    return out


# ======================================================
# PREDICTOR H6
# ======================================================

def run_predictor_h6(ticker: str):

    raw = get_price_history(
        ticker=ticker,
        period="2y",
        interval="1d"
    )

    if raw is None or len(raw) < 240:
        return None

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if len(df) < 230:
        return None

    feat = make_features_h6(df)

    feature_cols = [
        "range",
        "rv_25",
        "rsi_z",
        "ema_slope",
        "slope_25",
        "ret_lag_1",
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

    if len(clean) < 190:
        return None

    # PIPELINE
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("ridge", Ridge(alpha=ALPHA_H6, random_state=42))
    ])

    model.fit(clean[feature_cols], clean["y_fwd"])

    # PREDICCIÓN
    last_features = (
        feat[feature_cols]
        .iloc[-1:]
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    y_pred_log = model.predict(last_features)[0]

    # CLIP REALISTA
    y_pred_log = np.clip(y_pred_log, -0.12, 0.12)

    price_today = float(feat["Close"].iloc[-1])

    price_6d = price_today * np.exp(y_pred_log)

    # MÉTRICAS
    ridge_model = model.named_steps["ridge"]

    confidence = 1 / (1 + np.std(ridge_model.coef_))

    r2_train = ridge_model.score(
        clean[feature_cols],
        clean["y_fwd"]
    )

    return {
        "ticker": ticker,
        "predictor": "H6_v2.1",
        "horizon_days": HORIZON,
        "date_today": datetime.now().strftime("%Y-%m-%d"),
        "price_today": round(price_today, 4),
        "price_6d": round(price_6d, 4),
        "return_6d_pct": round(y_pred_log * 100, 4),
        "confidence": round(confidence, 3),
        "r2_train": round(r2_train, 4),
        "samples": len(clean),
        "model": {
            "alpha": ALPHA_H6,
            "pca_components": PCA_COMPONENTS,
            "features": len(feature_cols)
        },
        "timestamp": datetime.now().isoformat()
    }


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"

    print(f"🚀 H6 v2.1 Medio Plazo (6d) → {ticker}")

    result = run_predictor_h6(ticker)

    if result:

        filename = f"{ticker}_H6_6d_v2.json"

        path = os.path.join(DATA_OUTPUT_DIR, filename)

        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print("\n✅ H6 v2.1 - MEDIO PLAZO")

        print(f"📊 {ticker}:        ${result['price_today']:>8.2f}")
        print(f"📊 Predicción 6d:   ${result['price_6d']:>8.2f}")
        print(f"📈 Retorno:         {result['return_6d_pct']:>6.2f}%")
        print(f"🔥 Confianza:       {result['confidence']:>5.1f}")
        print(f"📁 {path}")

    else:

        print("❌ Datos insuficientes H6")

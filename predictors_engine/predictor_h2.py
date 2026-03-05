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
# CONFIGURACIÓN H2 - PREDICTOR 48 HORAS (PRODUCCIÓN)
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 2
ALPHA_H2 = 0.15
PCA_COMPONENTS = 10

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)


# ======================================================
# FEATURE ENGINEERING H2
# ======================================================

def make_features_h2(df: pd.DataFrame):

    out = df.copy()

    # BASE
    out["ret1"] = np.log(out["Close"]).diff()

    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.25)

    # LAGS CLAVE
    for k in [1, 2, 3, 5, 10]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # VOLATILIDAD
    past = out["ret1"].shift(1)

    out["rv_10"] = past.rolling(10).std().clip(1e-6)

    # MOMENTUM
    out["mom_2d"] = past.rolling(2).sum()
    out["mom_5d"] = past.rolling(5).sum()

    out["mom_ratio"] = (
        out["mom_5d"] /
        out["rv_10"].replace(0, 0.01)
    ).clip(-5, 5)

    # RSI
    delta = out["Close"].diff()

    gain = delta.clip(lower=0).rolling(9).mean()
    loss = (-delta.clip(upper=0)).rolling(9).mean()

    rs = gain / loss.replace(0, 1.0)

    out["rsi_9"] = 100 - (100 / (1 + rs))

    # MEAN REVERSION
    out["ma_10"] = out["Close"].rolling(10).mean()

    out["dist_ma10"] = np.log(
        out["Close"] / out["ma_10"]
    ).clip(-0.2, 0.2)

    # TENDENCIA
    out["trend_10"] = out["Close"].pct_change(10).fillna(0)

    return out


# ======================================================
# PREDICTOR H2
# ======================================================

def run_predictor_h2(ticker: str):

    raw = get_price_history(
        ticker=ticker,
        period="2y",
        interval="1d"
    )

    if raw is None or len(raw) < 160:
        return None

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    feat = make_features_h2(df)

    feature_cols = [
        "range",
        "rv_10",
        "mom_ratio",
        "rsi_9",
        "dist_ma10",
        "mom_2d",
        "mom_5d",
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "ret_lag_5",
        "ret_lag_10",
        "trend_10"
    ]

    # TARGET
    feat["y_fwd"] = np.log(
        feat["Close"].shift(-HORIZON) /
        feat["Close"]
    )

    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 140:
        return None

    # ==================================================
    # PIPELINE
    # ==================================================

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("ridge", Ridge(alpha=ALPHA_H2, random_state=42))
    ])

    model.fit(clean[feature_cols], clean["y_fwd"])

    # ==================================================
    # PREDICCIÓN
    # ==================================================

    last_features = (
        feat[feature_cols]
        .iloc[-1:]
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    y_pred_log = model.predict(last_features)[0]

    # PROTECCIÓN CONTRA PREDICCIONES EXTREMAS
    y_pred_log = np.clip(y_pred_log, -0.05, 0.05)

    price_today = float(feat["Close"].iloc[-1])

    price_48h = price_today * np.exp(y_pred_log)

    # ==================================================
    # MÉTRICAS
    # ==================================================

    ridge_model = model.named_steps["ridge"]

    confidence = 1 / (1 + np.std(ridge_model.coef_))

    r2_train = ridge_model.score(
        clean[feature_cols],
        clean["y_fwd"]
    )

    return {
        "ticker": ticker,
        "predictor": "H2_v2.0",
        "horizon_days": HORIZON,
        "date_today": datetime.now().strftime("%Y-%m-%d"),
        "price_today": round(price_today, 4),
        "price_48h": round(price_48h, 4),
        "return_48h_pct": round(y_pred_log * 100, 4),
        "confidence": round(confidence, 3),
        "r2_train": round(r2_train, 4),
        "samples": len(clean),
        "model": {
            "alpha": ALPHA_H2,
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

    print(f"🚀 H2 Predictor 48h → {ticker}")

    result = run_predictor_h2(ticker)

    if result:

        filename = f"{ticker}_H2_48h_v2.json"

        path = os.path.join(DATA_OUTPUT_DIR, filename)

        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print("\n✅ H2 v2.0 - 48H PREDICTED")
        print(f"📊 {ticker}:  ${result['price_today']:>8.2f}")
        print(f"📊 48h:       ${result['price_48h']:>8.2f}")
        print(f"📈 Retorno:   {result['return_48h_pct']:>6.2f}%")
        print(f"🔥 Confianza: {result['confidence']:>5.1f}")
        print(f"📁 {path}")

    else:

        print("❌ Datos insuficientes H2")

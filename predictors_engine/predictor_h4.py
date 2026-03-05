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
# CONFIGURACIÓN H4 - CONSOLIDACIÓN SEMANAL (PRODUCCIÓN)
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 4
ALPHA_H4 = 0.5
PCA_COMPONENTS = 10

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)


# ======================================================
# FEATURE ENGINEERING H4
# ======================================================

def make_features_h4(df: pd.DataFrame):

    out = df.copy()

    # BASE
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.30)

    past = out["ret1"].shift(1)

    # VOLATILIDAD 20d
    out["rv_20"] = past.rolling(20).std().clip(1e-6) * np.sqrt(252/20)

    # TREND EMA
    out["ema_5"] = out["Close"].ewm(span=5).mean()
    out["ema_20"] = out["Close"].ewm(span=20).mean()

    out["trend_diff"] = np.log(
        out["ema_5"] / out["ema_20"]
    ).clip(-0.15, 0.15)

    # MOMENTUM
    out["mom_5d"] = past.rolling(5).sum()
    out["mom_10d"] = past.rolling(10).sum()
    out["mom_20d"] = past.rolling(20).sum()

    # ATR RATIO
    out["atr_14"] = out["range"].rolling(14).mean().clip(1e-6)

    out["vol_ratio"] = (
        out["range"] /
        out["atr_14"]
    ).clip(0, 5)

    # RSI
    delta = out["Close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, 1)

    out["rsi_14"] = 100 - (100 / (1 + rs))

    # STOCHASTIC
    low_14 = out["Low"].rolling(14).min()
    high_14 = out["High"].rolling(14).max()

    out["stoch_k"] = (
        100 * (out["Close"] - low_14) /
        (high_14 - low_14).replace(0, 1)
    )

    out["stoch_k"] = out["stoch_k"].clip(0, 100)

    # DONCHIAN
    out["hi_20"] = out["High"].rolling(20).max()
    out["lo_20"] = out["Low"].rolling(20).min()

    out["dist_hi"] = np.log(
        out["hi_20"] / out["Close"]
    ).clip(0, 0.3)

    out["dist_lo"] = np.log(
        out["Close"] / out["lo_20"]
    ).clip(0, 0.3)

    # VOLUMEN
    out["vol_price"] = (out["Volume"] * out["ret1"]).clip(-1e7, 1e7)

    return out


# ======================================================
# PREDICTOR H4
# ======================================================

def run_predictor_h4(ticker: str):

    raw = get_price_history(
        ticker=ticker,
        period="2y",
        interval="1d"
    )

    if raw is None or len(raw) < 200:
        return None

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    feat = make_features_h4(df)

    feature_cols = [
        "range",
        "rv_20",
        "trend_diff",
        "vol_ratio",
        "rsi_14",
        "stoch_k",
        "dist_hi",
        "dist_lo",
        "mom_5d",
        "mom_10d",
        "mom_20d",
        "vol_price"
    ]

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
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("ridge", Ridge(alpha=ALPHA_H4, random_state=42))
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

    # CLIP REALISTA 4 DÍAS
    y_pred_log = np.clip(y_pred_log, -0.10, 0.10)

    price_today = float(feat["Close"].iloc[-1])
    price_96h = price_today * np.exp(y_pred_log)

    # MÉTRICAS
    ridge_model = model.named_steps["ridge"]

    confidence = 1 / (1 + np.std(ridge_model.coef_))

    r2_train = ridge_model.score(
        clean[feature_cols],
        clean["y_fwd"]
    )

    return {
        "ticker": ticker,
        "predictor": "H4_v2.1",
        "horizon_days": HORIZON,
        "date_today": datetime.now().strftime("%Y-%m-%d"),
        "price_today": round(price_today, 4),
        "price_96h": round(price_96h, 4),
        "return_96h_pct": round(y_pred_log * 100, 4),
        "confidence": round(confidence, 3),
        "r2_train": round(r2_train, 4),
        "samples": len(clean),
        "model": {
            "alpha": ALPHA_H4,
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

    print(f"🚀 H4 Consolidación Semanal (96h) → {ticker}")

    result = run_predictor_h4(ticker)

    if result:

        filename = f"{ticker}_H4_96h_v2.json"

        path = os.path.join(DATA_OUTPUT_DIR, filename)

        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print("\n✅ H4 v2.1 - INSTITUCIONAL SEMANAL")
        print(f"📊 {ticker}:           ${result['price_today']:>8.2f}")
        print(f"📊 Predicción 96h:     ${result['price_96h']:>8.2f}")
        print(f"📈 Retorno esperado:   {result['return_96h_pct']:>6.2f}%")
        print(f"🔥 Confianza:          {result['confidence']:>5.1f}")
        print(f"📁 {path}")

    else:

        print("❌ Datos insuficientes H4")

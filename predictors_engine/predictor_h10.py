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
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from data_provider import get_price_history


# ======================================================
# CONFIGURACIÓN GLOBAL
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 10
ALPHA_H10 = 2.0
K_NEIGHBORS = 30
PCA_TARGET = 25


# ======================================================
# UTILIDADES MATEMÁTICAS
# ======================================================

def hurst_rs(x: np.ndarray) -> float:

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)

    if n < 30:
        return 0.5

    x = x - x.mean()

    z = np.cumsum(x)

    r = z.max() - z.min()
    s = x.std(ddof=1)

    if s <= 1e-9 or r <= 1e-9:
        return 0.5

    return float(np.log((r + 1e-9) / (s + 1e-9)) / np.log(n))


def compute_rsi_wilder(price: pd.Series, period: int = 14) -> pd.Series:

    delta = price.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)

    return 100 - (100 / (1 + rs))


def zscore_rolling(s: pd.Series, window: int) -> pd.Series:

    mean = s.rolling(window).mean()
    std = s.rolling(window).std()

    return (s - mean) / (std + 1e-9)


# ======================================================
# FEATURE ENGINEERING H10
# ======================================================

def make_features_h10(df: pd.DataFrame):

    out = df.copy()

    out["ret1"] = np.log(out["Close"]).diff()

    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.50)

    out["vol_chg"] = np.log(out["Volume"].replace(0, np.nan)).diff().fillna(0)

    # LAGS
    for k in range(1, 21):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    past = out["ret1"].shift(1)

    out["rv_60"] = past.rolling(60).std().fillna(0)

    out["hurst_120"] = past.rolling(120).apply(
        lambda a: hurst_rs(a),
        raw=True
    ).fillna(0.5)

    close_past = out["Close"].shift(1)

    sma = close_past.rolling(20).mean()
    std = close_past.rolling(20).std()

    out["bb_pct_b_z"] = zscore_rolling(
        (close_past - (sma - 2 * std)) / (4 * std + 1e-9),
        252
    ).fillna(0)

    out["rsi_z"] = zscore_rolling(
        compute_rsi_wilder(close_past, 14),
        252
    ).fillna(0)

    return out.fillna(0)


# ======================================================
# MOTOR H10
# ======================================================

def run_predictor_h10(ticker: str):

    raw = get_price_history(ticker=ticker, period="max", interval="1d")

    if raw is None or len(raw) < 300:
        return None

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    feat = make_features_h10(df)

    feature_cols = [
        "range",
        "vol_chg",
        "rv_60",
        "hurst_120",
        "bb_pct_b_z",
        "rsi_z"
    ] + [f"ret_lag_{k}" for k in range(1, 21)]

    feat["y_fwd"] = np.log(
        feat["Close"].shift(-HORIZON) / feat["Close"]
    )

    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    X = clean[feature_cols].values
    y = clean["y_fwd"].values


    # ======================================================
    # PCA DINÁMICO
    # ======================================================

    n_samples, n_features = X.shape

    dynamic_pca = max(
        1,
        min(PCA_TARGET, n_features, n_samples - 1)
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=dynamic_pca, random_state=42)),
        ("ridge", Ridge(alpha=ALPHA_H10))
    ])

    pipe.fit(X, y)


    # ======================================================
    # PREDICCIÓN GLOBAL
    # ======================================================

    last_feat = feat[feature_cols].iloc[-1:].values

    y_global = float(pipe.predict(last_feat)[0])


    # ======================================================
    # kNN ANALOGÍA HISTÓRICA
    # ======================================================

    X_scaled = pipe.named_steps["scaler"].transform(X)
    X_pca = pipe.named_steps["pca"].transform(X_scaled)

    X_last_scaled = pipe.named_steps["scaler"].transform(last_feat)
    X_last_pca = pipe.named_steps["pca"].transform(X_last_scaled)

    k = min(K_NEIGHBORS, len(X_pca))

    nn = NearestNeighbors(n_neighbors=k)

    nn.fit(X_pca)

    distances, idx = nn.kneighbors(X_last_pca)

    weights = 1.0 / (distances[0] + 1e-9)

    y_knn = float(
        np.sum(y[idx[0]] * weights) /
        np.sum(weights)
    )


    # ======================================================
    # ENSEMBLE
    # ======================================================

    y_final = 0.5 * y_knn + 0.5 * y_global


    # ======================================================
    # CONDITIONING
    # ======================================================

    h120 = float(feat["hurst_120"].iloc[-1])
    rsi_z = float(feat["rsi_z"].iloc[-1])

    if h120 > 0.55:
        y_final *= 1.15

    if abs(rsi_z) > 1.5:
        y_final *= 1.10


    y_final = np.clip(y_final, -0.20, 0.20)


    price_today = float(feat["Close"].iloc[-1])

    price_10d = price_today * np.exp(y_final)


    return {

        "ticker": ticker,
        "predictor": "H10_v2.6_LongStructure",

        "price_now": round(price_today, 4),
        "price_10d": round(price_10d, 4),

        "ret_pct": round(y_final * 100, 4),

        "samples": len(clean),

        "model": {
            "pca_components": dynamic_pca,
            "k_neighbors": k
        },

        "timestamp": datetime.now().isoformat()
    }


# ======================================================
# EJECUCIÓN
# ======================================================

if __name__ == "__main__":

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"

    print(f"🚀 Iniciando H10 V2.6 → {ticker}")

    res = run_predictor_h10(ticker)

    if res:

        path = os.path.join(DATA_OUTPUT_DIR, f"{ticker}_H10.json")

        with open(path, "w") as f:
            json.dump(res, f, indent=2)

        print("")
        print("✅ H10 COMPLETADO")
        print("----------------------------------")
        print(f"📊 Precio: ${res['price_now']}")
        print(f"🎯 Target 10d: ${res['price_10d']}")
        print(f"📈 Retorno: {res['ret_pct']}%")
        print("----------------------------------")

    else:

        print("❌ Datos insuficientes para H10")

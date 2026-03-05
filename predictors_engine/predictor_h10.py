import numpy as np
import pandas as pd
import os
import json
import sys
from datetime import datetime
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from data_provider import get_price_history
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# ======================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS H10
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
if not os.path.exists(DATA_OUTPUT_DIR):
    os.makedirs(DATA_OUTPUT_DIR)

# PARÁMETROS H10 (Inercia Estructural)
HORIZON = 10
ALPHA_H10 = 2.0      # Máxima regularización para horizonte de 2 semanas
K_NEIGHBORS = 30     # Mayor vecindad para filtrar anomalías
PCA_TARGET = 25
THETA_BASE = 0.85    # Umbral de decisión más exigente para H10

# ======================================================
# MATH UTILS (V6.3)
# ======================================================

def hurst_rs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30: return np.nan
    x = x - x.mean()
    z = np.cumsum(x)
    r = z.max() - z.min()
    s = x.std(ddof=1)
    if s == 0 or r == 0: return np.nan
    return float(np.log(r / s) / np.log(n))

def compute_rsi_wilder(price: pd.Series, period: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def zscore_rolling(s: pd.Series, window: int) -> pd.Series:
    return (s - s.rolling(window).mean()) / s.rolling(window).std().replace(0, np.nan)

# ======================================================
# CORE ENGINE H10
# ======================================================

def make_features_h10(df: pd.DataFrame):
    out = df.copy()
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"])
    out["vol_chg"] = np.log(out["Volume"].replace(0, np.nan)).diff()

    # Lags (Ventana de 20 días para H10)
    for k in range(1, 21):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    past = out["ret1"].shift(1)
    out["rv_60"] = past.rolling(60).std()
    out["hurst_120"] = past.rolling(120).apply(lambda a: hurst_rs(a), raw=True)

    # Bollinger & RSI (Lógica V6.3)
    close_past = out["Close"].shift(1)
    sma = close_past.rolling(20).mean()
    std = close_past.rolling(20).std()
    out["bb_pct_b_z"] = zscore_rolling((close_past - (sma - 2*std)) / (4*std).replace(0, np.nan), 252)
    out["rsi_z"] = zscore_rolling(compute_rsi_wilder(close_past, 14), 252)
    
    return out

def run_predictor_h10(ticker: str):
    raw = get_price_history(ticker=ticker, period="max", interval="1d")
    if raw is None or raw.empty: return None

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h10(df)
    
    feature_cols = ["range", "vol_chg", "rv_60", "hurst_120", "bb_pct_b_z", "rsi_z"] + [f"ret_lag_{k}" for k in range(1, 21)]
    target_col = "y_fwd"
    feat[target_col] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    
    clean = feat.dropna(subset=feature_cols + [target_col])
    X = clean[feature_cols].values
    y = clean[target_col].values

    # Pipeline Ridge + PCA (V6.3)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(PCA_TARGET, len(feature_cols)), random_state=42)),
        ("ridge", Ridge(alpha=ALPHA_H10))
    ])
    pipe.fit(X, y)

    # Predicción Ensemble
    last = feat.iloc[-1]
    X_last = last[feature_cols].values.reshape(1, -1)
    y_global = float(pipe.predict(X_last)[0])

    # kNN Predictor
    X_pca = pipe.named_steps["pca"].transform(pipe.named_steps["scaler"].transform(X))
    X_last_pca = pipe.named_steps["pca"].transform(pipe.named_steps["scaler"].transform(X_last))
    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(X_pca)
    _, idx = nn.kneighbors(X_last_pca)
    y_knn = float(np.mean(y[idx[0]]))

    # Signal Conditioning
    y_final = (0.5 * y_knn + 0.5 * y_global) # Ensemble base
    
    # Ajuste por Hurst y RSI (Lógica suave V6.3)
    h120 = float(last.get("hurst_120", 0.5))
    if h120 > 0.55: y_final *= 1.15
    if abs(float(last.get("rsi_z", 0))) > 1.5: y_final *= 1.1

    price_now = float(last["Close"])
    price_pred = float(price_now * np.exp(y_final))

    return {
        "ticker": ticker,
        "horizon": HORIZON,
        "price_now": round(price_now, 2),
        "price_pred": round(price_pred, 2),
        "expected_ret_pct": round(y_final * 100, 4),
        "alpha_used": ALPHA_H10,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker_input = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    res = run_predictor_h10(ticker_input)
    if res:
        filename = f"{ticker_input}_h{HORIZON}.json"
        full_path = os.path.join(DATA_OUTPUT_DIR, filename)
        with open(full_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"✅ H10 FINALIZADO (V6.3-Long): {ticker_input} -> {full_path}")
  

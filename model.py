"""
model.py — V6.3 (PRECISIÓN MEJORADA, JSON 100% V4)
- Señal mejorada con Bollinger + RSI cuant
- Signal conditioning matemático suave
- Ensemble adaptativo robusto
- SIN romper contrato JSON
"""

import numpy as np
import pandas as pd
from datetime import datetime
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.offsets import DateOffset

# ======================================================
# Dataclasses (IGUAL V4)
# ======================================================

@dataclass
class ModelMeta:
    ticker: str
    horizon_days: int
    pca_target: int
    theta: float
    k_neighbors: int
    alpha: float
    period: str

@dataclass
class HistoricalStats:
    hit_rate_mean: Optional[float]
    mae_mean: Optional[float]
    rmse_mean: Optional[float]
    pca_dims: Optional[int]
    n_features: int
    n_windows: int

# ======================================================
# Utils
# ======================================================

def log_return(series: pd.Series) -> pd.Series:
    return np.log(series).diff()

def hurst_rs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    x = x - x.mean()
    z = np.cumsum(x)
    r = z.max() - z.min()
    s = x.std(ddof=1)
    if s == 0 or r == 0:
        return np.nan
    return float(np.log(r / s) / np.log(n))

def rolling_hurst(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).apply(lambda a: hurst_rs(a), raw=True)

def zscore_rolling(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std().replace(0, np.nan)
    return (s - mu) / sd

def compute_rsi_wilder(price: pd.Series, period: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ======================================================
# Feature engineering
# ======================================================

def make_features(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    out = df.copy()

    out["ret1"] = log_return(out["Close"])
    out["range"] = np.log(out["High"] / out["Low"])
    out["vol_chg"] = np.log(out["Volume"].replace(0, np.nan)).diff()

    for k in range(1, 21):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    past = out["ret1"].shift(1)
    out["rv_20"] = past.rolling(20).std()
    out["rv_60"] = past.rolling(60).std()
    out["hurst_60"] = rolling_hurst(past, 60)
    out["hurst_120"] = rolling_hurst(past, 120)

    # Bollinger
    close_past = out["Close"].shift(1)
    sma = close_past.rolling(20).mean()
    std = close_past.rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb_range = (upper - lower).replace(0, np.nan)

    out["bb_pct_b"] = (close_past - lower) / bb_range
    out["bb_width"] = bb_range / sma.replace(0, np.nan)

    # Estacionarización
    out["bb_pct_b_z"] = zscore_rolling(out["bb_pct_b"], 252)
    out["bb_width_z"] = zscore_rolling(out["bb_width"], 252)

    # RSI
    out["rsi"] = compute_rsi_wilder(close_past, 14)
    out["rsi_z"] = zscore_rolling(out["rsi"], 252)

    out["y_fwd"] = np.log(out["Close"].shift(-horizon) / out["Close"])

    return out

# ======================================================
# kNN
# ======================================================

def knn_predict(X_train, y_train, X_query, k):
    k_eff = min(k, len(X_train))
    if k_eff < 3:
        return 0.0
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(X_train)
    _, idx = nn.kneighbors(X_query)
    return float(np.mean(y_train[idx[0]]))

# ======================================================
# Walk Forward
# ======================================================

def walk_forward_train_test(data, feature_cols, target_col,
                            train_years=5, test_months=6, pca_target=50):

    clean = data.dropna(subset=feature_cols + [target_col]).sort_index()
    if len(clean) < 400:
        return pd.DataFrame()

    results = []
    cur_train_start = clean.index.min()

    n_features = len(feature_cols)
    n_pca = min(pca_target, max(1, n_features - 1))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=42)),
        ("ridge", Ridge(alpha=1.0))
    ])

    while True:
        cur_train_end = cur_train_start + DateOffset(years=train_years)
        cur_test_end = cur_train_end + DateOffset(months=test_months)

        train = clean[(clean.index >= cur_train_start) & (clean.index < cur_train_end)]
        test = clean[(clean.index >= cur_train_end) & (clean.index < cur_test_end)]

        if len(test) < 30:
            break

        if len(train) >= 300:
            X_train = train[feature_cols].values
            y_train = train[target_col].values
            X_test = test[feature_cols].values
            y_test = test[target_col].values

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "hit_rate": float((np.sign(y_pred) == np.sign(y_test)).mean()),
                "n_pca": n_pca,
                "n_features": n_features,
            })

        cur_train_start = cur_train_end

    return pd.DataFrame(results)

# ======================================================
# CORE ENGINE
# ======================================================

def _run_full_math_engine(ticker, horizon, pca_target,
                          theta, k_neighbors, alpha, period):

    raw = get_price_history(ticker=ticker, period=period, interval="1d")
    if raw is None or raw.empty:
        raise RuntimeError(f"Sin datos para {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)

    feat = make_features(df, horizon)

    feature_cols = [
        "range", "vol_chg", "rv_20", "rv_60", "hurst_60", "hurst_120",
        "bb_pct_b_z", "bb_width_z", "rsi_z",
        *[f"ret_lag_{k}" for k in range(1, 21)]
    ]

    res_df = walk_forward_train_test(feat, feature_cols, "y_fwd", pca_target)

    if res_df.empty:
        hist = HistoricalStats(None, None, None, None, len(feature_cols), 0)
    else:
        hist = HistoricalStats(
            float(res_df["hit_rate"].mean()),
            float(res_df["mae"].mean()),
            float(res_df["rmse"].mean()),
            int(res_df["n_pca"].iloc[0]),
            len(feature_cols),
            len(res_df),
        )

    train_df = feat.dropna(subset=["y_fwd"] + feature_cols)
    X = train_df[feature_cols].values
    y = train_df["y_fwd"].values

    n_features = len(feature_cols)
    n_pca = min(pca_target, max(1, n_features - 1))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=42)),
        ("ridge", Ridge(alpha=alpha))
    ])

    pipe.fit(X, y)

    last = feat.iloc[-1]
    X_last = last[feature_cols].values.reshape(1, -1)

    y_global = float(pipe.predict(X_last)[0])

    X_pca = pipe.named_steps["pca"].transform(pipe.named_steps["scaler"].transform(X))
    X_last_pca = pipe.named_steps["pca"].transform(pipe.named_steps["scaler"].transform(X_last))

    y_knn = knn_predict(X_pca, y, X_last_pca, k_neighbors)

    # Ensemble adaptativo
    hr = hist.hit_rate_mean if hist.hit_rate_mean else 0.5
    w = float(np.clip(hr, 0.4, 0.6))
    y_ens = w * y_knn + (1 - w) * y_global

    # Signal conditioning real (mejora precisión)
    hurst60 = float(last.get("hurst_60", np.nan))
    rsi_z = float(last.get("rsi_z", np.nan))
    bb_width_z = float(last.get("bb_width_z", np.nan))

    signal = y_ens

    if np.isfinite(hurst60):
        if hurst60 > 0.55:
            signal *= 1.15
        elif hurst60 < 0.45:
            signal *= 0.9

    if np.isfinite(rsi_z) and abs(rsi_z) > 1.5:
        signal *= 1.1

    if np.isfinite(bb_width_z) and bb_width_z < -1:
        signal *= 1.1

    y_final = signal

    recent_vol = float(last.get("rv_60", 0.02))
    vol_adj = np.clip(recent_vol * 50, 0.5, 2.0)
    qual_adj = np.clip(1.0 - (hr - 0.5), 0.8, 1.2)
    theta_dyn = theta * vol_adj * qual_adj

    price_now = float(last["Close"])
    price_pred = float(price_now * np.exp(y_final))

    recommendation = (
        "COMPRA" if y_final * 100 >= theta_dyn else
        "VENDE" if y_final * 100 <= -theta_dyn else
        "MANTÉN"
    )

    return {
        "meta": ModelMeta(ticker, horizon, pca_target, theta, k_neighbors, alpha, period).__dict__,
        "historical": {
            "hit_rate_mean": hist.hit_rate_mean,
            "mae_mean": hist.mae_mean,
            "rmse_mean": hist.rmse_mean,
            "pca_dims": hist.pca_dims,
            "n_features": hist.n_features,
            "n_windows": hist.n_windows,
        },
        "prediction": {
            "date_base": datetime.utcnow().date().isoformat(),
            "ret_global_pct": round(y_global * 100.0, 4),
            "ret_knn_pct": round(y_knn * 100.0, 4),
            "ret_ens_pct": round(y_final * 100.0, 4),
            "price_now": round(price_now, 2),
            "price_pred": round(price_pred, 2),
            "recommendation": recommendation,
            "theta_dynamic_pct": round(theta_dyn, 4),
            "pca_dims_effective": n_pca,
            "n_features": n_features,
        },
    }

# ======================================================
# API pública
# ======================================================

def run_model(ticker="SPY", horizon=1, pca_target=50,
              theta=0.75, k_neighbors=20, alpha=0.5,
              period="max"):
    return _run_full_math_engine(ticker, horizon, pca_target,
                                 theta, k_neighbors, alpha, period)

if __name__ == "__main__":
    print(run_model("SPY"))

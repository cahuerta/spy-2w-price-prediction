"""
predictor_h10.py
Adaptación de model.py para funcionar como H10 dentro del orchestrator.
NO cambia la lógica matemática.

FIX v10.1:
  [N1] feat.iloc[-1] puede tener NaN en hurst_120/lags para tickers
       con datos irregulares → ValueError: Input X contains NaN en PCA.
       Fix: buscar última fila válida sin NaN en feature_cols.

FIX v10.2:
  [N2] THETA_H10 subido de 0.75 a 1.5 — umbral anterior demasiado bajo
       (~0.3-0.4% efectivo tras vol_adjustment) generaba COMPRA en ruido.
       Con 1.5% el theta_dynamic efectivo es ~0.75-1.5%, filtrando señales débiles.
"""

import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from datetime import datetime
from data_provider import get_price_history
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.offsets import DateOffset

# ======================================================
# CONFIG H10
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON_H10  = 10
PCA_TARGET_H10 = 50
THETA_H10    = 1.5   # [N2] era 0.75 — subido para filtrar señales débiles
K_NEIGHBORS_H10 = 20
ALPHA_H10    = 0.5
PERIOD_H10   = "max"

# ======================================================
# Dataclasses
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
# Utils matemáticos
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

# ======================================================
# Feature engineering (SIN data leak)
# ======================================================

def make_features(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    out = df.copy()
    out["ret1"]    = log_return(out["Close"])
    out["range"]   = np.log(out["High"] / out["Low"])
    out["vol_chg"] = np.log(out["Volume"].replace(0, np.nan)).diff()
    for k in range(1, 21):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)
    past = out["ret1"].shift(1)
    out["rv_20"]     = past.rolling(20).std()
    out["rv_60"]     = past.rolling(60).std()
    out["hurst_60"]  = rolling_hurst(past, 60)
    out["hurst_120"] = rolling_hurst(past, 120)
    out["y_fwd"]     = np.log(out["Close"].shift(-horizon) / out["Close"])
    return out

# ======================================================
# kNN caótico
# ======================================================

def knn_caotico_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    k: int = 20
) -> float:
    k_eff = min(k, len(X_train))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(X_train)
    _, idx = nn.kneighbors(X_query)
    return float(np.mean(y_train[idx[0]]))

# ======================================================
# WALK-FORWARD
# ======================================================

def walk_forward_train_test(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_years: int = 5,
    test_months: int = 6,
    pca_target: int = 50
) -> pd.DataFrame:
    clean_data = data.dropna(subset=feature_cols + [target_col]).sort_index()
    if len(clean_data) < 400:
        return pd.DataFrame()

    results = []
    start = clean_data.index.min()
    cur_train_start = start
    n_features = len(feature_cols)
    n_pca = min(pca_target, max(1, n_features - 1))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, random_state=42)),
        ("ridge",  Ridge(alpha=1.0))
    ])

    while True:
        cur_train_end = cur_train_start + DateOffset(years=train_years)
        cur_test_end  = cur_train_end   + DateOffset(months=test_months)

        train = clean_data[(clean_data.index >= cur_train_start) & (clean_data.index < cur_train_end)]
        test  = clean_data[(clean_data.index >= cur_train_end)   & (clean_data.index < cur_test_end)]

        if len(test) < 30:
            break

        if len(train) >= 300:
            X_train = train[feature_cols].values
            y_train = train[target_col].values
            X_test  = test[feature_cols].values
            y_test  = test[target_col].values

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                "mae":      float(mean_absolute_error(y_test, y_pred)),
                "rmse":     float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "hit_rate": float((np.sign(y_pred) == np.sign(y_test)).mean()),
                "n_test":   len(test),
                "n_train":  len(train),
                "n_features": n_features,
                "n_pca":    n_pca,
            })

        cur_train_start = cur_train_end

    return pd.DataFrame(results)

# ======================================================
# Motor matemático completo
# ======================================================

def _run_full_math_engine(
    ticker: str,
    horizon: int,
    pca_target: int,
    theta: float,
    k_neighbors: int,
    alpha: float,
    period: str
) -> Dict:

    raw = get_price_history(ticker=ticker, period=period, interval="1d")
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"No se pudieron descargar datos para {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)

    if len(df) < 300:
        raise RuntimeError(f"Datos insuficientes: {len(df)} filas")

    feat = make_features(df, horizon=horizon)

    feature_cols = [
        "range", "vol_chg", "rv_20", "rv_60", "hurst_60", "hurst_120",
        *[f"ret_lag_{k}" for k in range(1, 21)]
    ]

    res_df = walk_forward_train_test(feat, feature_cols, "y_fwd", pca_target=pca_target)

    if res_df is None or len(res_df) == 0:
        hist = HistoricalStats(
            hit_rate_mean=None, mae_mean=None, rmse_mean=None,
            pca_dims=None, n_features=len(feature_cols), n_windows=0
        )
    else:
        hist = HistoricalStats(
            hit_rate_mean=float(res_df["hit_rate"].mean()),
            mae_mean=float(res_df["mae"].mean()),
            rmse_mean=float(res_df["rmse"].mean()),
            pca_dims=int(res_df["n_pca"].iloc[0]),
            n_features=int(res_df["n_features"].iloc[0]),
            n_windows=int(len(res_df))
        )

    n_features = len(feature_cols)
    n_pca = min(pca_target, max(1, n_features - 1))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, random_state=42)),
        ("ridge",  Ridge(alpha=1.0))
    ])

    train_df = feat.dropna(subset=["y_fwd"] + feature_cols)
    X = train_df[feature_cols].values
    y = train_df["y_fwd"].values
    model.fit(X, y)

    # [N1] Buscar última fila sin NaN en feature_cols
    feat_only  = feat[feature_cols]
    valid_rows = feat_only.dropna()
    if len(valid_rows) == 0:
        last_values = feat_only.ffill().fillna(0).iloc[-1].values
    else:
        last_values = valid_rows.iloc[-1].values

    if not np.all(np.isfinite(last_values)):
        last_values = np.nan_to_num(last_values, nan=0.0, posinf=0.0, neginf=0.0)

    X_last     = last_values.reshape(1, -1)
    price_now  = float(feat["Close"].dropna().iloc[-1])
    y_global   = float(model.predict(X_last)[0])

    X_pca = model.named_steps["pca"].transform(
        model.named_steps["scaler"].transform(X)
    )
    X_last_pca = model.named_steps["pca"].transform(
        model.named_steps["scaler"].transform(X_last)
    )
    y_knn = knn_caotico_predict(X_pca, y, X_last_pca, k_neighbors)

    w = hist.hit_rate_mean
    if w is not None:
        w = float(np.clip(w, 0.4, 0.6))
    else:
        w = alpha

    y_ens = w * y_knn + (1 - w) * y_global

    recent_vol = float(feat["rv_60"].dropna().iloc[-1])
    if not np.isfinite(recent_vol) or recent_vol <= 0:
        recent_vol = 0.02

    hit_rate         = hist.hit_rate_mean if hist.hit_rate_mean is not None else 0.5
    vol_adjustment   = np.clip(recent_vol * 50, 0.5, 2.0)
    quality_adjustment = np.clip(1.0 - (hit_rate - 0.5), 0.8, 1.2)

    # [N2] theta ahora es 1.5 en vez de 0.75 → theta_dynamic efectivo ~0.75-1.5%
    theta_dynamic = theta * vol_adjustment * quality_adjustment

    price_pred = float(price_now * np.exp(y_ens))

    recommendation = (
        "COMPRA" if y_ens * 100 >= theta_dynamic else
        "VENDE"  if y_ens * 100 <= -theta_dynamic else
        "MANTÉN"
    )

    return {
        "meta": ModelMeta(ticker, horizon, pca_target, theta, k_neighbors, alpha, period).__dict__,
        "historical": {
            "hit_rate_mean": hist.hit_rate_mean,
            "mae_mean":      hist.mae_mean,
            "rmse_mean":     hist.rmse_mean,
            "pca_dims":      hist.pca_dims,
            "n_features":    hist.n_features,
            "n_windows":     hist.n_windows,
        },
        "prediction": {
            "date_base":           datetime.utcnow().date().isoformat(),
            "ret_global_pct":      round(y_global * 100.0, 4),
            "ret_knn_pct":         round(y_knn * 100.0, 4),
            "ret_ens_pct":         round(y_ens * 100.0, 4),
            "price_now":           round(price_now, 2),
            "price_pred":          round(price_pred, 2),
            "recommendation":      recommendation,
            "theta_dynamic_pct":   round(theta_dynamic, 4),
            "pca_dims_effective":  n_pca,
            "n_features":          n_features,
        },
    }

# ======================================================
# API pública
# ======================================================

def run_predictor_h10(
    ticker: str = "SPY",
    horizon: int = 10,
    pca_target: int = 50,
    theta: float = THETA_H10,   # [N2] usa constante actualizada
    k_neighbors: int = 20,
    alpha: float = 0.5,
    period: str = "max"
):
    return _run_full_math_engine(ticker, horizon, pca_target, theta, k_neighbors, alpha, period)


def format_report(result: dict) -> str:
    m, h, p = result["meta"], result["historical"], result["prediction"]
    lines = ["=== RESUMEN HISTÓRICO (WALK-FORWARD OOS) ==="]
    if h["hit_rate_mean"] is None:
        lines.append("Sin ventanas walk-forward válidas.")
    else:
        lines.extend([
            f"✅ Ventanas OOS     : {h['n_windows']}",
            f"✅ Hit-rate OOS     : {h['hit_rate_mean']:.4f}",
            f"MAE retorno OOS    : {h['mae_mean']:.6f}",
            f"RMSE retorno OOS   : {h['rmse_mean']:.6f}",
            f"PCA dims           : {h['pca_dims']}",
        ])
    lines.extend([
        "",
        "=== PREDICCIÓN HOY (ENSEMBLE) ===",
        f"Activo              : {m['ticker']}",
        f"Horizonte           : {m['horizon_days']} días",
        f"Precio actual       : ${p['price_now']:,.2f}",
        f"Precio esperado     : ${p['price_pred']:,.2f}",
        f"Retorno ENSAMBLE    : {p['ret_ens_pct']:.2f}%",
        f"Theta dinámico      : {p['theta_dynamic_pct']:.2f}%",
        f"RECOMENDACIÓN FINAL : {p['recommendation']}",
    ])
    return "\n".join(lines)


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    result = run_predictor_h10(ticker=ticker, horizon=10)
    output_path = os.path.join(DATA_OUTPUT_DIR, f"{ticker}_H10.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(format_report(result))
    print(f"\n💾 Guardado: {output_path}")
          

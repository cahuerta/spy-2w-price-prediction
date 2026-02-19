# model.py
# Motor matemático completo + adaptador de compatibilidad
# Salidas 100% compatibles con el sistema actual

import numpy as np
import pandas as pf
from datetime import datetime
from data_provider import get_price_history
from typing import Dict, List, Optional
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ======================================================
# Dataclasses (solo internas, no afectan API)
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
    # Versión portable y segura
    return returns.rolling(window).apply(lambda a: hurst_rs(a), raw=True)


# ======================================================
# Feature engineering (SIN data leak)
# ======================================================
def make_features(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    out = df.copy()

    out["ret1"] = log_return(out["Close"])
    out["range"] = np.log(out["High"] / out["Low"])
    out["vol_chg"] = np.log(out["Volume"].replace(0, np.nan)).diff()

    for k in range(1, 21):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # Solo datos pasados
    past = out["ret1"].shift(1)
    out["rv_20"] = past.rolling(20).std()
    out["rv_60"] = past.rolling(60).std()
    out["hurst_60"] = rolling_hurst(past, 60)
    out["hurst_120"] = rolling_hurst(past, 120)

    # Target futuro (correcto)
    out["y_fwd"] = np.log(out["Close"].shift(-horizon) / out["Close"])

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
# Walk-forward training (solo métricas matemáticas)
# ======================================================
def walk_forward_train_test(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_years: int = 5,
    test_months: int = 6,
    pca_target: int = 50,
) -> pd.DataFrame:
    df = data.sort_index()
    start, end = df.index.min(), df.index.max()

    results = []
    cur_train_start = start
    cur_train_end = cur_train_start + pd.DateOffset(years=train_years)
    cur_test_end = cur_train_end + pd.DateOffset(months=test_months)

    n_features = len(feature_cols)
    n_pca = min(pca_target, max(1, n_features - 1))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=42)),
        ("ridge", Ridge(alpha=1.0))
    ])

    while cur_test_end <= end:
        train = df[(df.index >= cur_train_start) & (df.index < cur_train_end)]
        test = df[(df.index >= cur_train_end) & (df.index < cur_test_end)]

        if len(train) < 300 or len(test) < 30:
            break

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
            "n_test": len(test),
            "n_features": n_features,
            "n_pca": n_pca,
        })

        cur_train_start += pd.DateOffset(months=test_months)
        cur_train_end = cur_train_start + pd.DateOffset(years=train_years)
        cur_test_end = cur_train_end + pd.DateOffset(months=test_months)

    return pd.DataFrame(results)


# ======================================================
# Motor matemático completo (INTERNO)
# ======================================================
def _run_full_math_engine(
    ticker: str,
    horizon: int,
    pca_target: int,
    theta: float,
    k_neighbors: int,
    alpha: float,
    period: str,
) -> Dict:
    # Descarga vía data_provider (fachada única)
    raw = get_price_history(
        ticker=ticker,
        period=period,
        interval="1d"
    )

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

    # Walk-forward
    res_df = walk_forward_train_test(
        feat, feature_cols, "y_fwd", pca_target=pca_target
    )

    if res_df is None or len(res_df) == 0:
        hist = HistoricalStats(
            hit_rate_mean=None,
            mae_mean=None,
            rmse_mean=None,
            pca_dims=None,
            n_features=len(feature_cols),
            n_windows=0
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

    # Modelo final
    n_features = len(feature_cols)
    n_pca = min(pca_target, max(1, n_features - 1))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=42)),
        ("ridge", Ridge(alpha=1.0))
    ])

    feat = make_features(df, horizon=horizon)

    # 🔹 SOLO filas entrenables
    train_df = feat.dropna(subset=["y_fwd"])

    X = train_df[feature_cols].values
    y = train_df["y_fwd"].values
    model.fit(X, y)

    # 🔹 Última fila REAL (hoy)
    last_row = feat.iloc[-1]
    X_last = last_row[feature_cols].values.reshape(1, -1)

    y_global = float(model.predict(X_last)[0])

    X_pca = model.named_steps["pca"].transform(
        model.named_steps["scaler"].transform(X)
    )
    X_last_pca = model.named_steps["pca"].transform(
        model.named_steps["scaler"].transform(X_last)
    )

    y_knn = knn_caotico_predict(X_pca, y, X_last_pca, k_neighbors)

    # Ensemble levemente adaptativo (NO rompe contrato)
    w = hist.hit_rate_mean
    if w is not None:
        w = float(np.clip(w, 0.4, 0.6))
    else:
        w = alpha

    y_ens = w * y_knn + (1 - w) * y_global

    # ======================================================
    # THETA DINÁMICO (NO rompe contrato externo)
    # ======================================================

    # 1️⃣ Volatilidad reciente REAL (60 días)
    recent_vol = float(feat["rv_60"].iloc[-1])

    if not np.isfinite(recent_vol) or recent_vol <= 0:
        recent_vol = 0.02  # fallback técnico mínimo seguro

    # 2️⃣ Calidad histórica del modelo
    hit_rate = hist.hit_rate_mean if hist.hit_rate_mean is not None else 0.5

    # 3️⃣ Ajuste dinámico
    # Base: theta original (ej 0.75%)
    base_theta = theta

    # Vol alta → exige más convicción
    vol_adjustment = np.clip(recent_vol * 50, 0.5, 2.0)

    # Modelo bueno → permite menor theta
    quality_adjustment = np.clip(1.0 - (hit_rate - 0.5), 0.8, 1.2)

    theta_dynamic = base_theta * vol_adjustment * quality_adjustment

    price_now = float(last_row["Close"])
    price_pred = float(price_now * np.exp(y_ens))

    recommendation = (
        "COMPRA" if y_ens * 100 >= theta_dynamic else
        "VENDE" if y_ens * 100 <= -theta_dynamic else
        "MANTÉN"
    )

    return {
        "meta": ModelMeta(
            ticker, horizon, pca_target, theta, k_neighbors, alpha, period
        ).__dict__,
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
            "ret_global_pct": y_global * 100.0,
            "ret_knn_pct": y_knn * 100.0,
            "ret_ens_pct": y_ens * 100.0,
            "price_now": price_now,
            "price_pred": price_pred,
            "recommendation": recommendation,
            "pca_dims_effective": n_pca,
            "n_features": n_features,
        },
    }


# ======================================================
# Adaptador de compatibilidad (CONTRATO INTACTO)
# ======================================================
def _export_compat_result(full: dict) -> dict:
    return {
        "meta": full["meta"],
        "historical": full["historical"],
        "prediction": full["prediction"],
    }


# ======================================================
# API pública (NO CAMBIA)
# ======================================================
def run_model(
    ticker: str = "SPY",
    horizon: int = 10,
    pca_target: int = 50,
    theta: float = 0.75,
    k_neighbors: int = 20,
    alpha: float = 0.5,
    period: str = "max",
):
    full = _run_full_math_engine(
        ticker, horizon, pca_target, theta, k_neighbors, alpha, period
    )
    return _export_compat_result(full)


def format_report(result: dict) -> str:
    m, h, p = result["meta"], result["historical"], result["prediction"]

    lines = ["=== RESUMEN HISTÓRICO ==="]
    if h["hit_rate_mean"] is None:
        lines.append("Sin ventanas walk-forward.")
    else:
        lines.extend([
            f"Hit-rate prom : {h['hit_rate_mean']:.4f}",
            f"MAE retorno   : {h['mae_mean']:.6f}",
            f"RMSE retorno  : {h['rmse_mean']:.6f}",
            f"PCA dims      : {h['pca_dims']}",
            f"Ventanas      : {h['n_windows']}",
        ])

    lines.extend([
        "",
        "=== PREDICCIÓN ACTUAL (ENSAMBLE) ===",
        f"Activo              : {m['ticker']}",
        f"Fecha base          : {p['date_base']}",
        f"Retorno ENSAMBLE    : {p['ret_ens_pct']:.2f} %",
        f"Precio actual       : {p['price_now']:.2f} USD",
        f"Precio esperado     : {p['price_pred']:.2f} USD",
        f"RECOMENDACIÓN       : {p['recommendation']}",
    ])

    return "\n".join(lines)

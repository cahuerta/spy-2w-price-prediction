# model.py
# Misma matemática del main actual, modularizada (sin cambiar lógica).
# - No abre puertos
# - No while True
# - Devuelve resultados en dict (y opcional texto para logs/email)

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# Utils
# =========================
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


# =========================
# Feature engineering
# =========================
def make_features(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    out = df.copy()

    out["ret1"] = log_return(out["Close"])
    out["range"] = np.log(out["High"] / out["Low"])
    out["vol_chg"] = np.log(out["Volume"].replace(0, np.nan)).diff()

    for k in range(1, 21):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    out["rv_20"] = out["ret1"].rolling(20).std()
    out["rv_60"] = out["ret1"].rolling(60).std()

    out["hurst_60"] = rolling_hurst(out["ret1"], 60)
    out["hurst_120"] = rolling_hurst(out["ret1"], 120)

    out["y_fwd"] = np.log(out["Close"].shift(-horizon) / out["Close"])

    return out.dropna()


# =========================
# kNN CAÓTICO (dinámica local)
# =========================
def knn_caotico_predict(X_train, y_train, X_query, k=20) -> float:
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_train)
    _, indices = nn.kneighbors(X_query)
    return float(np.mean(y_train[indices[0]]))


# =========================
# Walk-forward training
# =========================
def walk_forward_train_test(
    data: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    train_years: int = 5,
    test_months: int = 6,
    pca_target: int = 50,
):
    df = data.sort_index()

    start = df.index.min()
    end = df.index.max()

    results = []
    preds_all = []

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
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "hit_rate": float((np.sign(y_pred) == np.sign(y_test)).mean()),
            "n_test": len(test),
            "n_features": n_features,
            "n_pca": n_pca
        })

        preds_all.append(pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred
        }, index=test.index))

        cur_train_start += pd.DateOffset(months=test_months)
        cur_train_end = cur_train_start + pd.DateOffset(years=train_years)
        cur_test_end = cur_train_end + pd.DateOffset(months=test_months)

    return pd.DataFrame(results), pd.concat(preds_all).sort_index()


# =========================
# Recomendación
# =========================
def recomendar(ret_pct: float, theta: float) -> str:
    if ret_pct >= theta:
        return "COMPRA"
    if ret_pct <= -theta:
        return "VENDE"
    return "MANTÉN"


# =========================
# Núcleo del modelo (misma lógica del main)
# =========================
def run_model(
    ticker: str = "SPY",
    horizon: int = 10,
    pca_target: int = 50,
    theta: float = 0.75,
    k_neighbors: int = 20,
    alpha: float = 0.5,
    period: str = "max",
):
    """
    Ejecuta:
    - Descarga datos
    - Features
    - Walk-forward (resumen)
    - Entrena final en todo y predice último punto
    - Ensamble global + kNN

    Devuelve dict con:
    - summary histórico
    - predicción actual
    - metadata
    """

    raw = yf.download(ticker, period=period, progress=False)

    if raw is None or len(raw) == 0:
        raise RuntimeError(f"No se pudieron descargar datos para {ticker}.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)

    feat = make_features(df, horizon=horizon)

    feature_cols = [
        "range", "vol_chg", "rv_20", "rv_60", "hurst_60", "hurst_120",
        *[f"ret_lag_{k}" for k in range(1, 21)]
    ]

    # =========================
    # WALK-FORWARD (misma lógica)
    # =========================
    res_df, _ = walk_forward_train_test(
        feat, feature_cols, "y_fwd", pca_target=pca_target
    )

    # Por si el walk-forward no generó ventanas (dataset raro / corto)
    if res_df is None or len(res_df) == 0:
        hist = {
            "hit_rate_mean": None,
            "mae_mean": None,
            "rmse_mean": None,
            "pca_dims": None,
            "n_features": len(feature_cols),
            "n_windows": 0
        }
    else:
        hist = {
            "hit_rate_mean": float(res_df["hit_rate"].mean()),
            "mae_mean": float(res_df["mae"].mean()),
            "rmse_mean": float(res_df["rmse"].mean()),
            "pca_dims": int(res_df["n_pca"].iloc[0]),
            "n_features": int(res_df["n_features"].iloc[0]),
            "n_windows": int(len(res_df))
        }

    # =========================
    # PREDICCIÓN ACTUAL (misma lógica)
    # =========================
    n_features = len(feature_cols)
    n_pca = min(pca_target, n_features - 1)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=42)),
        ("ridge", Ridge(alpha=1.0))
    ])

    X = feat[feature_cols].values
    y = feat["y_fwd"].values
    model.fit(X, y)

    X_pca = model.named_steps["pca"].transform(
        model.named_steps["scaler"].transform(X)
    )

    last_row = feat.iloc[-1]
    X_last = last_row[feature_cols].values.reshape(1, -1)
    X_last_pca = model.named_steps["pca"].transform(
        model.named_steps["scaler"].transform(X_last)
    )

    y_global = float(model.predict(X_last)[0])
    y_knn = knn_caotico_predict(X_pca, y, X_last_pca, k_neighbors)

    y_ens = alpha * y_knn + (1 - alpha) * y_global

    price_now = float(last_row["Close"])
    price_pred = float(price_now * np.exp(y_ens))

    accion = recomendar(y_ens * 100, theta)

    # Resultado estructurado
    result = {
        "meta": {
            "ticker": ticker,
            "horizon_days": horizon,
            "pca_target": pca_target,
            "theta": theta,
            "k_neighbors": k_neighbors,
            "alpha": alpha,
            "period": period,
        },
        "historical": hist,
        "prediction": {
            "date_base": str(pd.to_datetime(last_row.name).date()),
            "ret_global_pct": float(y_global * 100.0),
            "ret_knn_pct": float(y_knn * 100.0),
            "ret_ens_pct": float(y_ens * 100.0),
            "price_now": price_now,
            "price_pred": price_pred,
            "recommendation": accion,
            "pca_dims_effective": int(n_pca),
            "n_features": int(n_features),
        }
    }

    return result


def format_report(result: dict) -> str:
    """
    Texto listo para logs/email (opcional).
    No afecta la lógica, solo formatea.
    """
    m = result.get("meta", {})
    h = result.get("historical", {})
    p = result.get("prediction", {})

    lines = []
    lines.append("=== RESUMEN HISTÓRICO ===")
    if h.get("hit_rate_mean") is None:
        lines.append("Sin ventanas walk-forward (dataset corto o sin suficiente data).")
    else:
        lines.append(f"Hit-rate prom : {h['hit_rate_mean']:.4f}")
        lines.append(f"MAE retorno   : {h['mae_mean']:.6f}")
        lines.append(f"RMSE retorno  : {h['rmse_mean']:.6f}")
        lines.append(f"PCA dims      : {h['pca_dims']}")
        lines.append(f"Ventanas      : {h['n_windows']}")

    lines.append("")
    lines.append("=== PREDICCIÓN ACTUAL (ENSAMBLE) ===")
    lines.append(f"Activo              : {m.get('ticker')}")
    lines.append(f"Fecha base          : {p.get('date_base')}")
    lines.append(f"Dimensión efectiva  : PCA = {p.get('pca_dims_effective')}")
    lines.append(f"Retorno global      : {p.get('ret_global_pct'):.2f} %")
    lines.append(f"Retorno kNN caótico : {p.get('ret_knn_pct'):.2f} %")
    lines.append(f"Retorno ENSAMBLE    : {p.get('ret_ens_pct'):.2f} %")
    lines.append(f"Precio actual       : {p.get('price_now'):.2f} USD")
    lines.append(f"Precio esperado     : {p.get('price_pred'):.2f} USD")
    lines.append(f"RECOMENDACIÓN       : {p.get('recommendation')}")
    return "\n".join(lines)

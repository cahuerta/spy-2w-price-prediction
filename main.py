import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# Utils
# =========================
def log_return(series: pd.Series) -> pd.Series:
    return np.log(series).diff()


def hurst_rs(x: np.ndarray) -> float:
    """
    Estimador simple Hurst (R/S).
    Retorna np.nan si no alcanza datos.
    """
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

    # Retornos y rangos
    out["ret1"] = log_return(out["Close"])
    out["range"] = np.log(out["High"] / out["Low"])
    out["vol_chg"] = np.log(out["Volume"].replace(0, np.nan)).diff()

    # Embedding de retornos
    for k in range(1, 21):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # Volatilidad realizada
    out["rv_20"] = out["ret1"].rolling(20).std()
    out["rv_60"] = out["ret1"].rolling(60).std()

    # Fractales
    out["hurst_60"] = rolling_hurst(out["ret1"], 60)
    out["hurst_120"] = rolling_hurst(out["ret1"], 120)

    # Target: retorno a 10 días
    out["y_fwd"] = np.log(out["Close"].shift(-horizon) / out["Close"])

    return out.dropna()


# =========================
# Walk-forward training
# =========================
def walk_forward_train_test(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_years: int = 5,
    test_months: int = 6,
):
    df = data.sort_index()

    start = df.index.min()
    end = df.index.max()

    results = []
    preds_all = []

    cur_train_start = start
    cur_train_end = cur_train_start + pd.DateOffset(years=train_years)
    cur_test_end = cur_train_end + pd.DateOffset(months=test_months)

    # 🔥 PIPELINE CON PCA (50 DIMENSIONES)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=50, random_state=42)),
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
            "train_start": train.index.min().date(),
            "train_end": train.index.max().date(),
            "test_start": test.index.min().date(),
            "test_end": test.index.max().date(),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "hit_rate": float((np.sign(y_pred) == np.sign(y_test)).mean()),
            "n_test": len(test)
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
# Main
# =========================
def main():
    ticker = "SPY"
    horizon = 10

    print(f"Descargando {ticker}...")
    raw = yf.download(ticker, period="max", progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)

    feat = make_features(df, horizon=horizon)

    feature_cols = [
        "range", "vol_chg", "rv_20", "rv_60", "hurst_60", "hurst_120",
        *[f"ret_lag_{k}" for k in range(1, 21)]
    ]

    res_df, pred_df = walk_forward_train_test(
        feat,
        feature_cols=feature_cols,
        target_col="y_fwd"
    )

    print("\n=== Resumen global ===")
    print(f"MAE promedio : {res_df['mae'].mean():.6f}")
    print(f"RMSE promedio: {res_df['rmse'].mean():.6f}")
    print(f"Hit-rate prom: {res_df['hit_rate'].mean():.4f}")
    print(f"Total tests  : {int(res_df['n_test'].sum())}")

    close_aligned = feat.loc[pred_df.index, "Close"]
    price_pred = close_aligned * np.exp(pred_df["y_pred"])
    price_true = close_aligned * np.exp(pred_df["y_true"])

    mae_price = float(np.mean(np.abs(price_true - price_pred)))
    print(f"MAE precio aprox (USD): {mae_price:.4f}")

    res_df.to_csv("walkforward_folds.csv", index=False)
    out = pred_df.copy()
    out["close_t"] = close_aligned
    out["price_pred_h"] = price_pred
    out["price_true_h"] = price_true
    out.to_csv("preds_timeseries.csv")

    print("\nGuardado: walkforward_folds.csv y preds_timeseries.csv")


if __name__ == "__main__":
    main()

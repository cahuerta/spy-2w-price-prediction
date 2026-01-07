import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# Utils
# =========================
def log_return(series: pd.Series) -> pd.Series:
    return np.log(series).diff()

def hurst_rs(x: np.ndarray) -> float:
    """
    Estimador simple Hurst (R/S). No es el más sofisticado, pero sirve para partir.
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

    # H ≈ log(R/S) / log(n)
    return float(np.log(r / s) / np.log(n))

def rolling_hurst(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).apply(lambda a: hurst_rs(a), raw=True)

def make_features(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """
    df debe tener columnas: Open, High, Low, Close, Volume
    horizon: días hábiles hacia adelante para el target
    """
    out = df.copy()

    # Base series
    out["ret1"] = log_return(out["Close"])
    out["range"] = np.log(out["High"] / out["Low"])
    out["vol_chg"] = np.log(out["Volume"].replace(0, np.nan)).diff()

    # Features: lags de retornos (embedding básico)
    for k in range(1, 21):
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # Volatilidad realizada simple (rolling std)
    out["rv_20"] = out["ret1"].rolling(20).std()
    out["rv_60"] = out["ret1"].rolling(60).std()

    # Fractales: Hurst en ventanas
    out["hurst_60"] = rolling_hurst(out["ret1"], 60)
    out["hurst_120"] = rolling_hurst(out["ret1"], 120)

    # Target: retorno a horizonte (10 días hábiles)
    out["y_fwd"] = np.log(out["Close"].shift(-horizon) / out["Close"])

    # Limpieza: elimina filas incompletas
    out = out.dropna()

    return out

def walk_forward_train_test(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_years: int = 5,
    test_months: int = 6,
):
    """
    Walk-forward con ventanas:
    - Entrena con ~train_years años
    - Testea con ~test_months meses
    Avanza y repite hasta terminar

    Esto evita "mirar el futuro".
    """
    df = data.copy()
    df = df.sort_index()

    # Fechas para corte
    start = df.index.min()
    end = df.index.max()

    results = []
    preds_all = []

    cur_train_start = start
    cur_train_end = cur_train_start + pd.DateOffset(years=train_years)
    cur_test_end = cur_train_end + pd.DateOffset(months=test_months)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])

    while cur_test_end <= end:
        train = df[(df.index >= cur_train_start) & (df.index < cur_train_end)]
        test  = df[(df.index >= cur_train_end) & (df.index < cur_test_end)]

        if len(train) < 300 or len(test) < 30:
            # Si queda poco, corta
            break

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test  = test[feature_cols].values
        y_test  = test[target_col].values

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold = {
            "train_start": train.index.min().date(),
            "train_end": train.index.max().date(),
            "test_start": test.index.min().date(),
            "test_end": test.index.max().date(),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "hit_rate": float((np.sign(y_pred) == np.sign(y_test)).mean()),
            "n_test": len(test)
        }
        results.append(fold)

        preds_fold = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred,
        }, index=test.index)
        preds_all.append(preds_fold)

        # Avanza ventana: “rolling origin”
        cur_train_start = cur_train_start + pd.DateOffset(months=test_months)
        cur_train_end = cur_train_start + pd.DateOffset(years=train_years)
        cur_test_end = cur_train_end + pd.DateOffset(months=test_months)

    res_df = pd.DataFrame(results)
    pred_df = pd.concat(preds_all).sort_index() if preds_all else pd.DataFrame()

    return res_df, pred_df


def main():
    ticker = "SPY"
    horizon = 10  # ~2 semanas (10 días hábiles)

    print(f"Descargando {ticker}...")
    raw = yf.download(ticker, period="max", auto_adjust=False, progress=False)

    if raw.empty:
        raise RuntimeError("No se pudo descargar data (revisa internet / ticker).")

    # yfinance a veces trae MultiIndex; normalizamos
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)

    feat = make_features(df, horizon=horizon)

    # Define features (sin target)
    feature_cols = [
        "range", "vol_chg", "rv_20", "rv_60", "hurst_60", "hurst_120",
        *[f"ret_lag_{k}" for k in range(1, 21)]
    ]

    # Walk-forward
    res_df, pred_df = walk_forward_train_test(
        feat,
        feature_cols=feature_cols,
        target_col="y_fwd",
        train_years=5,
        test_months=6
    )

    print("\n=== Resultados por fold (walk-forward) ===")
    if res_df.empty:
        print("No se generaron folds (quizás pocos datos luego de features).")
        return

    print(res_df.to_string(index=False))

    print("\n=== Resumen global ===")
    print(f"MAE promedio : {res_df['mae'].mean():.6f}")
    print(f"RMSE promedio: {res_df['rmse'].mean():.6f}")
    print(f"Hit-rate prom: {res_df['hit_rate'].mean():.4f}")
    print(f"Total tests  : {int(res_df['n_test'].sum())}")

    # Convertir predicción a precio esperado del horizonte (opcional)
    # P_{t+h} = P_t * exp(y_pred)
    # Para eso necesitamos el Close del mismo índice
    close_aligned = feat.loc[pred_df.index, "Close"]
    price_pred = close_aligned * np.exp(pred_df["y_pred"])
    price_true = close_aligned * np.exp(pred_df["y_true"])

    # Errores de precio (aprox) solo como referencia
    mae_price = float(np.mean(np.abs(price_true - price_pred)))
    print(f"MAE precio aprox (USD): {mae_price:.4f}")

    # Guarda outputs
    res_df.to_csv("walkforward_folds.csv", index=False)
    pred_out = pred_df.copy()
    pred_out["close_t"] = close_aligned
    pred_out["price_pred_h"] = price_pred
    pred_out["price_true_h"] = price_true
    pred_out.to_csv("preds_timeseries.csv")
    print("\nGuardado: walkforward_folds.csv y preds_timeseries.csv")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import os
import json
import sys
from datetime import datetime
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURACIÓN H7 - PREDICTOR SEMANAL COMPLETO
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 7
ALPHA_H7 = 1.0
PCA_COMPONENTS = 12

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)


# ======================================================
# HURST EXPONENT
# ======================================================

def get_hurst_exponent(ts, max_lag=20):

    ts = np.array(ts)

    lags = range(2, min(max_lag, len(ts)//4))

    if len(lags) < 2:
        return 0.5

    tau = [
        np.sqrt(np.std(ts[lag:] - ts[:-lag]))
        for lag in lags
    ]

    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    hurst = poly[0] * 2.0

    return hurst


# ======================================================
# FEATURES
# ======================================================

def make_features_h7(df: pd.DataFrame):

    out = df.copy()

    # BASE
    out["ret1"] = np.log(out["Close"]).diff()

    out["range"] = np.log(
        out["High"] / out["Low"]
    ).clip(0, 0.35)

    # LAGS
    for k in [1, 2, 3, 5, 10, 20, 30]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # VOLATILIDAD 30d
    past = out["ret1"].shift(1)

    out["rv_30"] = (
        past.rolling(30)
        .std()
        .clip(1e-6)
    ) * np.sqrt(252/30)

    # HURST
    out["hurst_60"] = (
        out["ret1"]
        .rolling(60)
        .apply(get_hurst_exponent, raw=True)
        .fillna(0.5)
    )

    out["hurst_60"] = out["hurst_60"].clip(0.2, 0.9)

    # DISTANCIA SMA50
    sma50 = out["Close"].rolling(50).mean()

    out["dist_sma50"] = np.log(
        out["Close"] / sma50
    ).clip(-0.3, 0.3)

    # MONEY FLOW NORMALIZADO
    out["money_flow"] = (
        out["Volume"] * out["ret1"]
    ) / out["Volume"].rolling(20).mean().replace(0,1)

    out["money_flow"] = out["money_flow"].clip(-5,5)

    return out


# ======================================================
# PREDICTOR
# ======================================================

def run_predictor_h7(ticker: str):

    raw = get_price_history(
        ticker=ticker,
        period="2y",
        interval="1d"
    )

    if raw is None or len(raw) < 260:
        return None

    df = raw[["Open","High","Low","Close","Volume"]].dropna()

    if len(df) < 250:
        return None

    feat = make_features_h7(df)

    feature_cols = [
        "range",
        "rv_30",
        "hurst_60",
        "dist_sma50",
        "money_flow",
        "ret_lag_1",
        "ret_lag_3",
        "ret_lag_5",
        "ret_lag_10",
        "ret_lag_20",
        "ret_lag_30"
    ]

    # TARGET
    feat["y_fwd"] = np.log(
        feat["Close"].shift(-HORIZON) /
        feat["Close"]
    )

    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 200:
        return None

    # PIPELINE
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("ridge", Ridge(alpha=ALPHA_H7, random_state=42))
    ])

    model.fit(clean[feature_cols], clean["y_fwd"])

    # PREDICCIÓN
    last_features = (
        feat[feature_cols]
        .iloc[-1:]
        .ffill()
        .bfill()
    )

    y_pred_log = model.predict(last_features)[0]

    # CLIP SEMANAL
    y_pred_log = np.clip(y_pred_log, -0.15, 0.15)

    price_today = float(feat["Close"].iloc[-1])

    price_7d = price_today * np.exp(y_pred_log)

    # MÉTRICAS
    ridge_model = model.named_steps["ridge"]

    confidence = 1 / (1 + np.std(ridge_model.coef_))

    r2_train = ridge_model.score(
        clean[feature_cols],
        clean["y_fwd"]
    )

    hurst_regime = float(feat["hurst_60"].iloc[-1])

    regime = (
        "TREND" if hurst_regime > 0.55
        else "MEAN-REV" if hurst_regime < 0.45
        else "RANDOM"
    )

    return {
        "ticker": ticker,
        "predictor": "H7_v2.1_Persistence",
        "horizon_days": HORIZON,
        "date_today": datetime.now().strftime("%Y-%m-%d"),
        "price_today": round(price_today, 4),
        "price_7d": round(price_7d, 4),
        "return_7d_pct": round(y_pred_log * 100, 4),
        "confidence": round(confidence, 3),
        "r2_train": round(r2_train, 4),
        "hurst_exponent": round(hurst_regime, 3),
        "hurst_regime": regime,
        "samples": len(clean),
        "model": {
            "alpha": ALPHA_H7,
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

    print(f"🚀 H7 v2.1 Predictor Semanal Completo → {ticker}")

    result = run_predictor_h7(ticker)

    if result:

        filename = f"{ticker}_H7_7d_v2.json"

        path = os.path.join(DATA_OUTPUT_DIR, filename)

        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print("\n✅ H7 v2.1 - SEMANAL PERSISTENCIA")
        print(f"📊 {ticker}:           ${result['price_today']:>8.2f}")
        print(f"📊 Predicción 7d:      ${result['price_7d']:>8.2f}")
        print(f"📈 Retorno:            {result['return_7d_pct']:>6.2f}%")
        print(f"🔥 Confianza:          {result['confidence']:>5.1f}")
        print(f"📊 Hurst: {result['hurst_exponent']} ({result['hurst_regime']})")
        print(f"📁 {path}")

    else:

        print("❌ Datos insuficientes H7")

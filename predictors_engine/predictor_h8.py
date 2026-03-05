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
# CONFIGURACIÓN H8 - FUERZA DE TENDENCIA
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 8
ALPHA_H8 = 1.25
PCA_COMPONENTS = 10

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)


# ======================================================
# FEATURES
# ======================================================

def make_features_h8(df: pd.DataFrame):

    out = df.copy()

    # BASE
    out["ret1"] = np.log(out["Close"]).diff()

    out["range"] = np.log(
        out["High"] / out["Low"]
    ).clip(0, 0.40)

    # LAGS
    for k in [1, 3, 5, 10, 20, 30]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # VOLATILIDAD
    past = out["ret1"].shift(1)

    out["rv_30"] = (
        past.rolling(30)
        .std()
        .clip(1e-6)
    ) * np.sqrt(252/30)

    # ADX
    high_diff = out["High"].diff()
    low_diff = out["Low"].diff()

    plus_dm = np.where(
        (high_diff > low_diff) & (high_diff > 0),
        high_diff,
        0
    )

    minus_dm = np.where(
        (low_diff > high_diff) & (low_diff > 0),
        low_diff,
        0
    )

    tr1 = out["High"] - out["Low"]
    tr2 = np.abs(out["High"] - out["Close"].shift(1))
    tr3 = np.abs(out["Low"] - out["Close"].shift(1))

    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    alpha = 1/14

    plus_dm = pd.Series(plus_dm, index=out.index)
    minus_dm = pd.Series(minus_dm, index=out.index)
    true_range = pd.Series(true_range, index=out.index)

    plus_di = (
        plus_dm.ewm(alpha=alpha).mean() /
        true_range.ewm(alpha=alpha).mean().replace(0,1)
    ) * 100

    minus_di = (
        minus_dm.ewm(alpha=alpha).mean() /
        true_range.ewm(alpha=alpha).mean().replace(0,1)
    ) * 100

    dx = (
        np.abs(plus_di - minus_di) /
        (plus_di + minus_di).replace(0,1)
    ) * 100

    out["adx_14"] = dx.ewm(alpha=alpha).mean().fillna(25)
    out["adx_14"] = out["adx_14"].clip(0,100)

    # EMA50 SLOPE
    ema50 = out["Close"].ewm(span=50).mean()

    out["ema50_slope"] = (
        (ema50 - ema50.shift(5)) /
        ema50.shift(5)
    ).clip(-0.1,0.1)

    # FUERZA DE TENDENCIA
    out["trend_strength"] = out["adx_14"] * np.sign(out["ema50_slope"])

    return out


# ======================================================
# PREDICTOR
# ======================================================

def run_predictor_h8(ticker: str):

    raw = get_price_history(
        ticker=ticker,
        period="2y",
        interval="1d"
    )

    if raw is None or len(raw) < 280:
        return None

    df = raw[["Open","High","Low","Close","Volume"]].dropna()

    if len(df) < 270:
        return None

    feat = make_features_h8(df)

    feature_cols = [
        "range",
        "rv_30",
        "adx_14",
        "ema50_slope",
        "trend_strength",
        "ret_lag_1",
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

    if len(clean) < 230:
        return None

    # PIPELINE
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("ridge", Ridge(alpha=ALPHA_H8, random_state=42))
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

    # CLIP
    y_pred_log = np.clip(y_pred_log, -0.15, 0.15)

    price_today = float(feat["Close"].iloc[-1])

    price_8d = price_today * np.exp(y_pred_log)

    # MÉTRICAS
    ridge_model = model.named_steps["ridge"]

    confidence = 1 / (1 + np.std(ridge_model.coef_))

    r2_train = ridge_model.score(
        clean[feature_cols],
        clean["y_fwd"]
    )

    adx_current = float(feat["adx_14"].iloc[-1])

    trend_regime = (
        "STRONG" if adx_current > 35
        else "WEAK" if adx_current < 20
        else "NEUTRAL"
    )

    return {
        "ticker": ticker,
        "predictor": "H8_v2.1_TrendStrength",
        "horizon_days": HORIZON,
        "date_today": datetime.now().strftime("%Y-%m-%d"),
        "price_today": round(price_today, 4),
        "price_8d": round(price_8d, 4),
        "return_8d_pct": round(y_pred_log * 100, 4),
        "confidence": round(confidence, 3),
        "r2_train": round(r2_train, 4),
        "adx_14": round(adx_current, 1),
        "trend_regime": trend_regime,
        "samples": len(clean),
        "model": {
            "alpha": ALPHA_H8,
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

    print(f"🚀 H8 v2.1 Fuerza de Tendencia (8d) → {ticker}")

    result = run_predictor_h8(ticker)

    if result:

        filename = f"{ticker}_H8_8d_v2.json"

        path = os.path.join(DATA_OUTPUT_DIR, filename)

        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print("\n✅ H8 v2.1 - TREND STRENGTH")
        print(f"📊 {ticker}:           ${result['price_today']:>8.2f}")
        print(f"📊 Predicción 8d:      ${result['price_8d']:>8.2f}")
        print(f"📈 Retorno:            {result['return_8d_pct']:>6.2f}%")
        print(f"🔥 ADX: {result['adx_14']} ({result['trend_regime']})")
        print(f"📁 {path}")

    else:

        print("❌ Datos insuficientes H8")

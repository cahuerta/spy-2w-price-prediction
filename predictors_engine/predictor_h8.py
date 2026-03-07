import numpy as np
import pandas as pd
import os
import json
import sys
from datetime import datetime

# ======================================================
# RUTAS
# ======================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 8
ALPHA_H8 = 1.25
MAX_PCA_COMPONENTS = 10

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)

# ======================================================
# FEATURE ENGINEERING H8
# ======================================================
def make_features_h8(df: pd.DataFrame):

    out = df.copy()

    # --------------------------------------------------
    # BASE RETURNS
    # --------------------------------------------------
    out["ret1"] = np.log(out["Close"]).diff()

    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.40)

    # --------------------------------------------------
    # LAGS
    # --------------------------------------------------
    for k in [1, 3, 5, 10, 20, 30]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # --------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------
    past_ret = out["ret1"].shift(1)

    out["rv_30"] = (
        past_ret
        .rolling(30)
        .std()
        .clip(1e-6)
        * np.sqrt(252)
    )

    # --------------------------------------------------
    # ADX SIN LEAKAGE
    # --------------------------------------------------
    h_past = out["High"].shift(1)
    l_past = out["Low"].shift(1)
    c_prev = out["Close"].shift(2)

    up_move = h_past - h_past.shift(1)
    down_move = l_past.shift(1) - l_past

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0
    )

    tr = np.maximum(
        h_past - l_past,
        np.maximum(
            np.abs(h_past - c_prev),
            np.abs(l_past - c_prev)
        )
    )

    alpha = 1 / 14

    plus_dm_s = pd.Series(plus_dm, index=out.index)
    minus_dm_s = pd.Series(minus_dm, index=out.index)
    tr_s = pd.Series(tr, index=out.index)

    plus_di = 100 * (
        plus_dm_s.ewm(alpha=alpha).mean()
        /
        tr_s.ewm(alpha=alpha).mean().replace(0, 1)
    )

    minus_di = 100 * (
        minus_dm_s.ewm(alpha=alpha).mean()
        /
        tr_s.ewm(alpha=alpha).mean().replace(0, 1)
    )

    dx = 100 * (
        np.abs(plus_di - minus_di)
        /
        (plus_di + minus_di).replace(0, 1)
    )

    out["adx_14"] = (
        dx
        .ewm(alpha=alpha)
        .mean()
        .clip(0, 100)
    )

    # --------------------------------------------------
    # EMA SLOPE
    # --------------------------------------------------
    ema50 = out["Close"].shift(1).ewm(span=50).mean()

    out["ema50_slope"] = (
        (ema50 - ema50.shift(5))
        /
        ema50.shift(5)
    ).clip(-0.1, 0.1)

    # --------------------------------------------------
    # TREND STRENGTH
    # --------------------------------------------------
    out["trend_strength"] = out["adx_14"] * np.sign(out["ema50_slope"])

    return out


# ======================================================
# PREDICTOR H8
# ======================================================
def run_predictor_h8(ticker: str):

    raw = get_price_history(
        ticker=ticker,
        period="2y",
        interval="1d"
    )

    if raw is None or len(raw) < 280:
        return None

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

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

    feat["y_fwd"] = np.log(
        feat["Close"].shift(-HORIZON)
        /
        feat["Close"]
    )

    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 230:
        return None

    X = clean[feature_cols].values
    y = clean["y_fwd"].values

    # --------------------------------------------------
    # PCA DINÁMICO
    # --------------------------------------------------
    n_samples, n_features = X.shape

    dynamic_pca = max(
        1,
        min(MAX_PCA_COMPONENTS, n_features, n_samples - 1)
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=dynamic_pca, random_state=42)),
        ("ridge", Ridge(alpha=ALPHA_H8, random_state=42))
    ])

    model.fit(X, y)

    # --------------------------------------------------
    # PREDICCIÓN
    # --------------------------------------------------
    last_features = feat[feature_cols].iloc[-1:]

    if last_features.isna().any().any():
        return None

    y_pred_log = float(model.predict(last_features)[0])

    y_pred_log = np.clip(y_pred_log, -0.15, 0.15)

    price_today = float(feat["Close"].iloc[-1])

    price_8d = price_today * np.exp(y_pred_log)

    # --------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------
    adx_now = float(feat["adx_14"].iloc[-1])

    if adx_now > 35:
        regime = "STRONG"
    elif adx_now < 20:
        regime = "WEAK"
    else:
        regime = "NEUTRAL"

    # confidence robusta
    try:
        coef = model.named_steps["ridge"].coef_
        coef_std = float(np.std(coef))
    except Exception:
        coef_std = 1.0

    confidence = 1 / (1 + coef_std)

    return {
        "ticker": ticker,
        "predictor": "H8_v2.3_TrendStrength",
        "horizon_days": HORIZON,
        "price_today": round(price_today, 4),
        "price_8d": round(price_8d, 4),
        "return_8d_pct": round(y_pred_log * 100, 4),
        "adx": round(adx_now, 1),
        "regime": regime,
        "confidence": round(float(confidence), 3),
        "timestamp": datetime.utcnow().isoformat()
    }


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"

    print(f"🚀 Iniciando H8 Trend Strength → {ticker}")

    result = run_predictor_h8(ticker)

    if result:

        filename = f"{ticker}_H8_8d_v2.json"

        path = os.path.join(DATA_OUTPUT_DIR, filename)

        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        print("\n✅ H8 COMPLETADO")
        print("-" * 40)
        print(
            f"📊 {ticker}: ${result['price_today']:,.2f} → "
            f"${result['price_8d']:,.2f}"
        )
        print(
            f"📈 Retorno: {result['return_8d_pct']}% "
            f"| ADX: {result['adx']} ({result['regime']})"
        )
        print(f"🔥 Confianza: {result['confidence']}")
        print("-" * 40)

    else:
        print("⚠️ H8 no generó resultado")

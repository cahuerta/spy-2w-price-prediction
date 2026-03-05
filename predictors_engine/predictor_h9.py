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
# CONFIGURACIÓN H9 - CICLO DE CAPITAL
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 9
ALPHA_H9 = 1.5
PCA_COMPONENTS = 12

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)


# ======================================================
# FEATURES
# ======================================================

def make_features_h9(df: pd.DataFrame):

    out = df.copy()

    # BASE
    out["ret1"] = np.log(out["Close"]).diff()

    out["range"] = np.log(
        out["High"] / out["Low"]
    ).clip(0, 0.40)

    # LAGS
    for k in [1, 3, 5, 10, 20, 30]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # ==================================================
    # MONEY FLOW INDEX
    # ==================================================

    typical_price = (
        out["High"] +
        out["Low"] +
        out["Close"]
    ) / 3

    raw_money_flow = typical_price * out["Volume"]

    pos_flow = raw_money_flow.where(
        typical_price > typical_price.shift(1), 0
    )

    neg_flow = raw_money_flow.where(
        typical_price < typical_price.shift(1), 0
    )

    pos_sum = pos_flow.rolling(14).sum()
    neg_sum = neg_flow.rolling(14).sum().replace(0,1)

    money_ratio = pos_sum / neg_sum

    out["mfi_14"] = 100 - (100 / (1 + money_ratio))

    out["mfi_14"] = out["mfi_14"].clip(5,95)

    # ==================================================
    # BOLLINGER DISTANCE
    # ==================================================

    sma20 = out["Close"].rolling(20).mean()

    std20 = out["Close"].rolling(20).std().clip(1e-6)

    out["bb_dist"] = (
        (out["Close"] - sma20) /
        (2 * std20)
    ).clip(-3,3)

    # ==================================================
    # CAPITAL FLOW
    # ==================================================

    out["capital_flow"] = (
        (out["mfi_14"] - 50) / 50
    ).clip(-1,1)

    return out


# ======================================================
# PREDICTOR
# ======================================================

def run_predictor_h9(ticker: str):

    raw = get_price_history(
        ticker=ticker,
        period="2y",
        interval="1d"
    )

    if raw is None or len(raw) < 300:
        return None

    df = raw[["Open","High","Low","Close","Volume"]].dropna()

    if len(df) < 290:
        return None

    feat = make_features_h9(df)

    feature_cols = [
        "range",
        "mfi_14",
        "bb_dist",
        "capital_flow",
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

    if len(clean) < 250:
        return None

    # PIPELINE
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("ridge", Ridge(alpha=ALPHA_H9, random_state=42))
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
    y_pred_log = np.clip(y_pred_log, -0.18, 0.18)

    price_today = float(feat["Close"].iloc[-1])

    price_9d = price_today * np.exp(y_pred_log)

    # MÉTRICAS
    ridge_model = model.named_steps["ridge"]

    confidence = 1 / (1 + np.std(ridge_model.coef_))

    r2_train = ridge_model.score(
        clean[feature_cols],
        clean["y_fwd"]
    )

    mfi_current = float(feat["mfi_14"].iloc[-1])
    bb_pos = float(feat["bb_dist"].iloc[-1])

    mfi_regime = (
        "OVERBOUGHT" if mfi_current > 80
        else "OVERSOLD" if mfi_current < 20
        else "NEUTRAL"
    )

    bb_regime = (
        "UPPER_BAND" if bb_pos > 2
        else "LOWER_BAND" if bb_pos < -2
        else "MIDDLE"
    )

    return {
        "ticker": ticker,
        "predictor": "H9_v2.1_CapitalCycle",
        "horizon_days": HORIZON,
        "date_today": datetime.now().strftime("%Y-%m-%d"),
        "price_today": round(price_today, 4),
        "price_9d": round(price_9d, 4),
        "return_9d_pct": round(y_pred_log * 100, 4),
        "confidence": round(confidence, 3),
        "r2_train": round(r2_train, 4),
        "mfi_14": round(mfi_current, 1),
        "bb_position": round(bb_pos, 2),
        "mfi_regime": mfi_regime,
        "bb_regime": bb_regime,
        "samples": len(clean),
        "model": {
            "alpha": ALPHA_H9,
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

    print(f"🚀 H9 v2.1 Capital Cycle (9d) → {ticker}")

    result = run_predictor_h9(ticker)

    if result:

        filename = f"{ticker}_H9_9d_v2.json"

        path = os.path.join(DATA_OUTPUT_DIR, filename)

        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print("\n✅ H9 v2.1 - CAPITAL CYCLE")
        print(f"📊 {ticker}:           ${result['price_today']:>8.2f}")
        print(f"📊 Predicción 9d:      ${result['price_9d']:>8.2f}")
        print(f"📈 Retorno:            {result['return_9d_pct']:>6.2f}%")
        print(f"💰 MFI: {result['mfi_14']} ({result['mfi_regime']})")
        print(f"📏 BB:  {result['bb_position']} ({result['bb_regime']})")
        print(f"📁 {path}")

    else:
        print("❌ Datos insuficientes H9")

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
# CONFIGURACIÓN H1 - PREDICTOR DÍA 1 (PRODUCCIÓN)
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 1
ALPHA_H1 = 0.08
MAX_PCA_COMPONENTS = 8

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)


# ======================================================
# FEATURE ENGINEERING (Lógica Original Intacta)
# ======================================================

def make_features_h1(df: pd.DataFrame):
    out = df.copy()

    # Retornos base
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.25)

    # Lags críticos
    out["ret_lag_1"] = out["ret1"].shift(1)
    out["ret_lag_2"] = out["ret1"].shift(2)
    out["ret_lag_5"] = out["ret1"].shift(5)

    # Volatilidad (Respetando bfill original)
    past_returns = out["ret1"].shift(1)
    out["rv_short"] = past_returns.rolling(3).std().bfill()
    out["vol_ema"] = past_returns.ewm(span=5).std().bfill()

    # Momentum
    out["mom_3d"] = past_returns.rolling(3).sum()
    out["mom_vol"] = out["mom_3d"] / out["rv_short"].replace(0, 0.01)

    # RSI rápido
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(3).mean()
    loss = (-delta.clip(upper=0)).rolling(3).mean()
    rs = gain / loss.replace(0, 1.0)
    out["rsi_fast"] = 100 - (100 / (1 + rs))

    # Fuerza relativa y tendencia
    out["price_strength"] = out["ret1"] / out["rv_short"].replace(0, 0.01)
    out["trend_5"] = out["Close"].pct_change(5).fillna(0)

    return out


# ======================================================
# PREDICTOR
# ======================================================

def run_predictor_h1(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    if raw is None or len(raw) < 150:
        return None

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if len(df) < 140:
        return None

    feat = make_features_h1(df)
    
    feature_cols = [
        "range", "ret_lag_1", "ret_lag_2", "ret_lag_5",
        "rv_short", "vol_ema", "mom_vol", "rsi_fast",
        "price_strength", "trend_5"
    ]

    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    
    # Limpieza estricta original
    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 130:
        return None

    # ======================================================
    # PCA DINÁMICO SEGURO (El único cambio solicitado)
    # ======================================================
    n_samples = len(clean)
    n_features = len(feature_cols)

    dynamic_pca = int(min(MAX_PCA_COMPONENTS, n_samples - 1, n_features))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=dynamic_pca)),
        ("ridge", Ridge(alpha=ALPHA_H1))
    ])

    model.fit(clean[feature_cols], clean["y_fwd"])

    # ======================================================
    # PREDICCIÓN Y MÉTRICAS (Respetando tu estructura)
    # ======================================================
    last_features = feat[feature_cols].iloc[-1:]

    if last_features.isna().any().any():
        return None

    y_pred_log = model.predict(last_features)[0]
    price_today = float(feat["Close"].iloc[-1])
    price_tomorrow = price_today * np.exp(y_pred_log)

    r2_train = model.score(clean[feature_cols], clean["y_fwd"])
    ridge_model = model.named_steps["ridge"]
    coef_std = float(np.std(ridge_model.coef_))
    confidence = float(1 / (1 + coef_std))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "ticker": ticker,
        "predictor": "H1_v2.1",
        "horizon_days": HORIZON,
        "date_today": datetime.now().strftime("%Y-%m-%d"),
        "price_today": round(price_today, 4),
        "price_tomorrow": round(price_tomorrow, 4),
        "return_pct": round(y_pred_log * 100, 4),
        "confidence": round(confidence, 3),
        "r2_train": round(r2_train, 4),
        "samples": len(clean),
        "model": {
            "alpha": ALPHA_H1,
            "pca_components": dynamic_pca,
            "features_count": n_features
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    print(f"🚀 H1 v2.1 → {ticker}")

    result = run_predictor_h1(ticker)

    if result:
        filename = f"{ticker}_H1_dia1_v2.json"
        path = os.path.join(DATA_OUTPUT_DIR, filename)
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print("\n✅ H1 v2.1 - PRODUCCIÓN")
        print(f"📊 {ticker}:      $ {result['price_today']:>8.2f}")
        print(f"📊 Predicción:   $ {result['price_tomorrow']:>8.2f}")
        print(f"📈 Retorno:        {result['return_pct']:>6.2f}%")
        print(f"🔥 Confianza:      {result['confidence']:>5.2f}")
        print(f"📁 {path}")
    else:
        print("❌ Datos insuficientes para entrenamiento")

import numpy as np
import pandas as pd
import os
import json
import sys
from datetime import datetime

# Configuración de rutas
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURACIÓN H6 - ESTRUCTURAL
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 6
ALPHA_H6 = 0.8
MAX_PCA_COMPONENTS = 12

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)

# ======================================================
# FEATURE ENGINEERING H6
# ======================================================
def make_features_h6(df: pd.DataFrame):
    out = df.copy()

    # Base y Retornos
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.35)

    # Lags Estructurales
    for k in [1, 2, 3, 5, 10, 20]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    past_ret = out["ret1"].shift(1)
    
    # Volatilidad Anualizada (Ventana de 25 días)
    out["rv_25"] = past_ret.rolling(25).std().clip(1e-6) * np.sqrt(252)

    # RSI Z-SCORE (Sin Leakage)
    close_past = out["Close"].shift(1)
    delta = close_past.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    
    rsi = 100 - (100 / (1 + (gain / loss.replace(0, 1.0))))
    
    # Normalización estadística del RSI (ventana de 3 meses de trading aprox)
    out["rsi_z"] = (
        (rsi - rsi.rolling(60).mean()) / 
        rsi.rolling(60).std().replace(0, 1.0)
    ).clip(-3, 3)

    # EMA SLOPE (Inercia de la media móvil)
    ema_20 = close_past.ewm(span=20).mean()
    out["ema_slope"] = ((ema_20 - ema_20.shift(5)) / ema_20.shift(5)).clip(-0.05, 0.05)

    # TREND SLOPE (Regresión lineal sobre 25 días)
    def calc_slope(x):
        if len(x) < 25: return 0.0
        return np.polyfit(range(25), x, 1)[0] / (x.mean() + 1e-9)

    out["slope_25"] = close_past.rolling(25).apply(calc_slope, raw=True)

    return out

# ======================================================
# PREDICTOR H6
# ======================================================
def run_predictor_h6(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    if raw is None or len(raw) < 240: # Necesitamos margen para el RSI Z-Score (60d)
        return None

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h6(df)

    feature_cols = [
        "range", "rv_25", "rsi_z", "ema_slope", "slope_25",
        "ret_lag_1", "ret_lag_3", "ret_lag_5", "ret_lag_10", "ret_lag_20"
    ]

    # Target T+6
    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 190:
        return None

    X = clean[feature_cols].values
    y = clean["y_fwd"].values

    # PCA Dinámico
    n_samples, n_features = X.shape
    dynamic_pca = max(1, min(MAX_PCA_COMPONENTS, n_features, n_samples - 1))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=dynamic_pca, random_state=42)),
        ("ridge", Ridge(alpha=ALPHA_H6, random_state=42))
    ])

    model.fit(X, y)

    # Predicción final
    last_features = feat[feature_cols].iloc[-1:]
    if last_features.isna().any().any():
        return None

    y_pred_log = float(model.predict(last_features)[0])
    y_pred_log = np.clip(y_pred_log, -0.12, 0.12) # Clip realista para 6 días

    price_today = float(feat["Close"].iloc[-1])
    price_6d = price_today * np.exp(y_pred_log)

    # Métricas de salida
    ridge_model = model.named_steps["ridge"]
    confidence = 1 / (1 + np.std(ridge_model.coef_))

    return {
        "ticker": ticker,
        "predictor": "H6",
        "horizon_days": HORIZON,
        "date_today": datetime.now().strftime("%Y-%m-%d"),
        "price_today": round(price_today, 4),
        "price_pred": round(price_6d, 4),
        "return_pct": round(y_pred_log * 100, 4),
        "confidence": round(float(confidence), 3),
        "samples": len(clean),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    print(f"🚀 Ejecutando H6 Structural (6d) para {ticker}...")

    result = run_predictor_h6(ticker)

    if result:
        filename = f"{ticker}_H6_6d_v2.json"
        path = os.path.join(DATA_OUTPUT_DIR, filename)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n✅ H6 COMPLETADO")
        print("-" * 35)
        print(f"📊 {ticker}: ${result['price_today']:,.2f} → Est. 6d: ${result['price_6d']:,.2f}")
        print(f"📈 Retorno Esperado: {result['return_6d_pct']}%")
        print(f"🔥 Confianza del Modelo: {result['confidence']}")
        print("-" * 35)
    else:
        print("❌ Datos insuficientes.")

import numpy as np
import pandas as pd
import os
import json
import sys
from datetime import datetime

# Configuración de Rutas
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURACIÓN H2 v2.1 (PRODUCCIÓN)
# ======================================================

DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 2               # Predicción a 48h (T+2)
ALPHA_H2 = 0.15           # Regularización Ridge
MAX_PCA_COMPONENTS = 10   # Límite superior de PCA

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)

# ======================================================
# FEATURE ENGINEERING (COHERENTE Y SIN LEAKAGE)
# ======================================================

def make_features_h2(df: pd.DataFrame):
    """
    Genera indicadores técnicos evitando el look-ahead bias.
    """
    out = df.copy()

    # Retornos logarítmicos y Rango
    out["ret1"] = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.25)

    # Lags de retornos (Información pasada)
    for k in [1, 2, 3, 5, 10]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    # Volatilidad (Rolling Std sobre retornos pasados)
    past_ret = out["ret1"].shift(1)
    out["rv_10"] = past_ret.rolling(10).std().clip(1e-6)

    # Momentum y Ratio
    out["mom_2d"] = past_ret.rolling(2).sum()
    out["mom_5d"] = past_ret.rolling(5).sum()
    out["mom_ratio"] = (out["mom_5d"] / out["rv_10"].replace(0, 0.01)).clip(-5, 5)

    # RSI (Relative Strength Index)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(9).mean()
    loss = (-delta.clip(upper=0)).rolling(9).mean()
    rs = gain / loss.replace(0, 1.0)
    out["rsi_9"] = 100 - (100 / (1 + rs))

    # Mean Reversion (Distancia a la Media Móvil)
    out["ma_10"] = out["Close"].rolling(10).mean()
    out["dist_ma10"] = np.log(out["Close"] / out["ma_10"]).clip(-0.2, 0.2)

    # Tendencia porcentual
    out["trend_10"] = out["Close"].pct_change(10).fillna(0)

    return out

# ======================================================
# PREDICTOR H2
# ======================================================

def run_predictor_h2(ticker: str):
    # 1. Obtención de datos
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    if raw is None or len(raw) < 160:
        return None

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h2(df)

    feature_cols = [
        "range", "rv_10", "mom_ratio", "rsi_9", "dist_ma10",
        "mom_2d", "mom_5d", "ret_lag_1", "ret_lag_2",
        "ret_lag_3", "ret_lag_5", "ret_lag_10", "trend_10"
    ]

    # 2. Definición del Target (Log-Return a futuro)
    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])

    # Limpieza de NaNs generados por lags y rolling windows
    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 140:
        return None

    # 3. PCA DINÁMICO SEGURO (Tu lógica integrada)
    # Calculamos n_pca basándonos en el tamaño real del dataset limpio
    X_train = clean[feature_cols].values
    y_train = clean["y_fwd"].values
    
    n_samples, n_features = X_train.shape
    # Evita pedir más de lo que hay (samples-1 o features)
    dynamic_n_pca = max(1, min(MAX_PCA_COMPONENTS, n_features, n_samples - 1))

    # 4. Pipeline de Entrenamiento
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=dynamic_n_pca, random_state=42)),
        ("ridge", Ridge(alpha=ALPHA_H2))
    ])

    model.fit(X_train, y_train)

    # 5. Predicción con la última fila disponible
    last_row = feat[feature_cols].iloc[-1:]
    
    if last_row.isna().any().any():
        return None

    y_pred_log = model.predict(last_row)[0]

    # Protección contra anomalías (Max +/- 5% en 2 días)
    y_pred_log = np.clip(y_pred_log, -0.05, 0.05)

    price_today = float(feat["Close"].iloc[-1])
    price_48h = price_today * np.exp(y_pred_log)

    # 6. Métricas de Calidad
    r2_train = model.score(X_train, y_train)
    # Confianza basada en la estabilidad de los coeficientes del Ridge
    ridge_coefs = model.named_steps["ridge"].coef_
    confidence = 1 / (1 + np.std(ridge_coefs))

    return {
        "ticker": ticker,
        "predictor": "H2",
        "horizon_days": HORIZON,
        "date_today": datetime.now().strftime("%Y-%m-%d"),
        "price_today": round(price_today, 4),
        "price_pred": round(price_48h, 4),
        "return_pct": round(y_pred_log * 100, 4),
        "confidence": round(float(confidence), 3),
        "r2_train": round(float(r2_train), 4),
        "samples": len(clean),
        "model_params": {
            "alpha": ALPHA_H2,
            "pca_components_used": dynamic_n_pca,
            "features_input": n_features
        },
        "timestamp": datetime.now().isoformat()
    }

# ======================================================
# EXECUTION INTERFACE
# ======================================================

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    print(f"🚀 Iniciando Predictor H2 (PCA Dinámico) para: {ticker}")

    result = run_predictor_h2(ticker)

    if result:
        filename = f"{ticker}_H2.json"
        path = os.path.join(DATA_OUTPUT_DIR, filename)

        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n✅ PROCESO EXITOSO")
        print("-" * 30)
        print(f"📊 Precio Actual:  ${result['price_today']:,.2f}")
        print(f"🎯 Predicción 48h: ${result['price_48h']:,.2f}")
        print(f"📈 Retorno Est.:   {result['return_48h_pct']}%")
        print(f"🧠 R2 (Training):  {result['r2_train']}")
        print(f"🔒 Confianza:      {result['confidence']}")
        print("-" * 30)
        print(f"📁 Datos guardados en: {path}")
    else:
        print(f"❌ ERROR: No hay datos suficientes para {ticker} o el provider falló.")

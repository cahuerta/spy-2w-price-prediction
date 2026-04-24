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

# ── DARWIN ────────────────────────────────────────────
try:
    from darwin_engine.predictor_genome import load_active_genome
    DARWIN_PREDICTOR = True
except ImportError:
    DARWIN_PREDICTOR = False
# ─────────────────────────────────────────────────────

# ======================================================
# CONFIGURACIÓN H4
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 4
ALPHA_H4 = 0.5
MAX_PCA_COMPONENTS = 10
CLIP_RET = 0.10

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)

# ======================================================
# FEATURE ENGINEERING
# ======================================================
def make_features_h4(df: pd.DataFrame):
    out = df.copy()

    out["ret1"]  = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.30)

    past = out["ret1"].shift(1)
    out["rv_20"] = past.rolling(20).std().clip(1e-6) * np.sqrt(252)

    close_past   = out["Close"].shift(1)
    out["ema_5"]  = close_past.ewm(span=5).mean()
    out["ema_20"] = close_past.ewm(span=20).mean()
    out["trend_diff"] = np.log(out["ema_5"] / out["ema_20"]).clip(-0.15, 0.15)

    out["mom_5d"]  = past.rolling(5).sum()
    out["mom_10d"] = past.rolling(10).sum()
    out["mom_20d"] = past.rolling(20).sum()

    out["atr_14"]   = out["range"].rolling(14).mean().clip(1e-6)
    out["vol_ratio"] = (out["range"] / out["atr_14"]).clip(0, 5)

    delta = out["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, 1)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    low_14  = out["Low"].shift(1).rolling(14).min()
    high_14 = out["High"].shift(1).rolling(14).max()
    out["stoch_k"] = (100 * (out["Close"] - low_14) / (high_14 - low_14).replace(0, 1)).clip(0, 100)

    out["hi_20"]   = out["High"].rolling(20).max()
    out["lo_20"]   = out["Low"].rolling(20).min()
    out["dist_hi"] = np.log(out["hi_20"] / out["Close"]).clip(0, 0.3)
    out["dist_lo"] = np.log(out["Close"] / out["lo_20"]).clip(0, 0.3)

    out["vol_price"] = (np.log(out["Volume"]) * out["ret1"]).clip(-5, 5)

    # Features adicionales para evolución
    for k in [1, 3, 5, 10, 20]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)
    out["rv_10"]   = past.rolling(10).std().clip(1e-6)
    out["vol_chg"] = np.log(out["Volume"].replace(0, np.nan)).diff()

    return out

# ======================================================
# PREDICTOR H4
# ======================================================
def run_predictor_h4(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    # ── DARWIN ────────────────────────────────────────
    _alpha    = ALPHA_H4
    _max_pca  = MAX_PCA_COMPONENTS
    _clip_ret = CLIP_RET
    _feature_override = None

    if DARWIN_PREDICTOR:
        try:
            genome    = load_active_genome(HORIZON)
            _alpha    = genome.model_params.get("alpha_ridge", ALPHA_H4)
            _max_pca  = genome.model_params.get("max_pca",     MAX_PCA_COMPONENTS)
            _clip_ret = genome.model_params.get("clip_ret",    CLIP_RET)
            if genome.data.get("n_evaluations", 0) >= 20:
                _feature_override = genome.features
        except Exception:
            pass
    # ─────────────────────────────────────────────────

    if raw is None or len(raw) < 200:
        return None

    df   = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h4(df)

    _base_features = [
        "range", "rv_20", "trend_diff", "vol_ratio", "rsi_14",
        "stoch_k", "dist_hi", "dist_lo", "mom_5d", "mom_10d",
        "mom_20d", "vol_price"
    ]

    if _feature_override:
        feature_cols = [f for f in _feature_override if f in feat.columns]
        if len(feature_cols) < 4:
            feature_cols = _base_features
    else:
        feature_cols = _base_features

    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 160:
        return None

    X = clean[feature_cols].values
    y = clean["y_fwd"].values

    n_samples, n_features = X.shape
    dynamic_pca = max(1, min(_max_pca, n_features, n_samples - 1))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=dynamic_pca, random_state=42)),
        ("ridge",  Ridge(alpha=_alpha))
    ])
    model.fit(X, y)

    last_features = feat[feature_cols].iloc[-1:]
    if last_features.isna().any().any():
        return None

    y_pred_log  = float(model.predict(last_features)[0])
    y_pred_log  = np.clip(y_pred_log, -_clip_ret, _clip_ret)
    price_today = float(feat["Close"].iloc[-1])
    price_4d    = price_today * np.exp(y_pred_log)

    r2_train   = model.score(X, y)
    confidence = 1 / (1 + np.std(model.named_steps["ridge"].coef_))

    return {
        "ticker":       ticker,
        "predictor":    "H4",
        "horizon_days": HORIZON,
        "date_today":   datetime.now().strftime("%Y-%m-%d"),
        "price_today":  round(price_today, 4),
        "price_pred":   round(price_4d, 4),
        "return_pct":   round(y_pred_log * 100, 4),
        "confidence":   round(float(confidence), 3),
        "r2_train":     round(float(r2_train), 4),
        "samples":      len(clean),
        "model_params": {
            "alpha":          _alpha,
            "pca_components": dynamic_pca,
            "genome_active":  DARWIN_PREDICTOR,
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    print(f"🚀 H4 → {ticker}")
    result = run_predictor_h4(ticker)
    if result:
        path = os.path.join(DATA_OUTPUT_DIR, f"{ticker}_H4.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ H4 | ${result['price_today']:,.2f} → ${result['price_pred']:,.2f} | {result['return_pct']}%")
    else:
        print("❌ Datos insuficientes")
            

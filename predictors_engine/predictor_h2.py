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
# CONFIGURACIÓN H2
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 2
ALPHA_H2 = 0.15
MAX_PCA_COMPONENTS = 10
CLIP_RET = 0.05

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)

# ======================================================
# FEATURE ENGINEERING
# ======================================================
def make_features_h2(df: pd.DataFrame):
    out = df.copy()

    out["ret1"]  = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.25)

    for k in [1, 2, 3, 5, 10]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    past_ret     = out["ret1"].shift(1)
    out["rv_10"] = past_ret.rolling(10).std().clip(1e-6)

    out["mom_2d"]    = past_ret.rolling(2).sum()
    out["mom_5d"]    = past_ret.rolling(5).sum()
    out["mom_ratio"] = (out["mom_5d"] / out["rv_10"].replace(0, 0.01)).clip(-5, 5)

    delta = out["Close"].diff()
    gain  = delta.clip(lower=0).rolling(9).mean()
    loss  = (-delta.clip(upper=0)).rolling(9).mean()
    rs    = gain / loss.replace(0, 1.0)
    out["rsi_9"] = 100 - (100 / (1 + rs))

    out["ma_10"]      = out["Close"].rolling(10).mean()
    out["dist_ma10"]  = np.log(out["Close"] / out["ma_10"]).clip(-0.2, 0.2)
    out["trend_10"]   = out["Close"].pct_change(10).fillna(0)

    # Features adicionales para evolución
    out["ret_lag_20"] = out["ret1"].shift(20)
    out["rv_20"]      = past_ret.rolling(20).std().clip(1e-6)
    out["vol_chg"]    = np.log(out["Volume"].replace(0, np.nan)).diff()

    gain14 = delta.clip(lower=0).rolling(14).mean()
    loss14 = (-delta.clip(upper=0)).rolling(14).mean()
    rs14   = gain14 / loss14.replace(0, 1.0)
    out["rsi_14"] = 100 - (100 / (1 + rs14))

    return out

# ======================================================
# PREDICTOR H2
# ======================================================
def run_predictor_h2(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    # ── DARWIN: cargar genoma activo ──────────────────
    _alpha    = ALPHA_H2
    _max_pca  = MAX_PCA_COMPONENTS
    _clip_ret = CLIP_RET
    _decay    = 0.0               # [SW1] default neutro — sin cambio de comportamiento
    _feature_override = None

    if DARWIN_PREDICTOR:
        try:
            genome    = load_active_genome(HORIZON)
            _alpha    = genome.model_params.get("alpha_ridge",         ALPHA_H2)
            _max_pca  = genome.model_params.get("max_pca",             MAX_PCA_COMPONENTS)
            _clip_ret = genome.model_params.get("clip_ret",            CLIP_RET)
            _decay    = genome.model_params.get("sample_weight_decay", 0.0)  # [SW1]
            if genome.data.get("n_evaluations", 0) >= 20:
                _feature_override = genome.features
        except Exception:
            pass
    # ─────────────────────────────────────────────────

    if raw is None or len(raw) < 160:
        return None

    df   = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h2(df)

    _base_features = [
        "range", "rv_10", "mom_ratio", "rsi_9", "dist_ma10",
        "mom_2d", "mom_5d", "ret_lag_1", "ret_lag_2",
        "ret_lag_3", "ret_lag_5", "ret_lag_10", "trend_10"
    ]

    if _feature_override:
        feature_cols = [f for f in _feature_override if f in feat.columns]
        if len(feature_cols) < 4:
            feature_cols = _base_features
    else:
        feature_cols = _base_features

    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 140:
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

    # [SW1] Sample weights exponenciales — Darwin controla _decay
    # _decay=0.0 → todos los días pesan igual (comportamiento actual, sin cambio)
    # _decay>0.0 → datos recientes pesan más (Darwin lo activa si detecta bias)
    if _decay > 0.0:
        sw = np.exp(np.linspace(-_decay, 0, len(y)))
        model.fit(X, y, ridge__sample_weight=sw)
    else:
        model.fit(X, y)

    last_row = feat[feature_cols].iloc[-1:]
    if last_row.isna().any().any():
        return None

    y_pred_log  = float(model.predict(last_row)[0])
    y_pred_log  = np.clip(y_pred_log, -_clip_ret, _clip_ret)
    price_today = float(feat["Close"].iloc[-1])
    price_2d    = price_today * np.exp(y_pred_log)

    r2_train   = model.score(X, y)
    confidence = 1 / (1 + np.std(model.named_steps["ridge"].coef_))

    return {
        "ticker":       ticker,
        "predictor":    "H2",
        "horizon_days": HORIZON,
        "date_today":   datetime.now().strftime("%Y-%m-%d"),
        "price_today":  round(price_today, 4),
        "price_pred":   round(price_2d, 4),
        "return_pct":   round(y_pred_log * 100, 4),
        "confidence":   round(float(confidence), 3),
        "r2_train":     round(float(r2_train), 4),
        "samples":      len(clean),
        "model_params": {
            "alpha":               _alpha,
            "pca_components_used": dynamic_pca,
            "features_input":      len(feature_cols),
            "genome_active":       DARWIN_PREDICTOR,
            "weight_decay":        _decay,  # [SW1] trazabilidad
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    print(f"🚀 H2 → {ticker}")
    result = run_predictor_h2(ticker)
    if result:
        path = os.path.join(DATA_OUTPUT_DIR, f"{ticker}_H2.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ H2 | ${result['price_today']:,.2f} → ${result['price_pred']:,.2f} | {result['return_pct']}% | decay={_decay}")
    else:
        print("❌ Datos insuficientes")

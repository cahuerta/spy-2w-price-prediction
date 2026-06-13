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
# CONFIGURACIÓN H1
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 1
ALPHA_H1 = 0.08
MAX_PCA_COMPONENTS = 8
CLIP_RET = 0.05

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)

# ======================================================
# FEATURE ENGINEERING
# ======================================================
def make_features_h1(df: pd.DataFrame):
    out = df.copy()
    out["ret1"]  = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.25)

    out["ret_lag_1"] = out["ret1"].shift(1)
    out["ret_lag_2"] = out["ret1"].shift(2)
    out["ret_lag_5"] = out["ret1"].shift(5)

    past_returns = out["ret1"].shift(1)
    out["rv_short"] = past_returns.rolling(3).std().bfill()
    out["vol_ema"]  = past_returns.ewm(span=5).std().bfill()

    out["mom_3d"]  = past_returns.rolling(3).sum()
    out["mom_vol"] = out["mom_3d"] / out["rv_short"].replace(0, 0.01)

    delta = out["Close"].diff()
    gain  = delta.clip(lower=0).rolling(3).mean()
    loss  = (-delta.clip(upper=0)).rolling(3).mean()
    rs    = gain / loss.replace(0, 1.0)
    out["rsi_fast"] = 100 - (100 / (1 + rs))

    out["price_strength"] = out["ret1"] / out["rv_short"].replace(0, 0.01)
    out["trend_5"]        = out["Close"].pct_change(5).fillna(0)

    # Features adicionales disponibles para evolución
    out["ret_lag_3"]  = out["ret1"].shift(3)
    out["ret_lag_10"] = out["ret1"].shift(10)
    out["rv_10"]      = past_returns.rolling(10).std().clip(1e-6)
    out["vol_chg"]    = np.log(out["Volume"].replace(0, np.nan)).diff()

    return out

# ======================================================
# PREDICTOR H1
# ======================================================
def run_predictor_h1(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    # ── DARWIN: cargar genoma activo ──────────────────
    _alpha    = ALPHA_H1
    _max_pca  = MAX_PCA_COMPONENTS
    _clip_ret = CLIP_RET
    _decay    = 0.0               # [SW1] default neutro — sin cambio de comportamiento
    _feature_override = None

    if DARWIN_PREDICTOR:
        try:
            genome    = load_active_genome(HORIZON)
            _alpha    = genome.model_params.get("alpha_ridge",         ALPHA_H1)
            _max_pca  = genome.model_params.get("max_pca",             MAX_PCA_COMPONENTS)
            _clip_ret = genome.model_params.get("clip_ret",            CLIP_RET)
            _decay    = genome.model_params.get("sample_weight_decay", 0.0)  # [SW1]
            if genome.data.get("n_evaluations", 0) >= 20:
                _feature_override = genome.features
        except Exception:
            pass
    # ─────────────────────────────────────────────────

    if raw is None or len(raw) < 150:
        return None

    df   = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(df) < 140:
        return None

    feat = make_features_h1(df)

    # Features base originales
    _base_features = [
        "range", "ret_lag_1", "ret_lag_2", "ret_lag_5",
        "rv_short", "vol_ema", "mom_vol", "rsi_fast",
        "price_strength", "trend_5"
    ]

    if _feature_override:
        feature_cols = [f for f in _feature_override if f in feat.columns]
        if len(feature_cols) < 4:
            feature_cols = _base_features
    else:
        feature_cols = _base_features

    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 130:
        return None

    X = clean[feature_cols].values
    y = clean["y_fwd"].values

    n_samples, n_features = X.shape
    dynamic_pca = max(1, min(_max_pca, n_features, n_samples - 1))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=dynamic_pca)),
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

    last_features = feat[feature_cols].iloc[-1:]
    if last_features.isna().any().any():
        return None

    y_pred_log  = float(model.predict(last_features)[0])
    y_pred_log  = np.clip(y_pred_log, -_clip_ret, _clip_ret)
    price_today = float(feat["Close"].iloc[-1])
    price_1d    = price_today * np.exp(y_pred_log)

    r2_train    = model.score(X, y)
    coef_std    = float(np.std(model.named_steps["ridge"].coef_))
    confidence  = float(max(0.0, min(1.0, 1 / (1 + coef_std))))

    return {
        "ticker":        ticker,
        "predictor":     "H1",
        "horizon_days":  HORIZON,
        "date_today":    datetime.now().strftime("%Y-%m-%d"),
        "price_today":   round(price_today, 4),
        "price_pred":    round(price_1d, 4),
        "return_pct":    round(y_pred_log * 100, 4),
        "confidence":    round(confidence, 3),
        "r2_train":      round(float(r2_train), 4),
        "samples":       len(clean),
        "model": {
            "alpha":          _alpha,
            "pca_components": dynamic_pca,
            "features_count": len(feature_cols),
            "genome_active":  DARWIN_PREDICTOR,
            "weight_decay":   _decay,   # [SW1] trazabilidad
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    print(f"🚀 H1 → {ticker}")
    result = run_predictor_h1(ticker)
    if result:
        path = os.path.join(DATA_OUTPUT_DIR, f"{ticker}_H1.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"✅ H1 | ${result['price_today']:,.2f} → ${result['price_pred']:,.2f} | {result['return_pct']}% | conf={result['confidence']} | decay={_decay}")
    else:
        print("❌ Datos insuficientes")

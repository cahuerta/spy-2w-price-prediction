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
# CONFIGURACIÓN H6 — MEJOR HIT RATE ACTUAL (54.3%)
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 6
ALPHA_H6 = 0.8
MAX_PCA_COMPONENTS = 12
CLIP_RET = 0.12

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

    out["ret1"]  = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.35)

    for k in [1, 2, 3, 5, 10, 20]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    past_ret     = out["ret1"].shift(1)
    out["rv_25"] = past_ret.rolling(25).std().clip(1e-6) * np.sqrt(252)

    close_past = out["Close"].shift(1)
    delta      = close_past.diff()
    gain       = delta.clip(lower=0).rolling(14).mean()
    loss       = (-delta.clip(upper=0)).rolling(14).mean()
    rsi        = 100 - (100 / (1 + (gain / loss.replace(0, 1.0))))
    out["rsi_z"] = (
        (rsi - rsi.rolling(60).mean()) /
        rsi.rolling(60).std().replace(0, 1.0)
    ).clip(-3, 3)

    ema_20 = close_past.ewm(span=20).mean()
    out["ema_slope"] = ((ema_20 - ema_20.shift(5)) / ema_20.shift(5)).clip(-0.05, 0.05)

    def calc_slope(x):
        if len(x) < 25:
            return 0.0
        return np.polyfit(range(25), x, 1)[0] / (x.mean() + 1e-9)

    out["slope_25"] = close_past.rolling(25).apply(calc_slope, raw=True)

    # Features adicionales para evolución
    out["rv_20"]   = past_ret.rolling(20).std().clip(1e-6) * np.sqrt(252)
    out["rv_60"]   = past_ret.rolling(60).std().clip(1e-6) * np.sqrt(252)
    out["vol_chg"] = np.log(out["Volume"].replace(0, np.nan)).diff()

    ema50 = close_past.ewm(span=50).mean()
    out["ema_slope_50_10"] = ((ema50 - ema50.shift(10)) / ema50.shift(10)).clip(-0.10, 0.10)

    def calc_slope50(x):
        if len(x) < 50:
            return 0.0
        return np.polyfit(range(50), x, 1)[0] / x.mean()
    out["slope_50"] = close_past.rolling(50).apply(calc_slope50, raw=True)

    return out

# ======================================================
# PREDICTOR H6
# ======================================================
def run_predictor_h6(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    # ── DARWIN: cargar genoma activo ──────────────────
    _alpha    = ALPHA_H6
    _max_pca  = MAX_PCA_COMPONENTS
    _clip_ret = CLIP_RET
    _decay    = 0.0               # [SW1] default neutro — sin cambio de comportamiento
    _feature_override = None

    if DARWIN_PREDICTOR:
        try:
            genome    = load_active_genome(HORIZON)
            _alpha    = genome.model_params.get("alpha_ridge",         ALPHA_H6)
            _max_pca  = genome.model_params.get("max_pca",             MAX_PCA_COMPONENTS)
            _clip_ret = genome.model_params.get("clip_ret",            CLIP_RET)
            _decay    = genome.model_params.get("sample_weight_decay", 0.0)  # [SW1]
            if genome.data.get("n_evaluations", 0) >= 20:
                _feature_override = genome.features
        except Exception:
            pass
    # ─────────────────────────────────────────────────

    if raw is None or len(raw) < 240:
        return None

    df   = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h6(df)

    _base_features = [
        "range", "rv_25", "rsi_z", "ema_slope", "slope_25",
        "ret_lag_1", "ret_lag_3", "ret_lag_5", "ret_lag_10", "ret_lag_20"
    ]

    if _feature_override:
        feature_cols = [f for f in _feature_override if f in feat.columns]
        if len(feature_cols) < 4:
            feature_cols = _base_features
    else:
        feature_cols = _base_features

    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 190:
        return None

    X = clean[feature_cols].values
    y = clean["y_fwd"].values

    n_samples, n_features = X.shape
    dynamic_pca = max(1, min(_max_pca, n_features, n_samples - 1))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=dynamic_pca, random_state=42)),
        ("ridge",  Ridge(alpha=_alpha, random_state=42))
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
    price_6d    = price_today * np.exp(y_pred_log)

    confidence = 1 / (1 + np.std(model.named_steps["ridge"].coef_))

    return {
        "ticker":       ticker,
        "predictor":    "H6",
        "horizon_days": HORIZON,
        "date_today":   datetime.now().strftime("%Y-%m-%d"),
        "price_today":  round(price_today, 4),
        "price_pred":   round(price_6d, 4),
        "return_pct":   round(y_pred_log * 100, 4),
        "confidence":   round(float(confidence), 3),
        "samples":      len(clean),
        "model": {
            "alpha":          _alpha,
            "pca_components": dynamic_pca,
            "genome_active":  DARWIN_PREDICTOR,
            "weight_decay":   _decay,  # [SW1] trazabilidad
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    print(f"🚀 H6 → {ticker}")
    result = run_predictor_h6(ticker)
    if result:
        path = os.path.join(DATA_OUTPUT_DIR, f"{ticker}_H6.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ H6 | ${result['price_today']:,.2f} → ${result['price_pred']:,.2f} | {result['return_pct']}% | conf={result['confidence']} | decay={_decay}")
    else:
        print("❌ Datos insuficientes")

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
# CONFIGURACIÓN H7
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 7
ALPHA_H7 = 1.0
MAX_PCA_COMPONENTS = 12
CLIP_RET = 0.15

try:
    from data_provider import get_price_history
except ImportError:
    print("❌ ERROR: data_provider.py no encontrado")
    sys.exit(1)

# ======================================================
# HURST EXPONENT
# ======================================================
def get_hurst_exponent(ts, max_lag=20):
    ts   = np.array(ts)
    lags = range(2, min(max_lag, len(ts) // 4))
    if len(lags) < 2:
        return 0.5
    tau  = [np.sqrt(np.std(ts[lag:] - ts[:-lag]) + 1e-9) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# ======================================================
# FEATURE ENGINEERING H7
# ======================================================
def make_features_h7(df: pd.DataFrame):
    out = df.copy()

    out["ret1"]  = np.log(out["Close"]).diff()
    out["range"] = np.log(out["High"] / out["Low"]).clip(0, 0.35)

    for k in [1, 2, 3, 5, 10, 20, 30]:
        out[f"ret_lag_{k}"] = out["ret1"].shift(k)

    past         = out["ret1"].shift(1)
    out["rv_30"] = past.rolling(30).std().clip(1e-6) * np.sqrt(252)

    out["hurst_60"] = (
        out["ret1"].rolling(60)
        .apply(get_hurst_exponent, raw=True)
        .fillna(0.5).clip(0.2, 0.9)
    )

    close_past     = out["Close"].shift(1)
    sma50          = close_past.rolling(50).mean()
    out["dist_sma50"] = np.log(close_past / sma50).clip(-0.3, 0.3)

    vol_log           = np.log(out["Volume"].shift(1) + 1)
    out["money_flow"] = (vol_log * past).clip(-5, 5)

    # Features adicionales para evolución
    out["rv_20"]    = past.rolling(20).std().clip(1e-6) * np.sqrt(252)
    out["vol_chg"]  = np.log(out["Volume"].replace(0, np.nan)).diff()

    out["hurst_120"] = (
        out["ret1"].rolling(120)
        .apply(get_hurst_exponent, raw=True)
        .fillna(0.5).clip(0.2, 0.9)
    )

    ema20 = close_past.ewm(span=20).mean()
    out["ema_slope"] = ((ema20 - ema20.shift(5)) / ema20.shift(5)).clip(-0.05, 0.05)

    def calc_slope25(x):
        if len(x) < 25:
            return 0.0
        return np.polyfit(range(25), x, 1)[0] / (x.mean() + 1e-9)
    out["slope_25"] = close_past.rolling(25).apply(calc_slope25, raw=True)

    return out

# ======================================================
# PREDICTOR H7
# ======================================================
def run_predictor_h7(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    # ── DARWIN ────────────────────────────────────────
    _alpha    = ALPHA_H7
    _max_pca  = MAX_PCA_COMPONENTS
    _clip_ret = CLIP_RET
    _feature_override = None

    if DARWIN_PREDICTOR:
        try:
            genome    = load_active_genome(HORIZON)
            _alpha    = genome.model_params.get("alpha_ridge", ALPHA_H7)
            _max_pca  = genome.model_params.get("max_pca",     MAX_PCA_COMPONENTS)
            _clip_ret = genome.model_params.get("clip_ret",    CLIP_RET)
            if genome.data.get("n_evaluations", 0) >= 20:
                _feature_override = genome.features
        except Exception:
            pass
    # ─────────────────────────────────────────────────

    if raw is None or len(raw) < 260:
        return None

    df   = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    feat = make_features_h7(df)

    _base_features = [
        "range", "rv_30", "hurst_60", "dist_sma50", "money_flow",
        "ret_lag_1", "ret_lag_3", "ret_lag_5", "ret_lag_10",
        "ret_lag_20", "ret_lag_30"
    ]

    if _feature_override:
        feature_cols = [f for f in _feature_override if f in feat.columns]
        if len(feature_cols) < 4:
            feature_cols = _base_features
    else:
        feature_cols = _base_features

    feat["y_fwd"] = np.log(feat["Close"].shift(-HORIZON) / feat["Close"])
    clean = feat.dropna(subset=feature_cols + ["y_fwd"])

    if len(clean) < 200:
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
    price_7d    = price_today * np.exp(y_pred_log)

    hurst_val  = float(feat["hurst_60"].iloc[-1])
    regime     = "PERSISTENT" if hurst_val > 0.55 else "MEAN-REV" if hurst_val < 0.45 else "RANDOM"
    confidence = 1 / (1 + np.std(model.named_steps["ridge"].coef_))

    return {
        "ticker":       ticker,
        "predictor":    "H7",
        "horizon_days": HORIZON,
        "price_today":  round(price_today, 4),
        "price_pred":   round(price_7d, 4),
        "return_pct":   round(y_pred_log * 100, 4),
        "hurst":        round(hurst_val, 3),
        "regime":       regime,
        "confidence":   round(float(confidence), 3),
        "samples":      len(clean),
        "model": {
            "alpha":          _alpha,
            "pca_components": dynamic_pca,
            "genome_active":  DARWIN_PREDICTOR,
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    print(f"🚀 H7 → {ticker}")
    result = run_predictor_h7(ticker)
    if result:
        path = os.path.join(DATA_OUTPUT_DIR, f"{ticker}_H7.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ H7 | ${result['price_today']:,.2f} → ${result['price_pred']:,.2f} | {result['return_pct']}% | Hurst={result['hurst']} ({result['regime']})")
    else:
        print("❌ Datos insuficientes")

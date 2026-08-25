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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit

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

# [DETREND] Mismo mecanismo aplicado a H1 (auditoría 2026-08-24): se
# resta el drift local (media móvil de retornos pasados) del target
# antes de entrenar, y se reincorpora al predecir. Sin esto, el Ridge
# converge al drift promedio de la muestra de entrenamiento en vez de
# aprender señal genuina de corto plazo.
DRIFT_WINDOW      = int(os.getenv("H2_DRIFT_WINDOW", "60"))
DRIFT_MIN_PERIODS = int(os.getenv("H2_DRIFT_MIN_PERIODS", "30"))

# [AUD-P10] Auditoría 2026-08-25 (Problema 4): ALPHA_H2=0.15 era una
# constante hardcodeada sin justificación estadística (hit rate
# 48.89%, peor que el azar). Se reemplaza por calibración walk-forward
# vía RidgeCV, por ticker, en cada corrida — igual variabilidad que si
# cada ticker tuviera su propio predictor, en vez de una constante
# única para los 191.
ALPHA_GRID_H2 = np.logspace(-1, 1.2, 15)  # ~0.1 a ~16

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

    # [DETREND] Drift local: media móvil de retornos PASADOS (usa
    # past_ret, ya shift(1) → no hay leakage). Se usa solo para
    # destrendear el target, no como feature del modelo.
    out["_drift_local"] = past_ret.rolling(
        DRIFT_WINDOW, min_periods=DRIFT_MIN_PERIODS
    ).mean()

    return out

# ======================================================
# [AUD-P10] CALIBRACIÓN DE ALPHA RIDGE — WALK-FORWARD POR TICKER
# ======================================================
def _select_alpha_ridge_cv(X: np.ndarray, y: np.ndarray, n_pca: int) -> float:
    """
    Calibra alpha_ridge con RidgeCV sobre el espacio PCA, usando
    TimeSeriesSplit para no filtrar información futura (walk-forward).
    Se ejecuta por ticker, en cada corrida — reemplaza la constante
    ALPHA_H2 hardcodeada. Si falla por cualquier motivo (pocos datos,
    problema numérico), cae de vuelta a ALPHA_H2 sin interrumpir la
    predicción.
    """
    try:
        n_samples = len(y)
        scaler_tmp = StandardScaler()
        X_scaled   = scaler_tmp.fit_transform(X)
        pca_tmp    = PCA(n_components=n_pca, random_state=42)
        X_pca      = pca_tmp.fit_transform(X_scaled)

        n_splits = min(5, max(2, n_samples // 30))
        tscv     = TimeSeriesSplit(n_splits=n_splits)

        ridge_cv = RidgeCV(alphas=ALPHA_GRID_H2, cv=tscv)
        ridge_cv.fit(X_pca, y)
        return float(ridge_cv.alpha_)
    except Exception:
        return ALPHA_H2


# ======================================================
# PREDICTOR H2
# ======================================================
def run_predictor_h2(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    # ── DARWIN: cargar genoma activo ──────────────────
    # [AUD-P10] _alpha_from_genome distingue "Darwin ya evolucionó un
    # alpha propio para este horizonte" (None si no) de "usar el
    # default". Si Darwin no tiene un valor propio todavía, el default
    # ya no es la constante ALPHA_H2 — se calibra vía RidgeCV más abajo,
    # una vez que X e y están listos.
    _alpha_from_genome = None
    _max_pca  = MAX_PCA_COMPONENTS
    _clip_ret = CLIP_RET
    _decay    = 0.0               # [SW1] default neutro — sin cambio de comportamiento
    _feature_override = None

    if DARWIN_PREDICTOR:
        try:
            genome    = load_active_genome(HORIZON)
            _alpha_from_genome = genome.model_params.get("alpha_ridge")  # None si Darwin no lo definió
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

    # [DETREND] Target de entrenamiento = retorno crudo menos el drift
    # local vigente en ese momento.
    # [FIX escala] _drift_local es la media de retornos DIARIOS; y_fwd
    # es el retorno acumulado a HORIZON=2 días. Sin escalar, se entrena
    # restando 1x el drift diario pero se reconstruye sumando 2x ese
    # mismo drift al predecir — desalineación real detectada en revisión.
    feat["y_fwd_excess"] = feat["y_fwd"] - (feat["_drift_local"] * HORIZON)

    clean = feat.dropna(subset=feature_cols + ["y_fwd_excess"])

    if len(clean) < 140:
        return None

    X = clean[feature_cols].values
    y = clean["y_fwd_excess"].values

    n_samples, n_features = X.shape
    dynamic_pca = max(1, min(_max_pca, n_features, n_samples - 1))

    # [AUD-P10] Si Darwin ya evolucionó un alpha propio para este
    # horizonte, se respeta (no se pisa la evolución). Si no, se
    # calibra por ticker vía RidgeCV en vez de usar la constante fija.
    if _alpha_from_genome is not None:
        _alpha = float(_alpha_from_genome)
    else:
        _alpha = _select_alpha_ridge_cv(X, y, dynamic_pca)

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

    # [DETREND] Drift local vigente HOY — se vuelve a sumar a la
    # predicción del modelo (que predice solo el exceso).
    local_drift_now = feat["_drift_local"].iloc[-1]
    if pd.isna(local_drift_now):
        local_drift_now = 0.0
    local_drift_now = float(local_drift_now)

    y_pred_excess_log = float(model.predict(last_row)[0])
    y_pred_log         = y_pred_excess_log + local_drift_now * HORIZON
    y_pred_log         = np.clip(y_pred_log, -_clip_ret, _clip_ret)

    price_today = float(feat["Close"].iloc[-1])
    price_2d    = price_today * np.exp(y_pred_log)

    r2_train   = model.score(X, y)   # r2 sobre el target destrendeado (excess)
    confidence = 1 / (1 + np.std(model.named_steps["ridge"].coef_))

    # JSON de salida — MISMO esquema que el original, sin campos nuevos.
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
        # [FIX] _decay es local a run_predictor_h2() — no existe en este scope
        # (mismo bug identificado y corregido en predictor_h1.py). Se lee
        # desde el propio resultado ya guardado en el JSON.
        print(
            f"✅ H2 | ${result['price_today']:,.2f} → ${result['price_pred']:,.2f} | "
            f"{result['return_pct']}% | decay={result['model_params']['weight_decay']}"
        )
    else:
        print("❌ Datos insuficientes")

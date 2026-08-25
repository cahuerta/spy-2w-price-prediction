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
# CONFIGURACIÓN H1
# ======================================================
DATA_OUTPUT_DIR = "predictions_data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

HORIZON = 1
ALPHA_H1 = 0.08
MAX_PCA_COMPONENTS = 8
CLIP_RET = 0.05

# [DETREND] Ventana de drift local — media móvil de retornos pasados
# que se resta del target antes de entrenar (y se vuelve a sumar al
# predecir). Auditoría 2026-08-24: sin esto, el Ridge converge al
# drift promedio de la muestra de entrenamiento (casi siempre positivo
# para acciones que siguen listadas), generando sesgo alcista
# sistemático (73.6% predicciones positivas vs 50.4% real) y un hit
# rate que solo coincide con el azar (~50-53%), no con edge genuino.
DRIFT_WINDOW      = int(os.getenv("H1_DRIFT_WINDOW", "60"))
DRIFT_MIN_PERIODS = int(os.getenv("H1_DRIFT_MIN_PERIODS", "30"))

# [AUD-P10] Auditoría 2026-08-25 (Problema 4): ALPHA_H1=0.08 era una
# constante hardcodeada sin justificación estadística — la
# regularización Ridge más débil de los 10 horizontes, propensa a
# sobreajustar ruido de 1 día (hit rate 47.44%, peor que el azar).
# Se reemplaza por calibración walk-forward vía RidgeCV, por ticker,
# en cada corrida — igual variabilidad que si cada ticker tuviera su
# propio predictor, en vez de una constante única para los 191.
ALPHA_GRID_H1 = np.logspace(-1, 1.2, 15)  # ~0.1 a ~16

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

    # [DETREND] Drift local: media móvil de retornos PASADOS (ya usa
    # past_returns, que está shift(1) → no hay leakage). Se calcula
    # aquí para poder restarlo del target y volverlo a sumar en la
    # predicción, sin usarlo como feature del modelo (si fuera feature,
    # reintroduciría el mismo sesgo por otra vía).
    out["_drift_local"] = past_returns.rolling(
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
    ALPHA_H1 hardcodeada. Si falla por cualquier motivo (pocos datos,
    problema numérico), cae de vuelta a ALPHA_H1 sin interrumpir la
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

        ridge_cv = RidgeCV(alphas=ALPHA_GRID_H1, cv=tscv)
        ridge_cv.fit(X_pca, y)
        return float(ridge_cv.alpha_)
    except Exception:
        return ALPHA_H1


# ======================================================
# PREDICTOR H1
# ======================================================
def run_predictor_h1(ticker: str):
    raw = get_price_history(ticker=ticker, period="2y", interval="1d")

    # ── DARWIN: cargar genoma activo ──────────────────
    # [AUD-P10] _alpha_from_genome distingue "Darwin ya evolucionó un
    # alpha propio para este horizonte" (None si no) de "usar el
    # default". Si Darwin no tiene un valor propio todavía, el default
    # ya no es la constante ALPHA_H1 — se calibra vía RidgeCV más abajo,
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

    # [DETREND] Target de entrenamiento = retorno crudo menos el drift
    # local vigente en ese momento. El modelo aprende a predecir el
    # EXCESO sobre el drift reciente, no el retorno absoluto.
    # [FIX escala] _drift_local es la media de retornos DIARIOS; y_fwd
    # es el retorno acumulado a HORIZON días. Hay que escalar el drift
    # por HORIZON para que sea consistente con lo que se le suma de
    # vuelta al predecir (local_drift_now * HORIZON, más abajo). Para
    # H1 (HORIZON=1) es cosmético; para horizontes mayores es necesario.
    feat["y_fwd_excess"] = feat["y_fwd"] - (feat["_drift_local"] * HORIZON)

    clean = feat.dropna(subset=feature_cols + ["y_fwd_excess"])

    if len(clean) < 130:
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

    # [DETREND] Drift local vigente HOY — se vuelve a sumar a la
    # predicción del modelo (que predice solo el exceso).
    local_drift_now = feat["_drift_local"].iloc[-1]
    if pd.isna(local_drift_now):
        local_drift_now = 0.0
    local_drift_now = float(local_drift_now)

    y_pred_excess_log = float(model.predict(last_features)[0])
    y_pred_log         = y_pred_excess_log + local_drift_now * HORIZON
    y_pred_log         = np.clip(y_pred_log, -_clip_ret, _clip_ret)

    price_today = float(feat["Close"].iloc[-1])
    price_1d    = price_today * np.exp(y_pred_log)

    r2_train    = model.score(X, y)   # r2 sobre el target destrendeado (excess)
    coef_std    = float(np.std(model.named_steps["ridge"].coef_))
    confidence  = float(max(0.0, min(1.0, 1 / (1 + coef_std))))

    # JSON de salida — MISMO esquema que el original, sin campos nuevos.
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
        # [FIX] _decay es local a run_predictor_h1() — no existe en este scope.
        # Se lee desde el propio resultado ya guardado en el JSON.
        print(
            f"✅ H1 | ${result['price_today']:,.2f} → ${result['price_pred']:,.2f} | "
            f"{result['return_pct']}% | conf={result['confidence']} | "
            f"decay={result['model']['weight_decay']}"
        )
    else:
        print("❌ Datos insuficientes")

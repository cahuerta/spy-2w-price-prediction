# =========================================================
# predictors_engine/calibration_utils.py
# =========================================================
# [AUD-P11][2026-09-23, auditoría Problema 4] ANTES: H1 y H2 tenían
# _select_alpha_ridge_cv() implementada de forma IDÉNTICA, cada uno
# con su propia copia-pegada del mismo código. H3-H10 nunca la
# recibieron — se quedaron con un ALPHA_HN fijo, elegido una vez a
# mano y jamás reevaluado contra el régimen de mercado actual,
# incluso siendo H10 el horizonte con el problema de precisión más
# grave de los diez.
#
# Este módulo extrae esa función UNA vez, parametrizada (recibe el
# alpha de fallback y el grid de búsqueda como argumentos en vez de
# leer una constante fija del módulo que la llama), para que los 10
# predictores usen la misma implementación en vez de que cada uno la
# reimplemente o simplemente no la tenga.
#
# No cambia el comportamiento de H1/H2 (mismo algoritmo, mismos
# defaults) — solo evita que el mismo código exista duplicado, y
# permite que H3-H10 lo adopten sin reescribirlo.
# =========================================================

import numpy as np
from typing import Optional

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit

# Mismo grid por defecto que ya usaban H1/H2 — ~0.1 a ~16,
# 15 puntos en escala logarítmica.
DEFAULT_ALPHA_GRID = np.logspace(-1, 1.2, 15)

# Separación entre train y test en el CV (días) = horizonte máximo H10.
CV_GAP_DAYS = 10


def select_alpha_ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_pca: int,
    fallback_alpha: float,
    alpha_grid: Optional[np.ndarray] = None,
) -> float:
    """
    Calibra alpha_ridge con RidgeCV sobre el espacio PCA, usando
    TimeSeriesSplit para no filtrar información futura (walk-forward).
    Se ejecuta por ticker, en cada corrida — reemplaza una constante
    ALPHA_HN hardcodeada.

    Si falla por cualquier motivo (pocos datos, problema numérico),
    cae de vuelta a `fallback_alpha` sin interrumpir la predicción —
    mismo comportamiento de seguridad que ya tenían H1/H2.

    Parámetros:
        X, y:            datos de entrenamiento (features, target)
        n_pca:           número de componentes PCA a usar
        fallback_alpha:  valor a devolver si la calibración falla
                         (típicamente la constante ALPHA_HN del
                         horizonte que llama, para no cambiar el
                         comportamiento de seguridad existente)
        alpha_grid:      grid de valores de alpha a probar; por
                         defecto el mismo que ya usaban H1/H2
    """
    try:
        grid = alpha_grid if alpha_grid is not None else DEFAULT_ALPHA_GRID

        n_samples  = len(y)
        scaler_tmp = StandardScaler()
        X_scaled   = scaler_tmp.fit_transform(X)
        pca_tmp    = PCA(n_components=n_pca, random_state=42)
        X_pca      = pca_tmp.fit_transform(X_scaled)

        n_splits = min(5, max(2, n_samples // 30))
        # [CAL-GAP][2026-09-26] gap = horizonte máximo (10 días): el target
        # de las últimas filas de cada fold de entrenamiento es el retorno
        # a h días hacia adelante, que se solapa con el fold de test. Sin
        # gap, esa fuga premia alphas bajos (sobreajuste) en el CV.
        tscv     = TimeSeriesSplit(n_splits=n_splits, gap=CV_GAP_DAYS)

        ridge_cv = RidgeCV(alphas=grid, cv=tscv)
        ridge_cv.fit(X_pca, y)
        return float(ridge_cv.alpha_)
    except Exception:
        return fallback_alpha

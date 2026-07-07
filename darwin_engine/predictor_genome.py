"""
darwin_engine/predictor_genome.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Genoma evolucionable para cada predictor H1-H10.

Cada H tiene su propio genoma independiente que controla:
  - Qué features usar (y con qué ventanas)
  - Parámetros del modelo (Ridge alpha, PCA components)
  - Parámetros de features (ventanas RSI, EMA, volatilidad, lags)
  - Clip del retorno predicho
  - sample_weight_decay [SW1]: cuánto peso dar a datos recientes

Los predictores leen su genoma activo al ejecutarse.
Si no existe genoma → usan sus parámetros originales (sin romper nada).

FIXES:
  [SW1] sample_weight_decay agregado a BASE_MODEL_PARAMS H1-H9.
        H10 excluido — ya tiene walk-forward que cumple la misma función.
        Valor inicial 0.0 = comportamiento idéntico al actual.
        Darwin lo muta cuando detecta bias_score alto (bias temporal).

  [SW2] bias_score agregado al genome como métrica de diagnóstico.
        bias_score = fracción de predicciones positivas que fallaron.
        GOOGL: 15/16 = 0.94 → bias alcista severo en caída.
        Darwin usa este score para decidir si mutar sample_weight_decay.

  [FG1] FEATURE_POOL y BASE_FEATURES_BY_HORIZON reescritos para reflejar
        las columnas REALES que cada predictor_hX.py calcula en su
        make_features_hX(). Antes, el genoma ofrecía nombres genéricos
        (rsi_z_14_60, ema_slope_20_5, slope_25, etc.) que NO existían
        en la mayoría de los predictores — Darwin evolucionaba genes
        que se descartaban en silencio por el filtro
        `[f for f in _feature_override if f in feat.columns]`,
        evaluando fitness sobre features distintas a las que creía
        estar probando. BASE_FEATURES_BY_HORIZON ahora es 1:1 idéntico
        a los `_base_features` hardcodeados en cada predictor (mismo
        comportamiento default, cero riesgo), y FEATURE_POOL incluye
        el pool real ampliado disponible para mutación futura.
"""

import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("predictor_genome")

DATA_PATH   = Path(os.getenv("DATA_PATH", "/data"))
REPO_PATH   = Path(os.getenv("REPO_PATH", "/opt/render/project/src"))
GENOME_BASE = REPO_PATH / "predictor_genomes"

# ══════════════════════════════════════════════════════
# FEATURE POOLS — todos los genes posibles [FG1]
# Nombres y metadata alineados a las columnas REALES que
# produce cada make_features_hX(). Metadata es descriptiva
# (documentación de qué representa cada gen), no se usa para
# calcular features — cada predictor calcula las suyas.
# ══════════════════════════════════════════════════════

FEATURE_POOL = {
    # Retornos y lags
    "ret_lag_1":   {"type": "lag", "lag": 1},
    "ret_lag_2":   {"type": "lag", "lag": 2},
    "ret_lag_3":   {"type": "lag", "lag": 3},
    "ret_lag_5":   {"type": "lag", "lag": 5},
    "ret_lag_8":   {"type": "lag", "lag": 8},
    "ret_lag_10":  {"type": "lag", "lag": 10},
    "ret_lag_20":  {"type": "lag", "lag": 20},
    "ret_lag_30":  {"type": "lag", "lag": 30},
    # Volatilidad realizada
    "rv_short": {"type": "rv", "window": 3},
    "rv_10":    {"type": "rv", "window": 10},
    "rv_15":    {"type": "rv", "window": 15},
    "rv_20":    {"type": "rv", "window": 20},
    "rv_25":    {"type": "rv", "window": 25},
    "rv_30":    {"type": "rv", "window": 30},
    "rv_60":    {"type": "rv", "window": 60},
    "vol_ema":  {"type": "rv_ema", "span": 5},
    "atr_14":   {"type": "atr", "window": 14},
    # Rango precio
    "range":  {"type": "range"},
    # Cambio de volumen
    "vol_chg": {"type": "vol_chg"},
    "vol_z":   {"type": "vol_zscore", "window": 20},
    "vol_price": {"type": "vol_price_interaction"},
    "vol_ratio": {"type": "range_vs_atr"},
    # RSI (variantes reales por horizonte — nombres tal cual los calcula cada predictor)
    "rsi_fast": {"type": "rsi", "window": 3},
    "rsi_9":    {"type": "rsi", "window": 9},
    "rsi_7":    {"type": "rsi", "window": 7},
    "rsi_14":   {"type": "rsi", "window": 14},
    "rsi_z":    {"type": "rsi_z", "rsi_window": 14, "norm_window": 60},
    # Momentum / trend
    "mom_2d":     {"type": "mom_sum", "window": 2},
    "mom_3d":     {"type": "mom_sum", "window": 3},
    "mom_5d":     {"type": "mom_sum", "window": 5},
    "mom_10d":    {"type": "mom_sum", "window": 10},
    "mom_20d":    {"type": "mom_sum", "window": 20},
    "mom_vol":    {"type": "mom_over_rv"},
    "mom_ratio":  {"type": "mom_over_rv"},
    "price_strength": {"type": "ret_over_rv"},
    "trend_5":    {"type": "pct_change", "window": 5},
    "trend_10":   {"type": "pct_change", "window": 10},
    "trend_20":   {"type": "pct_change", "window": 20},
    "trend_diff": {"type": "ema_diff", "fast": 5, "slow": 20},
    "trend_strength": {"type": "adx_x_slope_sign"},
    # Distancia a medias / bandas
    "dist_ma10":  {"type": "dist_sma", "window": 10},
    "dist_ma20":  {"type": "dist_sma", "window": 20},
    "dist_sma50": {"type": "dist_sma", "window": 50},
    "ema_20_dist": {"type": "dist_ema", "span": 20},
    "dist_hi":    {"type": "dist_high", "window": 20},
    "dist_lo":    {"type": "dist_low", "window": 20},
    "bb_dist":    {"type": "bollinger_dist", "window": 20},
    # EMA slope (nombres reales por horizonte)
    "ema_slope":        {"type": "ema_slope", "span": 20, "shift": 5},
    "ema_slope_10_3":   {"type": "ema_slope", "span": 10, "shift": 3},
    "ema_slope_50_10":  {"type": "ema_slope", "span": 50, "shift": 10},
    "ema50_slope":      {"type": "ema_slope", "span": 50, "shift": 5},
    # Trend slope
    "slope_20": {"type": "slope", "window": 20},
    "slope_25": {"type": "slope", "window": 25},
    "slope_50": {"type": "slope", "window": 50},
    # Hurst
    "hurst_60":  {"type": "hurst", "window": 60},
    "hurst_120": {"type": "hurst", "window": 120},
    # Osciladores / indicadores especializados por horizonte
    "stoch_k":       {"type": "stochastic", "window": 14},
    "adx_14":        {"type": "adx", "window": 14},
    "mfi_14":        {"type": "money_flow_index", "window": 14},
    "capital_flow":  {"type": "mfi_derived"},
    "money_flow":    {"type": "volume_x_return"},
}

# BASE_FEATURES_BY_HORIZON [FG1]: idéntico a los `_base_features`
# hardcodeados en cada predictor_hX.py — mismo comportamiento default,
# ahora también correcto como override evolucionado (100% de las
# columnas existen realmente en cada feat DataFrame).
BASE_FEATURES_BY_HORIZON = {
    1:  ["range", "ret_lag_1", "ret_lag_2", "ret_lag_5",
         "rv_short", "vol_ema", "mom_vol", "rsi_fast",
         "price_strength", "trend_5"],
    2:  ["range", "rv_10", "mom_ratio", "rsi_9", "dist_ma10",
         "mom_2d", "mom_5d", "ret_lag_1", "ret_lag_2",
         "ret_lag_3", "ret_lag_5", "ret_lag_10", "trend_10"],
    3:  ["range", "rv_15", "mom_ratio", "dist_ma20", "vol_z",
         "mom_5d", "mom_10d", "rsi_7", "rsi_14",
         "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
         "ret_lag_8", "trend_20"],
    4:  ["range", "rv_20", "trend_diff", "vol_ratio", "rsi_14",
         "stoch_k", "dist_hi", "dist_lo", "mom_5d", "mom_10d",
         "mom_20d", "vol_price"],
    5:  ["range", "rv_20", "rsi_14", "ema_20_dist", "slope_20",
         "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
         "ret_lag_10", "ret_lag_20"],
    6:  ["range", "rv_25", "rsi_z", "ema_slope", "slope_25",
         "ret_lag_1", "ret_lag_3", "ret_lag_5", "ret_lag_10", "ret_lag_20"],
    7:  ["range", "rv_30", "hurst_60", "dist_sma50", "money_flow",
         "ret_lag_1", "ret_lag_3", "ret_lag_5", "ret_lag_10",
         "ret_lag_20", "ret_lag_30"],
    8:  ["range", "rv_30", "adx_14", "ema50_slope", "trend_strength",
         "ret_lag_1", "ret_lag_5", "ret_lag_10", "ret_lag_20", "ret_lag_30"],
    9:  ["range", "mfi_14", "bb_dist", "capital_flow",
         "ret_lag_1", "ret_lag_5", "ret_lag_10", "ret_lag_20", "ret_lag_30"],
    10: ["range", "rv_20", "rv_60", "vol_chg", "hurst_60", "hurst_120",
         "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
         "ret_lag_10", "ret_lag_20"],
}

# [SW1] sample_weight_decay agregado a H1-H9
# H10 excluido — ya tiene walk-forward propio
BASE_MODEL_PARAMS = {
    1:  {"alpha_ridge": 0.5, "max_pca": 8,  "clip_ret": 0.05, "period": "1y",  "sample_weight_decay": 0.0},
    2:  {"alpha_ridge": 0.5, "max_pca": 8,  "clip_ret": 0.06, "period": "1y",  "sample_weight_decay": 0.0},
    3:  {"alpha_ridge": 0.6, "max_pca": 10, "clip_ret": 0.07, "period": "1y",  "sample_weight_decay": 0.0},
    4:  {"alpha_ridge": 0.7, "max_pca": 10, "clip_ret": 0.08, "period": "2y",  "sample_weight_decay": 0.0},
    5:  {"alpha_ridge": 0.7, "max_pca": 12, "clip_ret": 0.09, "period": "2y",  "sample_weight_decay": 0.0},
    6:  {"alpha_ridge": 0.8, "max_pca": 12, "clip_ret": 0.10, "period": "2y",  "sample_weight_decay": 0.0},
    7:  {"alpha_ridge": 0.8, "max_pca": 14, "clip_ret": 0.11, "period": "2y",  "sample_weight_decay": 0.0},
    8:  {"alpha_ridge": 0.9, "max_pca": 14, "clip_ret": 0.12, "period": "max", "sample_weight_decay": 0.0},
    9:  {"alpha_ridge": 1.0, "max_pca": 16, "clip_ret": 0.13, "period": "max", "sample_weight_decay": 0.0},
    10: {"alpha_ridge": 1.0, "max_pca": 16, "clip_ret": 0.15, "period": "max"},  # H10 sin decay
}

# [SW2] Límites de mutación para sample_weight_decay
# Darwin puede mutar entre 0.0 (sin ponderación) y 3.0 (recientes ~20x más peso)
DECAY_LIMITS = (0.0, 3.0)

# Umbral de bias_score para activar mutación de decay
# Si fracción de pred+ que falló >= 0.70 → modelo tiene bias temporal severo
BIAS_SCORE_THRESHOLD = 0.70


# ══════════════════════════════════════════════════════
# CLASE PRINCIPAL
# ══════════════════════════════════════════════════════

class PredictorGenome:

    def __init__(self, horizon: int, genome_dict: Dict = None):
        self.horizon = horizon
        if genome_dict:
            self.data = deepcopy(genome_dict)
        else:
            self.data = self._build_default(horizon)

    def _build_default(self, horizon: int) -> Dict:
        return {
            "genome_id":      f"H{horizon}_v1",
            "horizon":        horizon,
            "generation":     1,
            "parent_ids":     [],
            "fitness":        None,
            "hit_rate":       None,
            "bias_score":     None,   # [SW2] fracción pred+ que falló
            "n_evaluations":  0,
            "created_at":     datetime.now(timezone.utc).isoformat(),
            "features":       deepcopy(BASE_FEATURES_BY_HORIZON.get(horizon, [])),
            "model_params":   deepcopy(BASE_MODEL_PARAMS.get(horizon, {
                "alpha_ridge":         0.8,
                "max_pca":             12,
                "clip_ret":            0.10,
                "period":              "2y",
                "sample_weight_decay": 0.0,
            })),
        }

    # ── Accesores ────────────────────────────────────

    @property
    def genome_id(self) -> str:
        return self.data.get("genome_id", f"H{self.horizon}_v1")

    @property
    def features(self) -> List[str]:
        return self.data.get("features", [])

    @property
    def model_params(self) -> Dict:
        return self.data.get("model_params", {})

    @property
    def fitness(self) -> Optional[float]:
        return self.data.get("fitness")

    @fitness.setter
    def fitness(self, value: float):
        self.data["fitness"] = value

    @property
    def hit_rate(self) -> Optional[float]:
        return self.data.get("hit_rate")

    @hit_rate.setter
    def hit_rate(self, value: float):
        self.data["hit_rate"] = value

    @property
    def bias_score(self) -> Optional[float]:
        """[SW2] Fracción de predicciones positivas que fallaron."""
        return self.data.get("bias_score")

    @bias_score.setter
    def bias_score(self, value: float):
        self.data["bias_score"] = value

    @property
    def sample_weight_decay(self) -> float:
        """[SW1] Decay actual del genome. 0.0 = sin ponderación temporal."""
        return self.model_params.get("sample_weight_decay", 0.0)

    @property
    def has_temporal_bias(self) -> bool:
        """[SW2] True si el bias_score supera el umbral crítico."""
        bs = self.bias_score
        return bs is not None and bs >= BIAS_SCORE_THRESHOLD

    @property
    def generation(self) -> int:
        return self.data.get("generation", 1)

    # ── Serialización ────────────────────────────────

    def to_dict(self) -> Dict:
        return deepcopy(self.data)

    @classmethod
    def from_dict(cls, d: Dict) -> "PredictorGenome":
        horizon = int(d.get("horizon", 6))
        return cls(horizon=horizon, genome_dict=d)

    # ── Persistencia ─────────────────────────────────

    def save_as_champion(self) -> Path:
        path = GENOME_BASE / f"H{self.horizon}" / "champion.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        tmp.replace(path)
        logger.info(f"👑 Champion H{self.horizon} guardado: {self.genome_id}")
        return path

    def save_as_shadow(self) -> Path:
        path = GENOME_BASE / f"H{self.horizon}" / "shadow" / f"{self.genome_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        tmp.replace(path)
        logger.info(f"🌑 Shadow H{self.horizon} guardado: {self.genome_id}")
        return path

    @classmethod
    def load_champion(cls, horizon: int) -> "PredictorGenome":
        path = GENOME_BASE / f"H{horizon}" / "champion.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                # [SW1] Migración silenciosa: genomas antiguos sin sample_weight_decay
                if "model_params" in data and "sample_weight_decay" not in data["model_params"]:
                    if horizon < 10:  # H10 no necesita el campo
                        data["model_params"]["sample_weight_decay"] = 0.0
                        logger.info(f"🔄 H{horizon} migrado: sample_weight_decay=0.0 agregado")
                logger.info(f"✅ Champion H{horizon} cargado: {data.get('genome_id')}")
                return cls.from_dict(data)
            except Exception as e:
                logger.warning(f"⚠️ Error cargando champion H{horizon}: {e}")
        logger.info(f"🆕 Champion H{horizon} no existe → usando default")
        genome = cls(horizon=horizon)
        genome.save_as_champion()
        return genome

    @classmethod
    def load_shadow_genomes(cls, horizon: int) -> List["PredictorGenome"]:
        shadow_dir = GENOME_BASE / f"H{horizon}" / "shadow"
        if not shadow_dir.exists():
            return []
        genomes = []
        for path in shadow_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                genomes.append(cls.from_dict(data))
            except Exception:
                continue
        return genomes

    def describe(self) -> str:
        decay = self.model_params.get("sample_weight_decay", "N/A")
        bias  = f"{self.bias_score:.2f}" if self.bias_score is not None else "?"
        return (
            f"PredictorGenome [H{self.horizon}] {self.genome_id} "
            f"gen={self.generation} fit={self.fitness} hit={self.hit_rate} "
            f"bias={bias} decay={decay} "
            f"features={len(self.features)} "
            f"alpha={self.model_params.get('alpha_ridge')} "
            f"pca={self.model_params.get('max_pca')}"
        )

    def __repr__(self) -> str:
        return self.describe()


# ══════════════════════════════════════════════════════
# API PÚBLICA
# ══════════════════════════════════════════════════════

def load_active_genome(horizon: int) -> PredictorGenome:
    return PredictorGenome.load_champion(horizon)


def get_feature_cols(horizon: int) -> List[str]:
    return load_active_genome(horizon).features


def get_model_params(horizon: int) -> Dict:
    return load_active_genome(horizon).model_params


# ══════════════════════════════════════════════════════
# ACTUALIZACIÓN DE MÉTRICAS — llamada después de cada ciclo
# ══════════════════════════════════════════════════════

def update_genome_hit_rate(horizon: int, new_hit_rate: float, n_evals: int) -> None:
    """Actualiza hit_rate del campeón con datos frescos del evaluator."""
    genome = PredictorGenome.load_champion(horizon)
    genome.hit_rate = round(new_hit_rate, 4)
    genome.data["n_evaluations"] = n_evals
    genome.data["hit_rate_updated_at"] = datetime.now(timezone.utc).isoformat()
    genome.save_as_champion()
    logger.info(f"📊 H{horizon} hit_rate={new_hit_rate:.2%} ({n_evals} evals)")


def update_genome_bias_score(horizon: int, bias_score: float) -> None:
    """
    [SW2] Actualiza el bias_score del campeón.

    bias_score = predicciones_positivas_que_fallaron / total_predicciones_positivas

    Ejemplos:
      GOOGL H2: 15 pred+ fallaron de 16 pred+ → bias_score = 0.94
      AMAT  H2: 2  pred+ fallaron de 20 pred+ → bias_score = 0.10

    Darwin llama a esta función después de cada ciclo de evaluación.
    Si bias_score >= BIAS_SCORE_THRESHOLD → predictor_mutator muta decay.
    """
    genome = PredictorGenome.load_champion(horizon)
    genome.bias_score = round(bias_score, 4)
    genome.data["bias_score_updated_at"] = datetime.now(timezone.utc).isoformat()
    genome.save_as_champion()
    status = "⚠️ BIAS ALTO" if bias_score >= BIAS_SCORE_THRESHOLD else "✅ OK"
    logger.info(f"🎯 H{horizon} bias_score={bias_score:.2f} {status}")


# ══════════════════════════════════════════════════════
# CÁLCULO DE BIAS SCORE desde evaluaciones en disco
# ══════════════════════════════════════════════════════

def calc_bias_score_from_evals(ticker: str, horizon: int, min_evals: int = 10) -> Optional[float]:
    """
    [SW2] Calcula bias_score leyendo evaluaciones limpias del ticker.

    Solo usa evaluaciones con:
      - legacy_bad_horizon=False
      - price_real no None
      - weak_signal=False (tiene señal real)
      - hit_sign no None

    Retorna None si hay menos de min_evals evaluaciones válidas.
    """
    eval_dir = DATA_PATH / "evaluations" / ticker.upper()
    if not eval_dir.exists():
        return None

    pred_pos_total  = 0  # predicciones positivas con señal
    pred_pos_failed = 0  # predicciones positivas que fallaron

    for f in eval_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue

        if d.get("legacy_bad_horizon") is True:
            continue
        if d.get("price_real") is None:
            continue
        if d.get("weak_signal") is True:
            continue
        if d.get("hit_sign") is None:
            continue

        pred_ret = d.get("predicted_return_pct", 0) or 0
        hit      = d.get("hit_sign", False)

        if pred_ret > 0.05:  # predicción positiva con señal
            pred_pos_total += 1
            if not hit:
                pred_pos_failed += 1

    if pred_pos_total < min_evals:
        return None

    return round(pred_pos_failed / pred_pos_total, 4)


# ══════════════════════════════════════════════════════
# INICIALIZAR TODOS LOS GENOMAS
# ══════════════════════════════════════════════════════

def initialize_all_genomes() -> Dict[str, str]:
    """Crea los genomas default para H1-H10 si no existen."""
    results = {}
    for h in range(1, 11):
        path = GENOME_BASE / f"H{h}" / "champion.json"
        if not path.exists():
            genome = PredictorGenome(horizon=h)
            genome.save_as_champion()
            results[f"H{h}"] = "created"
            logger.info(f"🆕 Genome H{h} inicializado: {genome.genome_id}")
        else:
            results[f"H{h}"] = "exists"
    return results

# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧬 Inicializando genomas H1-H10...")
    results = initialize_all_genomes()
    for h, status in results.items():
        genome = PredictorGenome.load_champion(int(h[1:]))
        print(f"  {h}: {status} → {genome.describe()}")

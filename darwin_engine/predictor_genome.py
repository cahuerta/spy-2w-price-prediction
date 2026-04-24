"""
darwin_engine/predictor_genome.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHIVO NUEVO — darwin_engine/predictor_genome.py

Genoma evolucionable para cada predictor H1-H10.

Cada H tiene su propio genoma independiente que controla:
  - Qué features usar (y con qué ventanas)
  - Parámetros del modelo (Ridge alpha, PCA components)
  - Parámetros de features (ventanas RSI, EMA, volatilidad, lags)
  - Clip del retorno predicho

Los predictores leen su genoma activo al ejecutarse.
Si no existe genoma → usan sus parámetros originales (sin romper nada).

Ruta en repo: darwin_engine/predictor_genome.py
Genomas guardados en: predictor_genomes/H{n}/champion.json
                                         H{n}/shadow/genome_v{n}.json
"""

import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("predictor_genome")

DATA_PATH     = Path(os.getenv("DATA_PATH", "/data"))
REPO_PATH     = Path(os.getenv("REPO_PATH", "/opt/render/project/src"))
GENOME_BASE   = REPO_PATH / "predictor_genomes"

# ══════════════════════════════════════════════════════
# FEATURE POOLS — todos los genes posibles
# ══════════════════════════════════════════════════════

# Features disponibles para evolución
FEATURE_POOL = {
    # Retornos y lags
    "ret_lag_1":   {"type": "lag", "lag": 1},
    "ret_lag_2":   {"type": "lag", "lag": 2},
    "ret_lag_3":   {"type": "lag", "lag": 3},
    "ret_lag_5":   {"type": "lag", "lag": 5},
    "ret_lag_10":  {"type": "lag", "lag": 10},
    "ret_lag_20":  {"type": "lag", "lag": 20},

    # Volatilidad realizada
    "rv_10":  {"type": "rv", "window": 10},
    "rv_20":  {"type": "rv", "window": 20},
    "rv_25":  {"type": "rv", "window": 25},
    "rv_60":  {"type": "rv", "window": 60},

    # Rango precio
    "range":  {"type": "range"},

    # Cambio de volumen
    "vol_chg": {"type": "vol_chg"},

    # RSI Z-score
    "rsi_z_14_60":  {"type": "rsi_z", "rsi_window": 14, "norm_window": 60},
    "rsi_z_7_30":   {"type": "rsi_z", "rsi_window": 7,  "norm_window": 30},
    "rsi_z_21_90":  {"type": "rsi_z", "rsi_window": 21, "norm_window": 90},

    # EMA slope
    "ema_slope_20_5":  {"type": "ema_slope", "span": 20, "shift": 5},
    "ema_slope_50_10": {"type": "ema_slope", "span": 50, "shift": 10},
    "ema_slope_10_3":  {"type": "ema_slope", "span": 10, "shift": 3},

    # Trend slope (regresión lineal)
    "slope_10":  {"type": "slope", "window": 10},
    "slope_25":  {"type": "slope", "window": 25},
    "slope_50":  {"type": "slope", "window": 50},

    # Hurst (para horizontes largos)
    "hurst_60":  {"type": "hurst", "window": 60},
    "hurst_120": {"type": "hurst", "window": 120},
}

# Features base por horizonte (punto de partida evolutivo)
# Basado en lo que cada H usa actualmente
BASE_FEATURES_BY_HORIZON = {
    1:  ["range", "rv_10", "rsi_z_7_30",  "ema_slope_10_3",  "slope_10",
         "ret_lag_1", "ret_lag_2", "ret_lag_3"],
    2:  ["range", "rv_10", "rsi_z_7_30",  "ema_slope_10_3",  "slope_10",
         "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5"],
    3:  ["range", "rv_20", "rsi_z_14_60", "ema_slope_20_5",  "slope_25",
         "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5"],
    4:  ["range", "rv_20", "rsi_z_14_60", "ema_slope_20_5",  "slope_25",
         "ret_lag_1", "ret_lag_3", "ret_lag_5", "ret_lag_10"],
    5:  ["range", "rv_20", "rsi_z_14_60", "ema_slope_20_5",  "slope_25",
         "ret_lag_1", "ret_lag_3", "ret_lag_5", "ret_lag_10"],
    6:  ["range", "rv_25", "rsi_z_14_60", "ema_slope_20_5",  "slope_25",
         "ret_lag_1", "ret_lag_3", "ret_lag_5", "ret_lag_10", "ret_lag_20"],
    7:  ["range", "rv_25", "rsi_z_14_60", "ema_slope_20_5",  "slope_25",
         "ret_lag_1", "ret_lag_3", "ret_lag_5", "ret_lag_10", "ret_lag_20"],
    8:  ["range", "rv_25", "rsi_z_21_90", "ema_slope_50_10", "slope_50",
         "vol_chg", "ret_lag_1", "ret_lag_3", "ret_lag_5", "ret_lag_10", "ret_lag_20"],
    9:  ["range", "rv_60", "rsi_z_21_90", "ema_slope_50_10", "slope_50",
         "vol_chg", "hurst_60",
         "ret_lag_1", "ret_lag_3", "ret_lag_5", "ret_lag_10", "ret_lag_20"],
    10: ["range", "rv_20", "rv_60", "vol_chg", "hurst_60", "hurst_120",
         "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
         "ret_lag_10", "ret_lag_20"],
}

# Parámetros del modelo base por horizonte
BASE_MODEL_PARAMS = {
    1:  {"alpha_ridge": 0.5, "max_pca": 8,  "clip_ret": 0.05, "period": "1y"},
    2:  {"alpha_ridge": 0.5, "max_pca": 8,  "clip_ret": 0.06, "period": "1y"},
    3:  {"alpha_ridge": 0.6, "max_pca": 10, "clip_ret": 0.07, "period": "1y"},
    4:  {"alpha_ridge": 0.7, "max_pca": 10, "clip_ret": 0.08, "period": "2y"},
    5:  {"alpha_ridge": 0.7, "max_pca": 12, "clip_ret": 0.09, "period": "2y"},
    6:  {"alpha_ridge": 0.8, "max_pca": 12, "clip_ret": 0.10, "period": "2y"},
    7:  {"alpha_ridge": 0.8, "max_pca": 14, "clip_ret": 0.11, "period": "2y"},
    8:  {"alpha_ridge": 0.9, "max_pca": 14, "clip_ret": 0.12, "period": "max"},
    9:  {"alpha_ridge": 1.0, "max_pca": 16, "clip_ret": 0.13, "period": "max"},
    10: {"alpha_ridge": 1.0, "max_pca": 16, "clip_ret": 0.15, "period": "max"},
}


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
            "genome_id":    f"H{horizon}_v1",
            "horizon":      horizon,
            "generation":   1,
            "parent_ids":   [],
            "fitness":      None,
            "hit_rate":     None,
            "n_evaluations": 0,
            "created_at":   datetime.now(timezone.utc).isoformat(),

            # Genes principales
            "features":     deepcopy(BASE_FEATURES_BY_HORIZON.get(horizon, [])),
            "model_params": deepcopy(BASE_MODEL_PARAMS.get(horizon, {
                "alpha_ridge": 0.8,
                "max_pca":     12,
                "clip_ret":    0.10,
                "period":      "2y",
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
        return (
            f"PredictorGenome [H{self.horizon}] {self.genome_id} "
            f"gen={self.generation} fit={self.fitness} hit={self.hit_rate} "
            f"features={len(self.features)} "
            f"alpha={self.model_params.get('alpha_ridge')} "
            f"pca={self.model_params.get('max_pca')}"
        )

    def __repr__(self) -> str:
        return self.describe()


# ══════════════════════════════════════════════════════
# API PÚBLICA — usada por predictor_hN.py
# ══════════════════════════════════════════════════════

def load_active_genome(horizon: int) -> PredictorGenome:
    """
    Carga el genoma campeón para un horizonte dado.
    Si no existe → crea y guarda el default.

    Llamar al inicio de run_predictor_hN():
        genome = load_active_genome(6)
        feature_cols = genome.features
        alpha        = genome.model_params["alpha_ridge"]
    """
    return PredictorGenome.load_champion(horizon)


def get_feature_cols(horizon: int) -> List[str]:
    """Shortcut para obtener solo las features del campeón."""
    return load_active_genome(horizon).features


def get_model_params(horizon: int) -> Dict:
    """Shortcut para obtener solo los parámetros del modelo."""
    return load_active_genome(horizon).model_params


# ══════════════════════════════════════════════════════
# EVALUACIÓN DE HIT RATE — actualiza fitness del genome
# ══════════════════════════════════════════════════════

def update_genome_hit_rate(horizon: int, new_hit_rate: float, n_evals: int) -> None:
    """
    Actualiza el hit rate del genoma campeón con datos frescos del evaluator.
    Llamar desde el evaluator o desde predictor_arena después de cada ciclo.
    """
    genome = PredictorGenome.load_champion(horizon)
    genome.hit_rate = round(new_hit_rate, 4)
    genome.data["n_evaluations"] = n_evals
    genome.data["hit_rate_updated_at"] = datetime.now(timezone.utc).isoformat()
    genome.save_as_champion()
    logger.info(
        f"📊 H{horizon} hit_rate actualizado: {new_hit_rate:.2%} "
        f"({n_evals} evaluaciones)"
    )


# ══════════════════════════════════════════════════════
# INICIALIZAR TODOS LOS GENOMAS (primera vez)
# ══════════════════════════════════════════════════════

def initialize_all_genomes() -> Dict[str, str]:
    """
    Crea los genomas default para H1-H10 si no existen.
    Llamar una sola vez al hacer deploy.
    """
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

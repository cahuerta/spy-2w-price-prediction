"""
darwin_engine/executor_genome.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHIVO NUEVO — darwin_engine/executor_genome.py

El árbol que decide CÓMO usar las predicciones:
  - ¿Cuántos días mínimo mantener según confianza del H?
  - ¿Cuándo ignorar un alpha_kill si H de largo plazo confirma?
  - ¿Qué peso dar a cada H según su hit rate reciente?
  - ¿Cuándo es mejor salir antes del horizonte?

Este árbol EVOLUCIONA. El mutador lo modifica, el arena
promueve la versión que genera más PnL real en Alpaca.

Estructura:
  - ExecutorGenome: clase serializable con todas las reglas
  - evaluate(): dado un trade activo, retorna decisión HOLD/CLOSE
  - to_dict() / from_dict(): serialización para guardar en disco
  - save() / load(): persistencia en /data/darwin/genomes/
"""

import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger("executor_genome")

DATA_PATH   = Path(os.getenv("DATA_PATH", "/data"))
GENOME_DIR  = DATA_PATH / "darwin" / "genomes"

# ══════════════════════════════════════════════════════
# GENOME BASE — v1 calibrado con datos actuales
# ══════════════════════════════════════════════════════

DEFAULT_GENOME = {
    # ── Reglas de mantención mínima ───────────────────
    # Cuántos días mínimo mantener según confianza
    "min_hold_days": {
        "high":   7,    # confianza > high_threshold → aguantar 7 días
        "medium": 4,    # confianza 0.55-0.75 → aguantar 4 días
        "low":    1,    # confianza < 0.55 → salir rápido
    },
    "confidence_thresholds": {
        "high":   0.75,
        "medium": 0.55,
    },

    # ── Pesos por horizonte H ──────────────────────────
    # Cuánto confiar en cada H para la decisión de mantener
    # Basado en hit rates actuales: H6=54.9%, H7=52.8%, H8=53.2%
    "h_weights": {
        "H1":  0.04,
        "H2":  0.06,
        "H3":  0.07,
        "H4":  0.09,
        "H5":  0.10,
        "H6":  0.16,   # mejor actual
        "H7":  0.14,
        "H8":  0.13,
        "H9":  0.11,
        "H10": 0.10,
    },

    # ── Regla de escudo anti-kill ──────────────────────
    # Si estos H confirman la dirección → ignorar alpha_kill
    "kill_shield": {
        "min_confirmations": 2,        # cuántos H deben confirmar
        "min_h_horizon":     5,        # solo H5 en adelante (largo plazo)
        "min_hit_rate":      0.52,     # solo H con hit rate > esto
        "override_kill_below": -0.55,  # no ignorar kill si alpha < esto
    },

    # ── Regla de salida temprana ───────────────────────
    # Cuándo salir ANTES del horizonte (proteger ganancias)
    "early_exit": {
        "profit_lock_pct":   3.0,   # salir si ganancia >= 3%
        "loss_cut_pct":     -2.5,   # salir si pérdida <= -2.5%
        "enabled":           True,
    },

    # ── Regla de confirmación multi-H ─────────────────
    # Para abrir una posición, cuántos H deben estar de acuerdo
    "entry_confirmation": {
        "min_agreeing_h":    3,     # al menos 3 H en la misma dirección
        "min_horizon_for_entry": 4, # solo contar H4 en adelante
        "required_for_open": False, # si False: solo advierte, no bloquea
    },

    # ── Metadata del genome ───────────────────────────
    "genome_id":     "executor_v1",
    "generation":    1,
    "parent_ids":    [],
    "fitness":       None,
    "created_at":    None,
    "n_trades_eval": 0,
}


# ══════════════════════════════════════════════════════
# CLASE PRINCIPAL
# ══════════════════════════════════════════════════════

class ExecutorGenome:

    def __init__(self, genome_dict: Dict = None):
        base = deepcopy(DEFAULT_GENOME)
        if genome_dict:
            # Merge profundo preservando estructura
            for key, val in genome_dict.items():
                if isinstance(val, dict) and key in base and isinstance(base[key], dict):
                    base[key].update(val)
                else:
                    base[key] = val
        self.data = base

    # ── Accesores rápidos ─────────────────────────────

    @property
    def genome_id(self) -> str:
        return self.data.get("genome_id", "executor_v1")

    @property
    def generation(self) -> int:
        return self.data.get("generation", 1)

    @property
    def fitness(self) -> Optional[float]:
        return self.data.get("fitness")

    @fitness.setter
    def fitness(self, value: float):
        self.data["fitness"] = value

    # ══════════════════════════════════════════════════
    # DECISIÓN PRINCIPAL: HOLD o CLOSE
    # ══════════════════════════════════════════════════

    def evaluate(
        self,
        trade: Dict,
        current_pnl_pct: float,
        alpha_score: float,
        h_signals: Dict,
        h_hit_rates: Dict[str, float],
        days_held: int,
    ) -> Dict[str, Any]:
        """
        Dado un trade activo, decide si HOLD o CLOSE.

        Args:
            trade:           dict del trade desde tracker
            current_pnl_pct: PnL actual en %
            alpha_score:     alpha score actual del ticker
            h_signals:       señales H actuales {"H1": {...}, ...}
            h_hit_rates:     hit rates actuales {"H1": 0.41, ...}
            days_held:       días que lleva abierta la posición

        Returns:
            {"action": "HOLD"|"CLOSE", "reason": str, "confidence": float}
        """
        # ── Regla 1: Salida temprana por profit/loss ──
        early = self.data["early_exit"]
        if early.get("enabled", True):
            if current_pnl_pct >= early["profit_lock_pct"]:
                return {
                    "action":     "CLOSE",
                    "reason":     f"PROFIT_LOCK_{current_pnl_pct:.1f}pct",
                    "confidence": 0.90,
                }
            if current_pnl_pct <= early["loss_cut_pct"]:
                return {
                    "action":     "CLOSE",
                    "reason":     f"LOSS_CUT_{current_pnl_pct:.1f}pct",
                    "confidence": 0.85,
                }

        # ── Regla 2: Días mínimos según confianza ─────
        entry_confidence = float(
            trade.get("h_signals_at_entry", {})
                 .get("main", {})
                 .get("confidence", 0.5)
        )
        min_days = self._get_min_hold_days(entry_confidence)

        if days_held < min_days:
            return {
                "action":     "HOLD",
                "reason":     f"MIN_HOLD_{min_days}d_conf_{entry_confidence:.2f}",
                "confidence": 0.75,
            }

        # ── Regla 3: Escudo anti-kill ─────────────────
        shield = self.data["kill_shield"]
        if alpha_score <= shield["override_kill_below"]:
            # Alpha muy negativo — no importa qué digan los H
            return {
                "action":     "CLOSE",
                "reason":     f"KILL_OVERRIDE_alpha_{alpha_score:.3f}",
                "confidence": 0.95,
            }

        confirmations = self._count_h_confirmations(
            h_signals, h_hit_rates,
            trade.get("h_signals_at_entry", {}).get("main", {}).get("recommendation", "COMPRA"),
            min_horizon = shield["min_h_horizon"],
            min_hit_rate = shield["min_hit_rate"],
        )

        if confirmations >= shield["min_confirmations"]:
            # H de largo plazo confirman → proteger posición
            return {
                "action":     "HOLD",
                "reason":     f"SHIELD_{confirmations}_H_confirm_alpha_{alpha_score:.3f}",
                "confidence": 0.80,
            }

        # ── Regla 4: Score ponderado multi-H ──────────
        weighted_score = self._compute_weighted_h_score(h_signals, h_hit_rates)

        direction = trade.get("h_signals_at_entry", {}).get("main", {}).get("recommendation", "COMPRA")
        score_ok = (
            weighted_score > 0.05 if direction == "COMPRA"
            else weighted_score < -0.05
        )

        if score_ok:
            return {
                "action":     "HOLD",
                "reason":     f"H_WEIGHTED_SCORE_{weighted_score:.3f}",
                "confidence": min(0.5 + abs(weighted_score), 0.85),
            }
        else:
            return {
                "action":     "CLOSE",
                "reason":     f"H_SCORE_REVERSED_{weighted_score:.3f}",
                "confidence": 0.65,
            }

    # ── Helpers internos ──────────────────────────────

    def _get_min_hold_days(self, confidence: float) -> int:
        thresholds = self.data["confidence_thresholds"]
        hold_days  = self.data["min_hold_days"]
        if confidence >= thresholds["high"]:
            return hold_days["high"]
        elif confidence >= thresholds["medium"]:
            return hold_days["medium"]
        else:
            return hold_days["low"]

    def _count_h_confirmations(
        self,
        h_signals: Dict,
        h_hit_rates: Dict[str, float],
        direction: str,
        min_horizon: int,
        min_hit_rate: float,
    ) -> int:
        """Cuenta cuántos H de largo plazo confirman la dirección original."""
        confirmations = 0
        for h_key, signal in h_signals.items():
            if not h_key.startswith("H"):
                continue
            try:
                h_num = int(h_key[1:])
            except ValueError:
                continue
            if h_num < min_horizon:
                continue
            hit_rate = h_hit_rates.get(h_key, 0.0)
            if hit_rate < min_hit_rate:
                continue
            pred_ret = float(signal.get("pred_return", 0))
            confirms = (
                pred_ret > 0 if direction == "COMPRA"
                else pred_ret < 0
            )
            if confirms:
                confirmations += 1
        return confirmations

    def _compute_weighted_h_score(
        self,
        h_signals: Dict,
        h_hit_rates: Dict[str, float],
    ) -> float:
        """
        Score ponderado: combina retorno predicho × peso del H × hit rate.
        Positivo = señal de subida. Negativo = señal de bajada.
        """
        weights = self.data["h_weights"]
        total_weight = 0.0
        weighted_sum = 0.0

        for h_key, signal in h_signals.items():
            if h_key == "main" or not h_key.startswith("H"):
                continue
            weight   = weights.get(h_key, 0.05)
            hit_rate = h_hit_rates.get(h_key, 0.5)
            pred_ret = float(signal.get("pred_return", 0))

            # Ajustar peso por hit rate (H con mejor historial pesan más)
            adjusted_weight = weight * max(0.5, hit_rate * 2)
            weighted_sum   += pred_ret * adjusted_weight
            total_weight   += adjusted_weight

        if total_weight == 0:
            return 0.0
        return float(weighted_sum / total_weight)

    # ══════════════════════════════════════════════════
    # SERIALIZACIÓN
    # ══════════════════════════════════════════════════

    def to_dict(self) -> Dict:
        return deepcopy(self.data)

    @classmethod
    def from_dict(cls, d: Dict) -> "ExecutorGenome":
        return cls(genome_dict=d)

    def save(self, genome_id: str = None) -> Path:
        gid  = genome_id or self.genome_id
        path = GENOME_DIR / f"{gid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp  = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)
        )
        tmp.replace(path)
        logger.info(f"💾 Genome guardado: {path}")
        return path

    @classmethod
    def load(cls, genome_id: str) -> Optional["ExecutorGenome"]:
        path = GENOME_DIR / f"{genome_id}.json"
        if not path.exists():
            logger.warning(f"⚠️ Genome no encontrado: {genome_id}")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"❌ Error cargando genome {genome_id}: {e}")
            return None

    @classmethod
    def load_or_default(cls, genome_id: str = "executor_v1") -> "ExecutorGenome":
        """Carga el genome o crea el default si no existe."""
        genome = cls.load(genome_id)
        if genome is None:
            logger.info(f"🆕 Creando genome default: {genome_id}")
            genome = cls()
            genome.data["genome_id"]  = genome_id
            genome.data["created_at"] = datetime.now(timezone.utc).isoformat()
            genome.save(genome_id)
        return genome

    @classmethod
    def load_champion(cls) -> "ExecutorGenome":
        """Carga el genome con mejor fitness conocido."""
        if not GENOME_DIR.exists():
            return cls.load_or_default()

        best_genome  = None
        best_fitness = float("-inf")

        for path in GENOME_DIR.glob("executor_*.json"):
            try:
                data    = json.loads(path.read_text())
                fitness = data.get("fitness")
                if fitness is not None and float(fitness) > best_fitness:
                    best_fitness = float(fitness)
                    best_genome  = data
            except Exception:
                continue

        if best_genome:
            logger.info(
                f"🏆 Champion cargado: {best_genome['genome_id']} "
                f"fitness={best_fitness:.4f}"
            )
            return cls.from_dict(best_genome)

        return cls.load_or_default()

    def describe(self) -> str:
        """Descripción legible del genome para logs."""
        d     = self.data
        hold  = d["min_hold_days"]
        shield = d["kill_shield"]
        early  = d["early_exit"]
        return (
            f"ExecutorGenome [{self.genome_id}] gen={self.generation} "
            f"fitness={self.fitness} | "
            f"hold=({hold['low']}d/{hold['medium']}d/{hold['high']}d) | "
            f"shield={shield['min_confirmations']}H≥H{shield['min_h_horizon']} | "
            f"profit_lock={early['profit_lock_pct']}% "
            f"loss_cut={early['loss_cut_pct']}%"
        )

    def __repr__(self) -> str:
        return self.describe()


# ══════════════════════════════════════════════════════
# LISTA DE TODOS LOS GENOMAS DISPONIBLES
# ══════════════════════════════════════════════════════

def list_executor_genomes() -> List[Dict]:
    """Lista todos los executor genomes guardados con su fitness."""
    if not GENOME_DIR.exists():
        return []

    genomes = []
    for path in GENOME_DIR.glob("executor_*.json"):
        try:
            data = json.loads(path.read_text())
            genomes.append({
                "genome_id":  data.get("genome_id"),
                "generation": data.get("generation"),
                "fitness":    data.get("fitness"),
                "created_at": data.get("created_at"),
                "n_trades":   data.get("n_trades_eval", 0),
            })
        except Exception:
            continue

    genomes.sort(key=lambda x: x.get("fitness") or float("-inf"), reverse=True)
    return genomes

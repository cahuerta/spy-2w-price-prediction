"""
market_orchestrator.py — V2 PRODUCCIÓN REAL NUMÉRICO

ORQUESTADOR DE ENTORNO DE MERCADO

✔ Combina régimen cuantitativo + impacto numérico de noticias
✔ NO usa macro_bias
✔ NO inventa confidence
✔ Ajuste score estructural + impacto macro
✔ Hysteresis limpio
✔ Contratos intactos
"""

from dataclasses import dataclass, asdict
from typing import Dict, Literal
import datetime
import logging


# =================================================================
# LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =================================================================
# CONFIGURACIÓN
# =================================================================

HYSTERESIS_UP = 2

# Score base por régimen estructural
REGIME_BASE_SCORE = {
    "growth": 0.75,
    "neutral": 0.50,
    "defensive": 0.25,
}

# Umbrales score → modo
GROWTH_THRESHOLD = 0.65
DEFENSIVE_THRESHOLD = 0.35


# =================================================================
# DATACLASS DE SALIDA (NO SE MODIFICA)
# =================================================================

@dataclass
class MarketOrchestrationContext:
    market_mode: Literal["growth", "neutral", "defensive"]
    confidence: float
    reason: str
    timestamp: str
    source: Dict

    def to_dict(self) -> Dict:
        return asdict(self)


# =================================================================
# ORQUESTADOR
# =================================================================

class MarketOrchestrator:

    def __init__(self):
        self._last_mode: Literal["growth", "neutral", "defensive"] = "neutral"
        self._up_counter: int = 0
        logger.info("MarketOrchestrator inicializado (modo: neutral)")

    # -------------------------------------------------------------
    # FUNCIÓN PRINCIPAL
    # -------------------------------------------------------------
    def evaluate(self, quant_ctx: Dict, qual_ctx: Dict) -> MarketOrchestrationContext:

        if not quant_ctx or "regime" not in quant_ctx:
            logger.warning("quant_ctx inválido → fallback neutral")
            return self._fallback_neutral("quant_ctx inválido")

        timestamp = datetime.datetime.utcnow().isoformat()

        quant_regime = quant_ctx.get("regime", "neutral")
        base_score = REGIME_BASE_SCORE.get(quant_regime, 0.50)

        # Impacto cualitativo numérico
        impact_score = float(qual_ctx.get("impact_score", 0.0))
        qual_conf = float(qual_ctx.get("aggregated_confidence", 0.0))

        # ---------------------------------------------------------
        # SCORE FINAL AJUSTADO
        # ---------------------------------------------------------
        raw_score = base_score + impact_score
        final_score = max(0.0, min(raw_score, 1.0))

        # ---------------------------------------------------------
        # MAP SCORE → MODO
        # ---------------------------------------------------------
        if final_score >= GROWTH_THRESHOLD:
            computed_mode = "growth"
        elif final_score <= DEFENSIVE_THRESHOLD:
            computed_mode = "defensive"
        else:
            computed_mode = "neutral"

        next_mode = computed_mode
        reason_parts = [
            f"Base regime={quant_regime} ({base_score:.2f})",
            f"Impact={impact_score:.3f}",
            f"Score={final_score:.2f}",
        ]

        # ---------------------------------------------------------
        # HYSTERESIS SOLO PARA SUBIDA A GROWTH
        # ---------------------------------------------------------
        if computed_mode == "growth":
            if self._last_mode in ["neutral", "growth"]:
                self._up_counter += 1
                if self._up_counter < HYSTERESIS_UP:
                    next_mode = "neutral"
                    reason_parts.append(
                        f"Hysteresis growth warming ({self._up_counter}/{HYSTERESIS_UP})"
                    )
            else:
                # desde defensive → neutral primero
                next_mode = "neutral"
                self._up_counter = 1
                reason_parts.append("Recovery from defensive (hysteresis)")

        else:
            self._up_counter = 0

        # ---------------------------------------------------------
        # CONFIDENCE REAL (SIN INVENTAR 0.5)
        # ---------------------------------------------------------
        # Confidence estructural fuerte si score lejos de neutral
        structural_conf = abs(final_score - 0.5) * 2  # escala 0–1
        structural_conf = max(0.0, min(structural_conf, 1.0))

        confidence = round((structural_conf + qual_conf) / 2, 2)

        # ---------------------------------------------------------
        # LOGGING
        # ---------------------------------------------------------
        if next_mode != self._last_mode:
            logger.info(
                f"🎯 MODE CHANGE: {self._last_mode} → {next_mode} | score={final_score:.2f} | conf={confidence:.2f}"
            )
        else:
            logger.debug(f"Mode stable: {next_mode}")

        self._last_mode = next_mode

        # ---------------------------------------------------------
        # SALIDA (CONTRATO INTACTO)
        # ---------------------------------------------------------
        return MarketOrchestrationContext(
            market_mode=next_mode,
            confidence=confidence,
            reason=" | ".join(reason_parts),
            timestamp=timestamp,
            source={
                "quant_regime": quant_regime,
                "impact_score": impact_score,
                "aggregated_confidence": qual_conf,
                "base_score": base_score,
                "final_score": final_score,
                "last_mode": self._last_mode,
                "up_counter": self._up_counter,
                "raw_quant": quant_ctx,
                "raw_qual": qual_ctx,
            },
        )

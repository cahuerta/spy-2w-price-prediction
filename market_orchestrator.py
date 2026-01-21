"""
market_orchestrator.py — V1.1 PRODUCCIÓN

ORQUESTADOR DE ENTORNO DE MERCADO

✔ Combina evaluador cuantitativo + cualitativo
✔ Decide market_mode (NO compra / NO vende)
✔ IA puede FORZAR defensive en eventos extremos
✔ Hysteresis para evitar flip-flop
✔ Logging auditable + validaciones

Input: QuantMarketContext + QualitativeMarketContext
Output: MarketOrchestrationContext
"""

from dataclasses import dataclass, asdict
from typing import Dict, Literal
import datetime
import logging

# =================================================================
# LOGGING
# =================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Corregido: __name__ no "name"

# =================================================================
# CONFIGURACIÓN DE HYSTERESIS
# =================================================================
HYSTERESIS_UP = 2  # Ciclos mínimos para ↑ riesgo (defensive → neutral → growth)

# =================================================================
# DATACLASS DE SALIDA
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
    """Orquesta el estado del mercado combinando sensores cuantitativos y cualitativos.
    NO ejecuta decisiones de inversión. Solo contexto de entorno."""

    def __init__(self):
        self._last_mode: Literal["growth", "neutral", "defensive"] = "neutral"
        self._up_counter: int = 0
        logger.info("MarketOrchestrator inicializado (modo: neutral)")

    # -------------------------------------------------------------
    # FUNCIÓN PRINCIPAL
    # -------------------------------------------------------------
    def evaluate(self, quant_ctx: Dict, qual_ctx: Dict) -> MarketOrchestrationContext:
        """Evalúa entorno combinando quant + qual con hysteresis."""
        
        # Validaciones input
        if not quant_ctx or 'regime' not in quant_ctx:
            logger.warning("quant_ctx inválido → fallback neutral")
            return self._fallback_neutral("quant_ctx inválido")
        
        if not qual_ctx or 'macro_bias' not in qual_ctx:
            logger.warning("qual_ctx inválido → usar quant_ctx solo")
            # Continuar con quant_ctx válido
        
        timestamp = datetime.datetime.utcnow().isoformat()
        quant_regime = quant_ctx.get("regime", "neutral")
        qual_bias = qual_ctx.get("macro_bias", "neutral") 
        qual_conf = float(qual_ctx.get("confidence", 0.0))

        reason_parts = []
        next_mode = self._last_mode

        logger.debug(f"Eval: quant={quant_regime}, qual={qual_bias}({qual_conf:.2f}), last={self._last_mode}")

        # =========================================================
        # 1️⃣ VETO ESTRUCTURAL CUANTITATIVO (IMEDIATO)
        # =========================================================
        if quant_regime == "defensive":
            next_mode = "defensive"
            reason_parts.append("VETO: régimen cuantitativo defensive")
            self._up_counter = 0

        # =========================================================
        # 2️⃣ VETO CUALITATIVO (EVENTOS MACRO EXTREMOS)
        # =========================================================
        elif qual_bias == "risk_off" and qual_conf >= 0.7:
            next_mode = "defensive"
            reason_parts.append(f"VETO: risk_off fuerte (conf {qual_conf:.1f})")
            self._up_counter = 0

        # =========================================================
        # 3️⃣ GROWTH (solo con condiciones limpias + hysteresis)
        # =========================================================
        elif quant_regime == "growth" and qual_bias != "risk_off":
            if self._last_mode in ["neutral", "growth"]:
                self._up_counter += 1
                if self._up_counter >= HYSTERESIS_UP:
                    next_mode = "growth"
                    reason_parts.append("GROWTH: quant growth sostenido")
                else:
                    next_mode = "neutral"
                    reason_parts.append(f"NEUTRAL: calentando growth (ciclo {self._up_counter}/{HYSTERESIS_UP})")
            else:
                # Desde defensive → neutral primero
                self._up_counter = 1
                next_mode = "neutral"
                reason_parts.append("NEUTRAL: recuperación desde defensive")

        # =========================================================
        # 4️⃣ DEFAULT NEUTRAL
        # =========================================================
        else:
            next_mode = "neutral"
            reason_parts.append("NEUTRAL: señales mixtas")
            self._up_counter = 0

        # =========================================================
        # CALCULAR CONFIDENCE
        # =========================================================
        base_conf = qual_conf if qual_conf > 0 else 0.5
        confidence = round(
            base_conf if quant_regime == next_mode else base_conf * 0.8, 2
        )

        # =========================================================
        # LOGGING Y ACTUALIZAR ESTADO
        # =========================================================
        if next_mode != self._last_mode:
            logger.info(f"🎯 MODO CAMBIO: {self._last_mode} → {next_mode} | {confidence:.2f} | {reason_parts[-1]}")
        else:
            logger.debug(f"✅ MODO ESTABLE: {next_mode}")

        self._last_mode = next_mode

        return MarketOrchestrationContext(
            market_mode=next_mode,
            confidence=confidence,
            reason=" | ".join(reason_parts),
            timestamp=timestamp,
            source={
                "quant_regime": quant_regime,
                "qual_bias": qual_bias,
                "qual_confidence": qual_conf,
                "last_mode": self._last_mode,
                "up_counter": self._up_counter,
                "raw_quant": quant_ctx,
                "raw_qual": qual_ctx,
            },
        )

    def _fallback_neutral(self, reason: str) -> MarketOrchestrationContext:
        """Fallback seguro ante inputs inválidos."""
        timestamp = datetime.datetime.utcnow().isoformat()
        return MarketOrchestrationContext(
            market_mode="neutral",
            confidence=0.0,
            reason=f"FALLBACK: {reason}",
            timestamp=timestamp,
            source={},
        )

    def get_state(self) -> Dict:
        """Estado interno para monitoring."""
        return {
            "last_mode": self._last_mode,
            "up_counter": self._up_counter,
        }

# =================================================================
# USO (ejemplo)
# =================================================================
"""
# Inicializar (singleton recomendado)
orchestrator = MarketOrchestrator()

# Loop de producción
quant_ctx = market_state_evaluator.evaluate_market_state(...)
qual_ctx = market_qualitative_evaluator.evaluate_qualitative_market(quant_ctx.to_dict())

ctx = orchestrator.evaluate(quant_ctx.to_dict(), qual_ctx.to_dict())
print(ctx.to_dict())

# Monitoring
print(orchestrator.get_state())
"""

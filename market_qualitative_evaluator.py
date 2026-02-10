"""
market_qualitative_evaluator.py — V1.3 PRODUCCIÓN

EVALUADOR CUALITATIVO DE ENTORNO DE MERCADO (IA PURA)

✔ NO decide | NO asigna capital | NO predice precios
✔ SOLO contexto macro y sistémico
✔ Complementa market_state_evaluator
✔ Fallback conservador ante error IA
✔ Input validation + logging production-ready
✔ Blindado contra inputs no escalares
"""

from dataclasses import dataclass, asdict
from typing import Dict, Literal, Optional
import os
import json
import datetime
import logging
import numpy as np
from openai import OpenAI

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURACIÓN IA
# =========================================================
IA_MODEL = "gpt-4o-mini"
MAX_TOKENS = 300
TEMPERATURE = 0.15  # Bajo = estable, poco narrativo

# =========================================================
# DATACLASS SALIDA
# =========================================================
@dataclass
class QualitativeMarketContext:
    macro_bias: Literal["risk_on", "neutral", "risk_off"]
    confidence: float  # 0.0 – 1.0
    rationale: str     # 2–3 líneas máximo
    timestamp: str
    model: str = IA_MODEL

    def to_dict(self) -> Dict:
        return asdict(self)

# =========================================================
# UTILIDAD CRÍTICA — NORMALIZACIÓN ESCALAR
# =========================================================
def safe_float(x, default: float = 0.0) -> float:
    """
    Convierte cualquier input razonable a float escalar.

    Acepta:
    - float / int
    - np.nan
    - list / tuple / ndarray
    - pandas Series
    - None

    Nunca lanza excepción.
    """
    try:
        if x is None:
            return default

        if isinstance(x, (float, int)):
            return float(x)

        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            return float(np.nanmean(arr)) if arr.size else default

        # pandas Series o similares
        if hasattr(x, "mean"):
            v = x.mean()
            return float(v) if np.isfinite(v) else default

        return float(x)

    except Exception:
        return default

# =========================================================
# PROMPT INSTITUCIONAL
# =========================================================
SYSTEM_PROMPT = """Eres analista macroeconómico senior de un hedge fund institucional.

REGLAS ABSOLUTAS:
NO predecir precios
NO recomendar comprar o vender
NO asignar capital
NO dar trading advice

TAREA: Evaluar SOLO el CONTEXTO MACRO Y SISTÉMICO actual usando:
- datos cuantitativos objetivos
- noticias y eventos recientes

FORMATO DE SALIDA (JSON estricto): 
{ 
  "macro_bias": "risk_on | neutral | risk_off", 
  "confidence": 0.0–1.0, 
  "rationale": "2–3 líneas objetivas" 
}

CRITERIOS:
- risk_on: crecimiento, baja volatilidad, entorno favorable
- risk_off: recesión, estrés financiero, alta volatilidad, tensiones geopolíticas  
- neutral: señales mixtas o poco concluyentes

Si la información es ambigua o insuficiente → usar "neutral" con confidence < 0.6
"""

# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================
def evaluate_qualitative_market(
    quant_context: Dict,
    news_summary: Optional[str] = None
) -> QualitativeMarketContext:
    """Evaluación cualitativa basada en números reales + noticias"""

    timestamp = datetime.datetime.utcnow().isoformat()

    # --------------------
    # VALIDACIÓN MÍNIMA
    # --------------------
    required_fields = ["volatility", "drawdown_rolling", "trend_strength"]
    for field in required_fields:
        if field not in quant_context:
            logger.warning(f"Input faltante: {field}")
            return QualitativeMarketContext(
                macro_bias="neutral",
                confidence=0.0,
                rationale=f"Input faltante: {field}",
                timestamp=timestamp,
                model=IA_MODEL,
            )

    # --------------------
    # OPENAI CLIENT
    # --------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY no configurada")
        return QualitativeMarketContext(
            macro_bias="neutral",
            confidence=0.0,
            rationale="API key no disponible",
            timestamp=timestamp,
            model=IA_MODEL,
        )

    client = OpenAI(api_key=api_key)

    # --------------------
    # CONTEXTO CUANTITATIVO NORMALIZADO
    # --------------------
    vol = safe_float(quant_context.get("volatility"))
    dd = safe_float(quant_context.get("drawdown_rolling"))
    trend = safe_float(quant_context.get("trend_strength"))
    corr = safe_float(quant_context.get("cross_asset_correlation"))

    quant_block = f"""
CONTEXTO CUANTITATIVO ACTUAL:

Volatilidad: {vol:.1%}
Drawdown rolling: {dd:.1%}
Trend strength: {trend:.1%}
Cross-asset correlation: {corr:.1%}
Regime cuantitativo: {quant_context.get('regime', 'neutral')}
Downside risk: {quant_context.get('downside_risk', 'unknown')}
"""

    user_prompt = quant_block

    if news_summary:
        truncated = news_summary[:800]
        if len(news_summary) > 800:
            truncated += "... [truncado]"
        user_prompt += f"\n\nNOTICIAS RECIENTES:\n{truncated}"

    # --------------------
    # LLAMADA IA
    # --------------------
    try:
        response = client.chat.completions.create(
            model=IA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        result = json.loads(content)

        macro_bias = result.get("macro_bias", "neutral")
        confidence = safe_float(result.get("confidence"), 0.0)
        rationale = str(result.get("rationale", "Sin explicación"))[:200]

        if macro_bias not in ["risk_on", "neutral", "risk_off"]:
            raise ValueError(f"macro_bias inválido: {macro_bias}")

        confidence = min(max(confidence, 0.0), 1.0)

        logger.info(f"Qual eval OK: {macro_bias} (conf {confidence:.2f})")

        return QualitativeMarketContext(
            macro_bias=macro_bias,
            confidence=confidence,
            rationale=rationale,
            timestamp=timestamp,
            model=IA_MODEL,
        )

    except Exception as e:
        logger.error(f"Qual eval fallback: {str(e)[:120]}")
        return QualitativeMarketContext(
            macro_bias="neutral",
            confidence=0.0,
            rationale=f"Error IA: {str(e)[:120]}",
            timestamp=timestamp,
            model=IA_MODEL,
        )

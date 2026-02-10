"""
market_qualitative_evaluator.py — V1.2 PRODUCCIÓN

EVALUADOR CUALITATIVO DE ENTORNO DE MERCADO (IA PURA)

✔ NO decide | NO asigna capital | NO predice precios
✔ SOLO contexto macro y sistémico
✔ Complementa market_state_evaluator
✔ Fallback conservador ante error IA
✔ Input validation + logging production-ready

Input: contexto cuantitativo + noticias reales
Output: sesgo macro + confianza + racional breve (JSON)
"""

from dataclasses import dataclass, asdict
from typing import Dict, Literal, Optional
import os
import json
import datetime
import logging
from openai import OpenAI

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =================================================================
# CONFIGURACIÓN IA
# =================================================================
IA_MODEL = "gpt-4o-mini"
MAX_TOKENS = 300
TEMPERATURE = 0.15  # Bajo = estable, poco narrativo

# =================================================================
# DATACLASS SALIDA
# =================================================================
@dataclass
class QualitativeMarketContext:
    macro_bias: Literal["risk_on", "neutral", "risk_off"]
    confidence: float  # 0.0 – 1.0
    rationale: str     # 2–3 líneas máximo
    timestamp: str
    model: str = IA_MODEL

    def to_dict(self) -> Dict:
        return asdict(self)

# =================================================================
# PROMPT INSTITUCIONAL (ROL SISTEMA)
# =================================================================
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

Si la información es ambigua o insuficiente → usar "neutral" con confidence < 0.6"""

# =================================================================
# FUNCIÓN PRINCIPAL
# =================================================================
def evaluate_qualitative_market(
    quant_context: Dict, 
    news_summary: Optional[str] = None
) -> QualitativeMarketContext:
    """Evaluación cualitativa basada en números reales + noticias"""
    
    timestamp = datetime.datetime.utcnow().isoformat()
    
    # --------------------
    # VALIDACIÓN INPUT
    # --------------------
    required_fields = ['volatility', 'drawdown_rolling', 'trend_strength']
    for field in required_fields:
        if field not in quant_context or quant_context[field] is None:
            logger.warning(f"Input faltante: {field}")
            return QualitativeMarketContext(
                macro_bias="neutral",
                confidence=0.0,
                rationale=f"Input faltante: {field}",
                timestamp=timestamp,
                model=IA_MODEL,
            )
    
    # --------------------
    # Cliente OpenAI
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
    # Input cuantitativo formateado
    # --------------------
    quant_block = f"""
CONTEXTO CUANTITATIVO ACTUAL:

Volatilidad: {float(quant_context.get('volatility', 0)):.1%}
Drawdown rolling: {float(quant_context.get('drawdown_rolling', 0)):.1%}
Trend strength: {float(quant_context.get('trend_strength', 0)):.1%}
Cross-asset correlation: {float(quant_context.get('cross_asset_correlation', 0)):.1%}
Regime cuantitativo: {quant_context.get('regime', 'neutral')}
Downside risk: {quant_context.get('downside_risk', 'unknown')}
"""
    user_prompt = quant_block
    
    if news_summary:
        # Truncado inteligente
        truncated_news = news_summary[:800]
        if len(news_summary) > 800:
            truncated_news += "... [truncado]"
        user_prompt += f"""

NOTICIAS RECIENTES: {truncated_news}"""
    
    # --------------------
    # Llamada IA con fallback
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
        
        # Validar output IA
        macro_bias = result.get("macro_bias", "neutral")
        confidence = float(result.get("confidence", 0.0))
        rationale = result.get("rationale", "Sin explicación")[:200]  # Max 200 chars
        
        if macro_bias not in ["risk_on", "neutral", "risk_off"]:
            raise ValueError(f"macro_bias inválido: {macro_bias}")
        
        if not (0 <= confidence <= 1):
            raise ValueError(f"confidence inválido: {confidence}")
        
        logger.info(f"Qual eval OK: {macro_bias} (conf: {confidence:.2f})")
        
        return QualitativeMarketContext(
            macro_bias=macro_bias,
            confidence=confidence,
            rationale=rationale,
            timestamp=timestamp,
            model=IA_MODEL,
        )
        
    except Exception as e:
        logger.error(f"Qual eval fallback: {str(e)[:100]}")
        return QualitativeMarketContext(
            macro_bias="neutral",
            confidence=0.0,
            rationale=f"Error IA: {str(e)[:120]}",
            timestamp=timestamp,
            model=IA_MODEL,
        )

# =================================================================
# USO (ejemplo)
# =================================================================
"""
quant = market_state_evaluator.evaluate_market_state(...)
qual = evaluate_qualitative_market(quant.to_dict(), "Fed minutes + earnings")
print(qual.to_dict())
"""

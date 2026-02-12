# =========================================================
# screener_ia_module.py
# IA Context Enrichment Module para Screener — V2 PRO ✅
# Async + Cache + Rate Limit + JSON Parser Robusto
# =========================================================

from typing import List, Dict, Any
import os
import logging
import asyncio
import json
import time
from functools import lru_cache
from threading import Lock
import re

# =========================================================
# LOGGING
# =========================================================
logger = logging.getLogger("screener_ia")

# =========================================================
# CONFIG
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("SCREENER_IA_MODEL", "gpt-4o-mini")
MAX_CONCURRENT = int(os.getenv("IA_CONCURRENT", "5"))
REQUESTS_PER_MINUTE = int(os.getenv("IA_RATE_LIMIT", "30"))
CACHE_SIZE = int(os.getenv("IA_CACHE", "200"))

# Rate limiting
_request_lock = Lock()
_requests_count = 0
_last_reset = time.time()

# =========================================================
# OPENAI CLIENT (v1+)
# =========================================================
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# =========================================================
# ROBUST JSON PARSER
# =========================================================
def _extract_json_from_response(content: str) -> Dict[str, Any]:
    """Extrae JSON incluso si OpenAI devuelve markdown```
    Patterns comunes:
    - ```json ... ```
    - JSON puro
    - JSON al final después de texto
    """
    if not content:
        return {}

    # 1. Buscar bloques ```json
    json_match = re.search(r'```(?:json)?s*({.*?})s*```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 2. Buscar último { ... }
        json_match = re.search(r'{[^{}]*?(?:}[^{}]*)*}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return {}

    try:
        return json.loads(json_str)
    except:
        return {}

# =========================================================
# PROMPT BUILDER MEJORADO
# =========================================================
def _build_prompt(candidate: Dict[str, Any]) -> str:
    score = candidate.get("score", 0)
    trend = candidate.get("trend_3m_pct", 0)
    vol = candidate.get("volatility", 0)
    rsi = candidate.get("rsi_wilder", 50)
    sharpe = candidate.get("sharpe_ratio", 0)
    beta = candidate.get("beta_spy", 1.0)
    
    return f"""Analyze this stock for screener context (Feb 2026 market):

TICKER: {candidate.get("ticker")}
QUANT SCORE: {score:.2f}/100
TREND 3M: {trend:+.1f}%
VOLATILITY: {vol:.1%}
RSI: {rsi:.0f}
SHARPE: {sharpe:.2f}
BETA SPY: {beta:.2f}

Output ONLY valid JSON:
{{
  "context_score": 0.00-1.00,
  "event_risk": 0.00-1.00,
  "summary": "1 sentence max"
}}"""

# =========================================================
# RATE LIMITING
# =========================================================
async def _rate_limit():
    global _requests_count, _last_reset
    with _request_lock:
        now = time.time()
        if now - _last_reset > 60:  # Reset minuto
            _requests_count = 0
            _last_reset = now
        
        if _requests_count >= REQUESTS_PER_MINUTE:
            sleep_time = 60 - (now - _last_reset)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                _requests_count = 0
                _last_reset = time.time()
        
        _requests_count += 1

# =========================================================
# SINGLE ENRICH (CACHED)
# =========================================================
@lru_cache(maxsize=CACHE_SIZE)
def _build_cache_key(ticker: str, score: float, trend: float, rsi: float) -> str:
    return f"{ticker}_{score:.1f}_{trend:.1f}_{rsi:.0f}"

async def _enrich_one(client: AsyncOpenAI, candidate: Dict[str, Any]) -> Dict[str, Any]:
    ticker = candidate.get("ticker", "UNKNOWN")
    
    try:
        await _rate_limit()
        
        prompt = _build_prompt(candidate)
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Financial AI analyst. ONLY JSON response."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=150
        )

        content = response.choices[0].message.content or ""
        json_data = _extract_json_from_response(content)

        # Validación estricta
        context_score = float(json_data.get("context_score", 0.5))
        event_risk = float(json_data.get("event_risk", 0.5))
        
        # Clamp valores
        context_score = max(0.0, min(1.0, context_score))
        event_risk = max(0.0, min(1.0, event_risk))

        candidate["ia"] = {
            "context_score": context_score,
            "event_risk": event_risk,
            "summary": json_data.get("summary", "IA analysis complete")[:100]
        }
        
        logger.debug(f"✅ IA enriched: {ticker}")
        return candidate

    except Exception as e:
        logger.warning(f"❌ IA failed {ticker}: {str(e)[:50]}")
        candidate["ia"] = {
            "context_score": 0.5,
            "event_risk": 0.3,  # Neutral bajo riesgo
            "summary": "Fallback analysis"
        }
        return candidate

# =========================================================
# BATCH ENRICH OPTIMIZADO
# =========================================================
async def enrich_screener_candidates_batch(
    candidates: List[Dict[str, Any]],
    semaphore: asyncio.Semaphore = None
) -> List[Dict[str, Any]]:
    """
    Enriquecimiento IA batch con concurrency control.
    
    Args:
        semaphore: Para control externo de concurrency
    """
    if not OPENAI_AVAILABLE:
        logger.warning("❌ OpenAI SDK no disponible")
        return candidates

    if not OPENAI_API_KEY:
        logger.warning("❌ OPENAI_API_KEY missing")
        return candidates

    if not candidates:
        return candidates

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    semaphore = semaphore or asyncio.Semaphore(MAX_CONCURRENT)

    async def _semaphore_enrich(client, candidate):
        async with semaphore:
            return await _enrich_one(client, candidate)

    tasks = [_semaphore_enrich(client, c) for c in candidates]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    enriched = [r for r in results if isinstance(r, dict)]
    logger.info(f"🎯 IA batch complete: {len(enriched)}/{len(candidates)} enriched")
    
    return enriched

# =========================================================
# CLEAR CACHE
# =========================================================
def clear_ia_cache():
    """Limpia cache IA para refresh"""
    _build_cache_key.cache_clear()
    logger.info("🧹 IA cache limpiado")

# =========================================================
# TEST
# =========================================================
async def test_ia_module():
    """Test standalone"""
    candidates = [
        {
            "ticker": "AAPL",
            "score": 85.5,
            "trend_3m_pct": 12.3,
            "volatility": 0.25,
            "rsi_wilder": 65,
            "sharpe_ratio": 1.8,
            "beta_spy": 1.2
        },
        {
            "ticker": "TSLA", 
            "score": 92.1,
            "trend_3m_pct": 25.7,
            "volatility": 0.45,
            "rsi_wilder": 78,
            "sharpe_ratio": 2.1,
            "beta_spy": 1.8
        }
    ]
    
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        enriched = await enrich_screener_candidates_batch(candidates)
        for c in enriched:
            print(f"{c['ticker']}: {c['ia']}")
    else:
        print("Test: OpenAI no disponible")

if __name__ == "__main__":
    asyncio.run(test_ia_module())

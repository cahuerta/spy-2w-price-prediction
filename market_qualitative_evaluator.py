"""
market_qualitative_evaluator.py — V3 PRODUCCIÓN REAL

✔ RSS feeds múltiples
✔ Filtrado por dominio confiable
✔ Evaluación INDIVIDUAL por noticia
✔ Agregación ponderada (impact × severity × confidence)
✔ Impacto final acotado [-0.25, +0.15]
✔ Logging robusto
✔ Listo para cron diario
✔ NO mezcla cuantitativo
"""

from dataclasses import dataclass, asdict
from typing import Dict, List
import os
import json
import datetime
import logging
import feedparser
from openai import OpenAI

# =========================================================
# CONFIG
# =========================================================

IA_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1
MAX_TOKENS = 300

MAX_NEG_IMPACT = -0.25
MAX_POS_IMPACT = 0.15

MAX_NEWS_PER_FEED = 5
MAX_TOTAL_NEWS = 12

TRUSTED_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "federalreserve.gov",
    "ecb.europa.eu",
    "cnbc.com",
    "marketwatch.com"
]

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/Reuters/marketsNews",
    "https://www.ft.com/?format=rss",
    "https://feeds.reuters.com/Reuters/TopNews",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# =========================================================
# DATACLASS
# =========================================================

@dataclass
class QualitativeMarketImpact:
    impact_score: float
    events_detected: int
    aggregated_confidence: float
    rationale: str
    timestamp: str
    model: str = IA_MODEL

    def to_dict(self) -> Dict:
        return asdict(self)

# =========================================================
# FETCH RSS
# =========================================================

def fetch_market_news() -> List[str]:
    news_items = []

    for feed_url in RSS_FEEDS:
        try:
            logger.info(f"Fetching {feed_url}")
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:MAX_NEWS_PER_FEED]:
                link = entry.get("link", "").lower()

                if any(domain in link for domain in TRUSTED_DOMAINS):
                    title = entry.get("title", "")[:120]
                    summary = entry.get("summary", "")[:600]

                    news_item = f"Título: {title}\nResumen: {summary}"
                    news_items.append(news_item)

        except Exception as e:
            logger.warning(f"RSS error {feed_url}: {e}")

    logger.info(f"Total trusted news collected: {len(news_items)}")
    return news_items[:MAX_TOTAL_NEWS]

# =========================================================
# PROMPT
# =========================================================

SYSTEM_PROMPT = """
Eres analista macro institucional especializado en eventos sistémicos.

Evalúa UNA sola noticia macroeconómica.

IGNORA:
- Earnings individuales
- Opiniones de analistas
- Métricas técnicas
- Crypto o activos individuales

Escala impacto:
-0.25 shock sistémico severo
-0.15 evento monetario restrictivo fuerte
-0.10 macro negativo relevante
-0.05 negativo leve
 0.00 irrelevante
+0.05 macro positivo
+0.10 macro muy positivo
+0.15 estímulo fuerte inesperado

Devuelve JSON estricto:
{
  "impact_score": float,
  "severity": float [0-1],
  "confidence": float [0-1],
  "rationale": "máx 100 caracteres"
}
"""

# =========================================================
# EVALUAR NOTICIA INDIVIDUAL
# =========================================================

def evaluate_single_news(client: OpenAI, news_text: str) -> Dict:

    try:
        response = client.chat.completions.create(
            model=IA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": news_text[:2000]},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        impact = float(result.get("impact_score", 0.0))
        impact = max(min(impact, MAX_POS_IMPACT), MAX_NEG_IMPACT)

        severity = min(max(float(result.get("severity", 0.0)), 0.0), 1.0)
        confidence = min(max(float(result.get("confidence", 0.0)), 0.0), 1.0)

        rationale = str(result.get("rationale", ""))[:120]

        return {
            "impact": impact,
            "severity": severity,
            "confidence": confidence,
            "rationale": rationale
        }

    except Exception as e:
        logger.error(f"News eval error: {e}")
        return {
            "impact": 0.0,
            "severity": 0.0,
            "confidence": 0.0,
            "rationale": "evaluation error"
        }

# =========================================================
# CORE
# =========================================================

def evaluate_qualitative_market() -> QualitativeMarketImpact:

    timestamp = datetime.datetime.utcnow().isoformat()
    news_items = fetch_market_news()

    if not news_items:
        return QualitativeMarketImpact(
            impact_score=0.0,
            events_detected=0,
            aggregated_confidence=0.0,
            rationale="No trusted macro news detected",
            timestamp=timestamp,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return QualitativeMarketImpact(
            impact_score=0.0,
            events_detected=len(news_items),
            aggregated_confidence=0.0,
            rationale="OPENAI_API_KEY missing",
            timestamp=timestamp,
        )

    client = OpenAI(api_key=api_key)

    impacts = []
    confidences = []
    rationales = []

    for news in news_items:
        result = evaluate_single_news(client, news)

        weighted_impact = result["impact"] * result["severity"] * result["confidence"]

        impacts.append(weighted_impact)
        confidences.append(result["confidence"])

        if abs(result["impact"]) > 0.02:
            rationales.append(result["rationale"])

    if not impacts:
        return QualitativeMarketImpact(
            impact_score=0.0,
            events_detected=0,
            aggregated_confidence=0.0,
            rationale="No impactful macro events",
            timestamp=timestamp,
        )

    final_impact = sum(impacts)
    final_impact = max(min(final_impact, MAX_POS_IMPACT), MAX_NEG_IMPACT)

    aggregated_conf = round(sum(confidences) / len(confidences), 2)

    rationale_summary = (
        " | ".join(rationales[:3])
        if rationales
        else "No significant macro impact"
    )

    logger.info(f"FINAL IMPACT: {final_impact:.4f} | Conf: {aggregated_conf}")

    return QualitativeMarketImpact(
        impact_score=round(final_impact, 4),
        events_detected=len(news_items),
        aggregated_confidence=aggregated_conf,
        rationale=rationale_summary,
        timestamp=timestamp,
    )

# =========================================================
# CLI TEST
# =========================================================

if __name__ == "__main__":
    result = evaluate_qualitative_market()
    print(json.dumps(result.to_dict(), indent=2))

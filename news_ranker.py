# =========================================================
# news_ranker.py — RANKING DE NOTICIAS POR TICKER (V1)
# =========================================================
# Flujo:
#   1. Trae noticias del día (Alpaca News API, UNA sola llamada
#      cubriendo todo el mercado — no una llamada por ticker).
#   2. Un solo prompt al LLM con todas las noticias juntas →
#      devuelve top 15 alcista + top 15 bajista.
#   3. Alcistas fuera del universo: se validan contra Alpaca Assets
#      (solo acciones individuales operables, no ETF/cripto/índices)
#      antes de agregarse a tickers.json.
#   4. Bajistas fuera del universo: se ignoran (no se agregan).
#   5. Resultado persistido en /data/news_ranking.json para que
#      alpha_engine_v4.py (ajustar alpha_score) e intraday_tracker.py
#      (stop más ajustado en posiciones abiertas) lo consuman después.
#
# Diseño acordado en conversación 2026-08-25:
#   - Peso conservador inicial en el ensemble de alfa (a definir
#     cuando se integre a alpha_engine_v4.py).
#   - No reemplaza al screener técnico — es una fuente de candidatos
#     PARALELA, mismo destino (tickers.json).
#   - No toca el stop loss duro de los PM — solo el umbral de
#     "diverging" en intraday_tracker.py, más ajustado ese día.
# =========================================================

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [NEWS_RANKER] %(message)s")
logger = logging.getLogger("news_ranker")

# =========================================================
# CONFIG
# =========================================================

DATA_PATH  = Path(os.getenv("DATA_PATH", "/data"))
OUTPUT_FILE = DATA_PATH / "news_ranking.json"
TICKERS_FILE = DATA_PATH / "tickers.json"

ALPACA_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_TRADING_BASE = os.getenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets")

NEWS_URL   = "https://data.alpaca.markets/v1beta1/news"
ASSETS_URL = f"{ALPACA_TRADING_BASE}/v2/assets"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
IA_MODEL       = os.getenv("NEWS_RANKER_MODEL", "gpt-4o-mini")

NEWS_HOURS_BACK = int(os.getenv("NEWS_RANKER_HOURS_BACK", "24"))
NEWS_LIMIT      = int(os.getenv("NEWS_RANKER_LIMIT", "50"))
TOP_N           = int(os.getenv("NEWS_RANKER_TOP_N", "15"))

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# =========================================================
# HELPERS
# =========================================================

def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"⚠️ load_json {path}: {e}")
        return None


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _load_universe() -> List[str]:
    """Mismo formato que signals.py::load_universe()."""
    d = _load_json(TICKERS_FILE)
    if d is None:
        return []
    if isinstance(d, list):
        return [str(t).upper() for t in d]
    if isinstance(d, dict):
        return [str(t).upper() for t in d.get("tickers", [])]
    return []


def _add_to_universe(ticker: str) -> bool:
    """
    Agrega un ticker a tickers.json si no está ya. Retorna True si
    se agregó, False si ya existía.
    """
    d = _load_json(TICKERS_FILE)
    ticker = ticker.upper()

    if d is None:
        _save_json(TICKERS_FILE, [ticker])
        return True

    if isinstance(d, list):
        if ticker in [str(t).upper() for t in d]:
            return False
        d.append(ticker)
        _save_json(TICKERS_FILE, d)
        return True

    if isinstance(d, dict):
        tickers = d.get("tickers", [])
        if ticker in [str(t).upper() for t in tickers]:
            return False
        tickers.append(ticker)
        d["tickers"] = tickers
        _save_json(TICKERS_FILE, d)
        return True

    return False


# =========================================================
# 1. TRAER NOTICIAS (Alpaca News API — una sola llamada)
# =========================================================

def fetch_market_news() -> List[Dict[str, Any]]:
    """
    Trae las noticias más recientes de todo el mercado en UNA llamada
    (sin filtrar por símbolo — así el LLM puede descubrir tickers que
    ni siquiera están en nuestro universo todavía).
    """
    if not ALPACA_KEY or not ALPACA_SECRET:
        logger.error("❌ ALPACA_API_KEY / ALPACA_SECRET_KEY no configuradas")
        return []

    headers = {
        "APCA-API-KEY-ID":     ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }
    start = (datetime.now(timezone.utc) - timedelta(hours=NEWS_HOURS_BACK)).isoformat()
    params = {
        "start": start,
        "limit": NEWS_LIMIT,
        "sort":  "desc",
    }

    try:
        resp = requests.get(NEWS_URL, headers=headers, params=params, timeout=20)
        if resp.status_code != 200:
            logger.error(f"❌ Alpaca News API HTTP {resp.status_code}: {resp.text[:300]}")
            return []
        data = resp.json()
        news = data.get("news", [])
        logger.info(f"📰 {len(news)} noticias recibidas (últimas {NEWS_HOURS_BACK}h)")
        return news
    except Exception as e:
        logger.error(f"❌ fetch_market_news error: {e}")
        return []


# =========================================================
# 2. RANKING VÍA LLM — UNA sola llamada para todo el batch
# =========================================================

def _build_ranking_prompt(news_items: List[Dict[str, Any]]) -> str:
    lines = []
    for n in news_items:
        headline = (n.get("headline") or "")[:200]
        summary  = (n.get("summary") or "")[:400]
        symbols  = n.get("symbols") or []
        lines.append(f"- [{','.join(symbols)}] {headline} — {summary}")

    news_block = "\n".join(lines)

    return f"""Eres un analista financiero. A continuación hay una lista de noticias
de mercado de las últimas {NEWS_HOURS_BACK} horas, cada una con los símbolos
que menciona.

Tu tarea: identificar hasta {TOP_N} tickers con noticias claramente ALCISTAS
(catalizador positivo específico de esa empresa) y hasta {TOP_N} tickers con
noticias claramente BAJISTAS (catalizador negativo específico de esa empresa).

Reglas estrictas:
- IGNORA noticias de mercado general que mencionan muchos símbolos a la vez
  (resúmenes de mercado, "stock market today", listas de ETFs) — esas no
  cuentan como catalizador específico de una empresa.
- Solo incluye SÍMBOLOS DE ACCIONES INDIVIDUALES (no ETFs como QQQ/SPY/SMH,
  no cripto como BTCUSD, no índices).
- Si una noticia no tiene un catalizador claro y específico, no la uses.
- Ordena cada lista de mayor a menor convicción (rank 1 = más fuerte).

NOTICIAS:
{news_block}

Devuelve SOLO este JSON, sin texto adicional:
{{
  "bullish": [{{"ticker": "XXX", "rank": 1, "reason": "máx 15 palabras"}}, ...],
  "bearish": [{{"ticker": "XXX", "rank": 1, "reason": "máx 15 palabras"}}, ...]
}}"""


def rank_news_with_llm(news_items: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    empty = {"bullish": [], "bearish": []}

    if not news_items:
        logger.warning("⚠️ Sin noticias para rankear")
        return empty

    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        logger.error("❌ OpenAI no disponible o OPENAI_API_KEY faltante")
        return empty

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = _build_ranking_prompt(news_items)

    try:
        response = client.chat.completions.create(
            model=IA_MODEL,
            messages=[
                {"role": "system", "content": "Analista financiero. Responde SOLO JSON válido."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)

        bullish = result.get("bullish", [])[:TOP_N]
        bearish = result.get("bearish", [])[:TOP_N]

        # Normalizar tickers a mayúsculas
        for item in bullish + bearish:
            item["ticker"] = str(item.get("ticker", "")).upper().strip()

        bullish = [i for i in bullish if i["ticker"]]
        bearish = [i for i in bearish if i["ticker"]]

        logger.info(f"🎯 Ranking LLM: {len(bullish)} alcistas, {len(bearish)} bajistas")
        return {"bullish": bullish, "bearish": bearish}

    except Exception as e:
        logger.error(f"❌ rank_news_with_llm error: {e}")
        return empty


# =========================================================
# 3. VALIDAR TICKER NUEVO CONTRA ALPACA ASSETS
# =========================================================

def _is_tradable_equity(ticker: str) -> bool:
    """
    Confirma que el símbolo es una acción individual operable
    (class=us_equity, tradable=True) — descarta ETFs, cripto, índices
    sin necesidad de mantener una lista manual de exclusiones.
    """
    if not ALPACA_KEY or not ALPACA_SECRET:
        return False

    headers = {
        "APCA-API-KEY-ID":     ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }
    try:
        resp = requests.get(f"{ASSETS_URL}/{ticker}", headers=headers, timeout=10)
        if resp.status_code != 200:
            return False
        asset = resp.json()
        return (
            asset.get("class") == "us_equity"
            and asset.get("tradable") is True
            and asset.get("status") == "active"
        )
    except Exception as e:
        logger.debug(f"_is_tradable_equity {ticker}: {e}")
        return False


# =========================================================
# 4. CICLO PRINCIPAL
# =========================================================

def run_news_ranking() -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()

    news_items = fetch_market_news()
    ranking    = rank_news_with_llm(news_items)

    universe = set(_load_universe())

    bullish_result = []
    for item in ranking["bullish"]:
        ticker = item["ticker"]
        entry  = {"ticker": ticker, "rank": item.get("rank"), "reason": item.get("reason")}

        if ticker in universe:
            entry["status"] = "in_universe"
        else:
            if _is_tradable_equity(ticker):
                added = _add_to_universe(ticker)
                entry["status"] = "added_to_universe" if added else "in_universe"
                universe.add(ticker)
            else:
                entry["status"] = "rejected_not_tradable_equity"

        bullish_result.append(entry)

    bearish_result = []
    for item in ranking["bearish"]:
        ticker = item["ticker"]
        entry  = {"ticker": ticker, "rank": item.get("rank"), "reason": item.get("reason")}
        entry["status"] = "in_universe" if ticker in universe else "outside_universe_ignored"
        bearish_result.append(entry)

    result = {
        "generated_at":  timestamp,
        "n_news_fetched": len(news_items),
        "bullish": bullish_result,
        "bearish": bearish_result,
    }

    _save_json(OUTPUT_FILE, result)
    logger.info(
        f"✅ news_ranking.json guardado | "
        f"bullish={len(bullish_result)} bearish={len(bearish_result)}"
    )
    return result


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    result = run_news_ranking()
    print(json.dumps(result, indent=2, ensure_ascii=False))

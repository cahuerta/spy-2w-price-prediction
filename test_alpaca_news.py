"""
test_alpaca_news.py — PRUEBA de la Alpaca News API
=====================================================
Script de verificación, NO para producción. Confirma que:
  1. Las credenciales ALPACA_API_KEY/ALPACA_SECRET_KEY ya existentes
     tienen acceso a la News API (mismo plan, sin costo adicional).
  2. El endpoint devuelve noticias reales para un puñado de tickers.
  3. Una sola llamada trae noticias de VARIOS tickers a la vez
     (symbols separados por coma) — confirma que no hace falta
     una llamada por ticker.

Uso:
    python test_alpaca_news.py
    python test_alpaca_news.py AAPL,TSLA,NVDA
"""

import os
import sys
import json
import requests

ALPACA_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


def main():
    if not ALPACA_KEY or not ALPACA_SECRET:
        print("❌ ALPACA_API_KEY / ALPACA_SECRET_KEY no están seteadas en el entorno")
        sys.exit(1)

    symbols = sys.argv[1] if len(sys.argv) > 1 else "AAPL,TSLA,NVDA"

    headers = {
        "APCA-API-KEY-ID":     ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }
    params = {
        "symbols": symbols,
        "limit":   10,
        "sort":    "desc",
    }

    print(f"🔍 Consultando noticias para: {symbols}")
    resp = requests.get(NEWS_URL, headers=headers, params=params, timeout=15)

    print(f"HTTP status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"❌ Error: {resp.text[:500]}")
        sys.exit(1)

    data = resp.json()
    news = data.get("news", [])

    print(f"✅ {len(news)} noticias recibidas\n")

    for n in news:
        print(f"— {n.get('headline', '(sin título)')}")
        print(f"  symbols: {n.get('symbols')}")
        print(f"  fecha:   {n.get('created_at')}")
        print(f"  fuente:  {n.get('source')}")
        print()

    # Guardar el JSON crudo completo para revisión manual
    out_path = "/tmp/alpaca_news_test.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"💾 JSON completo guardado en: {out_path}")


if __name__ == "__main__":
    main()

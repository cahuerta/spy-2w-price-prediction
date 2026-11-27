"""
operar_paper.py — Paper trading (dinero ficticio en Alpaca) de la estrategia que pasó la prueba:
"Todas juntas" ESCALONADA: el dinero en 3 carteras de 10 acciones; cada una rota cada 3 meses, pero
en meses distintos (cartera 1: ene/abr/jul/oct; cartera 2: feb/may/ago/nov; cartera 3: mar/jun/sep/dic).
Así cada mes rota un tercio, que es la versión que mejor salió en la prueba.

Cada día hábil:
  1. Arma el ranking "Todas juntas" (Buffett/Munger, Graham, Lynch, Kiyosaki y Musk, a partes
     iguales) entre las 150 acciones más líquidas del S&P 500, con cifras de la SEC ya publicadas.
  2. Muestra el portafolio actual de la cuenta paper, el ranking de hoy y cuánto le va contra SPY.
  3. Calcula qué tiene cada cartera: las 10 mejores del ranking del día en que le tocó rotar
     (se recalcula con los datos que existían ese día, así no hace falta guardar nada).
  4. El primer día hábil de cada mes, si la cuenta no tiene acciones o si se pide "forzar": ajusta
     la cuenta a la suma de las 3 carteras (deja 1% en efectivo para costos).

Usa las mismas llaves de Alpaca (las de paper trading, que empiezan con PK). Nunca opera dinero real:
el cliente se crea siempre con paper=True.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import validar_red as vr

K = 10                 # acciones en el portafolio
LIQUIDAS = 150         # universo: las más líquidas del S&P 500
EFECTIVO = 0.01        # parte que se deja sin invertir
PARTES = 3             # carteras escalonadas
MIN_ORDEN = 25.0       # no se mandan órdenes menores a esto (USD)


def ranking_hoy():
    tickers = vr.obtener_universo("sp500")
    df = vr.descargar_alpaca(1.6, "paper_barras.csv", True, tickers + [vr.BENCH], raw=True)
    p = vr.a_paneles(df)
    p, buenos = vr.limpiar(p)
    c, v = p["close"], p["volume"]
    liq = (c[buenos] * v[buenos]).iloc[-252:].mean()
    liq = liq[c[buenos].iloc[-1] >= 10].sort_values(ascending=False)
    elegidas = list(liq.index[:LIQUIDAS])
    print(f"Universo: las {len(elegidas)} más líquidas del último año (precio >= 10 USD).")

    fund = vr.descargar_fundamentales(elegidas, "paper_sec.json", True)
    fechas = c.index
    Ff, grupos = vr.panel_fundamentales(fund, elegidas, fechas, c)
    puntajes = {g: Ff[-1][:, ix].mean(axis=1) for g, ix in grupos.items()}
    serie = np.mean([Ff[:, :, ix].mean(axis=2) for ix in grupos.values()], axis=0)   # [días, acciones]
    tabla = pd.DataFrame(puntajes, index=elegidas)
    tabla["Todas juntas"] = serie[-1]
    tabla["precio"] = c[elegidas].iloc[-1].values
    serie = pd.DataFrame(serie, index=fechas, columns=elegidas)
    return tabla.sort_values("Todas juntas", ascending=False), fechas[-1], c[vr.BENCH], serie


def carteras(serie):
    """Para cada cartera: el mes en que rotó por última vez y sus 10 acciones (ranking del último
    cierre antes de ese día)."""
    hoy = pd.Timestamp(datetime.now(vr.timezone.utc)).tz_convert("America/New_York").tz_localize(None).normalize()
    fechas = serie.index
    out = []
    for j in range(PARTES):
        mes = hoy.to_period("M")
        while (mes.month - 1) % PARTES != j:
            mes -= 1
        previas = fechas[fechas < mes.start_time]
        d = previas[-1]
        out.append((j + 1, mes, d, list(serie.loc[d].sort_values(ascending=False).index[:K])))
    return out


def es_inicio_mes(ultima_sesion):
    hoy = pd.Timestamp(datetime.now(vr.timezone.utc)).tz_convert("America/New_York").tz_localize(None).normalize()
    return ultima_sesion.to_period("M") != hoy.to_period("M")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forzar", action="store_true", help="rebalancear hoy aunque no sea inicio de trimestre")
    ap.add_argument("--solo-mostrar", action="store_true", help="no mandar órdenes, solo mostrar qué haría")
    args = ap.parse_args()

    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("Faltan ALPACA_API_KEY y ALPACA_SECRET_KEY.")
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    cli = TradingClient(key, secret, paper=True)          # siempre cuenta de práctica
    cuenta = cli.get_account()
    equity = float(cuenta.equity)
    posiciones = {ps.symbol: float(ps.market_value) for ps in cli.get_all_positions()}
    abierto = cli.get_clock().is_open

    tabla, ultima, spy, serie = ranking_hoy()
    carts = carteras(serie)
    objetivo = {}
    for _, _, _, accs in carts:
        for a in accs:
            objetivo[a] = objetivo.get(a, 0.0) + 1.0 / (K * PARTES)
    top = list(objetivo)

    print("\n" + "=" * 70)
    print(f"PAPER TRADING — 'Todas juntas' escalonada — datos al {ultima.date()}")
    print("=" * 70)
    print(f"Cuenta paper: {equity:,.2f} USD | efectivo {float(cuenta.cash):,.2f} USD | "
          f"mercado {'ABIERTO' if abierto else 'cerrado'}")
    try:
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        hist = cli.get_portfolio_history(GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
    except Exception:  # noqa: BLE001
        hist = None
    if hist is not None and hist.equity and len([e for e in hist.equity if e]) > 1:
        eq = [e for e in hist.equity if e]
        ts = [pd.Timestamp(t, unit="s").normalize() for t, e in zip(hist.timestamp, hist.equity) if e]
        spy_ini = spy.asof(ts[0])
        print(f"Desde {ts[0].date()}: cuenta {(eq[-1] / eq[0] - 1) * 100:+.2f}% | "
              f"SPY {(spy.iloc[-1] / spy_ini - 1) * 100:+.2f}% (mismo período)")

    print("\nPosiciones actuales:")
    if posiciones:
        for sym, mv in sorted(posiciones.items(), key=lambda x: -x[1]):
            marca = "" if sym in top else "   <- ya no está en ninguna cartera"
            print(f"  {sym:<6} {mv:>11,.2f} USD ({mv / equity * 100:4.1f}%){marca}")
    else:
        print("  (ninguna)")

    print("\nCarteras (cada una con 1/3 del dinero):")
    for j, mes, d, accs in carts:
        print(f"  Cartera {j}: rotó en {mes} (ranking del {d.date()}): {', '.join(accs)}")
    print("\nPeso objetivo de cada acción (suma de las 3 carteras):")
    for a, w in sorted(objetivo.items(), key=lambda x: -x[1]):
        print(f"  {a:<6} {w * 100:4.1f}%  ahora {posiciones.get(a, 0.0) / equity * 100:4.1f}%")
    print(f"\nRanking 'Todas juntas' de hoy (top {K + 5}; puntaje de -0.5 a +0.5):")
    cols = [c for c in tabla.columns if c not in ("precio",)]
    print("  " + "símbolo".ljust(8) + "".join(c.split("/")[0][:9].rjust(10) for c in cols))
    for sym, fila in tabla.head(K + 5).iterrows():
        print("  " + sym.ljust(8) + "".join(f"{fila[c]:>10.2f}" for c in cols) + ("  *" if sym in top else ""))

    toca = args.forzar or not posiciones or es_inicio_mes(ultima)
    print(f"\n¿Ajustar hoy? {'SÍ' if toca else 'no'} "
          f"({'forzado' if args.forzar else 'cuenta sin acciones' if not posiciones else 'primer día hábil del mes' if toca else 'se ajusta el primer día hábil de cada mes'})")
    if not toca:
        return

    invertible = equity * (1 - EFECTIVO)
    ordenes = []
    for sym, mv in posiciones.items():
        if sym not in objetivo:
            ordenes.append((sym, -mv, "cerrar"))
    for sym, w in objetivo.items():
        dif = invertible * w - posiciones.get(sym, 0.0)
        # solo se corrigen diferencias que valgan la pena (>1% de la cuenta) o acciones nuevas
        if abs(dif) >= max(MIN_ORDEN, 0.01 * equity) or (sym not in posiciones and dif >= MIN_ORDEN):
            ordenes.append((sym, dif, "ajustar"))
    ordenes.sort(key=lambda o: o[1])                   # primero ventas, después compras

    print("\nÓrdenes:")
    for sym, monto, tipo in ordenes:
        print(f"  {'vender' if monto < 0 else 'comprar'} {sym:<6} {abs(monto):>10,.2f} USD ({tipo})")
    if not ordenes:
        print("  (nada que hacer, ya está balanceado)")
    if args.solo_mostrar or not abierto:
        print("\nNo se mandaron órdenes (" + ("solo mostrar" if args.solo_mostrar else "mercado cerrado") + ").")
        return
    for sym, monto, tipo in ordenes:
        try:
            if tipo == "cerrar":
                cli.close_position(sym)
            else:
                cli.submit_order(MarketOrderRequest(
                    symbol=sym, notional=round(abs(monto), 2),
                    side=OrderSide.BUY if monto > 0 else OrderSide.SELL, time_in_force=TimeInForce.DAY))
            print(f"  enviada: {sym}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR en {sym}: {e}")


if __name__ == "__main__":
    main()

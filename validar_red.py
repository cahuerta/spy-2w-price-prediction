#!/usr/bin/env python3
"""
validar_red.py - ¿Una red neuronal gana dinero de verdad con acciones de EE.UU.?

Horizonte: 1 a 5 días (24 horas a 1 semana). Barras diarias de Alpaca (feed IEX, gratis).

Qué hace, todo en este archivo:
  1. Descarga ~8 años de barras diarias de todo el S&P 500 (lista actual de Wikipedia; si falla,
     ~100 acciones líquidas) + SPY + variables de contexto (volatilidad VIXY, tasas TLT/IEF/SHY,
     dólar UUP, oro, petróleo y los 11 ETF de sectores), todo del mismo feed gratis de Alpaca.
  2. Una red neuronal pequeña predice, para 1, 3 y 5 días, qué acciones rendirán más o menos
     que las demás (retorno contra SPY, ordenado cada día entre todas las acciones; no intenta
     adivinar hacia dónde va el mercado). Se promedian varias redes (semillas). Hay dos formas de tratar los números (opción --modelo):
       - fila:   una GRU lee las últimas 60 barras de cada acción por separado (SPY solo
                 como contexto).
       - bloque: la red recibe el bloque completo (60 días x ~100 acciones x características).
                 Una GRU lee cada acción y luego una capa de atención deja que cada acción
                 mire a todas las demás antes de dar su puntaje.
  3. Valida con walk-forward: entrena en el pasado, prueba en meses que la red nunca vio,
     con un hueco (embargo) entre ambos para que no se filtre información del futuro.
  4. Simula operar esas predicciones DESCONTANDO costos (spread/slippage y costo de pedir
     prestadas acciones para vender en corto).
  5. Compara dos formas de operar: solo compras (top K) y compras + ventas en corto.
  6. Entrega un veredicto contra la vara acordada: ganancia neta positiva en la mayoría de
     los períodos de prueba y mejor Sharpe que comprar SPY.

Cómo operar la señal (simulación):
  - Al cierre del día t se calcula la predicción. Se entra al día siguiente a la apertura (t+1)
    y se sale h días después, también a la apertura. Cada día se abre una nueva "tanda" con
    1/h del capital, así que siempre hay h tandas activas.

Uso:
    pip install torch alpaca-py pandas numpy
    export ALPACA_API_KEY=...  ALPACA_SECRET_KEY=...
    python validar_red.py                  # S&P 500 con Alpaca (IEX gratis), red en bloque
    python validar_red.py --universo top100  # solo las ~100 acciones líquidas (más rápido)
    python validar_red.py --modelo ambos   # también la red "fila" (o: fila, bloque)
    python validar_red.py --demo ruido     # prueba: datos SIN señal (no debe ganar)
    python validar_red.py --demo senal     # prueba: señal débil dentro de cada acción
    python validar_red.py --demo cruzada   # prueba: señal que solo se ve mirando varias acciones
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# PARÁMETROS (se pueden cambiar por línea de comandos cuando corresponde)
# ----------------------------------------------------------------------------
HORIZONTES = (1, 3, 5)      # días de tenencia: 24 horas hasta 1 semana
LOOKBACK = 60               # barras diarias que mira la red
TRAIN_DAYS = 504            # ~2 años para entrenar en cada tramo
TEST_DAYS = 126             # ~6 meses de prueba por tramo
VAL_FRAC = 0.2              # parte final del entrenamiento usada para parar a tiempo
WARMUP = 20                 # días iniciales sin features completas
BATCH = 512
LR = 2e-3
PACIENCIA = 3

UNIVERSO = list(dict.fromkeys([
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "MA", "UNH", "XOM", "LLY", "WMT", "JNJ", "PG", "HD", "COST", "ORCL",
    "NFLX", "BAC", "ABBV", "CRM", "CVX", "KO", "AMD", "MRK", "PEP", "TMO",
    "ADBE", "WFC", "LIN", "CSCO", "ACN", "MCD", "ABT", "TXN", "DHR", "QCOM",
    "GE", "IBM", "CAT", "AMGN", "INTU", "VZ", "PM", "NOW", "ISRG", "DIS",
    "NEE", "UBER", "GS", "SPGI", "RTX", "PFE", "HON", "UNP", "LOW", "AMAT",
    "BKNG", "T", "CMCSA", "AXP", "BLK", "MS", "ETN", "PGR", "TJX", "SYK",
    "LMT", "COP", "VRTX", "C", "BSX", "SCHW", "ADP", "MDT", "PANW", "MU",
    "ADI", "GILD", "CB", "MMC", "PLD", "LRCX", "KLAC", "SBUX", "DE", "BMY",
    "ANET", "MO", "SO", "INTC", "CME", "UPS", "ICE", "NKE", "SHW", "DUK",
    "TMUS", "APH", "WM", "CVS", "MDLZ",
]))
BENCH = "SPY"
# Variables de contexto (mismo feed de Alpaca, sin otra llave). VIXY = volatilidad esperada (VIX).
CONTEXTO = ["VIXY", "TLT", "IEF", "SHY", "UUP", "GLD", "USO",
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
SECTOR_ETF = {
    "Information Technology": "XLK", "Financials": "XLF", "Energy": "XLE", "Health Care": "XLV",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Industrials": "XLI",
    "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE", "Communication Services": "XLC",
}
SECTOR_DE = {}              # símbolo -> ETF de su sector (se llena al cargar el S&P 500)
OFFS = np.arange(-LOOKBACK + 1, 1)


# ----------------------------------------------------------------------------
# DATOS
# ----------------------------------------------------------------------------
def obtener_universo(modo):
    """S&P 500 actual desde Wikipedia (con su sector). Si falla, usa la lista corta."""
    if modo == "top100":
        return list(UNIVERSO)
    try:
        import io
        import requests
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                         headers={"User-Agent": "validar_red/1.0"}, timeout=30)
        r.raise_for_status()
        t = pd.read_html(io.StringIO(r.text))[0]
        syms = [str(x).strip().replace("-", ".") for x in t["Symbol"]]
        for sym, sec in zip(syms, t["GICS Sector"]):
            if sec in SECTOR_ETF:
                SECTOR_DE[sym] = SECTOR_ETF[sec]
        print(f"S&P 500 cargado de Wikipedia: {len(syms)} símbolos.")
        return list(dict.fromkeys(syms))
    except Exception as e:  # noqa: BLE001
        print(f"No pude leer la lista del S&P 500 ({e}); uso la lista corta de ~100 acciones.")
        return list(UNIVERSO)


def descargar_alpaca(years, cache, refrescar, tickers):
    if os.path.exists(cache) and not refrescar:
        print(f"Usando datos guardados en {cache} (usa --refresh para volver a descargar)")
        return pd.read_csv(cache)

    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("Faltan ALPACA_API_KEY y ALPACA_SECRET_KEY en el entorno.")

    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    cli = StockHistoricalDataClient(key, secret)
    fin = datetime.now(timezone.utc) - timedelta(days=1)
    ini = fin - timedelta(days=int(365.25 * years) + 10)
    todos = list(dict.fromkeys(tickers + [BENCH] + CONTEXTO))
    print(f"Descargando {len(todos)} símbolos desde Alpaca (feed IEX)...", flush=True)
    partes = []
    for i in range(0, len(todos), 100):
        lote_s = todos[i:i + 100]
        for intento in range(3):
            try:
                req = StockBarsRequest(symbol_or_symbols=lote_s, timeframe=TimeFrame.Day,
                                       start=ini, end=fin, adjustment=Adjustment.ALL,
                                       feed=DataFeed.IEX)
                d = cli.get_stock_bars(req).df.reset_index()
                partes.append(d)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  lote {i // 100 + 1}: error ({e}); reintento {intento + 1}/3", flush=True)
                time.sleep(5 * (intento + 1))
        print(f"  {min(i + 100, len(todos))}/{len(todos)}", flush=True)
    if not partes:
        sys.exit("No se pudo descargar ningún dato de Alpaca.")
    df = pd.concat(partes, ignore_index=True)
    df.to_csv(cache, index=False)
    return df


def a_paneles(df):
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["fecha"] = ts.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    return {
        c: df.pivot_table(index="fecha", columns="symbol", values=c, aggfunc="last").sort_index()
        for c in ("open", "high", "low", "close", "volume")
    }


def datos_demo(modo, seed=0):
    """Datos inventados. 'ruido': sin ninguna señal. 'senal': persistencia débil dentro de cada
    acción (corr ~0.06). 'cruzada': las 90 acciones "seguidoras" repiten con un día de retraso
    el movimiento promedio de 10 "líderes"; solo se ve mirando varias acciones a la vez."""
    rng = np.random.default_rng(seed)
    T, N = 1260, 100
    mkt = rng.normal(0.0004, 0.010, T)
    beta = rng.uniform(0.7, 1.3, N)
    idio = rng.normal(0.0, 0.015, (T, N))
    if modo == "senal":
        for t in range(1, T):
            idio[t] += 0.06 * idio[t - 1]
    if modo == "cruzada":
        lider = idio[:, :10].mean(axis=1)
        for t in range(1, T):
            idio[t, 10:] += 0.25 * lider[t - 1]
    ret = beta[None, :] * mkt[:, None] + idio
    close = 100 * np.exp(np.cumsum(ret, axis=0))
    spy_c = 100 * np.exp(np.cumsum(mkt))
    ctx_n = ["VIXY", "TLT", "XLK"]
    ctx_c = np.column_stack([100 * np.exp(np.cumsum(rng.normal(0, 0.01, T) + k * mkt)) for k in (-3, 0.2, 1.0)])
    close = np.column_stack([close, spy_c, ctx_c])
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.003, close.shape))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, close.shape)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, close.shape)))
    vol = np.exp(rng.normal(15, 0.4, close.shape))
    cols = [f"T{i:03d}" for i in range(N)] + [BENCH] + ctx_n
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T)
    mk = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    SECTOR_DE.clear()
    for i in range(0, N, 3):
        SECTOR_DE[f"T{i:03d}"] = "XLK"
    return {"open": mk(open_), "high": mk(high), "low": mk(low), "close": mk(close), "volume": mk(vol)}


def limpiar(p):
    cal = p["close"][BENCH].dropna().index
    for k in p:
        p[k] = p[k].reindex(cal)
    for k in ("open", "high", "low", "close"):
        p[k] = p[k].ffill(limit=3)
    p["volume"] = p["volume"].fillna(0.0)
    cols = [c for c in p["close"].columns if c != BENCH and c not in CONTEXTO]
    malos = [c for c in cols if any(p[k][c].isna().any() for k in ("open", "high", "low", "close"))]
    buenos = [c for c in cols if c not in malos]
    if malos:
        print(f"Descartados por datos incompletos ({len(malos)}): {', '.join(malos)}")
    if len(buenos) < 30:
        sys.exit("Quedaron muy pocas acciones con datos completos; no se puede validar.")
    return p, buenos


def filtrar_liquidas(p, tickers, n, precio_min=10.0):
    """Criterio más estricto que el del S&P 500: se quedan solo las N acciones más líquidas
    (valor transado diario promedio) y con precio de al menos 10 dólares. La elección usa SOLO la
    primera ventana de entrenamiento, así ninguna decisión mira el futuro de los períodos de
    prueba (elegir 'las que mejor rindieron' hasta hoy daría un resultado falso)."""
    if n <= 0 or n >= len(tickers):
        return tickers
    fin = LOOKBACK + WARMUP + TRAIN_DAYS
    c = p["close"][tickers].iloc[:fin]
    v = p["volume"][tickers].iloc[:fin]
    liq = (c * v).mean()
    ok = liq[c.median() >= precio_min].sort_values(ascending=False)
    elegidas = list(ok.index[:n])
    print(f"Filtro de liquidez: de {len(tickers)} acciones quedan las {len(elegidas)} más líquidas "
          f"(precio ≥ {precio_min:g} USD), elegidas con datos de {c.index[0].date()} a {c.index[-1].date()}.")
    return elegidas


def relaciones(rex, tickers, W=60, k=5, WL=250):
    """Variables de RELACIÓN entre acciones. Todo se calcula día a día con datos hasta ese día.
    rex: retornos diarios de cada acción contra SPY, forma [días, acciones].
      - vecinos: las k acciones que más se mueven igual que ella en los últimos W días. Se mide
        cómo les fue hoy y en 5 días, y si ella va atrasada o adelantada respecto de ellas.
      - líderes: las k acciones cuyo movimiento de AYER mejor anticipó el movimiento de HOY de
        ella (correlación con un día de rezago). Se mide cómo les fue hoy.
      - pares del sector: promedio de las demás acciones de su mismo sector.
      - estado del mercado: correlación promedio entre acciones y dispersión del día."""
    T, N = rex.shape
    k = max(1, min(k, N - 1))
    cs = np.cumsum(rex, axis=0)
    m5 = cs.copy()
    m5[5:] = cs[5:] - cs[:-5]
    nb1, nb5, ld1 = (np.zeros((T, N)) for _ in range(3))
    corr_prom = np.zeros(T)

    def zs(X):
        return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    for t in range(W, T):
        Z = zs(rex[t - W + 1:t + 1])                    # ventana que termina hoy
        C = Z.T @ Z / W                                 # quién se mueve como quién
        np.fill_diagonal(C, -9.0)
        corr_prom[t] = (C.sum() + 9.0 * N) / (N * (N - 1))
        nb = np.argpartition(-C, k - 1, axis=1)[:, :k]
        nb1[t] = rex[t][nb].mean(axis=1)
        nb5[t] = m5[t][nb].mean(axis=1)
        if t < WL:
            continue
        # quién de ayer anticipa a quién de hoy; ventana larga porque la relación es muy débil
        # y con pocos días se confunde con ruido
        L = zs(rex[t - WL + 1:t + 1]).T @ zs(rex[t - WL:t]) / WL
        np.fill_diagonal(L, -9.0)
        ld = np.argpartition(-L, k - 1, axis=1)[:, :k]
        ld1[t] = rex[t][ld].mean(axis=1)

    sec1, sec5 = np.zeros((T, N)), np.zeros((T, N))
    grupos = {}
    for j, tk in enumerate(tickers):
        grupos.setdefault(SECTOR_DE.get(tk), []).append(j)
    for g, idx in grupos.items():
        if g is None or len(idx) < 2:
            continue
        for src, dst in ((rex, sec1), (m5, sec5)):
            s = src[:, idx].sum(axis=1, keepdims=True)
            dst[:, idx] = (s - src[:, idx]) / (len(idx) - 1)

    disp = np.repeat(rex.std(axis=1)[:, None], N, axis=1)
    cp = np.repeat(corr_prom[:, None], N, axis=1)
    return [nb1, nb5, m5 - nb5, ld1, sec1, sec5, disp, cp]


def construir(p, tickers, usar_rel=True):
    ctx = [x for x in CONTEXTO if x in p["close"].columns]
    o, h, l, c, v = (p[k][tickers + [BENCH] + ctx] for k in ("open", "high", "low", "close", "volume"))
    ret1 = np.log(c / c.shift(1))
    spy_ret = ret1[BENCH]
    vol_rel = np.log((v + 1) / (v.rolling(20, min_periods=5).mean().shift(1) + 1))
    T, N = len(c), len(tickers)

    def cols(x):
        return x[tickers].values

    spy_b = np.repeat(spy_ret.values[:, None], N, axis=1)
    rt = ret1[tickers]
    rel_sec = np.zeros((T, N))
    for j, tk in enumerate(tickers):
        se = SECTOR_DE.get(tk)
        if se in ctx:
            rel_sec[:, j] = (rt[tk] - ret1[se]).values
    ctx_b = [np.repeat(ret1[x].values[:, None], N, axis=1) for x in ctx]
    mom5 = np.log(c / c.shift(5))
    mom20 = np.log(c / c.shift(20))
    mom60 = np.log(c / c.shift(60))
    vol20 = ret1.rolling(20, min_periods=10).std()

    def rk(x):  # posición de cada acción entre todas ese día, de -0.5 (la peor) a +0.5 (la mejor)
        return x[tickers].rank(axis=1, pct=True).values - 0.5

    # momentum de 5 días contra su sector
    mom5_sec = np.zeros((T, N))
    for j, tk in enumerate(tickers):
        se = SECTOR_DE.get(tk)
        if se in ctx:
            mom5_sec[:, j] = (mom5[tk] - mom5[se]).values
    # dónde cerró dentro del rango del día (0 = en el mínimo, 1 = en el máximo)
    pos_rango = ((c - l) / (h - l).replace(0, np.nan)).clip(0, 1) - 0.5
    # distancia al máximo de 60 días
    dist_max = np.log(c / c.rolling(60, min_periods=20).max())
    # volumen de la última semana contra el del último mes
    vol_trend = np.log((v.rolling(5, min_periods=3).mean() + 1) / (v.rolling(20, min_periods=10).mean() + 1))

    rel_feats = []
    if usar_rel:
        rex = np.nan_to_num(cols(ret1.sub(spy_ret, axis=0)), nan=0.0, posinf=0.0, neginf=0.0)
        rel_feats = relaciones(rex, tickers)

    F = np.stack(
        [
            cols(ret1),                                   # retorno del día
            cols(ret1.sub(spy_ret, axis=0)),              # retorno vs SPY
            cols(np.log(h / l)),                          # rango del día
            cols(np.log(o / c.shift(1))),                 # gap de apertura
            cols(vol_rel),                                # volumen relativo
            spy_b,                                        # contexto de mercado
            cols(mom5),                                   # momentum 5 días
            cols(mom20),                                  # momentum 20 días
            cols(vol20),                                  # volatilidad 20 días
            rel_sec,                                      # retorno vs su sector
            cols(mom60),                                  # momentum 60 días
            cols(mom20.sub(mom20[BENCH], axis=0)),        # momentum 20 días vs SPY
            mom5_sec,                                     # momentum 5 días vs su sector
            cols(pos_rango),                              # dónde cerró dentro de su rango
            cols(dist_max),                               # distancia al máximo de 60 días
            cols(vol_trend),                              # tendencia del volumen
            rk(ret1), rk(mom5), rk(mom20), rk(vol20),     # posición entre todas las acciones ese día
        ] + rel_feats + ctx_b,                                      # volatilidad, tasas, dólar, oro, petróleo, sectores
        axis=2,
    )
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    op = cols(o)
    spy_op = o[BENCH].values
    Ys = []
    for hz in HORIZONTES:
        y = np.full((T, N), np.nan)
        y_t = op[1 + hz:T] / op[1:T - hz] - 1.0
        y_s = spy_op[1 + hz:T] / spy_op[1:T - hz] - 1.0
        y[: T - 1 - hz] = y_t - y_s[:, None]
        Ys.append(y)
    Y = np.stack(Ys, axis=2)

    R = np.zeros((T, N))
    R[1:] = op[1:] / op[:-1] - 1.0
    spyR = np.zeros(T)
    spyR[1:] = spy_op[1:] / spy_op[:-1] - 1.0
    return F, Y, R, spyR, c.index


# ----------------------------------------------------------------------------
# RED
# ----------------------------------------------------------------------------
def crear_red(nf, nout):
    import torch.nn as nn

    class Red(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(nf, 32, batch_first=True)
            self.drop = nn.Dropout(0.2)
            self.out = nn.Linear(32, nout)

        def forward(self, x):
            _, hn = self.gru(x)
            return self.out(self.drop(hn[-1]))

    return Red()


def lote(Fn, ts, ns):
    import torch
    return torch.from_numpy(Fn[ts[:, None] + OFFS[None, :], ns[:, None], :])


def muestras(Y, t0, t1):
    ok = np.isfinite(Y[t0:t1 + 1]).all(axis=2)
    ti, ni = np.nonzero(ok)
    return (ti + t0).astype(np.int64), ni.astype(np.int64)


def entrenar(Fn, Ys, tr, va, seed, epochs):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = crear_red(Fn.shape[2], Ys.shape[2])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-3)
    lossf = nn.HuberLoss(delta=1.0)
    ttr, ntr = tr
    tva, nva = va
    mejor, estado, malas, ep_mejor = np.inf, None, 0, 0

    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(ttr))
        for i in range(0, len(perm), BATCH):
            j = perm[i:i + BATCH]
            x = lote(Fn, ttr[j], ntr[j])
            y = torch.from_numpy(Ys[ttr[j], ntr[j], :])
            opt.zero_grad()
            lossf(net(x), y).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(tva), 4096):
                x = lote(Fn, tva[i:i + 4096], nva[i:i + 4096])
                y = torch.from_numpy(Ys[tva[i:i + 4096], nva[i:i + 4096], :])
                tot += lossf(net(x), y).item() * len(x)
        vl = tot / len(tva)
        if vl < mejor - 1e-5:
            mejor, malas, ep_mejor = vl, 0, ep + 1
            estado = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            malas += 1
            if malas >= PACIENCIA:
                break
    net.load_state_dict(estado)
    net.eval()
    return net, ep_mejor, mejor


def predecir(net, Fn, ts, ns):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(ts), 4096):
            out.append(net(lote(Fn, ts[i:i + 4096], ns[i:i + 4096])).numpy())
    return np.concatenate(out, axis=0)


# ----------------------------------------------------------------------------
# RED EN BLOQUE: ve todas las acciones a la vez
# ----------------------------------------------------------------------------
def crear_red_bloque(nf, nout, hid=32, heads=4):
    import torch.nn as nn

    class RedBloque(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(nf, hid, batch_first=True)
            self.att = nn.MultiheadAttention(hid, heads, dropout=0.1, batch_first=True)
            self.n1 = nn.LayerNorm(hid)
            self.ff = nn.Sequential(nn.Linear(hid, 2 * hid), nn.GELU(), nn.Linear(2 * hid, hid))
            self.n2 = nn.LayerNorm(hid)
            self.drop = nn.Dropout(0.2)
            self.out = nn.Linear(hid, nout)

        def forward(self, x):  # x: [días, acciones, barras, características]
            b, n, l, f = x.shape
            _, hn = self.gru(x.reshape(b * n, l, f))
            e = hn[-1].reshape(b, n, -1)              # una "huella" por acción
            a, _ = self.att(e, e, e, need_weights=False)  # cada acción mira a todas las demás
            e = self.n1(e + self.drop(a))
            e = self.n2(e + self.drop(self.ff(e)))
            return self.out(e)                         # [días, acciones, horizontes]

    return RedBloque()


def lote_bloque(Fn, dias):
    import torch
    x = Fn[dias[:, None] + OFFS[None, :]]              # [días, barras, acciones, caract.]
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1, 3)))


def entrenar_bloque(Fn, Ys, dias_tr, dias_va, seed, epochs, batch_dias=8, paciencia=8, frac=1.0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = crear_red_bloque(Fn.shape[2], Ys.shape[2])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-3)
    lossf = nn.HuberLoss(delta=1.0)
    mejor, estado, malas, ep_mejor = np.inf, None, 0, 0

    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(dias_tr))[:max(batch_dias, int(len(dias_tr) * frac))]
        for i in range(0, len(perm), batch_dias):
            d = dias_tr[perm[i:i + batch_dias]]
            opt.zero_grad()
            lossf(net(lote_bloque(Fn, d)), torch.from_numpy(Ys[d])).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(dias_va), 16):
                d = dias_va[i:i + 16]
                tot += lossf(net(lote_bloque(Fn, d)), torch.from_numpy(Ys[d])).item() * len(d)
        vl = tot / len(dias_va)
        if vl < mejor - 1e-5:
            mejor, malas, ep_mejor = vl, 0, ep + 1
            estado = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            malas += 1
            if malas >= paciencia:
                break
    net.load_state_dict(estado)
    net.eval()
    return net, ep_mejor, mejor


def predecir_bloque(net, Fn, dias):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(dias), 16):
            out.append(net(lote_bloque(Fn, dias[i:i + 16])).numpy())
    return np.concatenate(out, axis=0)                 # [días, acciones, horizontes]


# ----------------------------------------------------------------------------
# WALK-FORWARD
# ----------------------------------------------------------------------------
def tramos(T, hmax):
    gap = hmax + 2
    t_min = LOOKBACK + WARMUP
    t_max = T - 2 - hmax
    a = t_min + TRAIN_DAYS + gap
    out = []
    while a + TEST_DAYS - 1 <= t_max:
        out.append((a, a + TEST_DAYS - 1))
        a += TEST_DAYS
    return out, gap


def objetivo_relativo(Y):
    """Lo que la red aprende: cada día, qué acciones rinden más o menos que las demás.
    Se resta el promedio de ese día y se divide por la dispersión de ese día (puntaje z).
    Así la red no gasta capacidad en adivinar hacia dónde va todo el mercado, solo en ordenar
    las acciones. Usa solo información del mismo día, no mezcla días."""
    Yr = np.full(Y.shape, np.nan)
    for t in range(Y.shape[0]):
        y = Y[t]                                   # [acciones, horizontes]
        if not np.isfinite(y).all():
            continue
        sd = y.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Yr[t] = (y - y.mean(axis=0)) / sd
    return np.clip(Yr, -5, 5).astype(np.float32)


def correr_walk_forward(F, Y, tr_list, gap, seeds, epochs, modelo="fila", frac=1.0):
    T, N, H = Y.shape
    pred = np.full((T, N, H), np.nan)
    val_len = int(TRAIN_DAYS * VAL_FRAC)
    for k, (a, b) in enumerate(tr_list, 1):
        t_fin = a - gap - 1
        t_ini = t_fin - TRAIN_DAYS + 1
        v_ini = t_fin - val_len + 1
        fit_fin = v_ini - gap - 1

        mu = F[t_ini:t_fin + 1].reshape(-1, F.shape[2]).mean(axis=0)
        sd = F[t_ini:t_fin + 1].reshape(-1, F.shape[2]).std(axis=0) + 1e-8
        Fn = np.clip((F - mu) / sd, -5, 5).astype(np.float32)

        tr = muestras(Y, t_ini, fit_fin)
        va = muestras(Y, v_ini, t_fin)
        Ys = objetivo_relativo(Y)

        if modelo == "bloque":
            dias_tr = np.arange(t_ini, fit_fin + 1)
            dias_tr = dias_tr[np.isfinite(Ys[dias_tr]).all(axis=(1, 2))]
            dias_va = np.arange(v_ini, t_fin + 1)
            dias_va = dias_va[np.isfinite(Ys[dias_va]).all(axis=(1, 2))]
            dias_te = np.arange(a, b + 1)
            acum_b = np.zeros((len(dias_te), N, H))
            for s in range(seeds):
                t0 = time.time()
                net, ep, vl = entrenar_bloque(Fn, Ys, dias_tr, dias_va, seed=s, epochs=epochs, frac=frac)
                acum_b += predecir_bloque(net, Fn, dias_te)
                print(f"  tramo {k}/{len(tr_list)} semilla {s}: {len(dias_tr)} días de entrenamiento, "
                      f"mejor época {ep}, pérdida val {vl:.4f}, {time.time() - t0:.0f}s", flush=True)
            pred[a:b + 1] = acum_b / seeds
            continue

        ts_te = np.repeat(np.arange(a, b + 1), N).astype(np.int64)
        ns_te = np.tile(np.arange(N), b - a + 1).astype(np.int64)
        acum = np.zeros((len(ts_te), H))
        for s in range(seeds):
            t0 = time.time()
            net, ep, vl = entrenar(Fn, Ys, tr, va, seed=s, epochs=epochs)
            acum += predecir(net, Fn, ts_te, ns_te)
            print(f"  tramo {k}/{len(tr_list)} semilla {s}: {len(tr[0])} muestras, "
                  f"mejor época {ep}, pérdida val {vl:.4f}, {time.time() - t0:.0f}s", flush=True)
        pred[ts_te, ns_te, :] = acum / seeds
    return pred


# ----------------------------------------------------------------------------
# SIMULACIÓN DE OPERACIÓN CON COSTOS
# ----------------------------------------------------------------------------
def rango(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    ra, rb = rango(a), rango(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def estadisticas(r):
    r = np.asarray(r, dtype=float)
    ann = r.mean() * 252
    vol = r.std(ddof=1) * np.sqrt(252)
    eq = np.cumprod(1 + r)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol > 0 else np.nan, "dd": dd}


def simular(pred_h, R, spyR, h, modo, tr_list, k, cost_bps, borrow):
    T, N = pred_h.shape
    W = np.zeros((T, N))
    for t in range(T):
        p = pred_h[t]
        if np.isnan(p).any():
            continue
        orden = np.argsort(p)
        if modo == "short":                  # solo diagnóstico: la canasta que la red menos quiere
            W[t, orden[:k]] = 1.0 / k
            continue
        W[t, orden[-k:]] = 1.0 / k
        if modo == "ls":
            W[t, orden[:k]] = -1.0 / k
    gross = 1.0 if modo in ("long", "short") else 2.0
    costo = 2.0 * cost_bps * 1e-4 * gross / h
    borr = borrow / 252.0 if modo == "ls" else 0.0

    brutos, netos, spys, por_tramo = [], [], [], []
    for a, b in tr_list:
        dias = np.arange(a + h + 1, b + 2)
        pnl = np.zeros(len(dias))
        for lag in range(2, h + 2):
            pnl += (W[dias - lag] * R[dias]).sum(axis=1) / h
        neto = pnl - costo - borr
        brutos.append(pnl)
        netos.append(neto)
        spys.append(spyR[dias])
        por_tramo.append(float(np.prod(1 + neto) - 1))
    return np.concatenate(brutos), np.concatenate(netos), np.concatenate(spys), por_tramo


def veredicto(por_tramo, st_neto, st_spy):
    pos = sum(1 for x in por_tramo if x > 0)
    ok = pos > len(por_tramo) / 2 and st_neto["ann"] > 0 and st_neto["sharpe"] > st_spy["sharpe"]
    return ok, pos


# ----------------------------------------------------------------------------
# PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", choices=["ruido", "senal", "cruzada"],
                    help="usar datos inventados para probar el sistema")
    ap.add_argument("--modelo", type=str.lower, choices=["fila", "bloque", "ambos"], default="bloque",
                    help="fila: una GRU por acción; bloque: la red ve todas las acciones a la vez")
    ap.add_argument("--universo", type=str.lower, choices=["sp500", "top100"], default="sp500",
                    help="sp500: todo el índice (Wikipedia); top100: lista corta")
    ap.add_argument("--liquidas", type=int, default=150,
                    help="quedarse solo con las N acciones más líquidas (y de precio >= 10 USD); 0 = sin filtro")
    ap.add_argument("--frac", type=float, default=0.5,
                    help="fracción de días de entrenamiento usada por época en la red en bloque (acelera)")
    ap.add_argument("--sin-relaciones", action="store_true",
                    help="no usar las variables de relación entre acciones (para comparar con y sin)")
    ap.add_argument("--years", type=float, default=8.0, help="años de historial a descargar")
    ap.add_argument("--k", type=int, default=10, help="acciones por lado en el portafolio")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="costo por lado en puntos base (spread + slippage)")
    ap.add_argument("--borrow", type=float, default=0.01, help="costo anual de pedir prestadas acciones (0.01 = 1%%)")
    ap.add_argument("--seeds", type=int, default=5, help="redes por tramo (se promedian)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="máximo de épocas por red (por defecto 20 en fila, 60 en bloque)")
    ap.add_argument("--cache", default="barras_diarias.csv")
    ap.add_argument("--refresh", action="store_true", help="volver a descargar aunque exista el archivo guardado")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(os.cpu_count() or 1)

    if args.demo:
        print(f"MODO PRUEBA con datos inventados ({args.demo}).")
        p = datos_demo(args.demo)
    else:
        p = a_paneles(descargar_alpaca(args.years, args.cache, args.refresh, obtener_universo(args.universo)))
    p, tickers = limpiar(p)
    tickers = filtrar_liquidas(p, tickers, args.liquidas)
    F, Y, R, spyR, fechas = construir(p, tickers, usar_rel=not args.sin_relaciones)
    T, N, H = Y.shape
    print(f"{N} acciones, {T} días ({fechas[0].date()} a {fechas[-1].date()}).")

    tr_list, gap = tramos(T, max(HORIZONTES))
    if len(tr_list) < 2:
        sys.exit("Historial insuficiente para al menos 2 tramos de prueba. Usa más años.")
    print(f"{len(tr_list)} tramos de prueba (entrena {TRAIN_DAYS}d, prueba {TEST_DAYS}d, embargo {gap}d):")
    for i, (a, b) in enumerate(tr_list, 1):
        print(f"  tramo {i}: {fechas[a].date()} a {fechas[b].date()}")

    modelos = ["fila", "bloque"] if args.modelo == "ambos" else [args.modelo]
    pasan_total = []
    for m in modelos:
        pasan_total += [(m, h, n) for h, n in ejecutar(m, args, F, Y, R, spyR, tr_list, gap)]
    resumen(pasan_total, len(modelos))


def ejecutar(modelo, args, F, Y, R, spyR, tr_list, gap):
    epocas = args.epochs or (60 if modelo == "bloque" else 20)
    print(f"\nEntrenando modelo {modelo}...", flush=True)
    pred = correr_walk_forward(F, Y, tr_list, gap, args.seeds, epocas, modelo, args.frac)

    print("\n" + "=" * 78)
    print(f"RESULTADOS modelo {modelo.upper()} (solo períodos de prueba; costo {args.cost_bps:g} pb por lado, "
          f"préstamo {args.borrow * 100:g}%/año, K={args.k} por lado)")
    print("=" * 78)
    pasan = []
    for hi, h in enumerate(HORIZONTES):
        ics = []
        for a, b in tr_list:
            for t in range(a, b + 1):
                if np.isfinite(Y[t, :, hi]).all() and np.isfinite(pred[t, :, hi]).all():
                    ics.append(spearman(pred[t, :, hi], Y[t, :, hi]))
        ics = np.array(ics)
        tstat = ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics) / h) if len(ics) > 2 and ics.std() > 0 else np.nan
        print(f"\nHorizonte {h} día(s) | correlación de ranking (IC) medio {ics.mean():+.4f} (t≈{tstat:+.1f})")
        print(f"  {'modo':<18}{'bruto/año':>10}{'neto/año':>10}{'Sharpe':>8}{'MaxDD':>8}"
              f"{'tramos+':>9}{'SPY/año':>9}{'SPY Sh.':>8}  veredicto")
        for modo, nombre in (("long", "solo compras"), ("ls", "compras+cortos")):
            bruto, neto, spy, por_tramo = simular(
                pred[:, :, hi], R, spyR, h, modo, tr_list, args.k, args.cost_bps, args.borrow)
            sb, sn, ss = estadisticas(bruto), estadisticas(neto), estadisticas(spy)
            ok, pos = veredicto(por_tramo, sn, ss)
            if ok:
                pasan.append((h, nombre))
            print(f"  {nombre:<18}{sb['ann'] * 100:>9.1f}%{sn['ann'] * 100:>9.1f}%{sn['sharpe']:>8.2f}"
                  f"{sn['dd'] * 100:>7.1f}%{pos:>5}/{len(por_tramo):<3}{ss['ann'] * 100:>8.1f}%{ss['sharpe']:>8.2f}"
                  f"  {'PASA' if ok else 'no pasa'}")
            if modo == "ls":
                por_ls = por_tramo

        # --- ¿de dónde sale la pérdida? Cada pierna por separado (retorno bruto de la canasta) ---
        ph = pred[:, :, hi]
        bl, _, sp, _ = simular(ph, R, spyR, h, "long", tr_list, args.k, 0.0, 0.0)
        bs, _, _, _ = simular(ph, R, spyR, h, "short", tr_list, args.k, 0.0, 0.0)
        beta = lambda x: float(np.cov(x, sp)[0, 1] / np.var(sp, ddof=1))
        print(f"  piernas (bruto): comprada top{args.k} {estadisticas(bl)['ann'] * 100:+.1f}%/año (beta {beta(bl):.2f}) | "
              f"vendida bottom{args.k} {estadisticas(bs)['ann'] * 100:+.1f}%/año (beta {beta(bs):.2f}) | "
              f"SPY {estadisticas(sp)['ann'] * 100:+.1f}%/año")
        print("    (si la vendida rinde MÁS que la comprada, la red está ordenando al revés)")

        # --- señal invertida (informativa; ver nota) ---
        inv = -ph
        causal = ph.copy()
        for i, (a, b) in enumerate(tr_list):
            if i > 0 and por_ls[i - 1] < 0:      # el tramo anterior perdió: este se opera al revés
                causal[a:b + 1] = -causal[a:b + 1]
        for etiqueta, pr in (("cortos INVERTIDO*", inv), ("ls signo tramo prev", causal)):
            bruto, neto, spy, por_tramo = simular(pr, R, spyR, h, "ls", tr_list, args.k, args.cost_bps, args.borrow)
            sb, sn, ss = estadisticas(bruto), estadisticas(neto), estadisticas(spy)
            ok, pos = veredicto(por_tramo, sn, ss)
            print(f"  {etiqueta:<18}{sb['ann'] * 100:>9.1f}%{sn['ann'] * 100:>9.1f}%{sn['sharpe']:>8.2f}"
                  f"{sn['dd'] * 100:>7.1f}%{pos:>5}/{len(por_tramo):<3}{ss['ann'] * 100:>8.1f}%{ss['sharpe']:>8.2f}"
                  f"  {'pasaría' if ok else 'no pasa'}")
        print("    * el signo se invirtió mirando estos mismos resultados: solo orienta, no vale como prueba.")
        print("      'signo tramo prev' decide el signo de cada tramo con el tramo anterior (sin mirar adelante).")

    return pasan


def resumen(pasan, n_modelos):
    print("\n" + "-" * 78)
    print("Vara: ganancia neta positiva en la mayoría de los tramos, retorno anual neto > 0 y")
    print("Sharpe mayor que el de comprar SPY en los mismos días.")
    if pasan:
        print("Configuraciones que pasan: " + ", ".join(f"modelo {m}, {n} a {h}d" for m, h, n in pasan))
        print(f"OJO: se probaron {6 * n_modelos} combinaciones sobre los mismos datos; elegir la mejor infla el")
        print("resultado. Además la lista de acciones son las del S&P 500 de HOY (sesgo de")
        print("supervivencia: las que salieron del índice no están). Si pasa, confirmar en paper trading antes de usar dinero real.")
    else:
        print("Ninguna configuración pasa la vara: con estos datos no hay evidencia de ventaja real.")


if __name__ == "__main__":
    main()

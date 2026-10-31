#!/usr/bin/env python3
"""
validar_red.py - ¿Una red neuronal gana dinero de verdad con acciones de EE.UU.?

Horizonte: 1 a 5 días (24 horas a 1 semana). Barras diarias de Alpaca (feed IEX, gratis).

Qué hace, todo en este archivo:
  1. Descarga ~5 años de barras diarias de ~100 acciones líquidas + SPY (o usa datos de prueba).
  2. Una sola red GRU pequeña mira las últimas 60 barras de cada acción y predice, para
     1, 3 y 5 días, el retorno de la acción MENOS el de SPY.
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
    python validar_red.py                  # datos reales de Alpaca (feed IEX gratis)
    python validar_red.py --demo ruido     # prueba: datos SIN señal (no debe ganar)
    python validar_red.py --demo senal     # prueba: datos con señal débil (debe detectarla)
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
OFFS = np.arange(-LOOKBACK + 1, 1)


# ----------------------------------------------------------------------------
# DATOS
# ----------------------------------------------------------------------------
def descargar_alpaca(years, cache, refrescar):
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
    print(f"Descargando {len(UNIVERSO) + 1} símbolos desde Alpaca (feed IEX)...")
    req = StockBarsRequest(
        symbol_or_symbols=UNIVERSO + [BENCH],
        timeframe=TimeFrame.Day,
        start=ini,
        end=fin,
        adjustment=Adjustment.ALL,
        feed=DataFeed.IEX,
    )
    df = cli.get_stock_bars(req).df.reset_index()
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
    """Datos inventados. 'ruido': sin ninguna señal. 'senal': persistencia débil (corr ~0.06)."""
    rng = np.random.default_rng(seed)
    T, N = 1260, 100
    mkt = rng.normal(0.0004, 0.010, T)
    beta = rng.uniform(0.7, 1.3, N)
    idio = rng.normal(0.0, 0.015, (T, N))
    if modo == "senal":
        for t in range(1, T):
            idio[t] += 0.06 * idio[t - 1]
    ret = beta[None, :] * mkt[:, None] + idio
    close = 100 * np.exp(np.cumsum(ret, axis=0))
    spy_c = 100 * np.exp(np.cumsum(mkt))
    close = np.column_stack([close, spy_c])
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.003, close.shape))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, close.shape)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, close.shape)))
    vol = np.exp(rng.normal(15, 0.4, close.shape))
    cols = [f"T{i:03d}" for i in range(N)] + [BENCH]
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T)
    mk = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    return {"open": mk(open_), "high": mk(high), "low": mk(low), "close": mk(close), "volume": mk(vol)}


def limpiar(p):
    cal = p["close"][BENCH].dropna().index
    for k in p:
        p[k] = p[k].reindex(cal)
    for k in ("open", "high", "low", "close"):
        p[k] = p[k].ffill(limit=3)
    p["volume"] = p["volume"].fillna(0.0)
    cols = [c for c in p["close"].columns if c != BENCH]
    malos = [c for c in cols if any(p[k][c].isna().any() for k in ("open", "high", "low", "close"))]
    buenos = [c for c in cols if c not in malos]
    if malos:
        print(f"Descartados por datos incompletos ({len(malos)}): {', '.join(malos)}")
    if len(buenos) < 30:
        sys.exit("Quedaron muy pocas acciones con datos completos; no se puede validar.")
    return p, buenos


def construir(p, tickers):
    o, h, l, c, v = (p[k][tickers + [BENCH]] for k in ("open", "high", "low", "close", "volume"))
    ret1 = np.log(c / c.shift(1))
    spy_ret = ret1[BENCH]
    vol_rel = np.log((v + 1) / (v.rolling(20, min_periods=5).mean().shift(1) + 1))
    T, N = len(c), len(tickers)

    def cols(x):
        return x[tickers].values

    spy_b = np.repeat(spy_ret.values[:, None], N, axis=1)
    F = np.stack(
        [
            cols(ret1),                                   # retorno del día
            cols(ret1.sub(spy_ret, axis=0)),              # retorno vs SPY
            cols(np.log(h / l)),                          # rango del día
            cols(np.log(o / c.shift(1))),                 # gap de apertura
            cols(vol_rel),                                # volumen relativo
            spy_b,                                        # contexto de mercado
        ],
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


def correr_walk_forward(F, Y, tr_list, gap, seeds, epochs):
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
        sdh = np.array([np.nanstd(Y[t_ini:fit_fin + 1, :, h]) for h in range(H)])
        Ys = np.clip(Y / sdh[None, None, :], -5, 5).astype(np.float32)

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
        W[t, orden[-k:]] = 1.0 / k
        if modo == "ls":
            W[t, orden[:k]] = -1.0 / k
    gross = 1.0 if modo == "long" else 2.0
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
    ap.add_argument("--demo", choices=["ruido", "senal"], help="usar datos inventados para probar el sistema")
    ap.add_argument("--years", type=float, default=5.0, help="años de historial a descargar")
    ap.add_argument("--k", type=int, default=10, help="acciones por lado en el portafolio")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="costo por lado en puntos base (spread + slippage)")
    ap.add_argument("--borrow", type=float, default=0.01, help="costo anual de pedir prestadas acciones (0.01 = 1%%)")
    ap.add_argument("--seeds", type=int, default=3, help="redes por tramo (se promedian)")
    ap.add_argument("--epochs", type=int, default=20, help="máximo de épocas por red")
    ap.add_argument("--cache", default="barras_diarias.csv")
    ap.add_argument("--refresh", action="store_true", help="volver a descargar aunque exista el archivo guardado")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(os.cpu_count() or 1)

    if args.demo:
        print(f"MODO PRUEBA con datos inventados ({args.demo}).")
        p = datos_demo(args.demo)
    else:
        p = a_paneles(descargar_alpaca(args.years, args.cache, args.refresh))
    p, tickers = limpiar(p)
    F, Y, R, spyR, fechas = construir(p, tickers)
    T, N, H = Y.shape
    print(f"{N} acciones, {T} días ({fechas[0].date()} a {fechas[-1].date()}).")

    tr_list, gap = tramos(T, max(HORIZONTES))
    if len(tr_list) < 2:
        sys.exit("Historial insuficiente para al menos 2 tramos de prueba. Usa más años.")
    print(f"{len(tr_list)} tramos de prueba (entrena {TRAIN_DAYS}d, prueba {TEST_DAYS}d, embargo {gap}d):")
    for i, (a, b) in enumerate(tr_list, 1):
        print(f"  tramo {i}: {fechas[a].date()} a {fechas[b].date()}")

    print("Entrenando...", flush=True)
    pred = correr_walk_forward(F, Y, tr_list, gap, args.seeds, args.epochs)

    print("\n" + "=" * 78)
    print(f"RESULTADOS (solo períodos de prueba; costo {args.cost_bps:g} pb por lado, "
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

    print("\n" + "-" * 78)
    print("Vara: ganancia neta positiva en la mayoría de los tramos, retorno anual neto > 0 y")
    print("Sharpe mayor que el de comprar SPY en los mismos días.")
    if pasan:
        print("Configuraciones que pasan: " + ", ".join(f"{n} a {h}d" for h, n in pasan))
        print("OJO: se probaron 6 combinaciones sobre los mismos datos; elegir la mejor infla el")
        print("resultado. Además la lista de acciones son nombres grandes de hoy (sesgo de")
        print("supervivencia). Si pasa, confirmar en paper trading antes de usar dinero real.")
    else:
        print("Ninguna configuración pasa la vara: con estos datos no hay evidencia de ventaja real.")


if __name__ == "__main__":
    main()

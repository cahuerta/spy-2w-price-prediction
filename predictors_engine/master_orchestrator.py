import os
import json
import numpy as np
import pandas as pd
import sys
import importlib
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

DATA_DIR  = "predictions_data"
FINAL_DIR = "/data/predictions"
EVAL_DIR  = "/data/evaluations"

# ======================================================
# FIXES v7.3 → v7.5: (ver historial)
# FIXES v7.6:
#   [O11] price_curve enriquecida con upper_band, lower_band,
#         ret_path — calculadas desde dispersión entre modelos
#         por día. No modifica price_path ni ret_ens_pct.
#   [O12] curve_analysis agregado: peak_day, peak_ret_pct,
#         final_ret_pct, trajectory, consensus_std, confidence.
#         Consumible por intraday_tracker y dashboard.
#         Misma salida de siempre + dos campos nuevos.
# FIXES v7.7 [AUD-D2] (auditoría 2026-09-05, Problema 2):
#   [D2] adjust_factor era 0.15 fijo — H10 dominaba 85-100% del
#        precio final pese a tener el peor hit rate del ensemble
#        (48.07% global, peor que el azar). El sistema ya calculaba
#        el hit rate propio de H10 por ticker (_load_horizon_weights
#        incluye H10 en su rango) pero lo descartaba antes de usarlo
#        en este punto. Fix: adjust_factor ahora escala con
#        1 - hit_rate(H10) (clip 0.15-0.60) vía _h10_hit_rate() — si
#        H10 anda mal en ese ticker, la curva de H1-H9 puede
#        corregirlo más; si anda bien, domina casi igual que antes.
#        Sin muestras suficientes → 0.15 de siempre, sin cambio.
#        NO se modifica la forma del JSON de salida (compatibilidad
#        con el resto del sistema) — solo cambia cómo se calcula
#        internamente el valor de adjust_factor.
# ======================================================


# ======================================================
# UTILIDADES
# ======================================================

def extract_price_prediction(data: Dict) -> Optional[float]:
    val = data.get("price_pred")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    pred = data.get("prediction")
    if isinstance(pred, dict):
        val = pred.get("price_pred")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def extract_return_pct(data: Dict) -> float:
    for k in ["return_pct", "ret_ens_pct"]:
        val = data.get(k)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    pred = data.get("prediction")
    if isinstance(pred, dict):
        val = pred.get("ret_ens_pct")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


# ======================================================
# [O10] PONDERACIÓN DINÁMICA POR HORIZONTE
# ======================================================

def _load_horizon_weights(ticker: str) -> Dict[str, float]:
    eval_path = Path(EVAL_DIR) / ticker
    if not eval_path.exists():
        return {}

    hits: Dict[str, List[bool]] = {f"H{h}": [] for h in range(1, 11)}

    for f in sorted(eval_path.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            diag = data.get("models_diagnostics") or {}
            for h in range(1, 11):
                key = f"H{h}"
                if key in diag and diag[key].get("hit_sign") is not None:
                    hits[key].append(bool(diag[key]["hit_sign"]))
        except Exception:
            continue

    MIN_SAMPLES = 10
    rates: Dict[str, float] = {}
    for key, vals in hits.items():
        if len(vals) >= MIN_SAMPLES:
            rates[key] = float(np.mean(vals))

    if not rates:
        return {}

    keys   = list(rates.keys())
    values = np.array([rates[k] for k in keys])
    exp_v  = np.exp((values - values.mean()) * 2)
    norm   = exp_v / exp_v.sum()

    return {k: float(w) for k, w in zip(keys, norm)}


# ======================================================
# [AUD-D2] HIT RATE PROPIO DE H10 (2026-09-05, Problema 2)
# ======================================================
def _h10_hit_rate(ticker: str) -> Optional[float]:
    """
    [AUD-D2] Hit rate real de H10 por ticker (misma fuente y mismo
    MIN_SAMPLES=10 que _load_horizon_weights, pero SIN pasar por el
    softmax de normalización — acá se necesita el rate crudo (0-1)
    para decidir cuánto puede corregirlo la curva de H1-H9, no un
    peso relativo ya normalizado contra los demás horizontes).
    Retorna None si no hay 10 muestras — el llamador debe usar el
    comportamiento anterior (0.15 fijo) como fallback en ese caso.
    """
    eval_path = Path(EVAL_DIR) / ticker
    if not eval_path.exists():
        return None

    hits: List[bool] = []
    for f in sorted(eval_path.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            diag = (data.get("models_diagnostics") or {}).get("H10")
            if diag and diag.get("hit_sign") is not None:
                hits.append(bool(diag["hit_sign"]))
        except Exception:
            continue

    if len(hits) < 10:
        return None
    return float(np.mean(hits))


# ======================================================
# [O11] CURVA ENRIQUECIDA CON BANDAS
# ======================================================

def _build_curve_bands(
    results: List[Dict],
    price_path: List[float],
    price_today: float,
) -> Dict:
    """
    Calcula upper_band y lower_band usando la dispersión
    entre modelos en cada día de la curva.

    Lógica:
      - Para cada día d (1-9), recolectar precio predicho
        de cada modelo Hx que tenga horizonte >= d
      - std de esos precios = incertidumbre en ese día
      - band = price_path[d] ± 1.5 * std

    Si hay pocos modelos para un día → banda proporcional
    al retorno esperado (fallback conservador).
    """
    n_days = len(price_path)
    upper  = []
    lower  = []
    stds   = []

    for d in range(n_days):
        day = d + 1  # 1-based

        # Modelos con horizonte >= día actual
        prices_at_day = [
            r["price_pred"]
            for r in results
            if r["horizon"] >= day and r["horizon"] < 10
        ]

        if len(prices_at_day) >= 2:
            std = float(np.std(prices_at_day))
        else:
            # Fallback: std proporcional a la distancia desde hoy
            base_std = abs(price_path[d] - price_today) * 0.3
            std      = max(base_std, price_today * 0.005)  # mínimo 0.5%

        band_width = 1.5 * std
        upper.append(round(price_path[d] + band_width, 4))
        lower.append(round(price_path[d] - band_width, 4))
        stds.append(round(std, 4))

    # ret_path: retorno esperado % vs precio hoy para cada día
    ret_path = [
        round((p / price_today - 1) * 100, 4)
        for p in price_path
    ]

    return {
        "upper_band": upper,
        "lower_band": lower,
        "ret_path":   ret_path,
        "stds":       stds,
    }


# ======================================================
# [O12] ANÁLISIS DE TRAYECTORIA
# ======================================================

def _analyze_trajectory(
    price_path: List[float],
    price_today: float,
    ret_path: List[float],
) -> Dict:
    """
    Clasifica la trayectoria de la curva y extrae métricas clave.

    trajectory:
      "monotone_up"      → sube consistentemente
      "monotone_down"    → baja consistentemente
      "peak_and_revert"  → sube luego baja
      "valley_and_rise"  → baja luego sube
      "flat"             → poca variación
    """
    if not price_path or price_today <= 0:
        return {}

    rets       = np.array(ret_path)
    peak_idx   = int(np.argmax(rets))
    valley_idx = int(np.argmin(rets))
    peak_ret   = float(rets[peak_idx])
    final_ret  = float(rets[-1])

    # Clasificar trayectoria
    flat_threshold = 0.5  # menos de 0.5% de variación total → flat

    total_range = float(np.max(rets) - np.min(rets))

    if total_range < flat_threshold:
        trajectory = "flat"
    elif peak_idx > valley_idx:
        # Primero baja, luego sube
        trajectory = "valley_and_rise"
    elif peak_idx < len(rets) - 2 and rets[-1] < peak_ret * 0.7:
        # Sube y luego revierte significativamente
        trajectory = "peak_and_revert"
    elif final_ret > 0 and rets[0] >= 0:
        trajectory = "monotone_up"
    elif final_ret < 0 and rets[0] <= 0:
        trajectory = "monotone_down"
    else:
        trajectory = "peak_and_revert" if peak_ret > abs(final_ret) else "monotone_up"

    # Consenso: qué tan dispersos están los modelos en el día final
    # (ya calculado en bands, aquí solo resumimos)
    return {
        "peak_day":       peak_idx + 1,        # 1-based
        "peak_ret_pct":   round(peak_ret, 4),
        "valley_day":     valley_idx + 1,
        "final_ret_pct":  round(final_ret, 4),
        "trajectory":     trajectory,
        "best_entry_day": 1,                   # siempre día 1 para nuevas entradas
        "best_exit_day":  peak_idx + 1,        # día del pico como salida óptima
    }


# ======================================================
# CURVA PONDERADA (existente + enriquecida)
# ======================================================

def _weighted_price_curve(
    results: List[Dict],
    weights: Dict[str, float],
    price_today: float,
) -> Optional[Dict]:
    filtered = [r for r in results if r["horizon"] < 10]
    if len(filtered) < 2:
        return None

    horizons = np.array([r["horizon"] for r in filtered])
    prices   = np.array([r["price_pred"] for r in filtered])

    if weights:
        w_array = np.array([
            weights.get(f"H{r['horizon']}", 1.0 / len(filtered))
            for r in filtered
        ])
        w_array         = w_array / w_array.sum()
        weighted_prices = prices * w_array * len(filtered)
    else:
        weighted_prices = prices

    full_h       = np.arange(1, 10)
    interpolated = np.interp(full_h, horizons, weighted_prices)
    smoothed     = pd.Series(interpolated).rolling(3, center=True, min_periods=1).mean().values
    price_path   = smoothed.tolist()

    # [O11] Bandas de confianza
    bands    = _build_curve_bands(results, price_path, price_today)
    ret_path = bands["ret_path"]

    # [O12] Análisis de trayectoria
    analysis = _analyze_trajectory(price_path, price_today, ret_path)

    return {
        # Campos originales — sin cambios
        "days":       list(range(1, 10)),
        "price_path": price_path,
        "price_now":  price_today,
        # [O11] Nuevos — bandas de confianza
        "upper_band": bands["upper_band"],
        "lower_band": bands["lower_band"],
        "ret_path":   ret_path,
        "stds":       bands["stds"],
        # [O12] Nuevo — resumen de trayectoria
        "analysis":   analysis,
    }


# ======================================================
# ORCHESTRATOR
# ======================================================

class MasterOrchestrator:

    def __init__(self, ticker):
        self.ticker      = ticker.upper()
        self.results     = []
        self.price_today = None
        self.h10_json    = None

    def collect_results(self):
        curve = []

        for h in range(1, 11):
            possible_files = [
                f"{self.ticker}_H{h}.json",
                f"{self.ticker}_h{h}.json",
                f"{self.ticker}_H{h}_{h}d.json",
                f"{self.ticker}_H{h}_{h}d_v2.json",
                f"{self.ticker}_H{h}_{h}d_v3.json",
            ]

            for fname in possible_files:
                path = os.path.join(DATA_DIR, fname)
                if not os.path.exists(path):
                    continue

                try:
                    with open(path) as f:
                        data = json.load(f)

                    data["horizon"] = h
                    price = extract_price_prediction(data)
                    if price is None:
                        print(f"⚠️ H{h} sin price_pred válido — saltando")
                        continue

                    if self.price_today is None:
                        raw_price = (
                            data.get("price_now")
                            or data.get("price_today")
                            or (data.get("prediction") or {}).get("price_now")
                        )
                        if raw_price is None:
                            print(f"⚠️ H{h} sin price_today — saltando")
                            continue
                        self.price_today = float(raw_price)

                    data["price_pred"]     = price
                    data["normalized_ret"] = extract_return_pct(data)
                    curve.append(data)

                    if h == 10:
                        self.h10_json = data

                    print(f"   📊 H{h}: ${price:.2f}")
                    break

                except Exception as e:
                    print(f"⚠️ H{h} JSON error {e}")

        self.results = sorted(curve, key=lambda x: x["horizon"])
        print(f"📈 {len(self.results)}/10 modelos recolectados")
        return self.results

    def build_price_curve(self, weights: Dict[str, float] = None) -> Optional[Dict]:
        return _weighted_price_curve(
            self.results,
            weights or {},
            self.price_today,
        )

    def analyze_curve(self, curve) -> float:
        if curve is None:
            return 0.0
        curve_last = curve["price_path"][-1]
        return ((curve_last - self.price_today) / self.price_today) * 100

    def build_model_json(self, curve, weights: Dict[str, float] = None) -> Dict:
        if self.h10_json is None:
            raise RuntimeError("H10 missing")

        pred = self.h10_json.get("prediction")
        if not pred or not isinstance(pred, dict):
            raise RuntimeError(f"[{self.ticker}] H10 prediction inválido o None")

        required_fields = ["price_pred", "price_now", "ret_ens_pct", "theta_dynamic_pct"]
        null_fields = [f for f in required_fields if pred.get(f) is None]
        if null_fields:
            raise RuntimeError(f"[{self.ticker}] H10 campos críticos null: {null_fields}")

        final_json   = self.h10_json.copy()
        h10_price    = extract_price_prediction(self.h10_json)
        curve_adjust = self.analyze_curve(curve)

        ret_original  = extract_return_pct(self.h10_json)
        sign_original = float(np.sign(ret_original)) if ret_original != 0 else 0.0

        consensus        = np.std([r["price_pred"] for r in self.results if r["horizon"] < 10])
        consensus_weight = 1 / (1 + consensus / self.price_today)

        # [AUD-D2][2026-09-05] Antes: adjust_factor = 0.15 fijo, sin
        # relación con qué tan confiable es H10 en este ticker. Ahora:
        # se usa el hit rate real de H10 (mismo hit_sign que ya se
        # registra en evaluator.py) para decidir cuánto puede
        # corregirlo la curva de H1-H9 — si H10 anda mal en este
        # ticker, se le permite más corrección; si anda bien, domina
        # casi igual que antes. Sin muestras suficientes de H10 →
        # fallback al 0.15 de siempre (comportamiento sin cambios).
        h10_rate = _h10_hit_rate(self.ticker)
        if h10_rate is not None:
            base_adjust = float(np.clip(1.0 - h10_rate, 0.15, 0.60))
        else:
            base_adjust = 0.15
        adjust_factor = base_adjust * consensus_weight

        final_price  = h10_price * (1 + curve_adjust / 100 * adjust_factor)
        ret_adjusted = ((final_price - self.price_today) / self.price_today) * 100
        sign_adjusted = float(np.sign(ret_adjusted)) if ret_adjusted != 0 else 0.0

        if sign_original != 0 and sign_adjusted != sign_original:
            print(
                f"⚠️ [{self.ticker}] curve_adjust invirtió signo "
                f"({ret_original:.3f}% → {ret_adjusted:.3f}%) → forzando ret=0, MANTÉN"
            )
            ret_final   = 0.0
            final_price = self.price_today
        else:
            ret_final = round(ret_adjusted, 4)

        final_json["prediction"]["price_pred"]  = round(final_price, 2)
        final_json["prediction"]["ret_ens_pct"] = ret_final

        theta_dyn = float(final_json["prediction"]["theta_dynamic_pct"])
        final_json["prediction"]["recommendation"] = (
            "COMPRA" if ret_final >= theta_dyn else
            "VENDE"  if ret_final <= -theta_dyn else
            "MANTÉN"
        )

        # [O11][O12] price_curve ahora incluye bandas + análisis
        # Misma estructura que antes + campos nuevos
        final_json["price_curve"]     = curve
        final_json["ensemble_models"] = len(self.results)
        final_json["horizon_weights"] = weights or {}

        # [O12] curve_analysis también al nivel raíz para acceso directo
        if curve and curve.get("analysis"):
            final_json["curve_analysis"] = curve["analysis"]
            print(
                f"📈 Trayectoria: {curve['analysis'].get('trajectory')} | "
                f"Peak día {curve['analysis'].get('peak_day')} "
                f"({curve['analysis'].get('peak_ret_pct', 0):+.2f}%) | "
                f"Final: {curve['analysis'].get('final_ret_pct', 0):+.2f}%"
            )

        final_json["models_diagnostics"] = {
            f"H{r['horizon']}": {
                "pred_price":  round(r["price_pred"], 4),
                "pred_return": round(r["normalized_ret"], 4),
                "horizon":     r["horizon"],
                "weight":      round(weights.get(f"H{r['horizon']}", 1.0 / len(self.results)), 4)
                               if weights else None,
            }
            for r in self.results
        }

        return final_json

    def run(self):
        os.makedirs(DATA_DIR, exist_ok=True)

        for i in range(1, 11):
            try:
                print(f"🚀 Ejecutando H{i}")
                mod    = importlib.import_module(f"predictors_engine.predictor_h{i}")
                func   = getattr(mod, f"run_predictor_h{i}")
                result = func(self.ticker)

                if result is None:
                    print(f"⚠️ H{i} retornó None — datos insuficientes, saltando")
                    continue

                filename = f"{self.ticker}_H{i}.json"
                path     = os.path.join(DATA_DIR, filename)
                with open(path, "w") as f:
                    json.dump(result, f, indent=2, default=str)

            except AttributeError:
                print(f"❌ H{i} FALLÓ → función run_predictor_h{i}() no encontrada")
            except Exception as e:
                print(f"❌ H{i} FALLÓ: {e}")
            finally:
                gc.collect()

        self.collect_results()

        if len(self.results) < 5:
            raise RuntimeError(
                f"Ensemble inválido → solo {len(self.results)} modelos válidos (mínimo 5)"
            )

        missing = [h for h in range(1, 11) if h not in [r["horizon"] for r in self.results]]
        if missing:
            print(f"⚠️ Horizontes faltantes: {missing}")

        weights = _load_horizon_weights(self.ticker)
        if weights:
            print(f"⚖️  Pesos dinámicos: { {k: round(v,3) for k,v in weights.items()} }")
        else:
            print("⚖️  Sin historial — pesos iguales")

        curve = self.build_price_curve(weights)

        final_json = self.build_model_json(curve, weights)

        ticker_dir  = os.path.join(FINAL_DIR, self.ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        file_name   = datetime.utcnow().strftime("%Y-%m-%d") + ".json"
        output_file = os.path.join(ticker_dir, file_name)

        tmp_file = output_file + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(final_json, f, indent=2)
        os.replace(tmp_file, output_file)

        print(f"💾 ENSEMBLE guardado: {output_file}")
        return final_json


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "SPY").upper()
    print(f"🎯 MasterOrchestrator v7.6 → {ticker}")
    print("=" * 60)

    orch   = MasterOrchestrator(ticker)
    result = orch.run()

    print("\n🏆 RESULTADO FINAL:")
    print(json.dumps(result["prediction"], indent=2))
    print(f"\n🎯 RECOMENDACIÓN: {result['prediction']['recommendation']}")
    if result.get("curve_analysis"):
        print(f"\n📈 ANÁLISIS CURVA:")
        print(json.dumps(result["curve_analysis"], indent=2))
    if result.get("horizon_weights"):
        print(f"\n⚖️  Pesos usados: {result['horizon_weights']}")

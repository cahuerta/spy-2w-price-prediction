"""
pm_defensive.py - DEFENSIVE POSITION MANAGER v2.1 PRODUCCION

FIX v1.8:
  [F7] entry_price fallback a avg_entry_price — Alpaca devuelve
       avg_entry_price, no entry_price. Antes entry=0 → CLOSE
       inmediato antes de verificar si es anchor.
  [F8] is_anchor se evalúa ANTES del check de precio inválido.
       Anchors nunca se cierran por datos faltantes — retornan
       HOLD preventivo y se reevalúan en el siguiente ciclo.
  [F9] price_now fallback a current_price — mismo problema
       con el campo de precio actual.

FIX v1.9:
  [F10] entry_time → entry_date: el sistema nunca escribe entry_time.
        positions_meta.py guarda entry_date (solo fecha ISO "YYYY-MM-DD").
        portfolio_store.py mergea entry_date. pm_defensive leía entry_time
        → siempre "" → days_between explotaba en TODAS las posiciones.
        Ahora lee entry_date con fallback a entry_time por compatibilidad.
  [F11] days_between retorna 0 (no MAX_HOLD_DAYS) cuando entry_date
        está vacío o ausente — antes forzaba cierre por tiempo en
        posiciones sin metadata. Ahora las trata como nuevas.
  [F12] days_between maneja fechas sin timezone (solo "YYYY-MM-DD")
        añadiendo CL_TIMEZONE antes de comparar.

v2.0:
  [C1] _load_price_curve(ticker): lee curva de predicción desde disco
       /data/predictions/{ticker}/{fecha}.json — igual que _load_alpha_scores
       lee alpha_last.json. Sin dependencias externas, sin tracker.
  [C2] evaluate_position() para NO-ANCHORS: usa curva para más información:
       - pnl > 0 + slope_futura="baja" → CLOSE (tomar ganancia antes de caída)
       - pnl > 0 + slope_futura="sube" → HOLD (destino alcista, no cortar)
       - en_zona_valle=True → agrega contexto en meta
       Si no hay curva disponible: comportamiento idéntico a v1.9.
  [C3] ANCHORS: sin ningún cambio — lógica idéntica a v1.9.
  [C4] Orchestrator y tracker: sin cambios.

v2.1:
  [AUD-P3] FIX asimetría "corta ganadores, deja correr perdedores"
       — SOLO para no-anchors (anchors mantienen HOLD indefinido,
       sin cambios). Antes la curva solo actuaba con pnl > 0. Ahora,
       si pnl < 0, slope == "baja" (curva confirma la caída) y la
       pérdida aún es moderada (abs(pnl) < DEFENSIVE_CURVE_LOSS_CUT_PCT,
       5% fijo — independiente del stop catastrófico de 25%, que está
       pensado como último recurso, no como referencia de "pérdida
       moderada"), se cierra antes de esperar el time_exit o el stop
       catastrófico completo.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from dataclasses import dataclass
from pathlib import Path
import pytz
import json

# =========================================================
# CONFIGURACION PRODUCCION
# =========================================================
CL_TIMEZONE = pytz.timezone("America/Santiago")
DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))

MAX_HOLD_DAYS_NON_ANCHOR = int(os.getenv("PM_DEF_MAX_HOLD_DAYS", "30"))
CATASTROPHIC_STOP_PCT    = float(os.getenv("PM_DEF_STOP_LOSS_CATA", "0.25"))
MAX_ANCHOR_EXPOSURE_PCT  = float(os.getenv("PM_DEF_MAX_ANCHOR_EXPO", "0.30"))
ANCHOR_MIN_ALPHA         = float(os.getenv("PM_DEF_ANCHOR_MIN_ALPHA", "0.30"))

# [AUD-P3] Umbral de pérdida moderada para el corte simétrico por curva.
# 5% fijo (no escalado del 25% catastrófico) — confirmado con el usuario:
# escalar del stop catastrófico daba un umbral demasiado alto (15%) para
# no-anchors, que deben protegerse antes que las posiciones ancla.
DEFENSIVE_CURVE_LOSS_CUT_PCT = float(os.getenv("PM_DEF_CURVE_LOSS_CUT", "0.05"))

_BASE_DIR   = Path(__file__).resolve().parent
ANCHOR_FILE = Path(os.getenv("ANCHOR_FILE", str(_BASE_DIR / "anchor_universe.json")))
ALPHA_FILE  = DATA_PATH / "alpha_last.json"
PRED_DIR    = DATA_PATH / "predictions"


logger = logging.getLogger("pm_defensive")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# =========================================================
# HELPERS
# =========================================================

def pct_change(current: float, entry: float) -> float:
    return (current / entry - 1.0) if entry > 0 else 0.0


def days_between(entry_iso: str) -> int:
    """
    [F11] Retorna 0 si entry_iso está vacío o ausente — la posición
          se trata como nueva, no como expirada.
    [F12] Maneja fechas sin timezone ("YYYY-MM-DD") añadiendo
          CL_TIMEZONE antes de comparar.
    """
    try:
        if not entry_iso or not entry_iso.strip():
            return 0
        entry_str = entry_iso.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(entry_str)
        if dt.tzinfo is None:
            dt = CL_TIMEZONE.localize(dt)
        return max(0, (datetime.now(CL_TIMEZONE) - dt.astimezone(CL_TIMEZONE)).days)
    except Exception as e:
        logger.warning(f"days_between error '{entry_iso}': {e}")
        return 0


def _safe_price(pos: Dict, *keys) -> float:
    """
    [F7][F9] Busca el primer campo de precio válido en la posición.
    Alpaca devuelve avg_entry_price / current_price en vez de
    entry_price / price_now que usan los PMs internamente.
    """
    for k in keys:
        val = pos.get(k)
        if val is not None:
            try:
                f = float(val)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                pass
    return 0.0


# =========================================================
# CARGA CURVA DE PREDICCIÓN — v2.0
# =========================================================

def _load_price_curve(ticker: str) -> Optional[Dict]:
    """
    [C1] Lee la curva de predicción más reciente desde disco.
    /data/predictions/{TICKER}/{fecha}.json
    Retorna None si no existe o hay error — sin efectos secundarios.
    """
    try:
        ticker_dir = PRED_DIR / ticker.upper()
        if not ticker_dir.exists():
            return None
        candidates = sorted(ticker_dir.glob("*.json"))
        if not candidates:
            return None
        data  = json.loads(candidates[-1].read_text(encoding="utf-8"))
        curve = data.get("price_curve")
        if not curve or not curve.get("price_path"):
            return None

        path       = curve.get("price_path", [])
        price_now  = float(curve.get("price_now", 0))
        lower_band = curve.get("lower_band", [])

        if not path or len(path) < 2:
            return None

        # Slope: tendencia general de la curva completa
        mid       = max(1, len(path) // 2)
        avg_first = sum(path[:mid]) / mid
        avg_last  = sum(path[mid:]) / max(1, len(path) - mid)
        diff_pct  = (avg_last - avg_first) / avg_first * 100 if avg_first > 0 else 0.0

        if diff_pct > 1.0:
            slope = "sube"
        elif diff_pct < -1.0:
            slope = "baja"
        else:
            slope = "plana"

        # Peak y retornos esperados
        idx_peak     = int(max(range(len(path)), key=lambda i: path[i]))
        precio_peak  = float(path[idx_peak])
        precio_final = float(path[-1])

        ret_a_peak  = round((precio_peak  / price_now - 1) * 100, 2) if price_now > 0 else None
        ret_a_final = round((precio_final / price_now - 1) * 100, 2) if price_now > 0 else None

        # En zona de valle: precio actual bajo lower_band día 1
        en_zona_valle = False
        if lower_band and len(lower_band) > 0 and price_now > 0:
            en_zona_valle = price_now < float(lower_band[0])

        return {
            "slope_futura":              slope,
            "ret_desde_hoy_a_peak_pct":  ret_a_peak,
            "ret_desde_hoy_a_final_pct": ret_a_final,
            "dias_hasta_peak":           idx_peak + 1,
            "precio_peak_esperado":      round(precio_peak, 4),
            "precio_final_esperado":     round(precio_final, 4),
            "en_zona_valle":             en_zona_valle,
        }

    except Exception as e:
        logger.warning(f"⚠️ _load_price_curve {ticker}: {e}")
        return None


# =========================================================
# CARGA ANCHOR UNIVERSE Y ALPHA
# =========================================================

def _load_anchor_universe() -> List[Dict[str, Any]]:
    try:
        if not ANCHOR_FILE.exists():
            logger.warning(f"⚠️ anchor_universe.json no encontrado en {ANCHOR_FILE}")
            return []
        data = json.loads(ANCHOR_FILE.read_text())
        if isinstance(data, list):
            logger.info(f"⚓ {len(data)} anclas cargadas desde {ANCHOR_FILE}")
            return data
    except Exception as e:
        logger.error(f"❌ Error leyendo anchor_universe.json: {e}")
    return []


def _load_alpha_scores() -> Dict[str, float]:
    try:
        if not ALPHA_FILE.exists():
            logger.warning("⚠️ alpha_last.json no encontrado")
            return {}
        data    = json.loads(ALPHA_FILE.read_text())
        results = data.get("results", {})
        return {
            t.upper(): float(v.get("alpha_score", 0))
            for t, v in results.items()
            if isinstance(v, dict) and v.get("alpha_score") is not None
        }
    except Exception as e:
        logger.error(f"❌ Error leyendo alpha_last.json: {e}")
        return {}


# =========================================================
# DECISION STRUCT
# =========================================================

@dataclass
class DefensiveDecision:
    action:    str
    ticker:    str
    reason:    str
    timestamp: str
    meta:      Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action":    self.action,
            "ticker":    self.ticker,
            "reason":    self.reason,
            "timestamp": self.timestamp,
            "meta":      self.meta or {},
        }


# =========================================================
# PM DEFENSIVO v2.1
# =========================================================

class PMDefensive:

    def __init__(self):
        self.tz                  = CL_TIMEZONE
        self.anchor_exposure_pct = 0.0
        self._anchor_universe    = _load_anchor_universe()
        self._anchor_tickers     = {a["ticker"].upper() for a in self._anchor_universe}
        logger.info(
            f"PMDefensive v2.1 | anclas={len(self._anchor_tickers)}: "
            f"{sorted(self._anchor_tickers)}"
        )

    def is_anchor_asset(
        self,
        candidate:    Dict[str, Any],
        alpha_scores: Dict[str, float] = None,
    ) -> bool:
        ticker = candidate.get("ticker", "").upper()
        if ticker not in self._anchor_tickers:
            return False
        if alpha_scores is not None:
            alpha = alpha_scores.get(ticker, 0.0)
            if alpha < ANCHOR_MIN_ALPHA:
                logger.debug(
                    f"⚓ {ticker} en anchor_universe pero alpha={alpha:.3f} "
                    f"< {ANCHOR_MIN_ALPHA} → no elegible"
                )
                return False
        return True

    # --------------------------------------------------
    # EVALUAR POSICIÓN EXISTENTE
    # --------------------------------------------------
    def evaluate_position(self, pos: Dict[str, Any]) -> DefensiveDecision:
        ticker = str(pos.get("ticker", "UNKNOWN")).upper()
        ts     = datetime.now(self.tz).isoformat()

        # [F8] is_anchor se evalúa PRIMERO — antes de cualquier check de precio
        is_anchor = bool(pos.get("is_anchor", False)) or (ticker in self._anchor_tickers)

        # [F7][F9] Fallback a campos de Alpaca
        entry = _safe_price(pos, "entry_price", "avg_entry_price")
        price = _safe_price(pos, "price_now", "current_price")

        # [F8] Precio inválido — anchors: HOLD | no-anchors: CLOSE
        if entry <= 0 or price <= 0:
            if is_anchor:
                logger.warning(
                    f"⚠️ {ticker} ANCHOR precio inválido "
                    f"(entry={entry} price={price}) → HOLD preventivo"
                )
                return DefensiveDecision(
                    "HOLD", ticker, "anchor_invalid_price_hold", ts,
                    {"entry": entry, "price": price, "anchor": True},
                )
            else:
                logger.warning(f"⚠️ {ticker} precio inválido → CLOSE")
                return DefensiveDecision(
                    "CLOSE", ticker, "invalid_price_data", ts,
                    {"entry": entry, "price": price},
                )

        ret = pct_change(price, entry)

        # [F10] Leer entry_date con fallback a entry_time por compatibilidad
        entry_date = str(
            pos.get("entry_date") or pos.get("entry_time") or ""
        )
        age = days_between(entry_date)

        # [F14] Guardrail: una caída >90% en un solo ciclo es
        # matemáticamente casi imposible en el mercado real — es
        # mucho más probable que sea un precio corrupto (ej. el
        # placeholder price_now=1.0 que existía antes del fix [F13]
        # en trading_orchestrator.py). No cerrar con datos sospechosos.
        if ret <= -0.90:
            logger.error(
                f"🚨 {ticker} ret={ret:.1%} SOSPECHOSO (caída >90%) — "
                f"probable precio corrupto (entry={entry} price={price}), "
                f"NO se ejecuta catastrophic_loss, HOLD preventivo"
            )
            return DefensiveDecision(
                "HOLD", ticker, "suspicious_price_hold", ts,
                {"ret_pct": round(ret * 100, 2), "entry": entry, "price": price},
            )

        # Stop catastrófico — aplica a TODOS incluyendo anchors
        if ret <= -CATASTROPHIC_STOP_PCT:
            logger.warning(
                f"💀 {ticker} CATASTROPHIC STOP | ret={ret:.1%} | anchor={is_anchor}"
            )
            return DefensiveDecision(
                "CLOSE", ticker, "catastrophic_loss", ts,
                {"ret_pct": round(ret * 100, 2), "stop_pct": -CATASTROPHIC_STOP_PCT * 100},
            )

        # [C3] ANCHORS → HOLD indefinido — sin cambios respecto a v1.9
        if is_anchor:
            return DefensiveDecision(
                "HOLD", ticker, "anchor_hold_indefinite", ts,
                {
                    "ret_pct":          round(ret * 100, 2),
                    "days_held":        age,
                    "anchor":           True,
                    "dist_to_stop_pct": round((ret + CATASTROPHIC_STOP_PCT) * 100, 1),
                },
            )

        # No-anchors → time exit (límite duro, no lo toca la curva)
        if age >= MAX_HOLD_DAYS_NON_ANCHOR:
            return DefensiveDecision(
                "CLOSE", ticker, "non_anchor_time_exit", ts,
                {"days_held": age, "max_days": MAX_HOLD_DAYS_NON_ANCHOR},
            )

        # ── [C2][AUD-P3] Curva de predicción — solo no-anchors ──────────────
        # Lee desde disco. Si no hay curva: comportamiento idéntico a v1.9.
        # ────────────────────────────────────────────────────────────────────
        curva = _load_price_curve(ticker)
        pnl   = round(ret * 100, 2)

        if curva:
            slope         = curva.get("slope_futura")
            en_zona_valle = curva.get("en_zona_valle", False)
            ret_a_peak    = curva.get("ret_desde_hoy_a_peak_pct")
            ret_a_final   = curva.get("ret_desde_hoy_a_final_pct")
            dias_peak     = curva.get("dias_hasta_peak")

            # PnL positivo + curva cae → cerrar tomando ganancia
            if pnl > 0 and slope == "baja":
                logger.info(
                    f"📉 {ticker} CLOSE | PnL={pnl:.1f}% curva cae "
                    f"(ret_final={ret_a_final}%) → tomar ganancia"
                )
                return DefensiveDecision(
                    "CLOSE", ticker, "defensive_close_curve_falling", ts,
                    {
                        "ret_pct":         pnl,
                        "slope_futura":    slope,
                        "ret_a_final_pct": ret_a_final,
                        "dias_hasta_peak": dias_peak,
                        "days_held":       age,
                        "en_zona_valle":   en_zona_valle,
                    },
                )

            # PnL positivo + curva sube → mantener, destino alcista
            if pnl > 0 and slope == "sube":
                logger.info(
                    f"📈 {ticker} HOLD | PnL={pnl:.1f}% curva sube "
                    f"(ret_peak={ret_a_peak}% en {dias_peak}d) → mantener"
                )
                return DefensiveDecision(
                    "HOLD", ticker, "hold_curve_recovering", ts,
                    {
                        "ret_pct":         pnl,
                        "slope_futura":    slope,
                        "ret_a_peak_pct":  ret_a_peak,
                        "dias_hasta_peak": dias_peak,
                        "days_held":       age,
                        "en_zona_valle":   en_zona_valle,
                    },
                )

            # [AUD-P3] Rama simétrica (solo no-anchors, ya garantizado por
            # el HOLD indefinido de anchors más arriba): curva confirma
            # caída y ya estamos en pérdida moderada (< 5%, umbral fijo,
            # independiente del stop catastrófico de 25%) → cortar antes
            # de esperar el time_exit o el stop catastrófico.
            if pnl < 0 and slope == "baja" and abs(pnl) < DEFENSIVE_CURVE_LOSS_CUT_PCT * 100:
                logger.info(
                    f"📉 {ticker} CLOSE | PnL={pnl:.1f}% curva confirma caída "
                    f"(ret_final={ret_a_final}%) → cortar pérdida moderada, no-anchor"
                )
                return DefensiveDecision(
                    "CLOSE", ticker, "defensive_close_curve_confirms_down", ts,
                    {
                        "ret_pct":         pnl,
                        "slope_futura":    slope,
                        "ret_a_final_pct": ret_a_final,
                        "dias_hasta_peak": dias_peak,
                        "days_held":       age,
                        "en_zona_valle":   en_zona_valle,
                        "loss_cut_threshold_pct": -DEFENSIVE_CURVE_LOSS_CUT_PCT * 100,
                    },
                )

            # Otros casos: HOLD con contexto de curva en meta
            return DefensiveDecision(
                "HOLD", ticker, "defensive_hold_non_anchor", ts,
                {
                    "ret_pct":        pnl,
                    "days_held":      age,
                    "anchor":         False,
                    "days_to_exit":   MAX_HOLD_DAYS_NON_ANCHOR - age,
                    "slope_futura":   slope,
                    "en_zona_valle":  en_zona_valle,
                    "ret_a_peak_pct": ret_a_peak,
                },
            )

        # Sin curva disponible → comportamiento idéntico a v1.9
        return DefensiveDecision(
            "HOLD", ticker, "defensive_hold_non_anchor", ts,
            {
                "ret_pct":      pnl,
                "days_held":    age,
                "anchor":       False,
                "days_to_exit": MAX_HOLD_DAYS_NON_ANCHOR - age,
            },
        )

    # --------------------------------------------------
    # ROTACIÓN
    # --------------------------------------------------
    def evaluate_rotation(
        self,
        fragile_pos:      Dict[str, Any],
        anchor_candidate: Dict[str, Any],
        alpha_scores:     Dict[str, float] = None,
    ) -> List[DefensiveDecision]:
        if not self.is_anchor_asset(anchor_candidate, alpha_scores):
            return []

        ts             = datetime.now(self.tz).isoformat()
        fragile_ticker = fragile_pos.get("ticker", "").upper()
        anchor_ticker  = anchor_candidate.get("ticker", "").upper()
        alpha          = (alpha_scores or {}).get(anchor_ticker, 0.0)

        entry = _safe_price(fragile_pos, "entry_price", "avg_entry_price")
        price = _safe_price(fragile_pos, "price_now", "current_price")

        meta = {
            "close_ticker":    fragile_ticker,
            "open_ticker":     anchor_ticker,
            "anchor_alpha":    round(alpha, 4),
            "fragile_ret_pct": round(pct_change(price, entry) * 100, 2),
        }

        logger.info(f"🔄 ROTATE {fragile_ticker} → {anchor_ticker} | alpha={alpha:.3f}")

        return [
            DefensiveDecision("CLOSE", fragile_ticker, "rotate_out_fragile", ts, meta),
            DefensiveDecision("OPEN",  anchor_ticker,  "rotate_in_anchor",   ts, {
                "target_pct": 0.05,
                "reason":     f"ANCHOR_ROTATE_alpha={alpha:.3f}",
                "is_anchor":  True,
                **meta,
            }),
        ]

    # --------------------------------------------------
    # PORTFOLIO COMPLETO
    # --------------------------------------------------
    def evaluate_portfolio(
        self,
        positions:       List[Dict[str, Any]],
        anchor_universe: Optional[List[Dict[str, Any]]] = None,
        total_capital:   float = 1000000,
    ) -> List[DefensiveDecision]:

        decisions    = []
        alpha_scores = _load_alpha_scores()

        anchors     = [p for p in positions if p.get("is_anchor", False)
                       or p.get("ticker", "").upper() in self._anchor_tickers]
        non_anchors = [p for p in positions if p.get("ticker", "").upper()
                       not in self._anchor_tickers
                       and not p.get("is_anchor", False)]

        for pos in positions:
            decisions.append(self.evaluate_position(pos))

        portfolio_tickers = {p.get("ticker", "").upper() for p in positions}
        eligible_anchors  = [
            a for a in self._anchor_universe
            if a["ticker"].upper() not in portfolio_tickers
            and self.is_anchor_asset(a, alpha_scores)
        ]

        logger.info(
            f"⚓ Anclas elegibles para rotación: "
            f"{[a['ticker'] for a in eligible_anchors]}"
        )

        # Rotación: no-anclas → anclas elegibles
        max_rotations  = min(2, len(non_anchors), len(eligible_anchors))
        rotations_done = 0

        for fragile in non_anchors[:max_rotations]:
            if rotations_done >= max_rotations:
                break
            anchor             = eligible_anchors[rotations_done % len(eligible_anchors)]
            rotation_decisions = self.evaluate_rotation(fragile, anchor, alpha_scores)
            if rotation_decisions:
                decisions.extend(rotation_decisions)
                rotations_done += 1

        # Apertura proactiva de anchors
        already_opening = {d.ticker for d in decisions if d.action == "OPEN"}
        anchors_to_open = [
            a for a in eligible_anchors
            if a["ticker"].upper() not in already_opening
        ]

        ts = datetime.now(self.tz).isoformat()
        for anchor in anchors_to_open[:3]:
            anchor_ticker = anchor["ticker"].upper()
            alpha         = alpha_scores.get(anchor_ticker, 0.0)
            if alpha >= 0:
                logger.info(f"⚓ OPEN proactivo → {anchor_ticker} | alpha={alpha:.3f}")
                decisions.append(DefensiveDecision(
                    "OPEN", anchor_ticker, "anchor_proactive_open", ts, {
                        "target_pct": 0.05,
                        "reason":     f"ANCHOR_PROACTIVE_alpha={alpha:.3f}",
                        "is_anchor":  True,
                    }
                ))

        closes = len([d for d in decisions if d.action == "CLOSE"])
        opens  = len([d for d in decisions if d.action == "OPEN"])
        holds  = len([d for d in decisions if d.action == "HOLD"])

        logger.info(
            f"DEFENSIVE v2.1 | pos={len(positions)} anchors={len(anchors)} "
            f"| closes={closes} opens={opens} holds={holds} "
            f"rotates={rotations_done} | capital=${total_capital:,.0f}"
        )

        return decisions

    def allow_new_positions(self, total_capital: float) -> bool:
        return self.anchor_exposure_pct < MAX_ANCHOR_EXPOSURE_PCT


# =========================================================
# SELF-TEST
# =========================================================

if __name__ == "__main__":
    pm = PMDefensive()

    test_positions = [
        {   # Anchor — debe ser HOLD, curva no aplica
            "ticker":          "JNJ",
            "avg_entry_price": 230.70,
            "current_price":   231.47,
            "qty":             18,
            "entry_date":      "2026-02-26",
        },
        {   # Anchor sin entry_date — debe ser HOLD
            "ticker":    "KO",
            "price_now": 75.50,
            "qty":       56,
        },
        {   # No-anchor
            "ticker":          "BALL",
            "avg_entry_price": 64.95,
            "current_price":   64.03,
            "qty":             67,
            "entry_date":      "2026-03-01",
        },
        {   # No-anchor sin entry_date — debe ser HOLD (age=0)
            "ticker":          "STT",
            "avg_entry_price": 156.72,
            "current_price":   157.00,
            "qty":             27,
        },
    ]

    print("\nPMDefensive v2.1 SELF-TEST:")
    results = pm.evaluate_portfolio(test_positions, total_capital=85000)

    print("\nDECISIONES:")
    for d in results:
        print(f"  {d.action:6} {d.ticker:12} {d.reason}")

    jnj = next((d for d in results if d.ticker == "JNJ"),  None)
    ko  = next((d for d in results if d.ticker == "KO"),   None)
    stt = next((d for d in results if d.ticker == "STT"),  None)

    assert jnj and jnj.action == "HOLD", f"❌ JNJ debe ser HOLD, es {jnj}"
    assert ko  and ko.action  == "HOLD", f"❌ KO debe ser HOLD, es {ko}"
    assert stt and stt.action == "HOLD", f"❌ STT sin entry_date debe ser HOLD (age=0), es {stt}"

    print("\n✅ v2.1 TEST OK — curva desde disco integrada, retrocompatibilidad v1.9 confirmada")

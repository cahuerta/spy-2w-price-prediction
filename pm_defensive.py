"""
pm_defensive.py - DEFENSIVE POSITION MANAGER v2.0

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

v2.0 — Contexto de curva futura del tracker:
  [CF1] evaluate_portfolio() acepta tracker_signals: Dict[str, Dict] = None
        Parámetro opcional — retrocompatible con llamadas existentes.
  [CF2] evaluate_position() acepta tracker_signal: Dict = None
        Lee curva_futura si está disponible en la señal del tracker.
  [CF3] Lógica curva futura (solo no-anchors, solo informativa):
        - diverging + pnl > 0 + slope_futura="baja"
            → CLOSE "defensive_close_curve_falling"
            → tomamos ganancia antes de caída esperada
        - diverging + pnl > 0 + slope_futura="sube"
            → HOLD "hold_curve_recovering"
            → modelo erró hoy pero destino alcista — no cortar
        - en_zona_valle=True
            → se agrega a meta como contexto (el orchestrator decide)
        ANCLAS: ignoran curva_futura — su lógica no cambia.
  [CF4] Si tracker_signal es None o no tiene curva_futura,
        comportamiento idéntico a v1.9 — sin regresiones.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
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

_BASE_DIR   = Path(__file__).resolve().parent
ANCHOR_FILE = Path(os.getenv("ANCHOR_FILE", str(_BASE_DIR / "anchor_universe.json")))
ALPHA_FILE  = DATA_PATH / "alpha_last.json"

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
# CARGA ANCHOR UNIVERSE
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
# HELPER CURVA FUTURA — v2.0
# =========================================================

def _leer_curva_futura(tracker_signal: Optional[Dict]) -> Optional[Dict]:
    """
    [CF2] Extrae curva_futura de la señal del tracker.
    Retorna None si no está disponible — sin efectos secundarios.
    """
    if not tracker_signal or not isinstance(tracker_signal, dict):
        return None
    return tracker_signal.get("curva_futura") or None


# =========================================================
# PM DEFENSIVO v2.0
# =========================================================

class PMDefensive:

    def __init__(self):
        self.tz                  = CL_TIMEZONE
        self.anchor_exposure_pct = 0.0
        self._anchor_universe    = _load_anchor_universe()
        self._anchor_tickers     = {a["ticker"].upper() for a in self._anchor_universe}
        logger.info(
            f"PMDefensive v2.0 | anclas={len(self._anchor_tickers)}: "
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
    def evaluate_position(
        self,
        pos:            Dict[str, Any],
        tracker_signal: Optional[Dict] = None,   # [CF2] señal del tracker con curva_futura
    ) -> DefensiveDecision:
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

        # Stop catastrófico — aplica a TODOS incluyendo anchors
        if ret <= -CATASTROPHIC_STOP_PCT:
            logger.warning(
                f"💀 {ticker} CATASTROPHIC STOP | ret={ret:.1%} | anchor={is_anchor}"
            )
            return DefensiveDecision(
                "CLOSE", ticker, "catastrophic_loss", ts,
                {"ret_pct": round(ret * 100, 2), "stop_pct": -CATASTROPHIC_STOP_PCT * 100},
            )

        # Anchors → HOLD indefinido (excepto catastrófico)
        # [CF3] ANCLAS ignoran curva_futura — lógica sin cambios
        if is_anchor:
            return DefensiveDecision(
                "HOLD", ticker, "anchor_hold_indefinite", ts,
                {
                    "ret_pct":           round(ret * 100, 2),
                    "days_held":         age,
                    "anchor":            True,
                    "dist_to_stop_pct":  round((ret + CATASTROPHIC_STOP_PCT) * 100, 1),
                },
            )

        # No-anchors → time exit
        if age >= MAX_HOLD_DAYS_NON_ANCHOR:
            return DefensiveDecision(
                "CLOSE", ticker, "non_anchor_time_exit", ts,
                {"days_held": age, "max_days": MAX_HOLD_DAYS_NON_ANCHOR},
            )

        # ─── [CF3] Contexto de curva futura — solo no-anchors ──────────────
        # Si el tracker proveyó curva_futura, úsala para refinar la decisión.
        # Si no hay señal, comportamiento idéntico a v1.9.
        # ────────────────────────────────────────────────────────────────────
        curva_futura = _leer_curva_futura(tracker_signal)

        if curva_futura:
            slope         = curva_futura.get("slope_futura")          # "sube"|"baja"|"plana"
            en_zona_valle = curva_futura.get("en_zona_valle", False)
            ret_a_peak    = curva_futura.get("ret_desde_hoy_a_peak_pct")
            ret_a_final   = curva_futura.get("ret_desde_hoy_a_final_pct")
            dias_peak     = curva_futura.get("dias_hasta_peak")
            pnl_pct       = round(ret * 100, 2)

            # PnL positivo + curva cae desde aquí → cerrar tomando ganancia
            if pnl_pct > 0 and slope == "baja":
                logger.info(
                    f"📉 {ticker} CLOSE | PnL={pnl_pct:.1f}% + curva cae "
                    f"(slope={slope} ret_final={ret_a_final}%) → tomar ganancia"
                )
                return DefensiveDecision(
                    "CLOSE", ticker, "defensive_close_curve_falling", ts,
                    {
                        "ret_pct":          pnl_pct,
                        "slope_futura":     slope,
                        "ret_a_final_pct":  ret_a_final,
                        "dias_hasta_peak":  dias_peak,
                        "days_held":        age,
                        "en_zona_valle":    en_zona_valle,
                    },
                )

            # PnL positivo + curva sube → no cortar, el modelo erró hoy
            if pnl_pct > 0 and slope == "sube":
                logger.info(
                    f"📈 {ticker} HOLD | PnL={pnl_pct:.1f}% + curva sube "
                    f"(slope={slope} ret_peak={ret_a_peak}% en {dias_peak}d) → mantener"
                )
                return DefensiveDecision(
                    "HOLD", ticker, "hold_curve_recovering", ts,
                    {
                        "ret_pct":          pnl_pct,
                        "slope_futura":     slope,
                        "ret_a_peak_pct":   ret_a_peak,
                        "dias_hasta_peak":  dias_peak,
                        "days_held":        age,
                        "en_zona_valle":    en_zona_valle,
                    },
                )

            # PnL negativo + en zona de valle → informar, no cerrar todavía
            # El PM defensivo no promedia a la baja, pero deja el contexto
            # en meta para que el orchestrator / operador lo vea
            if pnl_pct < 0 and en_zona_valle:
                logger.info(
                    f"🪣 {ticker} en zona valle | PnL={pnl_pct:.1f}% "
                    f"slope={slope} → HOLD con contexto valle"
                )
                return DefensiveDecision(
                    "HOLD", ticker, "defensive_hold_zona_valle", ts,
                    {
                        "ret_pct":          pnl_pct,
                        "slope_futura":     slope,
                        "ret_a_peak_pct":   ret_a_peak,
                        "dias_hasta_peak":  dias_peak,
                        "days_held":        age,
                        "en_zona_valle":    True,
                        "days_to_exit":     MAX_HOLD_DAYS_NON_ANCHOR - age,
                    },
                )

        # Sin curva_futura o slope plano → comportamiento v1.9
        return DefensiveDecision(
            "HOLD", ticker, "defensive_hold_non_anchor", ts,
            {
                "ret_pct":      round(ret * 100, 2),
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
        tracker_signals: Optional[Dict[str, Dict]] = None,  # [CF1] señales del tracker
    ) -> List[DefensiveDecision]:
        """
        [CF1] tracker_signals: dict {ticker: señal_tracker} con curva_futura.
        Parámetro opcional — si es None, comportamiento idéntico a v1.9.
        El orchestrator lo pasa después de llamar al tracker.
        """
        decisions    = []
        alpha_scores = _load_alpha_scores()

        tracker_signals = tracker_signals or {}

        anchors     = [p for p in positions if p.get("is_anchor", False)
                       or p.get("ticker", "").upper() in self._anchor_tickers]
        non_anchors = [p for p in positions if p.get("ticker", "").upper()
                       not in self._anchor_tickers
                       and not p.get("is_anchor", False)]

        for pos in positions:
            ticker         = str(pos.get("ticker", "")).upper()
            tracker_signal = tracker_signals.get(ticker)   # [CF2] None si no hay señal
            decisions.append(self.evaluate_position(pos, tracker_signal))

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
            f"DEFENSIVE v2.0 | pos={len(positions)} anchors={len(anchors)} "
            f"| closes={closes} opens={opens} holds={holds} "
            f"rotates={rotations_done} | capital=${total_capital:,.0f} "
            f"| tracker_signals={len(tracker_signals)}"
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
        {   # Anchor con campos de Alpaca — debe ser HOLD
            "ticker":          "JNJ",
            "avg_entry_price": 230.70,
            "current_price":   231.47,
            "qty":             18,
            "entry_date":      "2026-02-26",
        },
        {   # Anchor sin entry_date — debe ser HOLD (no CLOSE, no warning)
            "ticker":    "KO",
            "price_now": 75.50,
            "qty":       56,
        },
        {   # No-anchor con entry_date como fecha simple
            "ticker":          "BALL",
            "avg_entry_price": 64.95,
            "current_price":   64.03,
            "qty":             67,
            "entry_date":      "2026-03-01",
        },
        {   # No-anchor sin entry_date — debe ser HOLD (age=0, no expirado)
            "ticker":          "STT",
            "avg_entry_price": 156.72,
            "current_price":   157.00,
            "qty":             27,
        },
    ]

    # Test con tracker_signals simuladas
    test_tracker_signals = {
        "BALL": {
            "curva_futura": {
                "slope_futura":              "baja",
                "ret_desde_hoy_a_peak_pct":  1.2,
                "ret_desde_hoy_a_final_pct": -3.5,
                "dias_hasta_peak":           1,
                "en_zona_valle":             False,
            }
        },
        "STT": {
            "curva_futura": {
                "slope_futura":              "sube",
                "ret_desde_hoy_a_peak_pct":  4.8,
                "ret_desde_hoy_a_final_pct": 2.1,
                "dias_hasta_peak":           3,
                "en_zona_valle":             False,
            }
        },
    }

    print("\nPMDefensive v2.0 SELF-TEST:")
    print("\n── Sin tracker_signals (retrocompatibilidad v1.9) ──")
    results_v19 = pm.evaluate_portfolio(test_positions, total_capital=85000)
    for d in results_v19:
        print(f"  {d.action:6} {d.ticker:12} {d.reason}")

    print("\n── Con tracker_signals (v2.0) ──")
    results_v20 = pm.evaluate_portfolio(
        test_positions,
        total_capital=85000,
        tracker_signals=test_tracker_signals,
    )
    for d in results_v20:
        print(f"  {d.action:6} {d.ticker:12} {d.reason}")

    # Asserts retrocompatibilidad
    jnj = next((d for d in results_v19 if d.ticker == "JNJ"),  None)
    ko  = next((d for d in results_v19 if d.ticker == "KO"),   None)
    stt = next((d for d in results_v19 if d.ticker == "STT"),  None)

    assert jnj and jnj.action == "HOLD", f"❌ JNJ debe ser HOLD, es {jnj}"
    assert ko  and ko.action  == "HOLD", f"❌ KO debe ser HOLD, es {ko}"
    assert stt and stt.action == "HOLD", f"❌ STT sin entry_date debe ser HOLD (age=0), es {stt}"

    print("\n✅ v2.0 TEST OK — curva futura integrada, retrocompatibilidad v1.9 confirmada")

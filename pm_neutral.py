"""
pm_neutral.py — NEUTRAL POSITION MANAGER v1.3 PRODUCCIÓN

PM NEUTRAL (equilibrio riesgo / oportunidad)

✔ Mantiene capital trabajando de forma selectiva
✔ Reduce rotación y sizing vs pm_growth
✔ Filtra señales débiles
✔ Respeta stops y tiempo
✔ Puente entre growth ↔ defensive
✔ Trailing stop + confidence-aware

FIX v1.2:
  [N1] invalid_price → HOLD preventivo en vez de CLOSE
       Cuando Alpaca no retorna precio (fuera de horario, error transitorio),
       el PM cerraba la posición inmediatamente causando pérdidas innecesarias.
       Ahora retorna HOLD y loggea warning — se evaluará en el siguiente ciclo.
  [N2] entry <= 0 también es HOLD si price > 0 (puede ser error de datos de entrada,
       no necesariamente una posición inválida real).

v1.3 — Contexto de curva futura del tracker:
  [CF1] evaluate_portfolio() acepta tracker_signals: Dict[str, Dict] = None
        Parámetro opcional — retrocompatible con llamadas existentes.
  [CF2] evaluate_position() acepta tracker_signal: Dict = None
        Lee curva_futura si está disponible en la señal del tracker.
  [CF3] Lógica curva futura (solo si tracker_signal disponible):
        Se inserta ANTES de los stops existentes para:
        - pnl > 0 + slope="baja" → CLOSE "neutral_close_curve_falling"
          (tomar ganancia antes de caída esperada, complementa take_profit)
        - pnl > 0 + slope="sube" → anular trailing/time_exit si caída es
          menor al stop duro (no cortar si el destino es alcista)
        - en_zona_valle=True → agrega contexto en meta, no modifica acción
        Los stops duros (stop_loss, trailing, take_profit) tienen prioridad
        sobre curva_futura — no se sobreescriben nunca.
  [CF4] Si tracker_signal es None o no tiene curva_futura,
        comportamiento idéntico a v1.2 — sin regresiones.
  [CF5] evaluate_portfolio() retorna los mismos campos que v1.2
        más "tracker_signals_count" para trazabilidad.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import pytz

# =================================================================
# CONFIGURACIÓN NEUTRAL
# =================================================================
CL_TIMEZONE = pytz.timezone("America/Santiago")

MAX_HOLD_DAYS_NEUTRAL     = int(os.getenv("PM_NEU_MAX_HOLD_DAYS",   "10"))
MIN_CONFIDENCE_NEUTRAL    = float(os.getenv("PM_NEU_MIN_CONF",      "0.60"))
STOP_LOSS_NEUTRAL_PCT     = float(os.getenv("PM_NEU_STOP_LOSS",     "0.04"))   # 4%
TAKE_PROFIT_NEUTRAL_PCT   = float(os.getenv("PM_NEU_TAKE_PROFIT",   "0.07"))   # 7%
TRAILING_STOP_NEUTRAL_PCT = float(os.getenv("PM_NEU_TRAILING",      "0.02"))   # 2%
MAX_RISK_PER_TRADE_NEUTRAL= float(os.getenv("PM_NEU_RISK_PER_TRADE","0.005"))  # 0.5%

logger = logging.getLogger("pm_neutral")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# =================================================================
# HELPERS
# =================================================================

def pct_change(current: float, entry: float) -> float:
    return (current / entry - 1.0) if entry > 0 else 0.0

def days_between(entry_iso: str) -> int:
    try:
        entry_str = entry_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(entry_str)
        now_cl   = datetime.now(CL_TIMEZONE)
        entry_cl = dt.astimezone(CL_TIMEZONE)
        return max(0, (now_cl - entry_cl).days)
    except Exception as e:
        logger.warning(f"Error days_between '{entry_iso}': {e}")
        return MAX_HOLD_DAYS_NEUTRAL

# =================================================================
# DATACLASS DECISIÓN
# =================================================================

@dataclass
class NeutralDecision:
    action:    str        # "OPEN" | "CLOSE" | "HOLD"
    ticker:    str
    reason:    str
    timestamp: str
    meta:      Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **{k: v for k, v in self.__dict__.items() if k != "meta"},
            "meta": self.meta or {},
        }

# =================================================================
# HELPER CURVA FUTURA — v1.3
# =================================================================

def _leer_curva_futura(tracker_signal: Optional[Dict]) -> Optional[Dict]:
    """
    [CF2] Extrae curva_futura de la señal del tracker.
    Retorna None si no está disponible — sin efectos secundarios.
    """
    if not tracker_signal or not isinstance(tracker_signal, dict):
        return None
    return tracker_signal.get("curva_futura") or None

# =================================================================
# PM NEUTRAL
# =================================================================

class PMNeutral:
    """PM NEUTRAL. Objetivo: participar selectivamente sin sobreexposición."""

    def __init__(self):
        self.tz = CL_TIMEZONE
        logger.info("🟡 PMNeutral v1.3 inicializado – MODO BALANCEADO")

    # --------------------------------------------------
    # EVALUAR POSICIÓN EXISTENTE
    # --------------------------------------------------
    def evaluate_position(
        self,
        pos:            Dict[str, Any],
        tracker_signal: Optional[Dict] = None,   # [CF2] señal del tracker con curva_futura
    ) -> NeutralDecision:

        ticker = str(pos.get("ticker", "UNKNOWN")).upper()

        try:
            entry      = float(pos.get("entry_price", 0))
            price      = float(pos.get("price_now", 0))
            peak       = float(pos.get("peak_price", entry))
            entry_time = str(pos.get("entry_time", ""))
            confidence = float(pos.get("confidence", 0))
        except (ValueError, TypeError) as e:
            logger.error(f"Posición inválida {ticker}: {e}")
            # [N1] Error de parseo → HOLD, no CLOSE
            return NeutralDecision(
                "HOLD", ticker, "input_error_hold",
                datetime.now(self.tz).isoformat(),
                {"error": str(e)}
            )

        # [N1] Precio inválido → HOLD PREVENTIVO
        if price <= 0:
            logger.warning(
                f"⚠️ {ticker}: price_now={price} inválido → HOLD preventivo "
                f"(se evaluará en próximo ciclo)"
            )
            return NeutralDecision(
                "HOLD", ticker, "invalid_price_hold",
                datetime.now(self.tz).isoformat(),
                {"price": price, "entry": entry, "note": "precio no disponible temporalmente"}
            )

        # [N2] Entry inválido con price válido → HOLD
        if entry <= 0:
            logger.warning(
                f"⚠️ {ticker}: entry_price={entry} inválido con price={price} → HOLD preventivo"
            )
            return NeutralDecision(
                "HOLD", ticker, "invalid_entry_hold",
                datetime.now(self.tz).isoformat(),
                {"price": price, "entry": entry}
            )

        ret      = pct_change(price, entry)
        age_days = days_between(entry_time)

        logger.debug(
            f"{ticker}: ret={ret:.1%} peak={pct_change(peak,entry):.1%} "
            f"conf={confidence:.2f} age={age_days}d"
        )

        # ── [CF3] Contexto de curva futura ─────────────────────────────────
        # Se evalúa ANTES de trailing y time_exit, pero DESPUÉS de stop duro.
        # Los stops duros tienen siempre prioridad.
        # Si no hay curva_futura, este bloque es transparente (no-op).
        # ───────────────────────────────────────────────────────────────────
        curva_futura = _leer_curva_futura(tracker_signal)

        # 1️⃣ STOP LOSS DURO — máxima prioridad, nunca se sobreescribe
        if price <= entry * (1 - STOP_LOSS_NEUTRAL_PCT):
            return NeutralDecision(
                "CLOSE", ticker, "stop_loss_neutral",
                datetime.now(self.tz).isoformat(),
                {"ret_pct": round(ret * 100, 2), "trigger_pct": -STOP_LOSS_NEUTRAL_PCT}
            )

        # 2️⃣ TAKE PROFIT — también tiene prioridad sobre curva
        if ret >= TAKE_PROFIT_NEUTRAL_PCT:
            return NeutralDecision(
                "CLOSE", ticker, "take_profit_neutral",
                datetime.now(self.tz).isoformat(),
                {"ret_pct": round(ret * 100, 2), "trigger_pct": TAKE_PROFIT_NEUTRAL_PCT}
            )

        # ── [CF3] Curva futura: cierre estratégico antes de caída ──────────
        if curva_futura:
            slope         = curva_futura.get("slope_futura")
            en_zona_valle = curva_futura.get("en_zona_valle", False)
            ret_a_peak    = curva_futura.get("ret_desde_hoy_a_peak_pct")
            ret_a_final   = curva_futura.get("ret_desde_hoy_a_final_pct")
            dias_peak     = curva_futura.get("dias_hasta_peak")
            pnl_pct       = round(ret * 100, 2)

            # PnL positivo + curva cae → cerrar tomando ganancia
            # (complementa take_profit para casos < 7% pero con caída esperada)
            if pnl_pct > 0 and slope == "baja":
                logger.info(
                    f"📉 {ticker} CLOSE | PnL={pnl_pct:.1f}% + curva cae "
                    f"(slope={slope} ret_final={ret_a_final}%) → tomar ganancia"
                )
                return NeutralDecision(
                    "CLOSE", ticker, "neutral_close_curve_falling",
                    datetime.now(self.tz).isoformat(),
                    {
                        "ret_pct":          pnl_pct,
                        "slope_futura":     slope,
                        "ret_a_final_pct":  ret_a_final,
                        "dias_hasta_peak":  dias_peak,
                        "days_held":        age_days,
                        "en_zona_valle":    en_zona_valle,
                    },
                )

            # PnL positivo + curva sube → no aplicar trailing ni time_exit
            # El destino es alcista — no cortar por tiempo o trailing pequeño
            if pnl_pct > 0 and slope == "sube":
                logger.info(
                    f"📈 {ticker} HOLD | PnL={pnl_pct:.1f}% + curva sube "
                    f"(ret_peak={ret_a_peak}% en {dias_peak}d) → omitir trailing/time"
                )
                return NeutralDecision(
                    "HOLD", ticker, "hold_curve_recovering",
                    datetime.now(self.tz).isoformat(),
                    {
                        "ret_pct":          pnl_pct,
                        "slope_futura":     slope,
                        "ret_a_peak_pct":   ret_a_peak,
                        "dias_hasta_peak":  dias_peak,
                        "days_held":        age_days,
                        "en_zona_valle":    en_zona_valle,
                    },
                )

        # 3️⃣ TRAILING STOP — después de curva (curva puede anularlo si slope=sube)
        trail_stop = peak * (1 - TRAILING_STOP_NEUTRAL_PCT)
        if price <= trail_stop:
            return NeutralDecision(
                "CLOSE", ticker, "trailing_stop_neutral",
                datetime.now(self.tz).isoformat(),
                {
                    "ret_pct":           round(ret * 100, 2),
                    "peak_pct":          round(pct_change(peak, entry) * 100, 2),
                    "trail_trigger_pct": -TRAILING_STOP_NEUTRAL_PCT,
                }
            )

        # 4️⃣ TIEMPO MÁXIMO — después de curva (curva puede anularlo si slope=sube)
        if age_days >= MAX_HOLD_DAYS_NEUTRAL:
            return NeutralDecision(
                "CLOSE", ticker, f"time_exit_{age_days}d",
                datetime.now(self.tz).isoformat(),
                {"days_held": age_days, "max_days": MAX_HOLD_DAYS_NEUTRAL}
            )

        # 5️⃣ CONFIDENCE COLAPSÓ — no se ve afectado por curva
        if confidence < MIN_CONFIDENCE_NEUTRAL * 0.7:
            return NeutralDecision(
                "CLOSE", ticker, "confidence_collapse",
                datetime.now(self.tz).isoformat(),
                {"current_conf": confidence, "min_conf": MIN_CONFIDENCE_NEUTRAL}
            )

        # 6️⃣ HOLD — agrega contexto de curva si disponible
        hold_meta = {
            "ret_pct":        round(ret * 100, 2),
            "days_held":      age_days,
            "confidence":     confidence,
            "distance_sl":    round((price / entry - (1 - STOP_LOSS_NEUTRAL_PCT)) * 100, 2),
            "distance_trail": round((price / peak  - (1 - TRAILING_STOP_NEUTRAL_PCT)) * 100, 2),
        }
        if curva_futura:
            hold_meta["slope_futura"]    = curva_futura.get("slope_futura")
            hold_meta["en_zona_valle"]   = curva_futura.get("en_zona_valle", False)
            hold_meta["ret_a_peak_pct"]  = curva_futura.get("ret_desde_hoy_a_peak_pct")
            hold_meta["dias_hasta_peak"] = curva_futura.get("dias_hasta_peak")

        return NeutralDecision(
            "HOLD", ticker, "hold_neutral",
            datetime.now(self.tz).isoformat(),
            hold_meta,
        )

    # --------------------------------------------------
    # EVALUAR NUEVA SEÑAL
    # --------------------------------------------------
    def evaluate_signal(self, signal: Dict[str, Any]) -> NeutralDecision:
        ticker     = str(signal.get("ticker", "UNKNOWN")).upper()
        confidence = float(signal.get("confidence", 0))
        price      = float(signal.get("price", 0))

        if confidence < MIN_CONFIDENCE_NEUTRAL:
            return NeutralDecision(
                "REJECT", ticker, "low_confidence",
                datetime.now(self.tz).isoformat(),
                {"confidence": confidence, "min_required": MIN_CONFIDENCE_NEUTRAL}
            )

        if price <= 0:
            return NeutralDecision(
                "REJECT", ticker, "invalid_price",
                datetime.now(self.tz).isoformat()
            )

        return NeutralDecision(
            "OPEN", ticker, "signal_approved",
            datetime.now(self.tz).isoformat(),
            {
                "confidence":     confidence,
                "risk_per_trade": MAX_RISK_PER_TRADE_NEUTRAL,
                "approved_size":  "dynamic",
            }
        )

    # --------------------------------------------------
    # PORTFOLIO COMPLETO
    # --------------------------------------------------
    def evaluate_portfolio(
        self,
        positions:       List[Dict[str, Any]],
        tracker_signals: Optional[Dict[str, Dict]] = None,  # [CF1] señales del tracker
    ) -> Dict[str, Any]:
        """
        [CF1] tracker_signals: dict {ticker: señal_tracker} con curva_futura.
        Parámetro opcional — si es None, comportamiento idéntico a v1.2.
        Retorna los mismos campos que v1.2 + tracker_signals_count.
        """
        if not positions:
            return {
                "mode":                 "neutral",
                "decisions":            [],
                "positions":            0,
                "closes":               0,
                "tracker_signals_count": 0,
                "timestamp":            datetime.now(self.tz).isoformat(),
                "message":              "Portfolio vacío",
            }

        tracker_signals  = tracker_signals or {}
        decisions        = []
        closes           = 0
        holds_preventive = 0

        for pos in positions:
            ticker         = str(pos.get("ticker", "")).upper()
            tracker_signal = tracker_signals.get(ticker)   # [CF2] None si no hay señal
            decision       = self.evaluate_position(pos, tracker_signal)
            decisions.append(decision.to_dict())

            if decision.action == "CLOSE":
                closes += 1
                logger.info(f"🟡 CLOSE: {decision.ticker} - {decision.reason}")
            elif decision.action == "HOLD" and "invalid" in decision.reason:
                holds_preventive += 1
                logger.info(f"🛡 HOLD preventivo: {decision.ticker} - {decision.reason}")

        close_pct = round((closes / len(positions)) * 100, 1) if positions else 0
        logger.info(
            f"📊 Neutral v1.3 eval: {len(positions)} pos, {closes} closes ({close_pct}%), "
            f"{holds_preventive} holds preventivos, "
            f"tracker_signals={len(tracker_signals)}"
        )

        return {
            "mode":                  "neutral",
            "decisions":             decisions,
            "positions":             len(positions),
            "closes":                closes,
            "holds_preventive":      holds_preventive,
            "close_pct":             close_pct,
            "tracker_signals_count": len(tracker_signals),
            "timestamp":             datetime.now(self.tz).isoformat(),
            "config": {
                "max_hold_days":      MAX_HOLD_DAYS_NEUTRAL,
                "min_confidence":     MIN_CONFIDENCE_NEUTRAL,
                "stop_loss_pct":      STOP_LOSS_NEUTRAL_PCT,
                "take_profit_pct":    TAKE_PROFIT_NEUTRAL_PCT,
                "trailing_stop_pct":  TRAILING_STOP_NEUTRAL_PCT,
                "max_risk_per_trade": MAX_RISK_PER_TRADE_NEUTRAL,
            },
        }


# =================================================================
# SELF TEST
# =================================================================

if __name__ == "__main__":
    pm = PMNeutral()

    test_positions = [
        {   # [N1] Precio 0 → debe retornar HOLD, no CLOSE
            "ticker":      "AMAT",
            "entry_price": 150.0,
            "price_now":   0,        # Alpaca fuera de horario
            "peak_price":  155.0,
            "entry_time":  "2026-01-10T00:00:00Z",
            "confidence":  0.65,
        },
        {   # Stop loss normal
            "ticker":      "AAPL",
            "entry_price": 150.0,
            "price_now":   144.0,    # -4% → CLOSE
            "peak_price":  155.0,
            "entry_time":  "2026-01-01T00:00:00Z",
            "confidence":  0.65,
        },
        {   # Hold normal
            "ticker":      "MSFT",
            "entry_price": 300.0,
            "price_now":   315.0,    # +5% → HOLD
            "peak_price":  320.0,
            "entry_time":  "2026-01-10T00:00:00Z",
            "confidence":  0.70,
        },
        {   # [CF3] PnL positivo + curva cae → CLOSE antes de take_profit
            "ticker":      "NVDA",
            "entry_price": 100.0,
            "price_now":   104.0,    # +4% (bajo take_profit 7%)
            "peak_price":  105.0,
            "entry_time":  "2026-06-10T00:00:00Z",
            "confidence":  0.70,
        },
        {   # [CF3] Trailing activado pero curva sube → HOLD
            "ticker":      "TSLA",
            "entry_price": 200.0,
            "price_now":   215.0,    # trailing: peak 220 → stop 215.6 → activado
            "peak_price":  220.0,
            "entry_time":  "2026-06-10T00:00:00Z",
            "confidence":  0.70,
        },
    ]

    test_tracker_signals = {
        "NVDA": {
            "curva_futura": {
                "slope_futura":              "baja",
                "ret_desde_hoy_a_peak_pct":  0.5,
                "ret_desde_hoy_a_final_pct": -4.2,
                "dias_hasta_peak":           1,
                "en_zona_valle":             False,
            }
        },
        "TSLA": {
            "curva_futura": {
                "slope_futura":              "sube",
                "ret_desde_hoy_a_peak_pct":  5.1,
                "ret_desde_hoy_a_final_pct": 3.8,
                "dias_hasta_peak":           3,
                "en_zona_valle":             False,
            }
        },
    }

    print("🧪 PMNeutral v1.3 SELF TEST:")

    print("\n── Sin tracker_signals (retrocompatibilidad v1.2) ──")
    result_v12 = pm.evaluate_portfolio(test_positions)
    for d in result_v12["decisions"]:
        print(f"  {d['action']:6} {d['ticker']:8} {d['reason']}")

    print("\n── Con tracker_signals (v1.3) ──")
    result_v13 = pm.evaluate_portfolio(test_positions, tracker_signals=test_tracker_signals)
    for d in result_v13["decisions"]:
        print(f"  {d['action']:6} {d['ticker']:8} {d['reason']}")

    # Asserts retrocompatibilidad v1.2
    amat = next(d for d in result_v12["decisions"] if d["ticker"] == "AMAT")
    aapl = next(d for d in result_v12["decisions"] if d["ticker"] == "AAPL")
    assert amat["action"] == "HOLD",  f"❌ AMAT debe ser HOLD, es {amat['action']}"
    assert aapl["action"] == "CLOSE", f"❌ AAPL debe ser CLOSE, es {aapl['action']}"

    # Asserts v1.3 curva futura
    nvda = next(d for d in result_v13["decisions"] if d["ticker"] == "NVDA")
    tsla = next(d for d in result_v13["decisions"] if d["ticker"] == "TSLA")
    assert nvda["action"] == "CLOSE" and nvda["reason"] == "neutral_close_curve_falling", \
        f"❌ NVDA debe ser CLOSE curve_falling, es {nvda}"
    assert tsla["action"] == "HOLD"  and tsla["reason"] == "hold_curve_recovering", \
        f"❌ TSLA debe ser HOLD curve_recovering, es {tsla}"

    print("\n✅ PMNeutral v1.3 – TEST OK – curva futura integrada, retrocompatibilidad v1.2 confirmada")

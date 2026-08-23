"""
pm_neutral.py — NEUTRAL POSITION MANAGER v1.4

PM NEUTRAL (equilibrio riesgo / oportunidad)

✔ Mantiene capital trabajando de forma selectiva
✔ Reduce rotación y sizing vs pm_growth
✔ Filtra señales débiles
✔ Respeta stops y tiempo
✔ Puente entre growth ↔ defensive
✔ Trailing stop + confidence-aware

FIX v1.2:
  [N1] invalid_price → HOLD preventivo en vez de CLOSE
  [N2] entry <= 0 también es HOLD si price > 0

v1.3:
  [C1] _load_price_curve(ticker): lee curva de predicción desde disco
  [C2] evaluate_position(): usa curva DESPUÉS de stops duros
  [C3] Stops duros: sin cambios, prioridad absoluta
  [C4] Orchestrator y tracker: sin cambios

v1.4:
  [N3] _safe_price(): helper propagado desde pm_defensive v1.8.
       Lee avg_entry_price / current_price como fallback de
       entry_price / price_now — Alpaca devuelve estos campos
       y sin el fallback entry=0 disparaba closes incorrectos.
  [N4] days_between(): retorna 0 (no MAX_HOLD_DAYS_NEUTRAL) cuando
       entry_iso está vacío — evita cierres por tiempo en posiciones
       sin metadata. Lee entry_date con fallback a entry_time.
  [N5] evaluate_position(): usa _safe_price() y lee entry_date
       con fallback a entry_time por compatibilidad.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import pytz

# =================================================================
# CONFIGURACIÓN NEUTRAL
# =================================================================
CL_TIMEZONE = pytz.timezone("America/Santiago")

DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
PRED_DIR  = DATA_PATH / "predictions"

MAX_HOLD_DAYS_NEUTRAL     = int(os.getenv("PM_NEU_MAX_HOLD_DAYS",   "10"))
MIN_CONFIDENCE_NEUTRAL    = float(os.getenv("PM_NEU_MIN_CONF",      "0.60"))
STOP_LOSS_NEUTRAL_PCT     = float(os.getenv("PM_NEU_STOP_LOSS",     "0.04"))
TAKE_PROFIT_NEUTRAL_PCT   = float(os.getenv("PM_NEU_TAKE_PROFIT",   "0.07"))
TRAILING_STOP_NEUTRAL_PCT = float(os.getenv("PM_NEU_TRAILING",      "0.02"))
MAX_RISK_PER_TRADE_NEUTRAL= float(os.getenv("PM_NEU_RISK_PER_TRADE","0.005"))

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


def _safe_price(pos: dict, *keys) -> float:
    """
    [N3] Busca el primer campo de precio válido en la posición.
    Alpaca devuelve avg_entry_price / current_price en vez de
    entry_price / price_now que usan los PMs internamente.
    Propagado desde pm_defensive.py v1.8.
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


def days_between(entry_iso: str) -> int:
    """
    [N4] Retorna 0 si entry_iso está vacío o ausente — la posición
         se trata como nueva, no como expirada.
         Maneja fechas sin timezone ("YYYY-MM-DD") añadiendo
         CL_TIMEZONE antes de comparar.
    """
    try:
        if not entry_iso or not entry_iso.strip():
            return 0
        entry_str = entry_iso.strip().replace("Z", "+00:00")
        dt        = datetime.fromisoformat(entry_str)
        if dt.tzinfo is None:
            dt = CL_TIMEZONE.localize(dt)
        now_cl    = datetime.now(CL_TIMEZONE)
        entry_cl  = dt.astimezone(CL_TIMEZONE)
        return max(0, (now_cl - entry_cl).days)
    except Exception as e:
        logger.warning(f"Error days_between '{entry_iso}': {e}")
        return 0


# =================================================================
# CARGA CURVA DE PREDICCIÓN — v1.3
# =================================================================

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

        idx_peak     = int(max(range(len(path)), key=lambda i: path[i]))
        precio_peak  = float(path[idx_peak])
        precio_final = float(path[-1])

        ret_a_peak  = round((precio_peak  / price_now - 1) * 100, 2) if price_now > 0 else None
        ret_a_final = round((precio_final / price_now - 1) * 100, 2) if price_now > 0 else None

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


# =================================================================
# DATACLASS DECISIÓN
# =================================================================

@dataclass
class NeutralDecision:
    action:    str
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
# PM NEUTRAL
# =================================================================

class PMNeutral:
    """PM NEUTRAL. Objetivo: participar selectivamente sin sobreexposición."""

    def __init__(self):
        self.tz = CL_TIMEZONE
        logger.info("🟡 PMNeutral v1.4 inicializado – MODO BALANCEADO")

    def evaluate_position(self, pos: Dict[str, Any]) -> NeutralDecision:
        """
        [N5] Usa _safe_price() para leer precios con fallback a campos
             de Alpaca (avg_entry_price / current_price).
             Lee entry_date con fallback a entry_time.
        """
        ticker = str(pos.get("ticker", "UNKNOWN")).upper()

        try:
            entry      = _safe_price(pos, "entry_price", "avg_entry_price")
            price      = _safe_price(pos, "price_now", "current_price")
            peak       = _safe_price(pos, "peak_price") or entry
            entry_time = str(pos.get("entry_date") or pos.get("entry_time") or "")
            confidence = float(pos.get("confidence", 0) or 0)
        except (ValueError, TypeError) as e:
            logger.error(f"Posición inválida {ticker}: {e}")
            return NeutralDecision(
                "HOLD", ticker, "input_error_hold",
                datetime.now(self.tz).isoformat(),
                {"error": str(e)}
            )

        # [N1] Precio inválido → HOLD PREVENTIVO
        if price <= 0:
            logger.warning(
                f"⚠️ {ticker}: price_now={price} inválido → HOLD preventivo"
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
        pnl      = round(ret * 100, 2)
        age_days = days_between(entry_time)

        # 1️⃣ STOP LOSS DURO — prioridad absoluta
        if price <= entry * (1 - STOP_LOSS_NEUTRAL_PCT):
            return NeutralDecision(
                "CLOSE", ticker, "stop_loss_neutral",
                datetime.now(self.tz).isoformat(),
                {"ret_pct": pnl, "trigger_pct": -STOP_LOSS_NEUTRAL_PCT}
            )

        # 2️⃣ TRAILING STOP — prioridad absoluta
        trail_stop = peak * (1 - TRAILING_STOP_NEUTRAL_PCT)
        if price <= trail_stop:
            return NeutralDecision(
                "CLOSE", ticker, "trailing_stop_neutral",
                datetime.now(self.tz).isoformat(),
                {
                    "ret_pct":           pnl,
                    "peak_pct":          round(pct_change(peak, entry) * 100, 2),
                    "trail_trigger_pct": -TRAILING_STOP_NEUTRAL_PCT,
                }
            )

        # 3️⃣ TAKE PROFIT — prioridad absoluta
        if ret >= TAKE_PROFIT_NEUTRAL_PCT:
            return NeutralDecision(
                "CLOSE", ticker, "take_profit_neutral",
                datetime.now(self.tz).isoformat(),
                {"ret_pct": pnl, "trigger_pct": TAKE_PROFIT_NEUTRAL_PCT}
            )

        # ── [C2] Curva desde disco — después de stops duros ────────────────
        curva = _load_price_curve(ticker)

        if curva:
            slope         = curva.get("slope_futura")
            en_zona_valle = curva.get("en_zona_valle", False)
            ret_a_peak    = curva.get("ret_desde_hoy_a_peak_pct")
            ret_a_final   = curva.get("ret_desde_hoy_a_final_pct")
            dias_peak     = curva.get("dias_hasta_peak")

            if pnl > 0 and slope == "baja":
                logger.info(
                    f"📉 {ticker} CLOSE | PnL={pnl:.1f}% curva cae → tomar ganancia"
                )
                return NeutralDecision(
                    "CLOSE", ticker, "neutral_close_curve_falling",
                    datetime.now(self.tz).isoformat(),
                    {
                        "ret_pct":         pnl,
                        "slope_futura":    slope,
                        "ret_a_final_pct": ret_a_final,
                        "dias_hasta_peak": dias_peak,
                        "days_held":       age_days,
                        "en_zona_valle":   en_zona_valle,
                    }
                )

            if pnl > 0 and slope == "sube":
                logger.info(
                    f"📈 {ticker} HOLD | PnL={pnl:.1f}% curva sube → mantener"
                )
                return NeutralDecision(
                    "HOLD", ticker, "hold_curve_recovering",
                    datetime.now(self.tz).isoformat(),
                    {
                        "ret_pct":         pnl,
                        "slope_futura":    slope,
                        "ret_a_peak_pct":  ret_a_peak,
                        "dias_hasta_peak": dias_peak,
                        "days_held":       age_days,
                        "confidence":      confidence,
                        "en_zona_valle":   en_zona_valle,
                    }
                )

        # 4️⃣ TIEMPO MÁXIMO — después de curva
        if age_days >= MAX_HOLD_DAYS_NEUTRAL:
            return NeutralDecision(
                "CLOSE", ticker, f"time_exit_{age_days}d",
                datetime.now(self.tz).isoformat(),
                {"days_held": age_days, "max_days": MAX_HOLD_DAYS_NEUTRAL}
            )

        # 5️⃣ CONFIDENCE COLAPSÓ — después de curva
        if confidence < MIN_CONFIDENCE_NEUTRAL * 0.7:
            return NeutralDecision(
                "CLOSE", ticker, "confidence_collapse",
                datetime.now(self.tz).isoformat(),
                {"current_conf": confidence, "min_conf": MIN_CONFIDENCE_NEUTRAL}
            )

        # 6️⃣ HOLD
        hold_meta = {
            "ret_pct":        pnl,
            "days_held":      age_days,
            "confidence":     confidence,
            "distance_sl":    round((price / entry - (1 - STOP_LOSS_NEUTRAL_PCT)) * 100, 2),
            "distance_trail": round((price / peak  - (1 - TRAILING_STOP_NEUTRAL_PCT)) * 100, 2),
        }
        if curva:
            hold_meta["slope_futura"]    = curva.get("slope_futura")
            hold_meta["en_zona_valle"]   = curva.get("en_zona_valle", False)
            hold_meta["ret_a_peak_pct"]  = curva.get("ret_desde_hoy_a_peak_pct")
            hold_meta["dias_hasta_peak"] = curva.get("dias_hasta_peak")

        return NeutralDecision(
            "HOLD", ticker, "hold_neutral",
            datetime.now(self.tz).isoformat(),
            hold_meta,
        )

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

    def evaluate_portfolio(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not positions:
            return {
                "mode":      "neutral",
                "decisions": [],
                "positions": 0,
                "closes":    0,
                "timestamp": datetime.now(self.tz).isoformat(),
                "message":   "Portfolio vacío",
            }

        decisions        = []
        closes           = 0
        holds_preventive = 0

        for pos in positions:
            decision = self.evaluate_position(pos)
            decisions.append(decision.to_dict())

            if decision.action == "CLOSE":
                closes += 1
                logger.info(f"🟡 CLOSE: {decision.ticker} - {decision.reason}")
            elif decision.action == "HOLD" and "invalid" in decision.reason:
                holds_preventive += 1
                logger.info(f"🛡 HOLD preventivo: {decision.ticker} - {decision.reason}")

        close_pct = round((closes / len(positions)) * 100, 1) if positions else 0
        logger.info(
            f"📊 Neutral v1.4 eval: {len(positions)} pos, {closes} closes ({close_pct}%), "
            f"{holds_preventive} holds preventivos"
        )

        return {
            "mode":             "neutral",
            "decisions":        decisions,
            "positions":        len(positions),
            "closes":           closes,
            "holds_preventive": holds_preventive,
            "close_pct":        close_pct,
            "timestamp":        datetime.now(self.tz).isoformat(),
            "config": {
                "max_hold_days":     MAX_HOLD_DAYS_NEUTRAL,
                "min_confidence":    MIN_CONFIDENCE_NEUTRAL,
                "stop_loss_pct":     STOP_LOSS_NEUTRAL_PCT,
                "take_profit_pct":   TAKE_PROFIT_NEUTRAL_PCT,
                "trailing_stop_pct": TRAILING_STOP_NEUTRAL_PCT,
                "max_risk_per_trade":MAX_RISK_PER_TRADE_NEUTRAL,
            },
        }


# =================================================================
# SELF TEST
# =================================================================

if __name__ == "__main__":
    pm = PMNeutral()

    test_positions = [
        {   # [N1] Precio 0 → debe retornar HOLD
            "ticker":          "AMAT",
            "avg_entry_price": 150.0,
            "current_price":   0,
            "peak_price":      155.0,
            "entry_date":      "2026-01-10",
            "confidence":      0.65,
        },
        {   # Stop loss normal
            "ticker":          "AAPL",
            "avg_entry_price": 150.0,
            "current_price":   144.0,
            "peak_price":      155.0,
            "entry_date":      "2026-01-01",
            "confidence":      0.65,
        },
        {   # Hold normal
            "ticker":          "MSFT",
            "avg_entry_price": 300.0,
            "current_price":   315.0,
            "peak_price":      320.0,
            "entry_date":      "2026-01-10",
            "confidence":      0.70,
        },
    ]

    print("🧪 PMNeutral v1.4 SELF TEST:")
    result = pm.evaluate_portfolio(test_positions)
    print(json.dumps(result, indent=2))

    amat = next(d for d in result["decisions"] if d["ticker"] == "AMAT")
    aapl = next(d for d in result["decisions"] if d["ticker"] == "AAPL")
    assert amat["action"] == "HOLD",  f"❌ AMAT debería ser HOLD, es {amat['action']}"
    assert aapl["action"] == "CLOSE", f"❌ AAPL debería ser CLOSE, es {aapl['action']}"

    print("\n✅ PMNeutral v1.4 – TEST OK")

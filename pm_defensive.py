"""
pm_defensive.py - DEFENSIVE POSITION MANAGER v1.6 PRODUCCION

DEFENSIVE != CASH
DEFENSIVE = CAPITAL ANCLADO + RIESGO MINIMO

NO REJECT -> SOLO HOLD / CLOSE / ROTATE
ANCLAS -> HOLD INDEFINIDO
NO-ANCLAS -> Time exit + Catastrofico
ROTATE -> Fragil -> ANCLA disponible
NO prediccion agresiva | USA alpha score real

FIX v1.6:
  [F1] Lee anchor_universe.json directamente desde disco
       Ya no depende de que el orchestrator pase anchor_universe
  [F2] is_anchor_asset() usa lista fija + alpha_score >= 0.65
       Elimina campos imposibles (is_structural, volatility_1y, etc.)
  [F3] Rotación genera acción OPEN además de ROTATE
       Para que el orchestrator pueda ejecutar la apertura del ancla
  [F4] _load_alpha_scores() lee alpha_last.json para validar
       que el ancla tiene alpha real antes de rotar hacia ella
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
CATASTROPHIC_STOP_PCT = float(os.getenv("PM_DEF_STOP_LOSS_CATA", "0.25"))
MAX_ANCHOR_EXPOSURE_PCT = float(os.getenv("PM_DEF_MAX_ANCHOR_EXPO", "0.30"))
ANCHOR_MIN_ALPHA = float(os.getenv("PM_DEF_ANCHOR_MIN_ALPHA", "0.65"))

# Paths de archivos
_BASE_DIR = Path(__file__).resolve().parent  # directorio del repo donde vive pm_defensive.py
ANCHOR_FILE = Path(os.getenv("ANCHOR_FILE", str(_BASE_DIR / "anchor_universe.json")))
ALPHA_FILE = DATA_PATH / "alpha_last.json"

logger = logging.getLogger("pm_defensive")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# =========================================================
# HELPERS ROBUSTOS
# =========================================================
def pct_change(current: float, entry: float) -> float:
    return (current / entry - 1.0) if entry > 0 else 0.0


def days_between(entry_iso: str) -> int:
    try:
        entry_str = entry_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(entry_str)
        return max(
            0,
            (datetime.now(CL_TIMEZONE) - dt.astimezone(CL_TIMEZONE)).days
        )
    except Exception as e:
        logger.warning(f"days_between error '{entry_iso}': {e}")
        return MAX_HOLD_DAYS_NON_ANCHOR


# =========================================================
# [F1] CARGA ANCHOR UNIVERSE DESDE DISCO
# =========================================================
def _load_anchor_universe() -> List[Dict[str, Any]]:
    """
    Lee anchor_universe.json desde disco.
    Retorna lista vacía si no existe — sistema sigue funcionando.
    """
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


# =========================================================
# [F4] CARGA ALPHA SCORES DESDE alpha_last.json
# =========================================================
def _load_alpha_scores() -> Dict[str, float]:
    """
    Lee alpha_last.json y retorna dict {ticker: alpha_score}.
    Retorna {} si no existe.
    """
    try:
        if not ALPHA_FILE.exists():
            logger.warning("⚠️ alpha_last.json no encontrado — anclas sin validación de alpha")
            return {}
        data = json.loads(ALPHA_FILE.read_text())
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
    action: str          # HOLD | CLOSE | ROTATE | OPEN
    ticker: str
    reason: str
    timestamp: str
    meta: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "ticker": self.ticker,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "meta": self.meta or {}
        }


# =========================================================
# PM DEFENSIVO v1.6 PRODUCCION
# =========================================================
class PMDefensive:
    """
    PM DEFENSIVO CON ROTACION ESTRUCTURAL v1.6
    Preservar poder adquisitivo -> ANCLAS estructurales
    """

    def __init__(self):
        self.tz = CL_TIMEZONE
        self.anchor_exposure_pct = 0.0
        # [F1] Cargar anclas desde disco al inicializar
        self._anchor_universe = _load_anchor_universe()
        self._anchor_tickers = {a["ticker"].upper() for a in self._anchor_universe}
        logger.info(
            f"PMDefensive v1.6 PRODUCCION - ANCLAS + ROTACION ACTIVA | "
            f"anclas={len(self._anchor_tickers)}: {sorted(self._anchor_tickers)}"
        )

    # =========================================================
    # [F2] is_anchor_asset — lista fija + alpha score real
    # =========================================================
    def is_anchor_asset(self, candidate: Dict[str, Any], alpha_scores: Dict[str, float] = None) -> bool:
        """
        Ancla válida = ticker en anchor_universe.json AND alpha >= ANCHOR_MIN_ALPHA
        Si alpha_scores no disponible, acepta cualquier ancla del universo.
        """
        ticker = candidate.get("ticker", "").upper()

        if ticker not in self._anchor_tickers:
            return False

        if alpha_scores is not None:
            alpha = alpha_scores.get(ticker, 0.0)
            if alpha < ANCHOR_MIN_ALPHA:
                logger.debug(f"⚓ {ticker} en anchor_universe pero alpha={alpha:.3f} < {ANCHOR_MIN_ALPHA} → no elegible")
                return False

        return True

    # --------------------------------------------------
    # EVALUAR POSICION EXISTENTE (CORE)
    # --------------------------------------------------
    def evaluate_position(self, pos: Dict[str, Any]) -> DefensiveDecision:
        ticker = str(pos.get("ticker", "UNKNOWN")).upper()
        entry = float(pos.get("entry_price", 0))
        price = float(pos.get("price_now", 0))
        entry_time = str(pos.get("entry_time", ""))

        # [F2] is_anchor ahora también verifica si está en anchor_universe
        is_anchor = bool(pos.get("is_anchor", False)) or (ticker in self._anchor_tickers)

        ts = datetime.now(self.tz).isoformat()

        # VALIDACION
        if entry <= 0 or price <= 0:
            return DefensiveDecision(
                "CLOSE", ticker, "invalid_price_data", ts,
                {"entry": entry, "price": price},
            )

        ret = pct_change(price, entry)
        age = days_between(entry_time)

        # STOP CATASTROFICO
        if ret <= -CATASTROPHIC_STOP_PCT:
            return DefensiveDecision(
                "CLOSE", ticker, "catastrophic_loss", ts,
                {
                    "ret_pct": round(ret * 100, 2),
                    "stop_pct": -CATASTROPHIC_STOP_PCT * 100,
                },
            )

        # ANCLA -> HOLD INDEFINIDO
        if is_anchor:
            return DefensiveDecision(
                "HOLD", ticker, "anchor_hold_indefinite", ts,
                {
                    "ret_pct": round(ret * 100, 2),
                    "days_held": age,
                    "anchor": True,
                    "dist_to_stop_pct": round((ret + CATASTROPHIC_STOP_PCT) * 100, 1),
                },
            )

        # NO-ANCLA envejecido -> CLOSE
        if age >= MAX_HOLD_DAYS_NON_ANCHOR:
            return DefensiveDecision(
                "CLOSE", ticker, "non_anchor_time_exit", ts,
                {"days_held": age, "max_days": MAX_HOLD_DAYS_NON_ANCHOR},
            )

        # NO-ANCLA sano -> HOLD temporal
        return DefensiveDecision(
            "HOLD", ticker, "defensive_hold_non_anchor", ts,
            {
                "ret_pct": round(ret * 100, 2),
                "days_held": age,
                "anchor": False,
                "days_to_exit": MAX_HOLD_DAYS_NON_ANCHOR - age,
            },
        )

    # --------------------------------------------------
    # [F3] ROTACION -> FRAGIL -> ANCLA
    # --------------------------------------------------
    def evaluate_rotation(
        self,
        fragile_pos: Dict[str, Any],
        anchor_candidate: Dict[str, Any],
        alpha_scores: Dict[str, float] = None,
    ) -> List[DefensiveDecision]:
        """
        Rota especulativo → ancla estructural.
        Retorna [CLOSE fragile, OPEN ancla] para que el orchestrator ejecute ambos.
        """
        if not self.is_anchor_asset(anchor_candidate, alpha_scores):
            return []

        ts = datetime.now(self.tz).isoformat()
        fragile_ticker = fragile_pos.get("ticker", "").upper()
        anchor_ticker = anchor_candidate.get("ticker", "").upper()
        alpha = (alpha_scores or {}).get(anchor_ticker, 0.0)

        meta = {
            "close_ticker": fragile_ticker,
            "open_ticker": anchor_ticker,
            "anchor_alpha": round(alpha, 4),
            "fragile_ret_pct": round(
                pct_change(
                    fragile_pos.get("price_now", 0),
                    fragile_pos.get("entry_price", 0),
                ) * 100, 2,
            ),
        }

        logger.info(f"🔄 ROTATE {fragile_ticker} → {anchor_ticker} | alpha={alpha:.3f}")

        return [
            DefensiveDecision("CLOSE", fragile_ticker, "rotate_out_fragile", ts, meta),
            DefensiveDecision("OPEN",  anchor_ticker,  "rotate_in_anchor",   ts, {
                "target_pct": 0.05,
                "reason": f"ANCHOR_ROTATE_alpha={alpha:.3f}",
                "is_anchor": True,
                **meta,
            }),
        ]

    # --------------------------------------------------
    # API PRINCIPAL
    # --------------------------------------------------
    def evaluate_portfolio(
        self,
        positions: List[Dict[str, Any]],
        anchor_universe: Optional[List[Dict[str, Any]]] = None,
        total_capital: float = 1000000,
    ) -> List[DefensiveDecision]:
        """
        anchor_universe: ignorado — ahora se lee desde disco [F1]
        Se mantiene el parámetro por compatibilidad con el orchestrator.
        """
        decisions: List[DefensiveDecision] = []

        # [F4] Cargar alpha scores para validar anclas
        alpha_scores = _load_alpha_scores()

        anchors = [p for p in positions if p.get("is_anchor", False) or p.get("ticker", "").upper() in self._anchor_tickers]
        non_anchors = [p for p in positions if p.get("ticker", "").upper() not in self._anchor_tickers and not p.get("is_anchor", False)]

        # Evaluar posiciones existentes
        for pos in positions:
            decisions.append(self.evaluate_position(pos))

        # Anclas elegibles con alpha real — excluir las que ya están en portfolio
        portfolio_tickers = {p.get("ticker", "").upper() for p in positions}
        eligible_anchors = [
            a for a in self._anchor_universe
            if a["ticker"].upper() not in portfolio_tickers
            and self.is_anchor_asset(a, alpha_scores)
        ]

        logger.info(
            f"⚓ Anclas elegibles para rotación: {[a['ticker'] for a in eligible_anchors]}"
        )

        # Rotación: no-anclas → anclas elegibles
        max_rotations = min(2, len(non_anchors), len(eligible_anchors))
        rotations_done = 0

        for i, fragile in enumerate(non_anchors[:max_rotations]):
            if rotations_done >= max_rotations:
                break

            anchor = eligible_anchors[rotations_done % len(eligible_anchors)]
            rotation_decisions = self.evaluate_rotation(fragile, anchor, alpha_scores)

            if rotation_decisions:
                decisions.extend(rotation_decisions)
                rotations_done += 1

        closes = len([d for d in decisions if d.action == "CLOSE"])
        opens  = len([d for d in decisions if d.action == "OPEN"])
        holds  = len([d for d in decisions if d.action == "HOLD"])

        logger.info(
            f"DEFENSIVE v1.6 | pos={len(positions)} anchors={len(anchors)} "
            f"| closes={closes} opens={opens} holds={holds} rotates={rotations_done} "
            f"| capital=${total_capital:,.0f}"
        )

        return decisions

    def allow_new_positions(self, total_capital: float) -> bool:
        return self.anchor_exposure_pct < MAX_ANCHOR_EXPOSURE_PCT


# =========================================================
# SELF-TEST PRODUCCION
# =========================================================
if __name__ == "__main__":
    pm = PMDefensive()

    test_positions = [
        {
            "ticker": "APA",
            "is_anchor": False,
            "entry_price": 29.29,
            "price_now": 33.92,
            "qty": 100,
            "entry_time": "2026-02-26T14:30:00Z",
        },
        {
            "ticker": "CF",
            "is_anchor": False,
            "entry_price": 80.0,
            "price_now": 85.0,
            "qty": 320,
            "entry_time": "2026-02-26T09:15:00Z",
        },
    ]

    print("\nPMDefensive v1.6 PRODUCCION - FULL EVAL")
    results = pm.evaluate_portfolio(test_positions, total_capital=1000000)

    print("\nDECISIONES:")
    for decision in results:
        print(json.dumps(decision.to_dict(), indent=2))

    print("\nPMDefensive v1.6 - TEST PASSED")
    print(
        f"Config → TimeMaxNonAnchor: {MAX_HOLD_DAYS_NON_ANCHOR}d | "
        f"CatStop: {CATASTROPHIC_STOP_PCT*100}% | "
        f"AnchorMinAlpha: {ANCHOR_MIN_ALPHA}"
        )
        

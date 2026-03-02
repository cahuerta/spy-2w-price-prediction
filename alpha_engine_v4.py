# =========================================================
# alpha_engine_enterprise_strict_v4.3.py
# =========================================================
# 🔥 V4.3 = V4.2 + V6.3 THETA BONUS + PRODUCTION READY
# ✔ Time decay + Liquidity gate + Disagreement haircut
# ✔ V6.3 theta_dynamic boost (+10% si supera umbral propio)
# ✔ Institutional-grade alpha filtering
# =========================================================

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from signals import compute_signal
from data_provider import get_price_history

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [ALPHA] %(message)s")
logger = logging.getLogger("alpha_engine")

# PATHS
DATA_PATH = Path(os.getenv("DATA_PATH", "/data"))
PRED_DIR = DATA_PATH / "predictions"
MARKET_FILE = DATA_PATH / "market_context.json"
ALPHA_FILE = DATA_PATH / "alpha_last.json"

# CONSTANTES INSTITUCIONALES
MAX_PRED_AGE_HOURS = 24
MIN_STRUCTURAL_LIQUIDITY = 0.20
DISAGREEMENT_HAIRCUT = 0.75
V6_3_THETA_BONUS = 1.10  # 🔥 NUEVO: Bonus si supera theta_dynamic

# =========================================================
# HELPERS ATÓMICOS
# =========================================================
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists(): return None
    try: return json.loads(path.read_text())
    except: return None

def save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

# =========================================================
# ESTRUCTURAL (PRIMERO POR PESO)
# =========================================================
def compute_structural_score(ticker: str) -> Dict[str, float]:
    raw = get_price_history(ticker, period="1y", interval="1d")
    if raw is None or len(raw) < 60:
        raise ValueError("Insuficiente historial estructural")

    closes = raw["Close"].values
    volumes = raw["Volume"].values
    returns = np.diff(closes) / closes[:-1]
    
    volatility = np.std(returns)
    trend = (closes[-1] / closes[0]) - 1
    liquidity = np.mean(closes * volumes)

    # Scores refinados
    s_trend = clip01(trend / 0.15)
    s_vol = clip01(1 - abs(volatility - 0.02) / 0.04)
    s_liq = clip01(liquidity / 75_000_000)

    final_struct = clip01(0.4 * s_trend + 0.3 * s_vol + 0.3 * s_liq)
    
    return {
        "score": final_struct,
        "is_uptrend": trend > 0,
        "liquidity": s_liq,
        "volatility": volatility,
        "trend_pct": trend
    }

# =========================================================
# 🔥 V4.3 CORE ENGINE (CON V6.3 THETA BONUS)
# =========================================================
def compute_alpha_for_ticker(ticker: str, market_ctx: Dict, universe_returns: List[float]):
    
    # 1. V6.3 PREDICTION + TIME DECAY
    pred_path = sorted((PRED_DIR / ticker).glob("*.json"))[-1] if (PRED_DIR / ticker).exists() else None
    if not pred_path: 
        raise FileNotFoundError(f"No prediction file for {ticker}")
    
    pred_data = load_json(pred_path)
    file_time = datetime.fromtimestamp(pred_path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - file_time).total_seconds() / 3600
    
    time_decay = clip01(1.0 - (age_hours / MAX_PRED_AGE_HOURS))
    pred_ret = pred_data["prediction"]["ret_ens_pct"]
    theta_dynamic = pred_data["prediction"]["theta_dynamic_pct"]

    # 🔥 V6.3 THETA BONUS (NUEVO!)
    v6_3_bonus = V6_3_THETA_BONUS if abs(pred_ret) >= theta_dynamic else 1.0

    # 2. SIGNAL & HISTÓRICO
    signal = compute_signal(ticker)
    confidence = signal.get("confidence", 0.0)
    metrics = signal.get("rolling_metrics", {"hit_rate": 0.45, "mae_return_pct": 10.0})
    
    s_hit = clip01((metrics["hit_rate"] - 0.45) / 0.30)
    s_mae = clip01(1.0 / (1.0 + metrics["mae_return_pct"] / 5.0))

    # 3. LIQUIDITY GATE (CRÍTICO)
    struct = compute_structural_score(ticker)
    if struct["liquidity"] < MIN_STRUCTURAL_LIQUIDITY:
        logger.warning(f"🚫 {ticker} BLOCKED: liquidity {struct['liquidity']:.3f} < {MIN_STRUCTURAL_LIQUIDITY}")
        return {
            "ticker": ticker, 
            "alpha_score": 0.0, 
            "reason": "liquidity_gate_triggered",
            "liquidity": struct["liquidity"]
        }

    # 4. FUNDAMENTAL & MARKET
    fundamental = signal.get("fundamental", {})
    mispricing = fundamental.get("mispricing_pct", 0.0) if fundamental.get("usable") else 0.0
    s_fund = clip01(abs(mispricing) / 25.0)
    
    m_raw = market_ctx.get("source", {}).get("raw_quant", {})
    vol_m = clip01(1 - abs(m_raw.get("volatility", 0.03) - 0.02) / 0.05)
    s_market = clip01(0.6 * vol_m + 0.4 * (1 + m_raw.get("drawdown_rolling", 0.0)))

    # 5. Z-SCORE RELATIVO (vs universo)
    avg_u = np.mean(universe_returns) if universe_returns else 0.0
    std_u = np.std(universe_returns) if universe_returns else 1.0
    s_ret_rel = clip01(0.5 + (pred_ret - avg_u) / (2 * std_u))

    # =====================================================
    # 🏆 ENSAMBLE V4.3 + V6.3 BONUS
    # =====================================================
    base_alpha = (
        0.25 * struct["score"] +        # Estructural (PRIMERO)
        0.20 * confidence +             # Confianza signals
        0.15 * s_ret_rel +              # V6.3 relativo
        0.15 * s_hit +                  # Hit-rate histórico
        0.10 * s_mae +                  # Precisión histórica
        0.10 * s_fund +                 # Fundamental
        0.05 * s_market                 # Contexto mercado
    )

    # ⚔️ DISAGREEMENT HAIRCUT
    is_pred_positive = pred_ret > 0
    disagreement = is_pred_positive != struct["is_uptrend"]
    if disagreement:
        base_alpha *= DISAGREEMENT_HAIRCUT
        logger.info(f"⚠️ {ticker} haircut: pred vs trend mismatch")

    # 🎯 FINAL: V6.3 Bonus × Time Decay
    final_alpha = clip01(base_alpha * v6_3_bonus * time_decay)

    return {
        "ticker": ticker,
        "alpha_score": round(final_alpha, 4),
        "components": {
            "structural": round(struct["score"], 3),
            "relative_return": round(s_ret_rel, 3),
            "confidence": round(confidence, 3),
            "v6_3_bonus": round(v6_3_bonus, 3),
            "time_decay": round(time_decay, 3),
            "hit_rate": round(s_hit, 3),
            "fundamental": round(s_fund, 3)
        },
        "flags": {
            "disagreement_penalty": disagreement,
            "liquidity_gate": False,
            "v6_3_theta_cleared": abs(pred_ret) >= theta_dynamic,
            "age_hours": round(age_hours, 1),
            "theta_dynamic_pct": round(theta_dynamic, 3)
        },
        "debug": {
            "pred_ret_pct": round(pred_ret, 3),
            "struct_trend": round(struct["trend_pct"], 3)
        }
    }

# =========================================================
# 🚀 BATCH PRODUCTION
# =========================================================
def compute_and_persist_alpha(tickers: List[str]):
    """Entrada: tickers.json → Salida: /data/alpha_last.json"""
    
    logger.info(f"🔬 AlphaEngine V4.3 iniciado | {len(tickers)} tickers")
    
    # Carga mercado
    market_ctx = load_json(MARKET_FILE) or {}
    
    # Pre-scan universo para Z-score
    universe_preds = []
    for t in tickers:
        try:
            path = sorted((PRED_DIR / t).glob("*.json"))[-1]
            d = load_json(path)
            universe_preds.append(d["prediction"]["ret_ens_pct"])
        except:
            continue

    results = {}
    valid_count = 0
    
    for t in tickers:
        try:
            result = compute_alpha_for_ticker(t, market_ctx, universe_preds)
            results[t] = result
            if result["alpha_score"] > 0:
                valid_count += 1
        except Exception as e:
            logger.error(f"❌ {t}: {e}")
            results[t] = {"error": str(e), "alpha_score": 0.0}

    # Persist atomic
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "4.3",
        "universe_size": len(tickers),
        "valid_alphas": valid_count,
        "alpha_threshold": 0.70,
        "results": results
    }

    save_json(ALPHA_FILE, payload)
    logger.info(f"✅ Alpha V4.3 COMPLETADO | {valid_count}/{len(tickers)} válidos")
    
    return payload

# =========================================================
# 🧪 TEST UNITARIO
# =========================================================
if __name__ == "__main__":
    from model import run_model  # Para test
    test_tickers = ["SPY", "TSLA", "AAPL"]
    compute_and_persist_alpha(test_tickers)

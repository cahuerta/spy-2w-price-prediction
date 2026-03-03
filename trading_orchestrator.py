# =========================================================
# trading_orchestrator.py — V2.5 HEDGE FUND EDITION
# =========================================================
# ✔ Integración con CapitalGovernor v2.4 (Hedge Fund)
# ✔ Sizing Atómico y Dinámico
# ✔ Priorización de CIERRES para liberar colateral
# ✔ Filtro Alpha Score para APERTURAS
# =========================================================

import logging
import asyncio
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from portfolio_store import (
    load_positions, register_open, register_close, register_rotate, portfolio_summary
)
from broker import get_trading_engine
from capital_governor import CapitalGovernor, CapitalState

# Importación de PMs
from pm_growth import PMGrowth
from pm_neutral import PMNeutral
from pm_defensive import PMDefensive

# Alpha Engine
from alpha_engine_v4 import compute_and_persist_alpha

logger = logging.getLogger("trading_orchestrator")

class TradingOrchestrator:
    def __init__(self):
        self.broker = get_trading_engine()
        self._last_run_file = Path("/tmp/trading_last_run.flag")
        
        # Configuración de límites y capital
        self.fixed_capital = float(os.getenv("FIXED_CAPITAL", "1000000"))
        self._daily_limit = int(os.getenv("MAX_ORDERS_DAY", "10"))
        self._alpha_threshold = float(os.getenv("ALPHA_THRESHOLD", "0.70"))
        
        # Inicializar el Gobernador Grado Hedge Fund
        self.governor = CapitalGovernor(fixed_capital=self.fixed_capital)
        logger.info(f"🧭 Orchestrator v2.5 Online | Capital: ${self.fixed_capital:,.0f}")

    async def run(
        self,
        market_ctx: Dict[str, Any],
        signals: Dict[str, Dict[str, Any]] | None = None,
        anchor_universe: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        
        # 1. Pre-flight checks
        if not self._check_daily_run():
            return {"status": "skipped_daily", "timestamp": datetime.utcnow().isoformat()}

        mode = market_ctx.get("market_mode", "neutral")
        signals = signals or {}
        positions = load_positions()
        
        logger.info(f"🚦 MODO: {mode.upper()} | Posiciones: {len(positions)}")

        # 2. Obtener Alpha Scores (Para filtrar aperturas)
        alpha_filtered_tickers = await self._get_alpha_filtered_tickers()

        # 3. Recopilar Intenciones de los PMs (Raw Decisions)
        raw_decisions = self._collect_pm_decisions(mode, positions, signals, anchor_universe)
        
        # 4. SEGREGACIÓN DE ÓRDENES
        # Los cierres son prioridad 1 (liberan capital y bajan riesgo)
        priority_closes = [d for d in raw_decisions if d['action'] == 'CLOSE']
        
        # Las aperturas y rotaciones son candidatos a "Gobernar"
        investment_candidates = [d for d in raw_decisions if d['action'] in ['OPEN', 'ROTATE']]

        # 5. PASAR POR EL GOBERNADOR (MAGIA DE SIZING)
        # El Governor ajusta los 'shares' de cada candidato según el ES y Cash Buffer
        logger.info(f"🏛 Enviando {len(investment_candidates)} candidatos al CapitalGovernor para Sizing...")
        validated_investments = self.governor.adjust_sizing(positions, investment_candidates)

        # 6. CONSOLIDAR Y FILTRAR POR ALPHA
        final_queue = priority_closes
        
        for cmd in validated_investments:
            ticker = cmd['ticker']
            # Solo aplicamos filtro alpha a nuevas aperturas (OPEN)
            if cmd['action'] == 'OPEN' and ticker not in alpha_filtered_tickers:
                logger.info(f"⛔ {ticker} bloqueado: Alpha < {self._alpha_threshold}")
                continue
            final_queue.append(cmd)

        # 7. EJECUCIÓN SERIALIZADA
        results = []
        executed_count = 0
        
        for order in final_queue[:self._daily_limit]:
            res = await self._execute_and_persist(order, market_ctx)
            results.append(res)
            if res["result"]["status"] == "executed":
                executed_count += 1
            await asyncio.sleep(0.5) # Evitar rate limits

        # 8. Finalizar
        self._last_run_file.write_text(datetime.utcnow().isoformat())
        
        return {
            "status": "ok",
            "executed_count": executed_count,
            "capital_state": self.governor.evaluate(load_positions()).to_dict(),
            "results": results
        }

    # --- HELPERS DE LÓGICA ---

    def _collect_pm_decisions(self, mode, positions, signals, anchor_universe) -> List[Dict]:
        """Extrae las decisiones de los PMs según el régimen"""
        if mode == "growth":
            pm = PMGrowth(fixed_capital=self.fixed_capital)
            return [{**pm.evaluate_position(p, signals.get(p['ticker'])), "pm": "GROWTH"} for p in positions]
        
        elif mode == "defensive":
            pm = PMDefensive()
            decisions = pm.evaluate_portfolio(positions, anchor_universe, total_capital=self.fixed_capital)
            return [{**d.to_dict(), "pm": "DEFENSIVE"} for d in decisions]
        
        else: # Neutral
            pm = PMNeutral()
            res = pm.evaluate_portfolio(positions)
            return [{**d, "pm": "NEUTRAL"} for d in res["decisions"]]

    async def _get_alpha_filtered_tickers(self) -> List[str]:
        """Retorna lista de tickers que pasan el umbral de Alpha"""
        try:
            # Cargar universo de tickers.json
            path = Path(os.getenv("DATA_PATH", "/data")) / "tickers.json"
            if not path.exists(): return []
            universe = json.loads(path.read_text())
            
            payload = await asyncio.to_thread(compute_and_persist_alpha, universe)
            scores = payload.get("results", {})
            
            return [t for t, v in scores.items() 
                    if isinstance(v, dict) and v.get("alpha_score", 0) >= self._alpha_threshold]
        except Exception as e:
            logger.error(f"Error en Alpha Engine: {e}")
            return []

    async def _execute_and_persist(self, decision: Dict, market_ctx: Dict) -> Dict:
        """Ejecuta en el broker y registra en PortfolioStore"""
        ticker = decision['ticker']
        action = decision['action']
        
        logger.info(f"▶️ Ejecutando {action} {ticker} | Shares: {decision.get('shares', 'N/A')}")
        
        try:
            # Llamada al broker físico
            broker_res = await asyncio.wait_for(self.broker.execute_decision(decision), timeout=30)
            
            if broker_res.status == "executed":
                if action == "OPEN": register_open(decision, broker_res.model_dump(), market_ctx)
                elif action == "CLOSE": register_close(decision, broker_res.model_dump())
                elif action == "ROTATE": register_rotate(decision, broker_res.model_dump(), broker_res.model_dump(), market_ctx)
                logger.info(f"✅ {ticker} {action} SUCCESS")
            
            return {"decision": decision, "result": broker_res.model_dump()}
        except Exception as e:
            logger.error(f"❌ Error ejecutando {ticker}: {e}")
            return {"decision": decision, "result": {"status": "error", "reason": str(e)}}

    def _check_daily_run(self) -> bool:
        if not self._last_run_file.exists(): return True
        last_date = datetime.fromisoformat(self._last_run_file.read_text().strip()).date()
        return last_date != datetime.utcnow().date()

    # =========================================================
        # PREVIEW EXECUTABILITY (NO EJECUTA)
        # =========================================================
        async def preview_executability(
            self,
            market_ctx: Dict[str, Any],
            signals: Dict[str, Dict[str, Any]] | None = None,
            anchor_universe: List[Dict[str, Any]] | None = None,
        ) -> Dict[str, Any]:

            signals = signals or {}
            positions = load_positions()

            mode = market_ctx.get("market_mode", "neutral")

            # 1️⃣ Alpha válidos
            alpha_filtered = await self._get_alpha_filtered_tickers()

            # 2️⃣ Decisiones crudas desde PM
            raw_decisions = self._collect_pm_decisions(
                mode, positions, signals, anchor_universe
            )

            results = {}

            # 3️⃣ Procesar CIERRES (siempre ejecutables)
            for cmd in raw_decisions:
                if cmd["action"] == "CLOSE":
                    ticker = cmd["ticker"]
                    results[ticker] = {
                        "action": "CLOSE",
                        "executable": True,
                        "reason": None
                    }

            # 4️⃣ Procesar OPEN / ROTATE (pasar por governor)
            investment_candidates = [
                d for d in raw_decisions
                if d["action"] in ["OPEN", "ROTATE"]
            ]

            validated = self.governor.adjust_sizing(
                positions,
                investment_candidates
            )

            for cmd in validated:
                ticker = cmd["ticker"]
                action = cmd["action"]

                # 🚫 Alpha bloquea solo OPEN
                if action == "OPEN" and ticker not in alpha_filtered:
                    results[ticker] = {
                        "action": action,
                        "executable": False,
                        "reason": "alpha_below_threshold"
                    }
                    continue

                # 🚫 Governor sizing bloquea si shares <= 0
                if cmd.get("shares", 0) <= 0:
                    results[ticker] = {
                        "action": action,
                        "executable": False,
                        "reason": "sizing_block"
                    }
                    continue

                # ✅ Ejecutable
                results[ticker] = {
                    "action": action,
                    "executable": True,
                    "reason": None
                }

            return {
                "results": results,
                "capital_state": self.governor.evaluate(positions).to_dict()
            }

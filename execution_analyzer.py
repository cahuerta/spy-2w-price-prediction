# =========================================================
# execution_analyzer.py — V1.0
# =========================================================
# Propósito: Calcular la efectividad real filtrando solo
# las órdenes que el broker efectivamente ejecutó (sys_).
# =========================================================

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
from fastapi import APIRouter
from broker import get_engine
from performance_router import list_evaluation_files, load_json

router = APIRouter(prefix="/dashboard/real-performance", tags=["analysis"])

@router.get("/effective-hit-rate")
async def get_effective_hit_rate(days: int = 14):
    engine = get_engine()
    files = list_evaluation_files()
    
    # 1. Obtener órdenes reales de Alpaca con prefijo 'sys_'
    # -------------------------------------------------------
    try:
        # Buscamos órdenes cerradas en el periodo
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            until=datetime.now(timezone.utc),
            after=datetime.now(timezone.utc) - timedelta(days=days)
        )
        closed_orders = engine.client.get_orders(filter=request)
        
        # Mapeamos órdenes del sistema: { "TICKER": [precios_ejecucion] }
        executed_sys_orders = {}
        for o in closed_orders:
            if o.client_order_id and o.client_order_id.startswith("sys_"):
                ticker = o.symbol.upper()
                if ticker not in executed_sys_orders:
                    executed_sys_orders[ticker] = []
                executed_sys_orders[ticker].append({
                    "id": o.client_order_id,
                    "filled_at": o.filled_at,
                    "side": o.side,
                    "price": float(o.filled_avg_price) if o.filled_avg_price else None
                })
    except Exception as e:
        return {"error": f"Error consultando Alpaca: {str(e)}"}

    # 2. Cruzar con Evaluaciones del Modelo
    # -------------------------------------------------------
    cut_date = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    
    real_hits = 0
    real_total = 0
    missed_signals = 0 # Señales que el modelo dio pero el broker/PM filtró

    for f in files:
        data = load_json(f)
        if not data or data.get("real_return_pct") is None:
            continue
            
        eval_date = str(data.get("evaluation_date") or f.stem)[:10]
        if eval_date < cut_date:
            continue

        ticker = data.get("ticker", "").upper()
        pred_ret = float(data.get("predicted_return_pct", 0))
        real_ret = float(data.get("real_return_pct", 0))

        # ¿Esta predicción se convirtió en una orden 'sys_'?
        was_executed = ticker in executed_sys_orders
        
        if was_executed:
            real_total += 1
            # Hit rate real basado en dirección
            if np.sign(real_ret) == np.sign(pred_ret):
                real_hits += 1
        else:
            missed_signals += 1

    # 3. Cálculo de métricas finales
    # -------------------------------------------------------
    effective_hr = round((real_hits / real_total * 100), 2) if real_total > 0 else 0
    
    return {
        "period_days": days,
        "effective_hit_rate": effective_hr,
        "executed_orders_count": real_total,
        "filtered_by_broker_count": missed_signals,
        "efficiency_ratio": round((real_total / (real_total + missed_signals) * 100), 2) if (real_total + missed_signals) > 0 else 0,
        "status": "Mejorando" if effective_hr > 50 else "Revisar Filtros"
    }

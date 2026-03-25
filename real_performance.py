import logging
from collections import defaultdict
from broker import get_engine

logger = logging.getLogger("real_performance")

async def get_alpaca_real_metrics():
    """
    Calcula el Win Rate REAL basado exclusivamente en el historial de Alpaca.
    Ignora predicciones teóricas y se enfoca en dinero ejecutado.
    """
    try:
        engine = get_engine()
        # Traemos las últimas 500 órdenes ejecutadas (filled)
        orders = await engine.get_orders(status='filled', limit=500)
        
        if not orders:
            return {
                "win_rate_real": 0,
                "total_trades_closed": 0,
                "status": "no_trades_found"
            }

        # Agrupar por ticker para reconstruir el ciclo Compra -> Venta
        trades_by_ticker = defaultdict(list)
        for o in orders:
            trades_by_ticker[o.symbol].append(o)

        real_hits = 0
        real_total_closed = 0
        real_pnl_pct_sum = 0

        for ticker, ticker_orders in trades_by_ticker.items():
            # Ordenar por fecha de ejecución para FIFO
            ticker_orders.sort(key=lambda x: x.filled_at)
            
            buys = [o for o in ticker_orders if o.side == 'buy']
            sells = [o for o in ticker_orders if o.side == 'sell']

            # Emparejamos cada venta con la compra previa más reciente
            for s in sells:
                # Buscamos una compra que haya ocurrido antes de esta venta
                match = next((b for b in reversed(buys) if b.filled_at < s.filled_at), None)
                
                if match:
                    real_total_closed += 1
                    entry_price = float(match.filled_avg_price)
                    exit_price = float(s.filled_avg_price)
                    
                    if exit_price > entry_price:
                        real_hits += 1
                    
                    # Cálculo de PnL por trade para tener una métrica extra
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    real_pnl_pct_sum += pnl_pct
                    
                    # Removemos la compra usada para no duplicar (FIFO simple)
                    buys.remove(match)

        win_rate = round((real_hits / real_total_closed) * 100, 2) if real_total_closed > 0 else 0
        avg_pnl = round(real_pnl_pct_sum / real_total_closed, 2) if real_total_closed > 0 else 0

        return {
            "win_rate_real": win_rate,
            "total_trades_closed": real_total_closed,
            "avg_pnl_per_trade_pct": avg_pnl,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error calculando métricas reales de Alpaca: {e}")
        return {"win_rate_real": 0, "error": str(e), "status": "error"}
      

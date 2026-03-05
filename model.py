# ======================================================
# model.py — WRAPPER PARA MASTER ORCHESTRATOR
# ======================================================
# Mantiene contrato original del sistema
# pero ejecuta el motor H1-H10
# ======================================================

import sys
from predictors_engine.master_orchestrator import MasterOrchestrator


def run_model(
    ticker="SPY",
    horizon=10,
    pca_target=50,
    theta=0.75,
    k_neighbors=20,
    alpha=0.5,
    period="max"
):

    maestro = MasterOrchestrator(ticker)

    maestro.run_all_predictors()

    maestro.collect_results()

    analysis = maestro.analyze_curve()

    result = maestro.build_model_json(analysis)

    return result


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    print(run_model(ticker))

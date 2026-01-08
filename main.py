# main.py
# Web Service FastAPI para Render
# - Abre puerto (Render OK)
# - Llama a model.py (misma matemática)
# - Endpoint REST /predict
# - Healthcheck /
# - Listo para cron, app Vercel, email y disk

import os
import json
from datetime import datetime

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from model import run_model, format_report
from evaluator import evaluate_all


# =========================
# App
# =========================
app = FastAPI(
    title="Prediction Service",
    description="Modelo predictivo PCA + Ridge + kNN caótico",
    version="1.0.0"
)


# =========================
# Healthcheck (OBLIGATORIO)
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "price-prediction-service",
        "time": datetime.utcnow().isoformat()
    }


# =========================
# Predicción principal
# =========================
@app.get("/predict")
def predict(
    ticker: str = Query("SPY"),
    horizon: int = Query(10),
    pca_target: int = Query(50),
    theta: float = Query(0.75),
    k_neighbors: int = Query(20),
    alpha: float = Query(0.5),
):
    """
    Ejecuta el modelo completo y devuelve:
    - métricas históricas
    - predicción actual
    - recomendación
    """
    try:
        result = run_model(
            ticker=ticker,
            horizon=horizon,
            pca_target=pca_target,
            theta=theta,
            k_neighbors=k_neighbors,
            alpha=alpha,
        )

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "timestamp": datetime.utcnow().isoformat(),
                "ticker": result["meta"]["ticker"],
                "recommendation": result["prediction"]["recommendation"],
                "ret_ens_pct": result["prediction"]["ret_ens_pct"],
                "price_pred": result["prediction"]["price_pred"],
                "result": result
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": str(e)
            }
        )


# =========================
# Endpoint texto (email/logs)
# =========================
@app.get("/predict/text")
def predict_text(
    ticker: str = Query("SPY")
):
    """
    Devuelve el reporte en texto plano
    (ideal para email o logs)
    """
    try:
        result = run_model(ticker=ticker)
        report = format_report(result)
        return {"text": report}

    except Exception as e:
        return {"error": str(e)}


# =========================
# Guardado en disco (cron)
# =========================
@app.get("/predict/save")
def predict_and_save(
    ticker: str = Query("SPY")
):
    """
    Ejecuta predicción y guarda resultado en:
    /data/predictions/{ticker}/YYYY-MM-DD.json
    """
    try:
        result = run_model(ticker=ticker)

        base_path = os.getenv("DATA_PATH", "/data")
        pred_dir = os.path.join(base_path, "predictions", ticker)
        os.makedirs(pred_dir, exist_ok=True)

        filename = f"{datetime.utcnow().date()}.json"
        path = os.path.join(pred_dir, filename)

        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        return {
            "ok": True,
            "ticker": ticker,
            "saved_to": path,
            "recommendation": result["prediction"]["recommendation"]
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


# =========================
# Evaluación automática
# =========================
@app.get("/evaluate")
def evaluate(
    ticker: str = Query(None)
):
    """
    Evalúa automáticamente predicciones pasadas
    y guarda resultados en /data/evaluations/{ticker}
    """
    try:
        result = evaluate_all(ticker=ticker)

        return {
            "ok": True,
            "ticker": ticker,
            "evaluated": result.get("evaluated", []),
            "skipped": result.get("skipped", []),
            "total_evaluated": len(result.get("evaluated", []))
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }

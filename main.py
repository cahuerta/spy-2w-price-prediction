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


# =========================
# App
# =========================
app = FastAPI(
    title="SPY Prediction Service",
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
        "service": "spy-2w-price-prediction",
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
def predict_text():
    """
    Devuelve el reporte en texto plano
    (ideal para email o logs)
    """
    try:
        result = run_model()
        report = format_report(result)
        return {"text": report}

    except Exception as e:
        return {"error": str(e)}


# =========================
# Guardado en disco (opcional, futuro)
# =========================
@app.get("/predict/save")
def predict_and_save():
    """
    Guarda resultado en /data si existe disk montado
    """
    try:
        result = run_model()
        report = format_report(result)

        base_path = os.getenv("DATA_PATH", "/data")
        os.makedirs(base_path, exist_ok=True)

        filename = f"prediction_{datetime.utcnow().date()}.json"
        path = os.path.join(base_path, filename)

        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        return {
            "ok": True,
            "saved_to": path,
            "recommendation": result["prediction"]["recommendation"]
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}

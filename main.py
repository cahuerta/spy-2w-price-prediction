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
import glob
import pandas as pd

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model2 import run_model_fundamental_dividend

from model import run_model, format_report
from evaluator import evaluate_all
from signals import compute_signal, compute_all_signals

# =========================
# App
# =========================
app = FastAPI(
    title="Prediction Service",
    description="Modelo predictivo PCA + Ridge + kNN caótico",
    version="1.0.0"
)

# =========================
# CORS (OBLIGATORIO PARA VERCEL)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # luego lo restringimos
    allow_credentials=True,
    allow_methods=["*"],          # ← ESTO PERMITE OPTIONS
    allow_headers=["*"],
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
# =========================
# Señales operativas
# =========================
@app.get("/signals")
def signals_all(
    window: int = Query(30)
):
    """
    Devuelve señales consolidadas para todos los tickers
    """
    try:
        data = compute_all_signals(window=window)
        return {
            "ok": True,
            "window": window,
            "signals": data
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


@app.get("/signals/{ticker}")
def signal_one(
    ticker: str,
    window: int = Query(30)
):
    """
    Devuelve señal consolidada para un ticker
    """
    try:
        data = compute_signal(ticker, window=window)
        return {
            "ok": True,
            "ticker": ticker,
            "signal": data
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }
# =========================
# Confianza del modelo
# =========================
@app.get("/confidence")
def confidence(
    ticker: str = Query("SPY")
):
    """
    Resume el desempeño histórico del modelo
    y entrega un nivel de confianza actual
    """
    try:
        base_path = os.getenv("DATA_PATH", "/data")
        eval_dir = os.path.join(base_path, "evaluations", ticker)

        if not os.path.exists(eval_dir):
            return {
                "ok": False,
                "ticker": ticker,
                "message": "No evaluations available"
            }

        files = sorted(glob.glob(os.path.join(eval_dir, "*.json")))

        if len(files) == 0:
            return {
                "ok": False,
                "ticker": ticker,
                "message": "No evaluations available"
            }

        records = []
        for f in files[-50:]:  # últimas 50
            with open(f, "r") as fh:
                records.append(json.load(fh))

        df = pd.DataFrame(records)

        hit_sign_pct = round(df["hit_sign"].mean() * 100, 1)
        decision_accuracy_pct = round(df["decision_correct"].mean() * 100, 1)
        avg_error_pct = round(df["error_price_pct"].mean(), 2)

        # Traducción humana
        if hit_sign_pct >= 60 and decision_accuracy_pct >= 55:
            confidence_level = "alta"
        elif hit_sign_pct >= 52:
            confidence_level = "moderada"
        else:
            confidence_level = "baja"

        return {
            "ok": True,
            "ticker": ticker,
            "n": len(df),
            "hit_sign_pct": hit_sign_pct,
            "decision_accuracy_pct": decision_accuracy_pct,
            "avg_error_pct": avg_error_pct,
            "confidence_level": confidence_level
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }
# =========================
# Modelo fundamental (dividendos)
# =========================
@app.get("/predict/fundamental")
def predict_fundamental(
    ticker: str = Query("SPY"),
    r: float = Query(0.09),   # tasa descuento
    g: float = Query(0.03)    # crecimiento dividendos
):
    """
    Modelo fundamental por dividendos.
    Entrega:
    - precio mercado
    - precio fundamental actual
    - precio fundamental T+10
    - mispricing
    """
    try:
        result = run_model_fundamental_dividend(
            ticker=ticker,
            discount_rate=r,
            growth_rate=g
        )

        return {
            "ok": True,
            **result
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }
# =========================
# Guardado masivo (cron)
# =========================
@app.get("/predict/save/all")
def predict_and_save_all():
    try:
        with open("tickers.json", "r") as f:
            tickers = json.load(f)

        # ✅ ÚNICA MODIFICACIÓN:
        # soporta tickers.json como {"tickers":[...]} o como lista directa [...]
        if isinstance(tickers, dict):
            tickers = tickers.get("tickers", [])
        elif not isinstance(tickers, list):
            tickers = []

        base_path = os.getenv("DATA_PATH", "/data")
        results = []

        for ticker in tickers:
            try:
                result = run_model(ticker=ticker)

                pred_dir = os.path.join(base_path, "predictions", ticker)
                os.makedirs(pred_dir, exist_ok=True)

                filename = f"{datetime.utcnow().date()}.json"
                path = os.path.join(pred_dir, filename)

                with open(path, "w") as fh:
                    json.dump(result, fh, indent=2)

                results.append({"ticker": ticker, "ok": True})

            except Exception as e:
                results.append({
                    "ticker": ticker,
                    "ok": False,
                    "error": str(e)
                })

        return {
            "ok": True,
            "n": len(results),
            "results": results
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
    }

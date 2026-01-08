# evaluator.py
# Evaluación automática de predicciones guardadas
# NO entrena modelos
# NO modifica datos históricos
# SOLO mide desempeño real

import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


DATA_PATH = os.getenv("DATA_PATH", "/data")


# =========================
# Utils
# =========================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_date(d):
    return datetime.strptime(d, "%Y-%m-%d").date()


# =========================
# Evaluación de una predicción
# =========================
def evaluate_prediction(prediction_path: str):
    """
    Evalúa una predicción individual:
    /data/predictions/{ticker}/YYYY-MM-DD.json
    """

    pred = load_json(prediction_path)

    meta = pred["meta"]
    p = pred["prediction"]

    ticker = meta["ticker"]
    horizon = meta["horizon_days"]

    base_date = parse_date(p["date_base"])
    target_date = base_date + timedelta(days=horizon)
    today = datetime.utcnow().date()

    # Aún no evaluable
    if today <= target_date:
        return None

    # Descargar precios reales
    df = yf.download(
        ticker,
        start=target_date - timedelta(days=5),
        end=target_date + timedelta(days=5),
        progress=False
    )

    if df is None or len(df) == 0:
        return None

    df.index = pd.to_datetime(df.index)
    df_eval = df[df.index.date >= target_date]

    if len(df_eval) == 0:
        return None

    real_price = float(df_eval.iloc[0]["Close"])

    price_now = p["price_now"]
    price_pred = p["price_pred"]

    real_ret = (real_price / price_now - 1) * 100
    pred_ret = p["ret_ens_pct"]

    # Métricas
    hit_sign = np.sign(real_ret) == np.sign(pred_ret)
    hit_threshold = abs(real_ret) >= meta["theta"]

    recommendation = p["recommendation"]

    if recommendation == "COMPRA":
        decision_correct = real_ret > 0
    elif recommendation == "VENDE":
        decision_correct = real_ret < 0
    else:  # MANTÉN
        decision_correct = abs(real_ret) < meta["theta"]

    error_price_pct = (price_pred / real_price - 1) * 100

    evaluation = {
        "meta": meta,
        "prediction_date": str(base_date),
        "evaluation_date": str(target_date),
        "price_now": price_now,
        "price_pred": price_pred,
        "price_real": real_price,
        "predicted_return_pct": pred_ret,
        "real_return_pct": real_ret,
        "error_price_pct": error_price_pct,
        "hit_sign": bool(hit_sign),
        "hit_threshold": bool(hit_threshold),
        "recommendation": recommendation,
        "decision_correct": bool(decision_correct),
        "evaluated_at": datetime.utcnow().isoformat()
    }

    return evaluation


# =========================
# Evaluar todas las pendientes
# =========================
def evaluate_all(ticker: str | None = None):
    """
    Evalúa predicciones pendientes.
    Si ticker=None → evalúa todos.
    """

    results = {
        "evaluated": [],
        "skipped": [],
    }

    pred_root = os.path.join(DATA_PATH, "predictions")
    eval_root = os.path.join(DATA_PATH, "evaluations")

    if not os.path.exists(pred_root):
        return results

    tickers = [ticker] if ticker else os.listdir(pred_root)

    for tkr in tickers:
        pred_dir = os.path.join(pred_root, tkr)
        eval_dir = os.path.join(eval_root, tkr)

        if not os.path.isdir(pred_dir):
            continue

        os.makedirs(eval_dir, exist_ok=True)

        for fname in sorted(os.listdir(pred_dir)):
            if not fname.endswith(".json"):
                continue

            pred_path = os.path.join(pred_dir, fname)
            eval_path = os.path.join(eval_dir, fname)

            # Ya evaluada
            if os.path.exists(eval_path):
                continue

            evaluation = evaluate_prediction(pred_path)

            if evaluation is None:
                results["skipped"].append(f"{tkr}/{fname}")
                continue

            save_json(eval_path, evaluation)
            results["evaluated"].append(f"{tkr}/{fname}")

    return results

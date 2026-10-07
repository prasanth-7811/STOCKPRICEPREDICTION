"""
api/index.py — FastAPI entrypoint for Vercel deployment.
Exposes REST endpoints that replicate the Streamlit dashboard logic.
"""

import os, sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing  import load_dataset, inspect_data, preprocess
from train_ann      import build_ann, train_ann, save_model, load_model
from train_baseline import train_linear_regression, train_random_forest
from evaluation     import compute_metrics, build_metrics_table

app = FastAPI(title="Stock Price Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "stock_data.csv")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "ann_model.keras")

# ── In-memory state (per cold start) ─────────────────────────────────────────
_state: dict = {}


class TrainRequest(BaseModel):
    features: List[str] = ["Open", "High", "Low", "Volume"]
    epochs: int = 150
    batch_size: int = 32


class PredictRequest(BaseModel):
    features: dict  # e.g. {"Open": 150.0, "High": 155.0, ...}


# ── Root — simple HTML dashboard link ────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Stock Price Prediction API</title>
      <style>
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: #cdd6f4;
               display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .card { background: #16213e; border-radius: 16px; padding: 2.5rem 3rem; max-width: 520px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
        h1 { color: #e94560; margin-top: 0; }
        a  { color: #e94560; }
        .ep { background: #0f3460; border-radius: 8px; padding: 0.4rem 0.8rem;
              margin: 0.3rem 0; display: block; font-family: monospace; font-size: 0.9rem; }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>📈 Stock Price Prediction API</h1>
        <p>FastAPI backend — ANN, Linear Regression, Random Forest</p>
        <p><strong>Endpoints:</strong></p>
        <span class="ep">GET  /api/data-info</span>
        <span class="ep">POST /api/train</span>
        <span class="ep">POST /api/predict</span>
        <span class="ep">GET  /api/metrics</span>
        <br>
        <a href="/docs">📄 Interactive API Docs (Swagger)</a>
      </div>
    </body>
    </html>
    """


# ── GET /api/data-info ────────────────────────────────────────────────────────
@app.get("/api/data-info")
def data_info():
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="stock_data.csv not found")
    df   = load_dataset(DATA_PATH)
    info = inspect_data(df)
    return {
        "rows":        info["shape"][0],
        "columns":     info["shape"][1],
        "duplicates":  info["duplicates"],
        "date_range":  info["date_range"],
        "missing":     info["missing_values"],
        "preview":     df.head(5).to_dict(orient="records"),
    }


# ── POST /api/train ───────────────────────────────────────────────────────────
@app.post("/api/train")
def train(req: TrainRequest):
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="stock_data.csv not found")

    df     = load_dataset(DATA_PATH)
    result = preprocess(df, req.features)
    X_train, X_test = result.X_train, result.X_test
    y_train, y_test = result.y_train, result.y_test
    scaler_y        = result.scaler_y

    # Train all three models
    lr_model  = train_linear_regression(X_train, y_train)
    rf_model  = train_random_forest(X_train, y_train)
    ann_model, history = train_ann(X_train, y_train, X_test, y_test,
                                   epochs=req.epochs, batch_size=req.batch_size)

    y_true    = scaler_y.inverse_transform(y_test)
    lr_pred   = scaler_y.inverse_transform(lr_model.predict(X_test).reshape(-1, 1))
    rf_pred   = scaler_y.inverse_transform(rf_model.predict(X_test).reshape(-1, 1))
    ann_pred  = scaler_y.inverse_transform(ann_model.predict(X_test, verbose=0))

    results = {
        "Linear Regression": compute_metrics(y_true, lr_pred),
        "Random Forest":     compute_metrics(y_true, rf_pred),
        "ANN":               compute_metrics(y_true, ann_pred),
    }
    metrics_df = build_metrics_table(results)

    # Persist state for /predict
    _state["scaler_X"]  = result.scaler_X
    _state["scaler_y"]  = scaler_y
    _state["ann_model"] = ann_model
    _state["features"]  = req.features + [
        f for f in ["Prev_Close", "Prev_Close2", "Price_Change"]
        if f not in req.features
    ]
    _state["metrics"]   = metrics_df.to_dict(orient="records")

    best = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]

    return {
        "status":       "trained",
        "epochs_run":   len(history.history["loss"]),
        "best_model":   best,
        "metrics":      metrics_df.to_dict(orient="records"),
        "train_loss":   history.history["loss"],
        "val_loss":     history.history["val_loss"],
        "actual":       y_true.ravel().tolist(),
        "ann_pred":     ann_pred.ravel().tolist(),
        "lr_pred":      lr_pred.ravel().tolist(),
        "rf_pred":      rf_pred.ravel().tolist(),
    }


# ── POST /api/predict ─────────────────────────────────────────────────────────
@app.post("/api/predict")
def predict(req: PredictRequest):
    if "ann_model" not in _state:
        raise HTTPException(status_code=400, detail="Model not trained yet. Call POST /api/train first.")

    features  = _state["features"]
    try:
        x_input = np.array([[req.features[f] for f in features]])
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")

    x_scaled    = _state["scaler_X"].transform(x_input)
    pred_scaled = _state["ann_model"].predict(x_scaled, verbose=0)
    pred_price  = _state["scaler_y"].inverse_transform(pred_scaled)[0][0]

    return {"predicted_close": round(float(pred_price), 4)}


# ── GET /api/metrics ──────────────────────────────────────────────────────────
@app.get("/api/metrics")
def metrics():
    if "metrics" not in _state:
        raise HTTPException(status_code=400, detail="No metrics yet. Call POST /api/train first.")
    return {"metrics": _state["metrics"]}

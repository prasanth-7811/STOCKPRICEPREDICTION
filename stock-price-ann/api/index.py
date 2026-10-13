"""
api/index.py — FastAPI entrypoint for Vercel deployment.
Uses pure-numpy ANN — no TensorFlow, stays well under 500 MB bundle limit.
"""

import os, sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing  import load_dataset, inspect_data, preprocess
from train_baseline import train_linear_regression, train_random_forest
from evaluation     import compute_metrics, build_metrics_table
from ann_numpy      import NumpyANN

app = FastAPI(title="Stock Price Prediction API", version="1.0.0")

from mangum import Mangum
handler = Mangum(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "stock_data.csv")

_state: dict = {}


class TrainRequest(BaseModel):
    features: List[str] = ["Open", "High", "Low", "Volume"]
    epochs: int = 150
    batch_size: int = 32


class PredictRequest(BaseModel):
    features: dict


# ── Favicon ───────────────────────────────────────────────────────────────────
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Stock Price Prediction API</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #0d1117;
      color: #c9d1d9;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }

    .card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 16px;
      padding: 2.5rem;
      width: 100%;
      max-width: 560px;
      box-shadow: 0 16px 48px rgba(0,0,0,0.5);
    }

    .header {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 0.5rem;
    }

    .header h1 {
      font-size: 1.5rem;
      color: #f0f6fc;
      font-weight: 700;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.75rem;
      background: #1a3a2a;
      color: #3fb950;
      border: 1px solid #238636;
      border-radius: 20px;
      padding: 0.2rem 0.7rem;
      margin-bottom: 1.25rem;
    }

    .status::before {
      content: '';
      width: 7px; height: 7px;
      background: #3fb950;
      border-radius: 50%;
      animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.3; }
    }

    .subtitle {
      font-size: 0.85rem;
      color: #8b949e;
      margin-bottom: 1.75rem;
    }

    .subtitle span {
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 4px;
      padding: 0.1rem 0.45rem;
      font-size: 0.78rem;
      color: #79c0ff;
      margin-right: 0.3rem;
    }

    .section-label {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #8b949e;
      margin-bottom: 0.6rem;
    }

    .endpoints { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1.75rem; }

    .ep {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      background: #0d1117;
      border: 1px solid #21262d;
      border-radius: 8px;
      padding: 0.6rem 0.9rem;
      font-family: 'Consolas', monospace;
      font-size: 0.85rem;
      color: #c9d1d9;
      transition: border-color 0.2s;
    }

    .ep:hover { border-color: #58a6ff; }

    .badge {
      font-size: 0.68rem;
      font-weight: 700;
      border-radius: 4px;
      padding: 0.15rem 0.45rem;
      min-width: 42px;
      text-align: center;
      letter-spacing: 0.03em;
    }

    .get  { background: #1a3a4a; color: #79c0ff; border: 1px solid #1f6feb; }
    .post { background: #1a3a2a; color: #56d364; border: 1px solid #238636; }

    .divider { border: none; border-top: 1px solid #21262d; margin: 1.5rem 0; }

    .actions { display: flex; gap: 0.75rem; }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.55rem 1.1rem;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      text-decoration: none;
      transition: opacity 0.2s;
    }

    .btn:hover { opacity: 0.85; }
    .btn-primary { background: #238636; color: #fff; }
    .btn-secondary { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; }
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <span style="font-size:1.6rem">📈</span>
      <h1>Stock Price Prediction API</h1>
    </div>

    <div class="status">Live</div>

    <p class="subtitle">
      <span>FastAPI</span>
      <span>NumPy ANN</span>
      <span>Linear Regression</span>
      <span>Random Forest</span>
    </p>

    <p class="section-label">Endpoints</p>
    <div class="endpoints">
      <div class="ep"><span class="badge get">GET</span>  /api/data-info</div>
      <div class="ep"><span class="badge post">POST</span> /api/train</div>
      <div class="ep"><span class="badge post">POST</span> /api/predict</div>
      <div class="ep"><span class="badge get">GET</span>  /api/metrics</div>
    </div>

    <hr class="divider" />

    <div class="actions">
      <a href="/docs" class="btn btn-primary">📄 Swagger UI</a>
      <a href="/redoc" class="btn btn-secondary">📘 ReDoc</a>
    </div>
  </div>
</body>
</html>
    """


# ── GET /api/data-info ────────────────────────────────────────────────────────
@app.get("/api/data-info")
def data_info():
    if not os.path.exists(DATA_PATH):
        raise HTTPException(404, "stock_data.csv not found")
    df   = load_dataset(DATA_PATH)
    info = inspect_data(df)
    return {
        "rows":       info["shape"][0],
        "columns":    info["shape"][1],
        "duplicates": info["duplicates"],
        "date_range": info["date_range"],
        "missing":    info["missing_values"],
        "preview":    df.head(5).to_dict(orient="records"),
    }


# ── POST /api/train ───────────────────────────────────────────────────────────
@app.post("/api/train")
def train(req: TrainRequest):
    if not os.path.exists(DATA_PATH):
        raise HTTPException(404, "stock_data.csv not found")

    df     = load_dataset(DATA_PATH)
    result = preprocess(df, req.features)
    X_train, X_test = result.X_train, result.X_test
    y_train, y_test = result.y_train, result.y_test
    scaler_y        = result.scaler_y

    # Baselines
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Numpy ANN
    ann = NumpyANN(input_dim=X_train.shape[1])
    history = ann.fit(X_train, y_train, X_test, y_test,
                      epochs=req.epochs, batch_size=req.batch_size)

    y_true   = scaler_y.inverse_transform(y_test)
    lr_pred  = scaler_y.inverse_transform(lr_model.predict(X_test).reshape(-1, 1))
    rf_pred  = scaler_y.inverse_transform(rf_model.predict(X_test).reshape(-1, 1))
    ann_pred = scaler_y.inverse_transform(ann.predict(X_test))

    results = {
        "Linear Regression": compute_metrics(y_true, lr_pred),
        "Random Forest":     compute_metrics(y_true, rf_pred),
        "ANN":               compute_metrics(y_true, ann_pred),
    }
    metrics_df = build_metrics_table(results)

    _state["scaler_X"] = result.scaler_X
    _state["scaler_y"] = scaler_y
    _state["ann"]      = ann
    _state["features"] = req.features + [
        f for f in ["Prev_Close", "Prev_Close2", "Price_Change"]
        if f not in req.features
    ]
    _state["metrics"]  = metrics_df.to_dict(orient="records")

    best = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]

    return {
        "status":     "trained",
        "epochs_run": len(history["loss"]),
        "best_model": best,
        "metrics":    metrics_df.to_dict(orient="records"),
        "train_loss": history["loss"],
        "val_loss":   history["val_loss"],
        "actual":     y_true.ravel().tolist(),
        "ann_pred":   ann_pred.ravel().tolist(),
        "lr_pred":    lr_pred.ravel().tolist(),
        "rf_pred":    rf_pred.ravel().tolist(),
    }


# ── POST /api/predict ─────────────────────────────────────────────────────────
@app.post("/api/predict")
def predict(req: PredictRequest):
    if "ann" not in _state:
        raise HTTPException(400, "Model not trained yet. Call POST /api/train first.")
    features = _state["features"]
    try:
        x_input = np.array([[req.features[f] for f in features]])
    except KeyError as e:
        raise HTTPException(422, f"Missing feature: {e}")

    x_scaled   = _state["scaler_X"].transform(x_input)
    pred_price = _state["scaler_y"].inverse_transform(_state["ann"].predict(x_scaled))[0][0]
    return {"predicted_close": round(float(pred_price), 4)}


# ── GET /api/metrics ──────────────────────────────────────────────────────────
@app.get("/api/metrics")
def metrics():
    if "metrics" not in _state:
        raise HTTPException(400, "No metrics yet. Call POST /api/train first.")
    return {"metrics": _state["metrics"]}

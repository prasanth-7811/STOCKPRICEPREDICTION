"""
app.py  —  Stock Price Prediction Using Artificial Neural Network
Streamlit interactive dashboard.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing  import load_dataset, inspect_data, preprocess
from train_ann      import build_ann, train_ann, save_model
from train_baseline import train_linear_regression, train_random_forest
from evaluation     import (compute_metrics, build_metrics_table, save_metrics_csv,
                             plot_closing_price, plot_volume, plot_training_loss,
                             plot_actual_vs_predicted, plot_model_comparison, plot_residuals)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Price Prediction — ANN", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .banner h1 { color: #e94560; margin: 0; font-size: 2rem; }
    .banner p  { color: #a8b2d8; margin: 0.4rem 0 0; font-size: 1rem; }

    .metric-card {
        background: #1e1e2e;
        border: 1px solid #2d2d44;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { color: #a8b2d8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #e94560; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #cdd6f4;
        border-left: 4px solid #e94560;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem;
    }

    .predict-box {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .predict-box .price { color: #a6e3a1; font-size: 2.2rem; font-weight: 800; }
    .predict-box .label { color: #a8b2d8; font-size: 0.9rem; margin-top: 4px; }

    div[data-testid="stSidebar"] { background: #1a1a2e; }
    div[data-testid="stSidebar"] * { color: #cdd6f4 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c23152);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        padding: 0.6rem;
    }
    .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
    <h1>📈 Stock Price Prediction</h1>
    <p>Comparing <strong>ANN</strong>, <strong>Linear Regression</strong>, and <strong>Random Forest</strong> for closing price forecasting</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**📂 Dataset**")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("**🧠 Features**")
    selected_features = st.multiselect(
        "Select input features",
        options=["Open", "High", "Low", "Volume"],
        default=["Open", "High", "Low", "Volume"],
        label_visibility="collapsed",
    )

    st.markdown("**🔧 Training**")
    epochs     = st.slider("Max Epochs", 50, 300, 150, step=10)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    st.markdown("---")
    train_button = st.button("🚀 Train Models")

# ── Data loading ──────────────────────────────────────────────────────────────
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "stock_data.csv")

def _find_local_csv():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if os.path.isdir(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(".csv"):
                return os.path.join(data_dir, f)
    return None

if uploaded:
    tmp_path = os.path.join(os.path.dirname(__file__), "data", "_uploaded.csv")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    pd.read_csv(uploaded).to_csv(tmp_path, index=False)
    data_path = tmp_path
    st.sidebar.success("✅ Dataset uploaded")
else:
    data_path = _find_local_csv()
    if data_path:
        st.sidebar.info(f"📂 {os.path.basename(data_path)}")
    else:
        st.sidebar.warning("No dataset found")

if data_path is None:
    st.warning("⚠️ Please upload a CSV dataset using the sidebar to begin.")
    st.stop()

try:
    df = load_dataset(data_path)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ── Section 1: Dataset Overview ───────────────────────────────────────────────
st.markdown('<div class="section-title">1 · Dataset Overview</div>', unsafe_allow_html=True)
info = inspect_data(df)

c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip(
    [c1, c2, c3, c4],
    ["Rows", "Columns", "Duplicates", "Date Range"],
    [info["shape"][0], info["shape"][1], info["duplicates"],
     f"{info['date_range'][0]} → {info['date_range'][1]}"],
):
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 Preview Data & Missing Values", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    missing_df = pd.DataFrame.from_dict(info["missing_values"], orient="index", columns=["Missing"])
    st.dataframe(missing_df.T, use_container_width=True)

# ── Section 2: Stock Price Trend ──────────────────────────────────────────────
st.markdown('<div class="section-title">2 · Stock Price Trend</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_closing_price(df))
with col2:
    st.pyplot(plot_volume(df))

# ── Section 3: Preprocessing ──────────────────────────────────────────────────
st.markdown('<div class="section-title">3 · Data Preprocessing</div>', unsafe_allow_html=True)
with st.expander("ℹ️ Preprocessing Steps", expanded=False):
    st.markdown("""
    - Dates converted to datetime and sorted chronologically
    - Lag features `Prev_Close`, `Prev_Close2`, `Price_Change` added
    - Missing values dropped; duplicates removed
    - **Chronological 80/20 train-test split** (no shuffling)
    - Features scaled with `MinMaxScaler` fitted **only on training data**
    """)

if not selected_features:
    st.warning("Select at least one feature in the sidebar.")
    st.stop()

try:
    result = preprocess(df, selected_features)
    X_train, X_test, y_train, y_test = result.X_train, result.X_test, result.y_train, result.y_test
    scaler_X, scaler_y, df_clean = result.scaler_X, result.scaler_y, result.df_clean
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

col1, col2 = st.columns(2)
col1.info(f"🏋️ Training samples: **{len(X_train)}**")
col2.info(f"🧪 Test samples: **{len(X_test)}**")

# ── Section 4: ANN Architecture ───────────────────────────────────────────────
st.markdown('<div class="section-title">4 · ANN Architecture</div>', unsafe_allow_html=True)
with st.expander("🧠 View Architecture", expanded=False):
    st.code("""
Input Layer  (features)
    ↓
Dense(64, relu) → Dropout(0.2)
    ↓
Dense(32, relu) → Dropout(0.2)
    ↓
Dense(16, relu)
    ↓
Dense(1, linear)  ← Predicted Close Price

Optimizer : Adam  |  Loss : MSE  |  EarlyStopping(patience=15)
""", language="text")

# ── Section 5: Training ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">5 · Model Training & Results</div>', unsafe_allow_html=True)

if train_button:
    with st.spinner("⏳ Training models — please wait..."):
        lr_model       = train_linear_regression(X_train, y_train)
        lr_pred        = scaler_y.inverse_transform(lr_model.predict(X_test).reshape(-1, 1))
        y_test_inv     = scaler_y.inverse_transform(y_test)

        rf_model       = train_random_forest(X_train, y_train)
        rf_pred        = scaler_y.inverse_transform(rf_model.predict(X_test).reshape(-1, 1))

        ann_model, history = train_ann(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
        save_model(ann_model, os.path.join(os.path.dirname(__file__), "models", "ann_model.keras"))
        ann_pred = scaler_y.inverse_transform(ann_model.predict(X_test, verbose=0))

    st.success("✅ All models trained successfully!")
    st.session_state.update({
        "trained": True, "y_test_inv": y_test_inv,
        "lr_pred": lr_pred, "rf_pred": rf_pred, "ann_pred": ann_pred,
        "history": history, "scaler_X": scaler_X, "scaler_y": scaler_y,
        "ann_model": ann_model, "df_clean": df_clean,
        "features": selected_features + ["Prev_Close"],
    })

if st.session_state.get("trained"):
    y_test_inv = st.session_state["y_test_inv"]
    lr_pred    = st.session_state["lr_pred"]
    rf_pred    = st.session_state["rf_pred"]
    ann_pred   = st.session_state["ann_pred"]
    history    = st.session_state["history"]
    scaler_y   = st.session_state["scaler_y"]
    ann_model  = st.session_state["ann_model"]
    features   = st.session_state["features"]

    # Actual vs Predicted
    st.markdown('<div class="section-title">6 · Actual vs Predicted</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🤖 ANN", "📉 Linear Regression", "🌲 Random Forest"])
    with tab1:
        st.pyplot(plot_actual_vs_predicted(y_test_inv, ann_pred, "ANN"))
        st.pyplot(plot_residuals(y_test_inv, ann_pred, "ANN"))
    with tab2:
        st.pyplot(plot_actual_vs_predicted(y_test_inv, lr_pred, "Linear Regression"))
    with tab3:
        st.pyplot(plot_actual_vs_predicted(y_test_inv, rf_pred, "Random Forest"))

    # Training loss
    st.markdown('<div class="section-title">7 · ANN Training Loss</div>', unsafe_allow_html=True)
    st.pyplot(plot_training_loss(history))

    # Metrics
    st.markdown('<div class="section-title">8 · Model Comparison & Metrics</div>', unsafe_allow_html=True)
    results = {
        "Linear Regression": compute_metrics(y_test_inv, lr_pred),
        "Random Forest":     compute_metrics(y_test_inv, rf_pred),
        "ANN":               compute_metrics(y_test_inv, ann_pred),
    }
    metrics_df = build_metrics_table(results)
    save_metrics_csv(metrics_df, os.path.join(os.path.dirname(__file__), "outputs", "metrics.csv"))

    st.dataframe(
        metrics_df.style
            .highlight_min(subset=["MAE", "MSE", "RMSE"], color="#1e4d2b")
            .highlight_max(subset=["R²"], color="#1e4d2b"),
        use_container_width=True,
    )
    st.pyplot(plot_model_comparison(metrics_df))

    best = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
    st.success(f"🏆 Best model by RMSE: **{best}**")

    # Single prediction
    st.markdown('<div class="section-title">9 · Predict a Single Closing Price</div>', unsafe_allow_html=True)
    cols = st.columns(len(features))
    input_vals = {}
    for i, feat in enumerate(features):
        default = float(st.session_state["df_clean"][feat].median()
                        if feat in st.session_state["df_clean"].columns else 0.0)
        input_vals[feat] = cols[i].number_input(feat, value=default, format="%.4f")

    if st.button("🔮 Predict Closing Price"):
        x_input  = np.array([[input_vals[f] for f in features]])
        x_scaled = st.session_state["scaler_X"].transform(x_input)
        pred_price = scaler_y.inverse_transform(ann_model.predict(x_scaled, verbose=0))[0][0]
        st.markdown(f"""
        <div class="predict-box">
            <div class="price">${pred_price:.2f}</div>
            <div class="label">Predicted Closing Price (ANN)</div>
        </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #a8b2d8;">
        <div style="font-size: 3rem;">🚀</div>
        <div style="font-size: 1.1rem; margin-top: 0.5rem;">Configure settings in the sidebar and click <strong>Train Models</strong> to begin</div>
    </div>""", unsafe_allow_html=True)

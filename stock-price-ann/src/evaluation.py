"""
evaluation.py  —  Metrics + 12 rich visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "figures")
sns.set_theme(style="darkgrid", palette="muted")


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    safe_name = os.path.basename(name)
    path = os.path.join(FIG_DIR, safe_name)
    if not os.path.abspath(path).startswith(os.path.abspath(FIG_DIR)):
        raise ValueError(f"Invalid file path: {name}")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    return path


def _style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    return {
        "MAE":  round(mae,  4),
        "MSE":  round(mse,  4),
        "RMSE": round(rmse, 4),
        "R2":   round(r2,   4),
        "MAPE": round(mape, 4),
    }


def build_metrics_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        rows.append({"Model": name, "MAE": m["MAE"], "MSE": m["MSE"],
                     "RMSE": m["RMSE"], "R²": m["R2"], "MAPE(%)": m["MAPE"]})
    return pd.DataFrame(rows)


def save_metrics_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# ── 1. Historical closing price with MAs ─────────────────────────────────────

def plot_closing_price(df_enriched: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_enriched["Date"], df_enriched["Close"],
            color="#2196F3", linewidth=1.2, label="Close Price", alpha=0.9)
    if "MA20" in df_enriched.columns:
        ax.plot(df_enriched["Date"], df_enriched["MA20"],
                color="#FF9800", linewidth=1.4, label="MA 20", linestyle="--")
    if "MA50" in df_enriched.columns:
        ax.plot(df_enriched["Date"], df_enriched["MA50"],
                color="#E91E63", linewidth=1.4, label="MA 50", linestyle="--")
    ax.fill_between(df_enriched["Date"], df_enriched["Close"],
                    alpha=0.08, color="#2196F3")
    _style_ax(ax, "Historical Stock Closing Price with Moving Averages", "Date", "Price ($)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "01_closing_price.png")
    return fig


# ── 2. Bollinger Bands ────────────────────────────────────────────────────────

def plot_bollinger_bands(df_enriched: pd.DataFrame) -> plt.Figure:
    d = df_enriched.dropna(subset=["MA20", "Upper_BB", "Lower_BB"])
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(d["Date"], d["Close"],    color="#2196F3", linewidth=1,   label="Close")
    ax.plot(d["Date"], d["MA20"],     color="#FF9800", linewidth=1.2, label="MA 20", linestyle="--")
    ax.plot(d["Date"], d["Upper_BB"], color="#4CAF50", linewidth=1,   label="Upper BB", linestyle=":")
    ax.plot(d["Date"], d["Lower_BB"], color="#F44336", linewidth=1,   label="Lower BB", linestyle=":")
    ax.fill_between(d["Date"], d["Upper_BB"], d["Lower_BB"], alpha=0.07, color="#9C27B0")
    _style_ax(ax, "Bollinger Bands (20-day)", "Date", "Price ($)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "02_bollinger_bands.png")
    return fig


# ── 3. Volume bar chart ───────────────────────────────────────────────────────

def plot_volume(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["#EF5350" if c < o else "#26A69A"
              for c, o in zip(df["Close"], df["Open"])]
    ax.bar(df["Date"], df["Volume"], color=colors, width=1.2, alpha=0.8)
    _style_ax(ax, "Trading Volume Over Time (Green=Up Day, Red=Down Day)", "Date", "Volume")
    fig.tight_layout()
    _save(fig, "03_volume.png")
    return fig


# ── 4. Daily return distribution ─────────────────────────────────────────────

def plot_daily_return_dist(df_enriched: pd.DataFrame) -> plt.Figure:
    ret = df_enriched["Daily_Return"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    sns.histplot(ret, bins=60, kde=True, color="#5C6BC0", ax=axes[0])
    axes[0].axvline(ret.mean(), color="red",    linestyle="--", label=f"Mean {ret.mean():.2f}%")
    axes[0].axvline(ret.std(),  color="orange", linestyle="--", label=f"Std  {ret.std():.2f}%")
    _style_ax(axes[0], "Daily Return Distribution", "Return (%)", "Frequency")
    axes[0].legend(fontsize=9)

    ret.plot(ax=axes[1], color="#5C6BC0", linewidth=0.7, alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    _style_ax(axes[1], "Daily Return Over Time", "Date", "Return (%)")

    fig.tight_layout()
    _save(fig, "04_daily_return.png")
    return fig


# ── 5. RSI indicator ─────────────────────────────────────────────────────────

def plot_rsi(df_enriched: pd.DataFrame) -> plt.Figure:
    d = df_enriched.dropna(subset=["RSI"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(d["Date"], d["Close"], color="#2196F3", linewidth=1)
    _style_ax(axes[0], "Close Price", ylabel="Price ($)")

    axes[1].plot(d["Date"], d["RSI"], color="#9C27B0", linewidth=1)
    axes[1].axhline(70, color="#F44336", linestyle="--", linewidth=0.9, label="Overbought (70)")
    axes[1].axhline(30, color="#4CAF50", linestyle="--", linewidth=0.9, label="Oversold (30)")
    axes[1].fill_between(d["Date"], d["RSI"], 70,
                         where=(d["RSI"] >= 70), alpha=0.2, color="#F44336")
    axes[1].fill_between(d["Date"], d["RSI"], 30,
                         where=(d["RSI"] <= 30), alpha=0.2, color="#4CAF50")
    _style_ax(axes[1], "RSI (14-day)", "Date", "RSI")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 100)

    fig.tight_layout()
    _save(fig, "05_rsi.png")
    return fig


# ── 6. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume",
                         "Daily_Return", "Daily_Range", "RSI", "MA20"]
            if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 9}, vmin=-1, vmax=1)
    _style_ax(ax, "Feature Correlation Heatmap")
    fig.tight_layout()
    _save(fig, "06_correlation_heatmap.png")
    return fig


# ── 7. ANN training loss ──────────────────────────────────────────────────────

def plot_training_loss(history) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history.history["loss"],     color="#2196F3", linewidth=1.5, label="Train Loss")
    axes[0].plot(history.history["val_loss"], color="#F44336", linewidth=1.5, label="Val Loss")
    _style_ax(axes[0], "Training vs Validation Loss (MSE)", "Epoch", "Loss")
    axes[0].legend(fontsize=9)

    axes[1].plot(history.history["mae"],     color="#4CAF50", linewidth=1.5, label="Train MAE")
    axes[1].plot(history.history["val_mae"], color="#FF9800", linewidth=1.5, label="Val MAE")
    _style_ax(axes[1], "Training vs Validation MAE", "Epoch", "MAE")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "07_training_loss.png")
    return fig


# ── 8. Actual vs Predicted (line) ─────────────────────────────────────────────

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str = "ANN") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(y_true))
    ax.plot(x, y_true.ravel(), color="#2196F3", linewidth=1.5, label="Actual", alpha=0.9)
    ax.plot(x, y_pred.ravel(), color="#FF5722", linewidth=1.5,
            label=f"Predicted ({model_name})", linestyle="--", alpha=0.9)
    ax.fill_between(x, y_true.ravel(), y_pred.ravel(), alpha=0.1, color="#9C27B0")
    _style_ax(ax, f"Actual vs Predicted Close Price — {model_name}",
              "Test Sample Index", "Price ($)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, f"08_actual_vs_predicted_{model_name.replace(' ', '_')}.png")
    return fig


# ── 9. Scatter: Actual vs Predicted ──────────────────────────────────────────

def plot_scatter_actual_pred(y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str = "ANN") -> plt.Figure:
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#5C6BC0", edgecolors="none")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect Prediction")
    _style_ax(ax, f"Actual vs Predicted Scatter — {model_name}",
              "Actual Price ($)", "Predicted Price ($)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, f"09_scatter_{model_name.replace(' ', '_')}.png")
    return fig


# ── 10. Residuals ─────────────────────────────────────────────────────────────

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   model_name: str = "ANN") -> plt.Figure:
    residuals = y_true.ravel() - y_pred.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(residuals, color="#7B1FA2", linewidth=0.8, alpha=0.8)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].fill_between(range(len(residuals)), residuals, 0,
                         where=(residuals > 0), alpha=0.2, color="#4CAF50", label="Over-predicted")
    axes[0].fill_between(range(len(residuals)), residuals, 0,
                         where=(residuals < 0), alpha=0.2, color="#F44336", label="Under-predicted")
    _style_ax(axes[0], f"Residuals Over Time — {model_name}",
              "Test Sample Index", "Residual ($)")
    axes[0].legend(fontsize=9)

    sns.histplot(residuals, bins=40, kde=True, color="#7B1FA2", ax=axes[1])
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    _style_ax(axes[1], "Residual Distribution", "Residual ($)", "Frequency")

    fig.tight_layout()
    _save(fig, f"10_residuals_{model_name.replace(' ', '_')}.png")
    return fig


# ── 11. Model comparison (4-metric bar chart) ─────────────────────────────────

def plot_model_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    metrics  = ["MAE", "RMSE", "R²", "MAPE(%)"]
    n_models = len(metrics_df)
    colors   = sns.color_palette("Set2", n_models)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        col = metric if metric in metrics_df.columns else metric.replace("²", "2")
        vals = metrics_df[col] if col in metrics_df.columns else metrics_df.get(metric, None)
        if vals is None:
            continue
        bars = axes[i].bar(metrics_df["Model"], vals, color=colors,
                           edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(vals) * 0.01,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        better = "lower" if metric != "R²" else "higher"
        _style_ax(axes[i], f"{metric}  ({better} is better)", "", metric)
        axes[i].set_xticklabels(metrics_df["Model"], rotation=10, fontsize=9)

    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "11_model_comparison.png")
    return fig


# ── 12. Radar / spider chart ──────────────────────────────────────────────────

def plot_radar_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    """Normalised radar chart — higher = better for all axes."""
    cats = ["MAE", "MSE", "RMSE", "MAPE(%)"]
    cats_present = [c for c in cats if c in metrics_df.columns]
    if len(cats_present) < 3:
        return plot_model_comparison(metrics_df)   # fallback

    # Invert error metrics so higher = better
    norm = metrics_df[cats_present].copy()
    for c in cats_present:
        col_max = norm[c].max()
        if col_max > 0:
            norm[c] = 1 - (norm[c] / col_max)

    if "R²" in metrics_df.columns:
        norm["R²"] = metrics_df["R²"].values
        cats_present.append("R²")

    N = len(cats_present)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for idx, row in metrics_df.iterrows():
        vals = norm.loc[idx, cats_present].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=row["Model"], color=colors[idx % 3])
        ax.fill(angles, vals, alpha=0.12, color=colors[idx % 3])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats_present, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("Model Comparison — Radar Chart\n(outer = better)", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    _save(fig, "12_radar_comparison.png")
    return fig

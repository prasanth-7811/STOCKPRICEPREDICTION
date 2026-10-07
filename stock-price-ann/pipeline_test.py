"""
pipeline_test.py  —  Quick end-to-end smoke test (no Streamlit needed).
Run: python pipeline_test.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from preprocessing   import load_dataset, inspect_data, preprocess
from train_ann       import train_ann, save_model
from train_baseline  import train_linear_regression, train_random_forest
from evaluation      import (compute_metrics, build_metrics_table, save_metrics_csv,
                              plot_closing_price, plot_volume, plot_training_loss,
                              plot_actual_vs_predicted, plot_model_comparison, plot_residuals)
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), "data", "stock_data.csv")

print("Loading dataset...")
df = load_dataset(DATA)
info = inspect_data(df)
print(f"  Shape      : {info['shape']}")
print(f"  Date range : {info['date_range'][0]}  to  {info['date_range'][1]}")
print(f"  Missing    : {sum(info['missing_values'].values())}")
print(f"  Duplicates : {info['duplicates']}")

print("\nPreprocessing...")
X_train, X_test, y_train, y_test, sX, sy, df_clean = preprocess(
    df, ["Open", "High", "Low", "Volume"]
)
print(f"  Train samples : {X_train.shape[0]}")
print(f"  Test  samples : {X_test.shape[0]}")
print(f"  Features      : {X_train.shape[1]}")

print("\nTraining Linear Regression...")
lr = train_linear_regression(X_train, y_train)

print("Training Random Forest...")
rf = train_random_forest(X_train, y_train)

print("Training ANN (50 epochs max)...")
ann, history = train_ann(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
save_model(ann, os.path.join(os.path.dirname(__file__), "models", "ann_model.keras"))
print("  ANN model saved.")

# Inverse-transform predictions back to original price scale
y_inv  = sy.inverse_transform(y_test)
lr_p   = sy.inverse_transform(lr.predict(X_test).reshape(-1, 1))
rf_p   = sy.inverse_transform(rf.predict(X_test).reshape(-1, 1))
ann_p  = sy.inverse_transform(ann.predict(X_test, verbose=0))

print("\nComputing metrics...")
results = {
    "Linear Regression": compute_metrics(y_inv, lr_p),
    "Random Forest":     compute_metrics(y_inv, rf_p),
    "ANN":               compute_metrics(y_inv, ann_p),
}
mdf = build_metrics_table(results)
os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
save_metrics_csv(mdf, os.path.join(os.path.dirname(__file__), "outputs", "metrics.csv"))

print("\n" + "="*55)
print(mdf.to_string(index=False))
print("="*55)

print("\nGenerating visualizations...")
plot_closing_price(df);                                  plt.close("all")
plot_volume(df);                                         plt.close("all")
plot_training_loss(history);                             plt.close("all")
plot_actual_vs_predicted(y_inv, ann_p, "ANN");           plt.close("all")
plot_model_comparison(mdf);                              plt.close("all")
plot_residuals(y_inv, ann_p, "ANN");                     plt.close("all")
print("  Figures saved to outputs/figures/")

print("\nAll tests PASSED. Run 'streamlit run app.py' to launch the dashboard.")

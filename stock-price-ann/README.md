# Stock Price Prediction Using Artificial Neural Network
### Academic Mini Project — 20 Marks

---

## Project Structure

```
stock-price-ann/
├── data/
│   └── stock_data.csv          ← Place your dataset here
├── notebooks/
│   └── stock_prediction.ipynb  ← Step-by-step Jupyter walkthrough
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        ← Load, clean, scale data
│   ├── train_ann.py            ← ANN model (TensorFlow/Keras)
│   ├── train_baseline.py       ← Linear Regression & Random Forest
│   └── evaluation.py           ← Metrics + 6 visualizations
├── models/
│   └── ann_model.keras         ← Saved after training
├── outputs/
│   ├── figures/                ← Auto-saved plots (PNG)
│   └── metrics.csv             ← Model comparison table
├── app.py                      ← Streamlit dashboard
├── generate_sample_data.py     ← Creates stock_data.csv for testing
├── requirements.txt
├── README.md
└── report_content.md           ← Full academic report (16 sections)
```

---

## Dataset

**Option A — Use the built-in sample generator (no download needed):**
```bash
python generate_sample_data.py
```
This creates `data/stock_data.csv` with 1,500 realistic synthetic trading days.

**Option B — Download a real dataset from Kaggle:**
- [Apple AAPL Stock](https://www.kaggle.com/datasets/tarunpaparaju/apple-aapl-historical-stock-data)
- [NSE India NIFTY50](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data)
- Any CSV with columns: `Date, Open, High, Low, Close, Volume`

Place the file in `data/` as `stock_data.csv`.

---

## Installation

```bash
cd stock-price-ann
pip install -r requirements.txt
```

---

## Run

### 1. Generate sample data (first time only)
```bash
python generate_sample_data.py
```

### 2. Launch the Streamlit app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 3. Run the Jupyter notebook (optional)
```bash
jupyter notebook notebooks/stock_prediction.ipynb
```

---

## Module Explanations

| File | Purpose |
|------|---------|
| `preprocessing.py` | Loads CSV, normalises columns, adds Prev_Close lag feature, performs chronological 80/20 split, fits MinMaxScaler on training data only |
| `train_ann.py` | Defines the 4-layer ANN (64→32→16→1) with Dropout, compiles with Adam+MSE, trains with EarlyStopping |
| `train_baseline.py` | Trains LinearRegression and RandomForestRegressor as comparison baselines |
| `evaluation.py` | Computes MAE/MSE/RMSE/R², builds comparison DataFrame, generates and saves all 6 plots |
| `app.py` | Streamlit dashboard — dataset upload, feature selection, model training, charts, single-price prediction |
| `generate_sample_data.py` | Creates a synthetic but realistic stock CSV using geometric Brownian motion |

---

## Expected Output

After clicking **Train Models** in the sidebar:

1. Dataset overview table with shape, missing values, date range
2. Historical closing price chart
3. Trading volume bar chart
4. Train/test split info
5. ANN architecture display
6. Actual vs Predicted charts for all 3 models (tabbed)
7. ANN training loss curve
8. Model comparison table (MAE, MSE, RMSE, R²) with green highlights
9. RMSE and R² bar charts
10. Single-price prediction input form

Files saved automatically:
- `outputs/metrics.csv` — comparison table
- `outputs/figures/01_closing_price.png` through `06_residuals_ANN.png`
- `models/ann_model.keras` — trained ANN weights

---

## Viva Questions & Answers

### Q1. What is an Artificial Neural Network?
An ANN is a computational model inspired by biological neurons. It consists of layers of
interconnected nodes (neurons). Each connection has a weight that is adjusted during training
via backpropagation to minimise a loss function. ANNs can approximate any continuous function,
making them suitable for regression tasks like stock price prediction.

### Q2. Why did you use a Dense (fully connected) ANN instead of LSTM?
For this academic project, a Dense ANN is sufficient to demonstrate the concept and compare
with baselines. LSTM is better for long sequential dependencies but requires more data and
tuning. The lag feature (Prev_Close) provides the temporal context that LSTM would otherwise
capture through its hidden state.

### Q3. What is the purpose of Dropout layers?
Dropout randomly sets a fraction (20% here) of neuron outputs to zero during each training
step. This prevents co-adaptation of neurons and forces the network to learn redundant
representations, reducing overfitting on the training data.

### Q4. Why did you use MinMaxScaler instead of StandardScaler?
MinMaxScaler maps all values to [0, 1], which is compatible with the sigmoid/relu activation
range and prevents features with large magnitudes (like Volume) from dominating the gradient
updates. StandardScaler could also work but MinMaxScaler is preferred when the output neuron
uses a linear activation and we want predictions in a bounded range before inverse-transforming.

### Q5. What is data leakage and how did you prevent it?
Data leakage occurs when information from the test set influences the training process,
leading to overly optimistic evaluation metrics. We prevented it by:
- Fitting the MinMaxScaler **only on training data** and applying it to test data.
- Using **chronological splitting** (no random shuffle) so future data never appears in training.

### Q6. Why is random shuffling wrong for time-series data?
Stock prices are temporally ordered. If we shuffle randomly, future prices can appear in the
training set and past prices in the test set. The model would then "see the future" during
training, producing unrealistically high accuracy — a form of look-ahead bias.

### Q7. What does R² Score mean?
R² (coefficient of determination) measures the proportion of variance in the target variable
explained by the model. R² = 1.0 means perfect prediction; R² = 0 means the model is no
better than predicting the mean; negative R² means the model is worse than the mean baseline.

### Q8. What is EarlyStopping and why is it used?
EarlyStopping monitors a metric (validation loss) during training and stops when it stops
improving for a specified number of epochs (patience=15). It prevents overfitting and saves
training time by restoring the best weights automatically.

### Q9. What is the difference between MAE and RMSE?
- MAE (Mean Absolute Error) treats all errors equally — it is the average absolute difference.
- RMSE (Root Mean Squared Error) squares errors before averaging, so large errors are penalised
  more heavily. RMSE is more sensitive to outliers than MAE.

### Q10. Why is Prev_Close (lag feature) important?
Stock prices exhibit strong autocorrelation — today's price is highly correlated with
yesterday's. Including Prev_Close gives the model direct access to the most recent price
signal, which is typically the strongest predictor of the next closing price.

### Q11. What optimizer did you use and why?
Adam (Adaptive Moment Estimation). It combines momentum (smooths gradient direction) and
RMSProp (adapts learning rate per parameter). Adam converges faster than plain SGD and
requires less manual learning rate tuning, making it the standard choice for ANNs.

### Q12. How does Random Forest work?
Random Forest builds many decision trees, each trained on a random bootstrap sample of the
data and a random subset of features. Predictions are averaged across all trees. This
ensemble approach reduces variance (overfitting) compared to a single decision tree.

### Q13. What are the limitations of your ANN model?
- Does not capture long-range temporal dependencies (LSTM would be better for that).
- Sensitive to hyperparameter choices (layers, neurons, dropout rate).
- Requires normalisation; raw features with different scales cause training instability.
- Black-box nature makes it hard to explain predictions to stakeholders.
- Performance degrades during market regime changes (e.g., COVID crash) not seen in training.

### Q14. What is the activation function used and why relu?
ReLU (Rectified Linear Unit): f(x) = max(0, x). It is used in hidden layers because:
- It does not suffer from the vanishing gradient problem (unlike sigmoid/tanh).
- It is computationally efficient.
- It introduces non-linearity, allowing the network to learn complex patterns.
The output layer uses **linear** activation because we are predicting a continuous value
(regression), not a probability.

### Q15. How would you improve this project further?
- Use LSTM or GRU for better temporal modelling.
- Add technical indicators (RSI, MACD, Bollinger Bands) as features.
- Incorporate news sentiment scores via NLP.
- Use Keras Tuner for automated hyperparameter optimisation.
- Integrate real-time data via the `yfinance` API.
- Extend to multi-step (multi-day) forecasting.

"""
preprocessing.py  —  Data loading, EDA stats, feature engineering, scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import NamedTuple


class PreprocessResult(NamedTuple):
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler_X: MinMaxScaler
    scaler_y: MinMaxScaler
    df_clean: pd.DataFrame


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().title() for c in df.columns]
    df.rename(columns={"Adj Close": "Adj_Close", "Adj. Close": "Adj_Close"}, inplace=True)

    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing columns: {missing_cols}")

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "date_range": (str(df["Date"].min().date()), str(df["Date"].max().date())),
    }


def compute_eda_stats(df: pd.DataFrame) -> dict:
    """Extended EDA: descriptive stats, daily return, volatility, price range."""
    numeric = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Daily return %
    df2 = df.copy()
    df2["Daily_Return"] = df2["Close"].pct_change() * 100
    df2["Daily_Range"]  = df2["High"] - df2["Low"]
    df2["HL_Ratio"]     = (df2["High"] - df2["Low"]) / df2["Close"]

    # Rolling 20-day stats
    df2["MA20"]   = df2["Close"].rolling(20).mean()
    df2["MA50"]   = df2["Close"].rolling(50).mean()
    df2["STD20"]  = df2["Close"].rolling(20).std()
    df2["Upper_BB"] = df2["MA20"] + 2 * df2["STD20"]
    df2["Lower_BB"] = df2["MA20"] - 2 * df2["STD20"]

    # Momentum: RSI (14-day)
    delta = df2["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df2["RSI"] = 100 - (100 / (1 + rs))

    return {
        "describe":      numeric.describe().round(2),
        "df_enriched":   df2,
        "avg_daily_ret": round(df2["Daily_Return"].mean(), 4),
        "volatility":    round(df2["Daily_Return"].std(), 4),
        "max_close":     round(df["Close"].max(), 2),
        "min_close":     round(df["Close"].min(), 2),
        "avg_volume":    int(df["Volume"].mean()),
        "total_days":    len(df),
    }


def preprocess(df: pd.DataFrame, feature_cols: list, target_col: str = "Close"):
    """
    Feature engineering + chronological split + scaling.
    Returns X_train, X_test, y_train, y_test, scaler_X, scaler_y, df_clean
    """
    df = df.copy()
    cols_needed = feature_cols + [target_col, "Date"]
    df.dropna(subset=cols_needed, inplace=True)
    df.drop_duplicates(inplace=True)

    # Lag features
    df["Prev_Close"]  = df["Close"].shift(1)
    df["Prev_Close2"] = df["Close"].shift(2)
    df["Price_Change"] = df["Close"].diff()
    df.dropna(inplace=True)

    features = list(feature_cols)
    features += [f for f in ["Prev_Close", "Prev_Close2", "Price_Change"] if f not in features]

    X = df[features].values
    y = df[[target_col]].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train  = scaler_X.fit_transform(X_train)
    X_test   = scaler_X.transform(X_test)
    y_train  = scaler_y.fit_transform(y_train)
    y_test   = scaler_y.transform(y_test)

    return PreprocessResult(X_train, X_test, y_train, y_test, scaler_X, scaler_y, df)

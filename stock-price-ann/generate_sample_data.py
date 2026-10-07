"""
generate_sample_data.py
Generates a realistic synthetic stock dataset for testing.
Run once: python generate_sample_data.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 1500  # ~6 years of trading days

dates = pd.bdate_range(start="2018-01-01", periods=n)

# Simulate a price series using geometric Brownian motion
price = 150.0
prices = [price]
for _ in range(n - 1):
    change = np.random.normal(0.0003, 0.015)
    price = max(price * (1 + change), 1.0)
    prices.append(round(price, 2))

prices = np.array(prices)

daily_range = prices * np.random.uniform(0.005, 0.025, n)
high   = np.round(prices + daily_range * np.random.uniform(0.3, 1.0, n), 2)
low    = np.round(prices - daily_range * np.random.uniform(0.3, 1.0, n), 2)
open_  = np.round(low + (high - low) * np.random.uniform(0, 1, n), 2)
volume = np.random.randint(5_000_000, 80_000_000, n)

df = pd.DataFrame({
    "Date":   dates,
    "Open":   open_,
    "High":   high,
    "Low":    low,
    "Close":  prices,
    "Volume": volume,
})

out_path = os.path.join(os.path.dirname(__file__), "data", "stock_data.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Sample dataset saved: {out_path}  ({len(df)} rows)")

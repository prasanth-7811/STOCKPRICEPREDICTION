"""
train_baseline.py
Trains Linear Regression and Random Forest baseline models.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train.ravel())
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train.ravel())
    return model

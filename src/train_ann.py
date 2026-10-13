"""
train_ann.py
Builds, trains, and saves the ANN model using TensorFlow/Keras.
"""

import os
import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def build_ann(input_dim: int) -> tf.keras.Model:
    """
    ANN Architecture:
        Input → Dense(64, relu) → Dropout(0.2)
              → Dense(32, relu) → Dropout(0.2)
              → Dense(16, relu)
              → Dense(1, linear)
    """
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_ann(X_train, y_train, X_test, y_test, epochs: int = 150, batch_size: int = 32):
    """Train the ANN with early stopping. Returns model and history."""
    model = build_ann(X_train.shape[1])

    early_stop = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    return model, history


def save_model(model: tf.keras.Model, path: str = "models/ann_model.keras"):
    """Save the trained Keras model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)


def load_model(path: str = "models/ann_model.keras") -> tf.keras.Model:
    return tf.keras.models.load_model(path)

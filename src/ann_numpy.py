"""
ann_numpy.py — Lightweight ANN using only numpy.
Architecture: Input → Dense(64,relu) → Dense(32,relu) → Dense(16,relu) → Dense(1,linear)
Trained with mini-batch SGD + Adam optimiser, EarlyStopping on val_loss.
"""

import numpy as np


def relu(x):       return np.maximum(0, x)
def relu_grad(x):  return (x > 0).astype(float)


class NumpyANN:
    def __init__(self, input_dim: int, layers=(64, 32, 16), lr=0.001):
        self.lr = lr
        dims = [input_dim] + list(layers) + [1]
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            self.W.append(np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i]))
            self.b.append(np.zeros((1, dims[i+1])))
        # Adam state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t  = 0

    def _forward(self, X):
        self._a = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self._a[-1] @ W + b
            self._a.append(relu(z) if i < len(self.W) - 1 else z)
        return self._a[-1]

    def _backward(self, y):
        m   = y.shape[0]
        dz  = (self._a[-1] - y) / m
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in reversed(range(len(self.W))):
            dW = self._a[i].T @ dz
            db = dz.sum(axis=0, keepdims=True)
            # Adam update
            self.mW[i] = b1*self.mW[i] + (1-b1)*dW
            self.vW[i] = b2*self.vW[i] + (1-b2)*dW**2
            self.mb[i] = b1*self.mb[i] + (1-b1)*db
            self.vb[i] = b2*self.vb[i] + (1-b2)*db**2
            mW_hat = self.mW[i] / (1 - b1**self.t)
            vW_hat = self.vW[i] / (1 - b2**self.t)
            mb_hat = self.mb[i] / (1 - b1**self.t)
            vb_hat = self.vb[i] / (1 - b2**self.t)
            self.W[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)
            if i > 0:
                dz = (dz @ self.W[i].T) * relu_grad(self._a[i])

    def fit(self, X_train, y_train, X_val, y_val,
            epochs=150, batch_size=32, patience=15):
        best_val, best_W, best_b, wait = np.inf, None, None, 0
        history = {"loss": [], "val_loss": []}
        np.random.seed(1)
        for epoch in range(epochs):
            idx = np.random.permutation(len(X_train))
            for start in range(0, len(X_train), batch_size):
                xb = X_train[idx[start:start+batch_size]]
                yb = y_train[idx[start:start+batch_size]]
                self._forward(xb)
                self._backward(yb)
            train_loss = float(np.mean((self._forward(X_train) - y_train)**2))
            val_loss   = float(np.mean((self._forward(X_val)   - y_val  )**2))
            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_W   = [w.copy() for w in self.W]
                best_b   = [b.copy() for b in self.b]
                wait     = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        self.W, self.b = best_W, best_b
        return history

    def predict(self, X):
        return self._forward(X)

    def get_weights(self):
        return {"W": [w.tolist() for w in self.W],
                "b": [b.tolist() for b in self.b]}

    def set_weights(self, data):
        self.W = [np.array(w) for w in data["W"]]
        self.b = [np.array(b) for b in data["b"]]

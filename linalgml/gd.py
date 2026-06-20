"""Linear regression fit by batch gradient descent (instead of the closed form).

The math runs through linalgpy.Matrix: predictions, residual, gradient, update.
"""
from __future__ import annotations

import numpy as np

from linalgpy import Matrix

from .features import design_matrix


class GDRegressor:
    """Least-squares fit by batch gradient descent.

    After fit():
      - self.weights      : Matrix (p[+1], 1)
      - self.coef_        : numpy feature weights
      - self.intercept_   : float
      - self.loss_history : list[float] of MSE per iteration (for convergence)
    """

    def __init__(self, lr: float = 0.1, n_iters: int = 2000, fit_intercept: bool = True):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.weights: Matrix | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X, y) -> "GDRegressor":
        """Minimize MSE by gradient descent."""
        Xm = Matrix.from_numpy(design_matrix(X, self.fit_intercept))
        ym = Matrix.from_numpy(np.asarray(y, float).reshape(-1, 1))
        n = Xm.rows
        p = Xm.cols
        w = Matrix.zeros(p, 1)

        for _ in range(self.n_iters):
            pred = Xm @ w
            resid = pred - ym
            grad = (Xm.T @ resid) * (2.0 / n)
            w = w - grad * self.lr
            MSE = np.mean(resid.to_numpy() ** 2)
            self.loss_history.append(MSE)
            
        self.weights = w
        wn = w.to_numpy().flatten()
        self.intercept_ = wn[0]
        self.coef_ = wn[1:]

        return self

    def predict(self, X) -> np.ndarray:
        Xm = Matrix.from_numpy(design_matrix(X, self.fit_intercept))
        return (Xm @ self.weights).to_numpy().flatten()

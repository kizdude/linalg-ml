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
        """Minimize MSE by gradient descent.

        TODO (you implement, using the Matrix API):
          - Xm = Matrix.from_numpy(design_matrix(X, self.fit_intercept))   # (n, p)
          - ym = Matrix.from_numpy(np.asarray(y, float).reshape(-1, 1))    # (n, 1)
          - n = number of samples; p = number of columns of the design matrix
          - initialise weights w = Matrix.zeros(p, 1)
          - repeat n_iters times:
              pred = Xm @ w
              resid = pred - ym                       # (n, 1)
              grad  = (Xm.T @ resid) * (2.0 / n)      # (p, 1)
              w     = w - grad * self.lr
              append the MSE to self.loss_history each iter (a test checks it falls)
              MSE = mean of resid^2; resid.to_numpy() then numpy mean is easiest
          - store self.weights = w, then split into intercept_ / coef_
            (same split logic as LinearRegression)
        Return self.
        """
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        Xm = Matrix.from_numpy(design_matrix(X, self.fit_intercept))
        return (Xm @ self.weights).to_numpy().flatten()

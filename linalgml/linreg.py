"""Multivariate linear regression via the normal equation.

The math runs through linalgpy.Matrix (your C library), not numpy: we form the
design matrix, then solve the normal equation

    (Xᵀ X) w = Xᵀ y

for the weight vector w. numpy is used only to shuttle data in and out.
"""
from __future__ import annotations

import numpy as np

from linalgpy import Matrix


class LinearRegression:
    """Ordinary least squares fit by the normal equation.

    After fit():
      - self.weights   : Matrix, shape (d[+1], 1) — full weight vector
      - self.coef_     : numpy array of the feature weights
      - self.intercept_: float (0.0 if fit_intercept=False)
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.weights: Matrix | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    # --- plumbing: build the design matrix (numpy), prepend a bias column ---
    def _design(self, X) -> np.ndarray:
        """X (n_samples, n_features) -> design matrix, with a leading column of
        1s if fit_intercept (so the first weight is the intercept)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack([ones, X])
        return X

    # --- solve the normal equation with Matrix ---
    def fit(self, X, y) -> "LinearRegression":
        """Fit weights solving (Xᵀ X) w = Xᵀ y."""
        Xd = self._design(X)
        y2 = np.asarray(y, float).reshape(-1, 1)

        Xm = Matrix.from_numpy(Xd)
        ym = Matrix.from_numpy(y2)

        Xt = Xm.T
        XtX = Xt @ Xm
        Xty = Xt @ ym
        w = XtX.solve(Xty)

        self.weights = w
        wn = w.to_numpy().flatten()
        if self.fit_intercept:
            self.intercept_ = float(wn[0])
            self.coef_ = wn[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = wn
        return self

    def predict(self, X) -> np.ndarray:
        """Return predictions X·w as a 1-D numpy array."""
        Xm = Matrix.from_numpy(self._design(X))
        return (Xm @ self.weights).to_numpy().flatten()

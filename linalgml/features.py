"""Feature transforms for linear models."""
from __future__ import annotations

import numpy as np


def design_matrix(X, fit_intercept: bool = True) -> np.ndarray:
    """Coerce X to a 2-D float array, prepending a column of 1s if fit_intercept."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if fit_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X


def polynomial_features(x, degree: int) -> np.ndarray:
    """Expand a single feature into polynomial columns.

    Input x: shape (n,) or (n, 1) — one feature.
    Output: shape (n, degree), with column j (0-indexed) equal to x**(j+1),
    i.e. [x, x**2, ..., x**degree]. The constant term is omitted because
    LinearRegression adds its own bias column.
    """
    x = np.asarray(x).ravel()
    if degree < 1:
        raise ValueError("degree must be >= 1")
    return np.vander(x, degree+1, increasing=True)[:, 1:]

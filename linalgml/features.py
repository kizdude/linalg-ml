"""Feature transforms for linear models."""
from __future__ import annotations

import numpy as np


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

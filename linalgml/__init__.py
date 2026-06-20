"""linalgml - minimal machine learning on top of the linalg C library."""
from __future__ import annotations

from .linreg import LinearRegression
from .gd import GDRegressor
from .features import polynomial_features

__all__ = ["LinearRegression", "GDRegressor", "polynomial_features"]

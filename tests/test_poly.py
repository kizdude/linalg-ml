"""Spec for polynomial_features + fitting a curve with LinearRegression."""
import numpy as np
import pytest

from linalgml import LinearRegression, polynomial_features


def test_shape_and_columns():
    x = np.array([2.0, 3.0])
    P = polynomial_features(x, 3)
    assert P.shape == (2, 3)
    # columns are x^1, x^2, x^3
    assert np.allclose(P[0], [2, 4, 8])
    assert np.allclose(P[1], [3, 9, 27])


def test_degree_one_is_identity_column():
    x = np.array([1.0, 5.0, -2.0])
    P = polynomial_features(x, 1)
    assert P.shape == (3, 1)
    assert np.allclose(P[:, 0], x)


def test_fits_cubic_exactly():
    # y = 2 + 3x - x^2 + 0.5x^3  (intercept 2; poly coefs 3, -1, 0.5)
    x = np.linspace(-3, 3, 50)
    y = 2 + 3 * x - x**2 + 0.5 * x**3
    model = LinearRegression().fit(polynomial_features(x, 3), y)
    assert model.intercept_ == pytest.approx(2.0, abs=1e-6)
    assert np.allclose(model.coef_, [3.0, -1.0, 0.5], atol=1e-6)
    assert np.allclose(model.predict(polynomial_features(x, 3)), y, atol=1e-6)


def test_underfit_degree_is_not_exact():
    # a cubic signal fit with degree 1 should NOT reproduce it
    x = np.linspace(-3, 3, 50)
    y = x**3
    model = LinearRegression().fit(polynomial_features(x, 1), y)
    assert not np.allclose(model.predict(polynomial_features(x, 1)), y, atol=1e-3)

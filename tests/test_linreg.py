"""Spec for LinearRegression. Cross-checked against numpy's least squares.

Run:  .venv/Scripts/python -m pytest   (after building the C lib + installs)
"""
import numpy as np
import pytest

from linalgml import LinearRegression


def _make_data(n=200, d=3, seed=0, noise=0.0, intercept=5.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = rng.normal(size=d)
    y = X @ true_w + intercept + noise * rng.normal(size=n)
    return X, y, true_w, intercept


def test_recovers_exact_weights_no_noise():
    X, y, true_w, true_b = _make_data(noise=0.0)
    model = LinearRegression().fit(X, y)
    assert np.allclose(model.coef_, true_w, atol=1e-6)
    assert model.intercept_ == pytest.approx(true_b, abs=1e-6)


def test_predict_matches_targets_no_noise():
    X, y, _, _ = _make_data(noise=0.0)
    model = LinearRegression().fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-6)


def test_matches_numpy_lstsq_with_noise():
    X, y, _, _ = _make_data(noise=0.5, seed=1)
    model = LinearRegression().fit(X, y)
    # numpy reference: solve with a bias column
    Xd = np.hstack([np.ones((X.shape[0], 1)), X])
    w_ref, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    got = np.concatenate([[model.intercept_], model.coef_])
    assert np.allclose(got, w_ref, atol=1e-6)


def test_no_intercept():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 2))
    true_w = np.array([2.0, -3.0])
    y = X @ true_w  # no intercept term
    model = LinearRegression(fit_intercept=False).fit(X, y)
    assert model.intercept_ == 0.0
    assert np.allclose(model.coef_, true_w, atol=1e-6)


def test_single_feature():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])  # y = 2x + 1
    model = LinearRegression().fit(X, y)
    assert model.intercept_ == pytest.approx(1.0, abs=1e-6)
    assert model.coef_[0] == pytest.approx(2.0, abs=1e-6)

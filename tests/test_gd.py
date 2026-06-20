"""Spec for GDRegressor (batch gradient descent)."""
import numpy as np
import pytest

from linalgml import GDRegressor, LinearRegression


def _data(n=200, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))          # standardized-ish features -> GD behaves
    true_w = rng.normal(size=d)
    y = X @ true_w + 4.0
    return X, y, true_w


def test_converges_to_true_weights():
    X, y, true_w = _data()
    model = GDRegressor(lr=0.1, n_iters=3000).fit(X, y)
    assert np.allclose(model.coef_, true_w, atol=1e-2)
    assert model.intercept_ == pytest.approx(4.0, abs=1e-2)


def test_matches_closed_form():
    X, y, _ = _data(seed=1)
    gd = GDRegressor(lr=0.1, n_iters=5000).fit(X, y)
    cf = LinearRegression().fit(X, y)
    assert np.allclose(gd.coef_, cf.coef_, atol=1e-2)
    assert gd.intercept_ == pytest.approx(cf.intercept_, abs=1e-2)


def test_loss_decreases():
    X, y, _ = _data(seed=2)
    model = GDRegressor(lr=0.1, n_iters=500).fit(X, y)
    assert len(model.loss_history) > 0
    assert model.loss_history[-1] < model.loss_history[0]


def test_predict_shape():
    X, y, _ = _data()
    model = GDRegressor(lr=0.1, n_iters=100).fit(X, y)
    assert model.predict(X).shape == (X.shape[0],)

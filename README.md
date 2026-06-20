# linalg-ml

Minimal machine learning built on the [linalg](https://github.com/kizdude/linalg)
C library, called from Python through the
[linalg-py](https://github.com/kizdude/linalg-py) ctypes bindings.

It provides:

- **`LinearRegression`** — least squares by the normal equation `(Xᵀ X) w = Xᵀ y`.
- **`GDRegressor`** — the same fit by batch gradient descent (iterative).
- **`polynomial_features`** — expand a feature into powers so a linear model fits curves.

Every matrix operation (`@`, `.T`, `.solve()`, `+`, scalar `*`) runs through the C
library, not numpy.

## Setup

`linalg-py` (and the `linalg` C library under it) are bundled as a recursive git
submodule. The C library is compiled once; then `linalgpy` is installed editable
so it can load the compiled `.dll`.

```sh
# 1. get the nested submodules
git submodule update --init --recursive

# 2. virtual environment
python -m venv .venv

# 3. build the bundled C library
.venv/Scripts/python external/linalg-py/build_lib.py

# 4. install the binding (editable) and this package
.venv/Scripts/python -m pip install -e external/linalg-py
.venv/Scripts/python -m pip install -e ".[dev,demo]"
```

## Use

```python
from linalgml import LinearRegression, GDRegressor, polynomial_features

# closed-form fit
model = LinearRegression().fit(X, y)        # X: (n, d), y: (n,)
print(model.coef_, model.intercept_)

# gradient descent
gd = GDRegressor(lr=0.1, n_iters=2000).fit(X, y)

# polynomial curve fitting (one feature)
poly = LinearRegression().fit(polynomial_features(x, degree=3), y)
```

## Demos & tests

```sh
.venv/Scripts/python examples/regression_demo.py   # multivariate fit, predicted-vs-actual
.venv/Scripts/python examples/poly_demo.py         # degree-3 curve fit
.venv/Scripts/python examples/gd_demo.py           # GD convergence across learning rates
.venv/Scripts/python -m pytest                     # cross-checked against numpy
```

## Layout

```
linalgml/linreg.py    LinearRegression (normal equation)
linalgml/gd.py        GDRegressor (gradient descent)
linalgml/features.py  polynomial_features, design_matrix
examples/             regression_demo.py, poly_demo.py, gd_demo.py
tests/                pytest suite (cross-checked against numpy)
external/linalg-py    the ctypes bindings (git submodule; contains linalg)
```

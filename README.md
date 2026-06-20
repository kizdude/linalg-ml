# linalg-ml

Minimal machine learning built on the [linalg](https://github.com/kizdude/linalg)
C library, called from Python through the
[linalg-py](https://github.com/kizdude/linalg-py) ctypes bindings.

The first model is **multivariate linear regression** solved by the normal
equation `(Xᵀ X) w = Xᵀ y` — every matrix operation runs through the C library
(`@`, `.T`, `.solve()`), not numpy.

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
from linalgml import LinearRegression

model = LinearRegression().fit(X, y)   # X: (n, d) array, y: (n,) array
print(model.coef_, model.intercept_)
preds = model.predict(X)
```

## Demo & tests

```sh
.venv/Scripts/python examples/regression_demo.py   # fits synthetic data, plots fit
.venv/Scripts/python -m pytest                      # cross-checked against numpy
```

## Layout

```
linalgml/linreg.py   LinearRegression (normal equation via linalgpy.Matrix)
examples/            regression_demo.py
tests/               pytest suite (cross-checked against numpy)
external/linalg-py   the ctypes bindings (git submodule; contains linalg)
```

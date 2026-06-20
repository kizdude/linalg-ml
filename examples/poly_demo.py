"""Demo: fit a polynomial curve to noisy data using polynomial_features +
LinearRegression (all matrix math via the linalg C library).

Run:  .venv/Scripts/python examples/poly_demo.py
"""
from __future__ import annotations

import numpy as np

from linalgml import LinearRegression, polynomial_features

DEGREE = 3


def true_curve(x):
    return 2.0 + 3.0 * x - 1.0 * x**2 + 0.5 * x**3


def main() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(-3, 3, 60)
    y = true_curve(x) + rng.normal(scale=2.0, size=x.shape)   # noisy samples

    model = LinearRegression().fit(polynomial_features(x, DEGREE), y)
    xs = np.linspace(-3, 3, 300)
    ys = model.predict(polynomial_features(xs, DEGREE))

    # --- plotting ---
    print(f"degree {DEGREE} fit:  intercept={model.intercept_:+.3f}  "
          f"coef={np.round(model.coef_, 3).tolist()}")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib for the plot: pip install -e '.[demo]')")
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=18, alpha=0.6, label="noisy data")
    plt.plot(xs, true_curve(xs), "g--", lw=1.5, label="true curve")
    plt.plot(xs, ys, "r-", lw=2, label=f"degree-{DEGREE} fit")
    plt.legend()
    plt.title(f"Polynomial regression (degree {DEGREE})")
    plt.tight_layout()
    plt.savefig("poly.png", dpi=110)
    print("saved plot -> poly.png")


if __name__ == "__main__":
    main()

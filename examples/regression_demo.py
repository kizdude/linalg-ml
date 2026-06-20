"""Demo: multivariate linear regression on synthetic data, fit via the normal
equation through the linalg C library.

Run:  .venv/Scripts/python examples/regression_demo.py
"""
from __future__ import annotations

import numpy as np

from linalgml import LinearRegression


def main() -> None:
    rng = np.random.default_rng(42)
    n, d = 300, 4
    X = rng.normal(size=(n, d))
    true_w = np.array([1.5, -2.0, 0.5, 3.0])
    true_b = 4.0
    y = X @ true_w + true_b + 0.3 * rng.normal(size=n)

    model = LinearRegression().fit(X, y)

    print("=== multivariate linear regression (normal equation) ===")
    print(f"true intercept : {true_b:+.3f}    learned: {model.intercept_:+.3f}")
    for i, (tw, lw) in enumerate(zip(true_w, model.coef_)):
        print(f"true w[{i}]      : {tw:+.3f}    learned: {lw:+.3f}")

    pred = model.predict(X)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    print(f"\nR^2 = {r2:.5f}")

    # predicted-vs-actual plot (optional; needs the 'demo' extra)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib for the plot: pip install -e '.[demo]')")
        return

    lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(y, pred, s=12, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], "r--", label="perfect")
    plt.xlabel("actual y")
    plt.ylabel("predicted y")
    plt.title(f"Linear regression fit (R² = {r2:.4f})")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig("regression.png", dpi=110)
    print("saved plot -> regression.png")


if __name__ == "__main__":
    main()

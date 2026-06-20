"""Demo: gradient-descent convergence at different learning rates.

Fits the same data with several learning rates and plots how the MSE loss
falls per iteration. Run: .venv/Scripts/python examples/gd_demo.py
"""
from __future__ import annotations

import numpy as np

from linalgml import GDRegressor

LEARNING_RATES = [0.01, 0.05, 0.1, 0.3]
N_ITERS = 200


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 4.0

    histories: dict[float, list[float]] = {}

    for lr in LEARNING_RATES:
        model = GDRegressor(lr=lr, n_iters=N_ITERS).fit(X, y)
        histories[lr] = model.loss_history
    
    # --- plotting ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib for the plot: pip install -e '.[demo]')")
        return

    plt.figure(figsize=(7, 5))
    for lr, hist in histories.items():
        plt.plot(hist, label=f"lr = {lr}")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("MSE (log scale)")
    plt.title("Gradient descent convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gd_convergence.png", dpi=110)
    print("saved plot -> gd_convergence.png")


if __name__ == "__main__":
    main()

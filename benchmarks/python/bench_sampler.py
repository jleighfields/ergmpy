"""Times the Gibbs sweep in pure Python and under numba, and checks correctness.

Correctness check: with theta_star2 = 0 the customers are independent, so each
customer's long-run marginal must equal the plain softmax of their attribute
utilities. Any error in the sweep's sampling shows up as drift from that.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ndcm import mple, sampler  # noqa: E402
from ndcm.predict import load, softmax_utilities  # noqa: E402


def setup(theta: np.ndarray):
    """Builds the flat arrays the sweep kernel operates on.

    Args:
        theta: (8,) parameter vector.

    Returns:
        choice_sets, current, degree, linear, theta_star2, design.
    """
    data = load(str(ROOT / "reference" / "Sampled_data_to_share.csv"))
    linear = data.design @ theta[:7]
    return (np.ascontiguousarray(data.choice_sets),
            data.chosen.copy().astype(np.int32),
            data.degree.copy(),
            linear,
            float(theta[7]),
            data.design)


def check_marginals(choice_sets, linear, n_sweeps: int = 2000) -> float:
    """Compares sampled marginals to the analytic ones when star2 is off.

    Args:
        choice_sets: (n_customers, set_size) product indices.
        linear: (n_products,) attribute utility per product.
        n_sweeps: Sweeps to average over.

    Returns:
        Largest absolute difference between sampled and analytic marginals.
    """
    n, k = choice_sets.shape
    subset = choice_sets[:400]
    current = subset[:, 0].copy().astype(np.int32)
    degree = np.bincount(current, minlength=len(linear)).astype(np.int64)
    counts = np.zeros((len(subset), k))
    np.random.seed(1)
    for _ in range(n_sweeps):
        sampler.run_sweeps(subset, current, degree, linear, 0.0, 1)
        counts += (subset == current[:, None])
    empirical = counts / n_sweeps
    analytic = softmax_utilities(linear[subset])
    return float(np.abs(empirical - analytic).max())


def main() -> None:
    """Fits MPLE for a realistic theta, then benchmarks both kernels."""
    data = load(str(ROOT / "reference" / "Sampled_data_to_share.csv"))
    theta = mple.fit(data).coef
    print(f"theta from MPLE: star2 = {theta[7]:.6f}\n")

    args = setup(theta)
    choice_sets, _, _, linear, theta_star2, _ = args

    print(f"marginal check (star2 = 0), max abs error: "
          f"{check_marginals(choice_sets, linear):.4f}\n")

    for label, kernel, sweeps in [("pure python", sampler.updates_python, 20),
                                  ("numba", sampler.updates_numba, 20)]:
        cs, current, degree, lin, ts, _ = setup(theta)
        if label == "numba":  # pay the compile once, outside the timed region
            t0 = time.perf_counter()
            sampler.run_sweeps(cs, current.copy(), degree.copy(), lin, ts, 1, kernel)
            print(f"numba compile + first call: {time.perf_counter() - t0:.2f} s")
        np.random.seed(7)
        t0 = time.perf_counter()
        sampler.run_sweeps(cs, current, degree, lin, ts, sweeps, kernel)
        elapsed = time.perf_counter() - t0
        per_sweep = elapsed / sweeps
        updates = choice_sets.shape[0] / per_sweep
        print(f"{label:>12}: {per_sweep * 1000:9.3f} ms/sweep   "
              f"{updates / 1e6:7.3f} M updates/s")


if __name__ == "__main__":
    main()

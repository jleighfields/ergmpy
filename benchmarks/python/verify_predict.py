"""Checks the Python choice probabilities against the R script's own output.

Uses the coefficients from the same saved ergm fit that produced R's
prob_y_star matrix, so any difference is implementation, not estimation.
"""

import time
from pathlib import Path

import numpy as np
import polars as pl

from ergmpy.choice.predict import (
    TERM_NAMES,
    choice_probabilities,
    load,
    top_n_accuracy,
)

ROOT = Path(__file__).resolve().parents[2]

R_RESULTS = ROOT / "results" / "r"
N_COMPARED = 200  # R's saved matrix only has rows for the customers it scored


def main() -> None:
    """Loads both sides, compares probabilities, and reports timings."""
    coefs = pl.read_csv(R_RESULTS / "coef_star_maxit2.csv")
    # The offset row is -Inf, so polars types the column as string; cast it.
    estimates = coefs["estimate"].cast(pl.Float64, strict=False).to_list()
    lookup = dict(zip(coefs["term"].to_list(), estimates, strict=True))
    # The seven linear terms; b2star2 is passed separately.
    theta_linear = np.array([lookup[t] for t in TERM_NAMES[:7]])
    theta_star2 = lookup["b2star2"]

    t0 = time.perf_counter()
    data = load(str(ROOT / "reference" / "test_data_to_share.csv"))
    load_s = time.perf_counter() - t0

    best = min(_timed(data, theta_linear, theta_star2) for _ in range(5))
    predict_s, probs = best

    # Most columns are all-zero for the scored rows, which defeats polars'
    # schema inference, so read the dense numeric matrix with numpy.
    r_probs = np.loadtxt(R_RESULTS / "prob_star_maxit2_n200.csv",
                         delimiter=",", skiprows=1)

    # R stores an (n_customers x n_products) matrix; gather the consideration-set
    # entries so the two are aligned the same way.
    rows = np.arange(N_COMPARED)[:, None]
    r_aligned = r_probs[rows, data.choice_sets[:N_COMPARED]]
    py_aligned = probs[:N_COMPARED]

    diff = np.abs(r_aligned - py_aligned)
    print(f"customers compared:    {N_COMPARED}")
    print(f"max abs difference:    {diff.max():.3e}")
    print(f"max rel difference:    {(diff / np.maximum(r_aligned, 1e-12)).max():.3e}")
    print(f"row sums (python):     min {probs.sum(axis=1).min():.12f} "
          f"max {probs.sum(axis=1).max():.12f}")
    print()
    print(f"python load:           {load_s * 1000:8.1f} ms")
    print(f"python predict (5000): {predict_s * 1000:8.1f} ms  (best of 5)")
    print(f"R predict (200):       {148950:8.1f} ms")
    print(f"R predict (5000, proj):{148950 * 25:8.1f} ms")
    print()
    for n in (1, 2, 3):
        print(f"top-{n} accuracy (5000 customers): {top_n_accuracy(data, probs, n):.4f}")


def _timed(data, theta_linear, theta_star2) -> tuple[float, np.ndarray]:
    t0 = time.perf_counter()
    p = choice_probabilities(data, theta_linear, theta_star2)
    return time.perf_counter() - t0, p


if __name__ == "__main__":
    main()

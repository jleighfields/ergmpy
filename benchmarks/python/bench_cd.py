"""Compares MCMLE seeded by pseudo-likelihood against MCMLE seeded by CD."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ndcm import cd, mcmle, mple  # noqa: E402
from ndcm.predict import load  # noqa: E402

R_COEF = np.array([-3.0451052, -0.0358261, 1.5947824, 1.2249765,
                   2.2061539, 1.2193177, 1.1847269, 0.0058188])


def distance(theta: np.ndarray) -> float:
    """Largest absolute deviation from ergm's converged estimates.

    Args:
        theta: (8,) parameter vector.

    Returns:
        The max absolute difference against R_COEF.
    """
    return float(np.abs(theta - R_COEF).max())


def main() -> None:
    """Times CD, then MCMLE from both starting points."""
    data = load(str(ROOT / "reference" / "Sampled_data_to_share.csv"))

    start = mple.fit(data).coef
    print(f"MPLE start          max|diff vs ergm| = {distance(start):.4f}  "
          f"b2star2 = {start[7]:.6f}")

    t0 = time.perf_counter()
    seed, history = cd.fit(data, start)
    cd_seconds = time.perf_counter() - t0
    print(f"CD seed ({cd_seconds:5.1f} s, {len(history)} iters)  "
          f"max|diff vs ergm| = {distance(seed):.4f}  b2star2 = {seed[7]:.6f}")
    print("  CD gap trace: " + " ".join(
        f"{h['max_standardized_gap']:.2f}" for h in history[:8]) + " ...")
    print()

    for label, theta0 in [("MPLE-seeded", start), ("CD-seeded", seed)]:
        t0 = time.perf_counter()
        result = mcmle.fit(data, theta0, max_iterations=120, n_draws=600,
                           burn_in=100, thin=30, tolerance=0.15)
        elapsed = time.perf_counter() - t0
        print(f"{label:>12} MCMLE: {result.n_iterations:3d} iterations, "
              f"{elapsed:6.1f} s, converged={result.converged}, "
              f"max|diff vs ergm| = {distance(result.coef):.5f}")


if __name__ == "__main__":
    main()

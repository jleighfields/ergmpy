"""Measures the full recipe: contrastive divergence as a seed, then MCMLE."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ergmpy import cd, mcmle
from ergmpy.choice import mple  # noqa: E402
from ergmpy.choice.predict import load  # noqa: E402

R_COEF = np.array([-3.0451052, -0.0358261, 1.5947824, 1.2249765,
                   2.2061539, 1.2193177, 1.1847269, 0.0058188])


def main() -> None:
    """Times MPLE, then CD, then MCMLE seeded by CD."""
    data = load(str(ROOT / "reference" / "Sampled_data_to_share.csv"))

    t0 = time.perf_counter()
    start = mple.fit(data).coef
    mple_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    seed, history = cd.fit(data, start, max_iterations=60, n_draws=300,
                           n_updates=50000)
    cd_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = mcmle.fit(data, seed, max_iterations=120, n_draws=600,
                       burn_in=100, thin=30, tolerance=0.15)
    mcmle_s = time.perf_counter() - t0

    print(f"MPLE          {mple_s:7.1f} s   max|diff| {np.abs(start - R_COEF).max():.5f}")
    print(f"CD (10 sweeps){cd_s:7.1f} s   max|diff| {np.abs(seed - R_COEF).max():.5f}")
    print(f"MCMLE         {mcmle_s:7.1f} s   max|diff| "
          f"{np.abs(result.coef - R_COEF).max():.5f}   "
          f"{result.n_iterations} iterations, converged={result.converged}")
    print(f"TOTAL         {mple_s + cd_s + mcmle_s:7.1f} s      "
          f"(ergm: 781.8 s on 4 cores)")
    print()
    print(result.summary())


if __name__ == "__main__":
    main()

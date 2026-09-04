"""Fits the star model under settings matched to the reference R script.

Writes results/python/matched_settings_fit.log. The settings, the stopping
rule and the objective all follow `ergm`; `docs/settings-comparison.md` records
what is matched and what is not.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

from ergmpy import contrastive_divergence as cd
from ergmpy import mcmle
from ergmpy.choice import mple
from ergmpy.choice.predict import load

ROOT = Path(__file__).resolve().parents[2]

# ergm's converged estimates at MCMLE.maxit = 200, from results/r/mcmle_star.csv.
ERGM = np.array([-3.0498114, -0.0381650, 1.5982802, 1.2313084,
                 2.2136847, 1.2216362, 1.1853674, 0.0057806])


def main() -> None:
    """Runs pseudo-likelihood, contrastive divergence, then MCMLE."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    data = load(str(ROOT / "reference" / "Sampled_data_to_share.csv"))

    started = time.perf_counter()
    start = mple.fit(data).coef
    mple_seconds = time.perf_counter() - started

    started = time.perf_counter()
    seed, _ = cd.fit(data, start, max_iterations=60, n_draws=300, n_updates=50000)
    cd_seconds = time.perf_counter() - started

    started = time.perf_counter()
    result = mcmle.fit(data, seed)
    mcmle_seconds = time.perf_counter() - started

    print(f"\nMPLE {mple_seconds:.1f}s | CD {cd_seconds:.1f}s | "
          f"MCMLE {mcmle_seconds:.1f}s | "
          f"total {mple_seconds + cd_seconds + mcmle_seconds:.1f}s")
    print(f"{result.n_iterations} iterations, converged {result.converged}, "
          f"{result.sweeps:,} sweeps drawn")
    print(f"largest disagreement with ergm: "
          f"{np.abs(result.coef - ERGM).max():.6f}")
    print("ergm's own star fit, for reference: 1027 s on 4 cores, "
          "converged after 34 iterations")
    print()
    print(result.summary())


if __name__ == "__main__":
    main()

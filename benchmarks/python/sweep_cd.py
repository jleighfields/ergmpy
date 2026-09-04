"""Sweeps the contrastive-divergence excursion length.

Too few updates and the excursions never leave the observed configuration, so
the gradient carries no signal; too many and CD approaches MCMLE and inherits
the instability it exists to avoid. This finds where the useful range is.
"""

import time
from pathlib import Path

import numpy as np

from ergmpy import contrastive_divergence as cd
from ergmpy.choice import mple
from ergmpy.choice.predict import load_choice_data

ROOT = Path(__file__).resolve().parents[2]

R_COEF = np.array([-3.0451052, -0.0358261, 1.5947824, 1.2249765,
                   2.2061539, 1.2193177, 1.1847269, 0.0058188])


def main() -> None:
    """Runs CD at several excursion lengths and reports the seed quality."""
    data = load_choice_data(ROOT / "reference" / "Sampled_data_to_share.csv")
    start = mple.fit(data).coef
    print(f"{'n_updates':>10}{'sweeps':>8}{'iters':>7}{'sec':>7}"
          f"{'max|diff|':>11}{'b2star2':>11}")
    print(f"{'(MPLE)':>10}{'-':>8}{'-':>7}{'-':>7}"
          f"{np.abs(start - R_COEF).max():>11.4f}{start[7]:>11.6f}")
    print(f"{'(ergm)':>10}{'-':>8}{'-':>7}{'-':>7}{0.0:>11.4f}{R_COEF[7]:>11.6f}")

    for n_updates in (500, 1250, 2500, 5000, 10000, 25000, 50000):
        t0 = time.perf_counter()
        with np.errstate(all="ignore"):
            seed, history = cd.fit(data, start, max_iterations=60,
                                   n_draws=300, n_updates=n_updates)
        elapsed = time.perf_counter() - t0
        print(f"{n_updates:>10}{n_updates / 5000:>8.1f}{len(history):>7}"
              f"{elapsed:>7.1f}{np.abs(seed - R_COEF).max():>11.4f}"
              f"{seed[7]:>11.6f}", flush=True)


if __name__ == "__main__":
    main()

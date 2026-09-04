"""Checks that the sampler reproduces the observed statistics at the published theta.

This is the strongest available check on the sampler: simulating at parameters
`ergm` converged to should recover the statistics of the network `ergm` was fit
to. It is a different check from the marginal comparison in `bench_sampler.py`,
which switches the dependence term off so the customers decouple and the answer
becomes analytic. Here the dependence is on, and there is no closed form -- the
reference is the observed network.

Writes results/python/sampler_at_published_theta.log.
"""

import time
from pathlib import Path

import numpy as np

from ergmpy.choice.predict import PUBLISHED_ESTIMATES, TERM_NAMES, load
from ergmpy.mcmle import observed_statistics, simulate

ROOT = Path(__file__).resolve().parents[2]

PUBLISHED = np.array(PUBLISHED_ESTIMATES)



def main() -> None:
    """Simulates at the published theta and reports the standardized gaps."""
    data = load(str(ROOT / "reference" / "Sampled_data_to_share.csv"))
    g_obs = observed_statistics(data)

    np.random.seed(123)
    started = time.perf_counter()
    draws, _ = simulate(data, PUBLISHED, n_draws=300, burn_in=100, thin=20)
    elapsed = time.perf_counter() - started

    mean = draws.mean(axis=0)
    spread = draws.std(axis=0)
    gap = np.abs(g_obs - mean) / np.where(spread > 1e-12, spread, np.inf)

    lines = [
        "Simulated at ergm's published star-model estimates, 300 draws, "
        "burn_in=100, thin=20.",
        f"Elapsed: {elapsed:.1f} s",
        "",
        f"{'term':<16}{'observed':>14}{'simulated mean':>16}"
        f"{'sd':>12}{'gap / sd':>11}",
    ]
    for name, observed, m, s, g in zip(TERM_NAMES, g_obs, mean, spread, gap, strict=True):
        lines.append(f"{name:<16}{observed:>14.1f}{m:>16.1f}{s:>12.1f}{g:>11.3f}")
    lines += ["", f"largest standardized gap: {gap.max():.3f} sd"]

    report = "\n".join(lines)
    print(report)
    out = ROOT / "results" / "python" / "sampler_at_published_theta.log"
    out.write_text(report + "\n")


if __name__ == "__main__":
    main()

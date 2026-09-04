"""Fits the star model under settings matched to the reference R script.

Prints to stdout; `results/python/matched_settings_fit.log` is that output
redirected, which is how the README runs it. The settings, the stopping rule
and the objective all follow `ergm`; `docs/settings-comparison.md` records what
is matched and what is not.
"""

import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

from ergmpy import contrastive_divergence as cd
from ergmpy import mcmle
from ergmpy.choice import mple
from ergmpy.choice.predict import TERM_NAMES, load

ROOT = Path(__file__).resolve().parents[2]

ERGM_COEFFICIENTS = ROOT / "results" / "r" / "mcmle_star.csv"
ERGM_METADATA = ROOT / "results" / "r" / "fit_metadata.csv"


def ergm_reference() -> tuple[np.ndarray, dict[str, str]]:
    """Reads ergm's converged fit and how it was run.

    Both files are written by benchmarks/r/export_fits.R from the fitted
    object, so nothing here transcribes a number.

    Returns:
        The (8,) coefficients in TERM_NAMES order, and the metadata row.

    Raises:
        FileNotFoundError: If either is missing, rather than comparing against
            nothing.
    """
    for path in (ERGM_COEFFICIENTS, ERGM_METADATA):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run benchmarks/r/export_fits.R"
            )
    estimates = {}
    for row in csv.DictReader(ERGM_COEFFICIENTS.open()):
        try:
            estimates[row["term"]] = float(row["estimate"])
        except ValueError:
            continue
    metadata = next(row for row in csv.DictReader(ERGM_METADATA.open())
                    if row["fit"] == "mcmle_star")
    return np.array([estimates[t] for t in TERM_NAMES]), metadata


def main() -> None:
    """Runs pseudo-likelihood, contrastive divergence, then MCMLE."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ergm, reference = ergm_reference()
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
          f"{np.abs(result.coef - ergm).max():.6f}")
    seconds = reference["seconds"]
    print(f"ergm's own star fit, for reference: "
          f"{'unrecorded' if seconds == 'NA' else format(float(seconds), '.0f') + ' s'}"
          f" on 4 cores, {reference['iterations']} iterations, "
          f"converged {reference['converged']}")
    print()
    print(result.summary())


if __name__ == "__main__":
    main()

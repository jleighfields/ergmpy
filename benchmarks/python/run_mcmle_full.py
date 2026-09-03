"""Fits the star model by MCMLE from the pseudo-likelihood start and reports."""

import time
from pathlib import Path

import numpy as np

from ergmpy import mcmle
from ergmpy.choice import mple
from ergmpy.choice.predict import choice_probabilities, load, top_n_accuracy

ROOT = Path(__file__).resolve().parents[2]

R_COEF = np.array([-3.0451052, -0.0358261, 1.5947824, 1.2249765,
                   2.2061539, 1.2193177, 1.1847269, 0.0058188])
R_SE = np.array([0.1007882, 0.0483922, 0.0727808, 0.0601395,
                 0.0721820, 0.0639832, 0.0790112, 0.0001343])


def main() -> None:
    """Runs the fit and prints a term-by-term comparison against ergm."""
    data = load(str(ROOT / "reference" / "Sampled_data_to_share.csv"))
    start = mple.fit(data).coef

    t0 = time.perf_counter()
    result = mcmle.fit(data, start, max_iterations=120, n_draws=600,
                       burn_in=100, thin=30, tolerance=0.15)
    elapsed = time.perf_counter() - t0

    print(f"===== Python MCMLE — {elapsed:.1f} s, {result.n_iterations} iterations, "
          f"converged={result.converged} =====")
    print(result.summary())
    print("\nlast 10 iterations:")
    for h in result.history[-10:]:
        print(f"  iter {h['iteration']:3d}  step {h['step_length']:.4f}  "
              f"gap {h['max_standardized_gap']:8.3f}")

    print(f"\n{'term':<16}{'python':>12}{'ergm':>12}{'diff':>10}"
          f"{'py SE':>11}{'ergm SE':>11}")
    for name, c, s, rc, rs in zip(mcmle.TERM_NAMES, result.coef,
                                  result.std_error, R_COEF, R_SE, strict=True):
        print(f"{name:<16}{c:>12.5f}{rc:>12.5f}{c - rc:>10.5f}{s:>11.5f}{rs:>11.5f}")

    probabilities = choice_probabilities(data, result.coef[:7], result.coef[7])
    acc = [top_n_accuracy(data, probabilities, n) for n in (1, 2, 3)]
    print(f"\ntop-1/2/3 accuracy: {acc[0]:.4f} / {acc[1]:.4f} / {acc[2]:.4f}")


if __name__ == "__main__":
    main()

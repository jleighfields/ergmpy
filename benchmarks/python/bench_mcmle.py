"""Fits the star model by MCMLE in Python and compares against ergm."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ndcm import mcmle, mple  # noqa: E402
from ndcm.predict import choice_probabilities, load, top_n_accuracy  # noqa: E402

PUBLISHED = np.array([-3.0567573, -0.0363712, 1.6013929, 1.2357093,
                      2.2218421, 1.2257136, 1.1918759, 0.0057696])
PUBLISHED_SE = np.array([0.0989747, 0.0476930, 0.0713491, 0.0604189,
                         0.0716060, 0.0616764, 0.0769932, 0.0001338])


def main() -> None:
    """Runs MPLE then MCMLE on the training data and prints the comparison."""
    data = load(str(ROOT / "reference" / "Sampled_data_to_share.csv"))

    start = mple.fit(data)
    print(f"MPLE start: {np.array2string(start.coef, precision=4)}\n")

    t0 = time.perf_counter()
    result = mcmle.fit(data, start.coef, max_iterations=30)
    elapsed = time.perf_counter() - t0

    print(f"===== Python MCMLE — {elapsed:.1f} s =====")
    print(result.summary())
    print("\niteration  step_length  max_standardized_gap")
    for h in result.history:
        print(f"{h['iteration']:>9}  {h['step_length']:>11.4f}  "
              f"{h['max_standardized_gap']:>20.4f}")

    print(f"\n{'term':<16}{'python':>12}{'ergm pub':>12}{'py SE':>11}{'ergm SE':>11}")
    for name, c, s, pc, ps in zip(mcmle.TERM_NAMES, result.coef, result.std_error,
                                  PUBLISHED, PUBLISHED_SE):
        print(f"{name:<16}{c:>12.5f}{pc:>12.5f}{s:>11.5f}{ps:>11.5f}")

    probabilities = choice_probabilities(data, result.coef[:7], result.coef[7])
    acc = [top_n_accuracy(data, probabilities, n) for n in (1, 2, 3)]
    print(f"\ntop-1/2/3 accuracy: {acc[0]:.4f} / {acc[1]:.4f} / {acc[2]:.4f}")


if __name__ == "__main__":
    main()

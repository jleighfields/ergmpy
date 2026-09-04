"""Fits the model by pseudo-likelihood and checks it against the R baselines.

Also verifies the analytic Hessian against a central-difference approximation,
since the standard errors depend entirely on it.
"""

import time
from pathlib import Path

import numpy as np
from published import published_estimates

from ergmpy.choice import mple
from ergmpy.choice.predict import (
    TERM_NAMES,
    choice_probabilities,
    load,
    top_n_accuracy,
)

ROOT = Path(__file__).resolve().parents[2]

PUBLISHED = published_estimates()


def check_hessian(result, data) -> float:
    """Compares the analytic observed information to central differences.

    Args:
        result: The fitted MPLEResult.
        data: The dataset it was fitted on.

    Returns:
        The largest relative discrepancy across all entries.
    """
    from ergmpy.choice.predict import change_statistics
    Z = change_statistics(data)
    slot = np.argmax(data.choice_sets == data.chosen[:, None], axis=1)
    step = 1e-5
    numeric = np.empty_like(result.hessian)
    for i in range(len(result.coef)):
        up, down = result.coef.copy(), result.coef.copy()
        up[i] += step
        down[i] -= step
        _, g_up = mple.negative_log_pseudo_likelihood(up, Z, slot)
        _, g_down = mple.negative_log_pseudo_likelihood(down, Z, slot)
        numeric[i] = (g_up - g_down) / (2 * step)
    scale = np.maximum(np.abs(result.hessian), 1.0)
    return float((np.abs(numeric - result.hessian) / scale).max())


def main() -> None:
    """Fits both datasets and prints the comparison tables."""
    for tag, filename in [("train", "Sampled_data_to_share.csv"),
                          ("test", "test_data_to_share.csv")]:
        data = load(str(ROOT / "reference" / filename))
        t0 = time.perf_counter()
        result = mple.fit(data)
        elapsed = time.perf_counter() - t0

        print(f"===== Python MPLE on {tag} — {elapsed * 1000:.1f} ms, "
              f"{result.n_iterations} iterations =====")
        print(result.summary())
        print(f"analytic vs finite-difference Hessian, max rel diff: "
              f"{check_hessian(result, data):.2e}")

        if tag == "train":
            print(f"\n{'term':<16}{'python MPLE':>14}{'ergm MCMLE pub':>16}{'ratio':>9}")
            for name, est in zip(TERM_NAMES, result.coef, strict=True):
                pub = PUBLISHED[name]
                print(f"{name:<16}{est:>14.6f}{pub:>16.7f}{est / pub:>9.3f}")

        probabilities = choice_probabilities(data, result.coef[:7], result.coef[7])
        acc = [top_n_accuracy(data, probabilities, n) for n in (1, 2, 3)]
        print(f"\ntop-1/2/3 accuracy on {tag}: "
              f"{acc[0]:.4f} / {acc[1]:.4f} / {acc[2]:.4f}\n")


if __name__ == "__main__":
    main()

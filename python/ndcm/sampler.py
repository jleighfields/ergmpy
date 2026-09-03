"""Gibbs sampler over the constrained sample space.

The b1degrees constraint plus the -Inf consideration-set offset make the sample
space a product of independent per-customer choice sets: each customer holds
exactly one purchase, drawn from their own six alternatives. Sampling from the
ERGM is therefore a Gibbs sweep in which each customer resamples their single
purchase from the conditional multinomial, with every other customer held fixed.

That conditional is the same softmax over change statistics the prediction and
pseudo-likelihood code uses. Only two product degrees move per update, so a
step costs O(set_size) rather than a network traversal, and no general ERGM
tie-toggling machinery is needed.

Both implementations here are deliberately the same code. `sweep_python` is the
readable reference; `sweep_numba` is the identical function under @njit, so the
two cannot drift and either can serve as the oracle for the other.
"""

import math

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only where numba is absent
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        """Falls back to the undecorated function when numba is unavailable."""
        def wrap(fn):
            return fn
        return wrap if not args else args[0]


def gibbs_sweep(choice_sets: np.ndarray, current: np.ndarray, degree: np.ndarray,
                linear: np.ndarray, theta_star2: float, n_sweeps: int) -> None:
    """Runs Gibbs sweeps in place over every customer.

    Args:
        choice_sets: (n_customers, set_size) product indices.
        current: (n_customers,) current purchase per customer; updated in place.
        degree: (n_products,) purchase counts; kept consistent with `current`.
        linear: (n_products,) precomputed attribute utility per product.
        theta_star2: Coefficient on the b2star2 statistic.
        n_sweeps: Number of full passes over all customers.
    """
    n_customers, set_size = choice_sets.shape
    weights = np.empty(set_size)
    for _ in range(n_sweeps):
        for i in range(n_customers):
            k = current[i]
            degree[k] -= 1

            largest = -1.0e308
            for s in range(set_size):
                j = choice_sets[i, s]
                u = linear[j] + theta_star2 * degree[j]
                weights[s] = u
                if u > largest:
                    largest = u

            total = 0.0
            for s in range(set_size):
                weights[s] = math.exp(weights[s] - largest)
                total += weights[s]

            target = np.random.random() * total
            cumulative = 0.0
            picked = set_size - 1
            for s in range(set_size):
                cumulative += weights[s]
                if target <= cumulative:
                    picked = s
                    break

            j = choice_sets[i, picked]
            current[i] = j
            degree[j] += 1


sweep_python = gibbs_sweep
sweep_numba = njit(cache=True, fastmath=False)(gibbs_sweep)


def network_statistics(choice_sets: np.ndarray, current: np.ndarray,
                       design: np.ndarray, n_products: int) -> np.ndarray:
    """Computes the full statistic vector of one sampled configuration.

    Args:
        choice_sets: Unused; present so callers can pass the same arguments.
        current: (n_customers,) purchase per customer.
        design: (n_products, 7) product design matrix.
        n_products: Number of products.

    Returns:
        (8,) statistics: the seven attribute sums, then b2star2.
    """
    degree = np.bincount(current, minlength=n_products)
    attribute_sums = design[current].sum(axis=0)
    star2 = (degree * (degree - 1) // 2).sum()
    return np.concatenate([attribute_sums, [float(star2)]])

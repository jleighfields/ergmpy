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

`updates_python` and `updates_numba` are built from one source definition --
`gibbs_updates` bare, and the same function passed through @njit -- so the
readable reference and the compiled kernel cannot drift, and either can serve
as the oracle for the other. Setting NUMBA_DISABLE_JIT=1 runs the compiled
path as plain Python.

The utility computed below is the same change statistic that
`predict.change_statistics` builds, written out inline as a scalar expression:
a numba kernel cannot call that array-based function. That makes this the
second copy of the formula, and the two must be changed together.
"""

import math
from collections.abc import Callable

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only where numba is absent
    HAVE_NUMBA = False

    def njit(*args: object, **kwargs: object) -> Callable:
        """Falls back to the undecorated function when numba is unavailable.

        Args:
            *args: The decorated function, when used bare as `@njit`.
            **kwargs: Decorator options, all ignored.

        Returns:
            The function itself, or a decorator returning it.
        """
        def wrap(fn: Callable) -> Callable:
            return fn
        return wrap if not args else args[0]


def gibbs_updates(choice_sets: np.ndarray, current: np.ndarray, degree: np.ndarray,
                  linear: np.ndarray, theta_star2: float,
                  customers: np.ndarray) -> None:
    """Resamples the listed customers' purchases in place, in the order given.

    Taking the customers to visit as an argument lets a full sweep and a short
    contrastive-divergence excursion share one kernel: a sweep passes every
    customer once, CD passes a random handful.

    Args:
        choice_sets: (n_customers, set_size) product indices.
        current: (n_customers,) current purchase per customer; updated in place.
        degree: (n_products,) purchase counts; kept consistent with `current`.
        linear: (n_products,) precomputed attribute utility per product.
        theta_star2: Coefficient on the b2star2 statistic.
        customers: Indices of the customers to resample, visited in order.
    """
    set_size = choice_sets.shape[1]
    weights = np.empty(set_size)
    for idx in range(len(customers)):
        i = customers[idx]
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


updates_python = gibbs_updates
updates_numba = njit(cache=True, fastmath=False)(gibbs_updates)


def run_sweeps(choice_sets: np.ndarray, current: np.ndarray, degree: np.ndarray,
               linear: np.ndarray, theta_star2: float, n_sweeps: int,
               kernel: Callable = updates_numba) -> None:
    """Passes over every customer `n_sweeps` times.

    Args:
        choice_sets: (n_customers, set_size) product indices.
        current: (n_customers,) current purchase per customer; updated in place.
        degree: (n_products,) purchase counts; kept consistent with `current`.
        linear: (n_products,) precomputed attribute utility per product.
        theta_star2: Coefficient on the b2star2 statistic.
        n_sweeps: Number of full passes.
        kernel: Which kernel to call, `updates_numba` or `updates_python`.
    """
    order = np.arange(choice_sets.shape[0], dtype=np.int32)
    for _ in range(n_sweeps):
        kernel(choice_sets, current, degree, linear, theta_star2, order)


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

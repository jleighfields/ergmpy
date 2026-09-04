"""Compares the numba kernel against the vectorised NumPy alternatives.

The sweep cannot be vectorised across customers: each update reads product
degrees the previous update changed. It can be vectorised two other ways, and
this measures both, which is where the README's tooling table comes from.

Vectorising one customer's alternatives turns out slower than plain Python,
because the arrays are six elements long and NumPy's per-call overhead exceeds
what the interpreter was costing. Vectorising across chains does work -- the
dependency is between customers, not chains -- and overtakes numba somewhere
past a hundred chains, far more than this package's sampling design uses.
"""

import time
from pathlib import Path

import numpy as np

from ergmpy import sampler
from ergmpy.choice.predict import load_choice_data

ROOT = Path(__file__).resolve().parents[2]
STAR2 = 0.01


def time_numba(data, choice_sets, linear, order, n_sweeps: int) -> float:
    """Times the compiled scalar kernel.

    Args:
        data: The dataset.
        choice_sets: (n_customers, set_size) product indices.
        linear: (n_products,) attribute utility per product.
        order: Customers to visit, in order.
        n_sweeps: Sweeps to average over.

    Returns:
        Milliseconds per sweep.
    """
    current = data.chosen.astype(np.int32).copy()
    degree = np.bincount(current, minlength=data.n_products).astype(np.int64)
    started = time.perf_counter()
    for _ in range(n_sweeps):
        sampler.updates_numba(choice_sets, current, degree, linear, STAR2, order)
    return (time.perf_counter() - started) / n_sweeps * 1000


def time_numpy_per_customer(data, choice_sets, linear, n_sweeps: int) -> float:
    """Times the obvious vectorisation: a customer's alternatives as an array.

    Args:
        data: The dataset.
        choice_sets: (n_customers, set_size) product indices.
        linear: (n_products,) attribute utility per product.
        n_sweeps: Sweeps to average over.

    Returns:
        Milliseconds per sweep.
    """
    n_customers, set_size = choice_sets.shape
    current = data.chosen.astype(np.int32).copy()
    degree = np.bincount(current, minlength=data.n_products).astype(np.int64)
    generator = np.random.default_rng(0)
    started = time.perf_counter()
    for _ in range(n_sweeps):
        for i in range(n_customers):
            k = current[i]
            degree[k] -= 1
            alternatives = choice_sets[i]
            utility = linear[alternatives] + STAR2 * degree[alternatives]
            utility = utility - utility.max()
            weights = np.exp(utility)
            weights = weights / weights.sum()
            j = alternatives[generator.choice(set_size, p=weights)]
            current[i] = j
            degree[j] += 1
    return (time.perf_counter() - started) / n_sweeps * 1000


def time_numpy_across_chains(data, choice_sets, linear, n_chains: int,
                             n_sweeps: int) -> float:
    """Times one array operation per customer-step, spanning every chain.

    Args:
        data: The dataset.
        choice_sets: (n_customers, set_size) product indices.
        linear: (n_products,) attribute utility per product.
        n_chains: Chains advanced together.
        n_sweeps: Sweeps to average over.

    Returns:
        Milliseconds per sweep of equivalent work, since `n_chains` sweeps
        happen at once.
    """
    n_customers, set_size = choice_sets.shape
    current = np.tile(data.chosen.astype(np.int64), (n_chains, 1))
    degree = np.stack([
        np.bincount(chain, minlength=data.n_products) for chain in current
    ]).astype(np.int64)
    rows = np.arange(n_chains)
    generator = np.random.default_rng(0)

    started = time.perf_counter()
    for _ in range(n_sweeps):
        for i in range(n_customers):
            degree[rows, current[:, i]] -= 1
            alternatives = choice_sets[i]
            utility = linear[alternatives] + STAR2 * degree[:, alternatives]
            utility = utility - utility.max(axis=1, keepdims=True)
            weights = np.exp(utility)
            weights = weights / weights.sum(axis=1, keepdims=True)
            drawn = (weights.cumsum(axis=1)
                     < generator.random((n_chains, 1))).sum(axis=1)
            j = alternatives[np.minimum(drawn, set_size - 1)]
            current[:, i] = j
            degree[rows, j] += 1
    return (time.perf_counter() - started) / (n_sweeps * n_chains) * 1000


def main() -> None:
    """Runs each formulation and prints milliseconds per sweep."""
    data = load_choice_data(ROOT / "reference" / "Sampled_data_to_share.csv")
    choice_sets = np.ascontiguousarray(data.choice_sets)
    linear = np.ascontiguousarray(data.design @ np.full(7, 0.3))
    order = np.arange(choice_sets.shape[0], dtype=np.int32)

    print(f"{'numba (compiled scalar loop)':<40}"
          f"{time_numba(data, choice_sets, linear, order, 200):8.3f} ms/sweep")
    print(f"{'numpy, vectorised per customer':<40}"
          f"{time_numpy_per_customer(data, choice_sets, linear, 1):8.3f} ms/sweep")
    for n_chains in (16, 64, 256):
        label = f"numpy, vectorised across {n_chains} chains"
        print(f"{label:<40}"
              f"{time_numpy_across_chains(data, choice_sets, linear, n_chains, 1):8.3f}"
              " ms/sweep-equivalent")


if __name__ == "__main__":
    main()

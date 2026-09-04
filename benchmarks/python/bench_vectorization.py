"""Compares numba against the vectorised NumPy formulations that are possible."""
import time

import numpy as np

from ergmpy import sampler
from ergmpy.choice.predict import load

d = load('reference/Sampled_data_to_share.csv')
cs = np.ascontiguousarray(d.choice_sets)
lin = np.ascontiguousarray(d.design @ np.full(7, 0.3))
order = np.arange(cs.shape[0], dtype=np.int32)
n_cust, set_size = cs.shape
star2 = 0.01


def numba_sweeps(n):
    cur = d.chosen.astype(np.int32).copy()
    deg = np.bincount(cur, minlength=d.n_products).astype(np.int64)
    t0 = time.perf_counter()
    for _ in range(n):
        sampler.updates_numba(cs, cur, deg, lin, star2, order)
    return (time.perf_counter() - t0) / n * 1000


def numpy_per_customer(n):
    """Vectorise the 6 alternatives; still a Python loop over customers."""
    cur = d.chosen.astype(np.int32).copy()
    deg = np.bincount(cur, minlength=d.n_products).astype(np.int64)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(n):
        for i in range(n_cust):
            k = cur[i]; deg[k] -= 1
            alts = cs[i]
            u = lin[alts] + star2 * deg[alts]
            u -= u.max()
            p = np.exp(u); p /= p.sum()
            j = alts[rng.choice(set_size, p=p)]
            cur[i] = j; deg[j] += 1
    return (time.perf_counter() - t0) / n * 1000


def numpy_across_chains(n_chains, n_sweeps):
    """One array op per customer-step, spanning every chain at once."""
    cur = np.tile(d.chosen.astype(np.int64), (n_chains, 1))
    deg = np.stack([np.bincount(c, minlength=d.n_products) for c in cur]).astype(np.int64)
    rows = np.arange(n_chains)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(n_sweeps):
        for i in range(n_cust):
            k = cur[:, i]
            deg[rows, k] -= 1
            alts = cs[i]                        # (set_size,) same for every chain
            u = lin[alts] + star2 * deg[:, alts]     # (n_chains, set_size)
            u -= u.max(axis=1, keepdims=True)
            p = np.exp(u); p /= p.sum(axis=1, keepdims=True)
            draw = (p.cumsum(axis=1) < rng.random((n_chains, 1))).sum(axis=1)
            j = alts[np.minimum(draw, set_size - 1)]
            cur[:, i] = j
            deg[rows, j] += 1
    # Cost per sweep of equivalent work: n_chains sweeps happened at once.
    return (time.perf_counter() - t0) / (n_sweeps * n_chains) * 1000


print(f"  {'numba (compiled scalar loop)':<38} {numba_sweeps(200):8.3f} ms/sweep")
print(f"  {'numpy, vectorised per customer':<38} {numpy_per_customer(1):8.3f} ms/sweep")
for c in (16, 64, 256):
    print(f"  {'numpy, vectorised across ' + str(c) + ' chains':<38} "
          f"{numpy_across_chains(c, 1):8.3f} ms/sweep-equivalent")

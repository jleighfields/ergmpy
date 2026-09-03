"""Convex-hull shrink factor by linear programming, matching ergm's shrink_into_CH.

ergm uses this inside .Hummel.steplength: given simulated network statistics M
and an observed vector p, it finds the largest gamma such that gamma * p stays
inside the convex hull of M. The test is the separating-hyperplane dual --
minimize x'z subject to Mz >= -1 with z free -- and gamma = |-1/optimum|.
A bounded optimum means a separating hyperplane exists, so p lies outside and
the step must be shortened.

References:
    Hummel, R.M., Hunter, D.R. and Handcock, M.S. (2012). Improving
        simulation-based algorithms for fitting ERGMs. Journal of
        Computational and Graphical Statistics, 21(4), 920-939.
    The function reimplemented here is ergm's `shrink_into_CH`, from the
        Statnet Project's ergm package.
"""

import numpy as np
from scipy.optimize import linprog


def shrink_into_ch(p: np.ndarray, M: np.ndarray, m: np.ndarray | None = None) -> float:
    """Finds the largest scaling of p that stays inside the convex hull of M.

    Args:
        p: Test point(s), shape (d,) or (np, d).
        M: Sampled points, shape (n, d).
        m: Centering vector; defaults to the column means of M, as ergm does.

    Returns:
        The shrink factor. Values above 1 mean p is strictly interior; ergm
        does not clamp here, and neither does this, so the two agree exactly.
    """
    p = np.atleast_2d(np.asarray(p, dtype=float))
    M = np.asarray(M, dtype=float)
    if m is None:
        m = M.mean(axis=0)
    p = p - m
    M = M - m
    n, d = M.shape

    g = np.inf
    for x in p:
        # ergm skips test points that are numerically at the centroid.
        if np.all(np.abs(x) <= np.sqrt(np.finfo(float).eps)):
            continue
        # linprog states constraints as A_ub @ z <= b_ub, so negate M z >= -1.
        res = linprog(c=x, A_ub=-M, b_ub=np.ones(n), bounds=[(None, None)] * d,
                      method="highs")
        if not res.success:
            # An unbounded objective means no separating hyperplane: p is interior.
            if res.status == 3:
                return 1.0
            raise RuntimeError(f"LP failed: {res.message}")
        g = min(g, abs(-1.0 / res.fun)) if res.fun != 0 else 0.0
    return float(g)

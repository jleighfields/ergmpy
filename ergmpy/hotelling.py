"""ergm's convergence test, ported.

`ergm` terminates on `MCMLE.termination = "confidence"`. The test asks whether
the gap between observed and simulated statistics is *significantly inside* a
tolerance region -- not whether it is indistinguishable from zero. The null is
that the fit has not converged, so a noisier sample raises the p-value and
`ergm` refuses; a test of "indistinguishable from zero" would do the opposite
and let an under-sampled iteration declare success.

The pieces, as `ergm` names them:

    Vm    = MCMLE.MCMC.precision * cov(esteq)   the tolerance ellipsoid
    d2e   = estdiff' Vm^+ estdiff               must be < 1 to test at all
    T2    = the squared distance from estdiff to the nearest point on Vm's
            boundary, measured in the metric of estcov, the covariance of the
            mean
    pval  = P(T2 exceeded), converge when it falls below 1 - MCMLE.confidence

For this model the estimating equations are the statistics themselves: the
model is linear in them with no curved terms, so `esteq` is the draw matrix.

Ported from `ergm:::ergm.MCMLE` and the helpers `xTAx_seigen`, `xTAx_eigen`,
`.ellipsoid_mahalanobis`, `.ptsq`, in ergm 4.12.0.
"""

import numpy as np
import scipy.optimize
import scipy.stats

TOLERANCE = np.sqrt(np.finfo(float).eps)


def quadratic_form(x: np.ndarray, a: np.ndarray,
                   tol: float = TOLERANCE) -> tuple[float, int]:
    """Computes `x' A^+ x` through a symmetric eigendecomposition.

    Ports `xTAx_eigen`. Directions where `A` has no variance are dropped rather
    than inverted, and `x` must lie in what remains -- a component of `x`
    outside `A`'s span means the two describe different spaces, which `ergm`
    raises on.

    Args:
        x: Vector to measure.
        a: Symmetric matrix whose pseudo-inverse defines the metric.
        tol: Relative threshold below which an eigenvalue counts as zero.

    Returns:
        The quadratic form, and the nullity -- how many directions were
        dropped.

    Raises:
        ValueError: If `x` has a component outside `a`'s span.
    """
    values, vectors = np.linalg.eigh(a)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    keep = values > max(tol * values[0], 0.0)
    projected = vectors.T @ x
    if not keep.all() and not np.all(np.abs(projected[~keep]) < tol):
        raise ValueError("x is not in the span of A")

    kept = projected[keep]
    return float(kept @ (kept / values[keep])), int((~keep).sum())


def standardized_quadratic_form(x: np.ndarray, a: np.ndarray,
                                tol: float = TOLERANCE) -> tuple[float, int]:
    """Computes `x' A^+ x` after putting `A` on the correlation scale.

    Ports `xTAx_seigen`. Rescaling by the inverse square root of the diagonal
    before the eigendecomposition keeps statistics of wildly different
    magnitude from dominating which directions survive the rank cut -- here
    b2star2 runs to 3e5 while the attribute sums are near 3e3.

    Args:
        x: Vector to measure.
        a: Symmetric matrix whose pseudo-inverse defines the metric.
        tol: Relative threshold below which an eigenvalue counts as zero.

    Returns:
        The quadratic form, and the nullity.
    """
    diagonal = np.diag(a).astype(float).copy()
    scale = np.where(diagonal > 0, 1.0 / np.sqrt(np.where(diagonal > 0,
                                                          diagonal, 1.0)), 0.0)
    return quadratic_form(x * scale, a * np.outer(scale, scale), tol)


def ellipsoid_mahalanobis(y: np.ndarray, w: np.ndarray, u: np.ndarray,
                          tol: float = TOLERANCE) -> tuple[float, int]:
    """Squared distance from `y` to the boundary of ellipsoid `u`, in metric `w`.

    Ports `.ellipsoid_mahalanobis`. The nearest boundary point solves
    `x(l) = (I + l W U^+)^-1 y` for the `l` that puts `x` on the boundary,
    which is a one-dimensional root find over `l` in `(-1/max eigenvalue, 0)`.
    The answer is how far inside the tolerance region the gap sits, measured in
    units of Monte Carlo error -- so a precise sample makes this large and a
    noisy one makes it small.

    Args:
        y: The gap, which must lie inside the ellipsoid.
        w: Covariance defining the metric distance is measured in.
        u: Matrix defining the ellipsoid `{x : x' U^+ x = 1}`.
        tol: Relative threshold below which an eigenvalue counts as zero.

    Returns:
        The squared distance, and the nullity of the metric.

    Raises:
        ValueError: If `y` is not strictly inside the ellipsoid, which is the
            case `ergm` checks with `d2e < 1` before calling this.
    """
    y = np.asarray(y, dtype=float).ravel()
    if standardized_quadratic_form(y, u, tol)[0] >= 1.0:
        raise ValueError("point is not in the interior of the ellipsoid")

    identity = np.eye(len(y))
    scaled = (np.linalg.pinv(u, rcond=tol) @ w).T

    def boundary_point(multiplier: float) -> np.ndarray:
        return np.linalg.solve(identity + multiplier * scaled, y)

    def on_boundary(multiplier: float) -> float:
        return standardized_quadratic_form(boundary_point(multiplier), u, tol)[0] - 1.0

    largest = np.real(np.linalg.eigvals(scaled)).max()
    lower = -1.0 / largest if largest > 0 else -1e12
    # The root is approached from inside the interval; nudge off the pole.
    multiplier = scipy.optimize.brentq(on_boundary, lower * (1 - 1e-9), 0.0,
                                       xtol=1e-14)
    return standardized_quadratic_form(y - boundary_point(multiplier), w, tol)


def nonconvergence_pvalue(t_squared: float, n_parameters: int,
                          df: float) -> float:
    """Upper-tail probability of Hotelling's T-squared.

    Ports `.ptsq(..., lower.tail = FALSE)`. Small values mean the gap is
    confidently inside the tolerance region, which is when `ergm` stops.

    Args:
        t_squared: The test statistic.
        n_parameters: Free parameters, after removing any nullity.
        df: Degrees of freedom, from the effective sample size.

    Returns:
        The probability of a larger statistic under non-convergence.
    """
    if n_parameters < 1 or df - n_parameters + 1 <= 0:
        return 1.0
    f_statistic = t_squared * (df - n_parameters + 1) / (n_parameters * df)
    return float(scipy.stats.f.sf(f_statistic, n_parameters,
                                  df - n_parameters + 1))

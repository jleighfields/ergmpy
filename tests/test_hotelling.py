"""ergm's confidence-test helpers, checked where the answer is available exactly.

Every check here avoids sampling. The ellipsoid cases are ones whose nearest
boundary point can be written down, the rank cases have a null direction put
there on purpose, and the tail is compared against `scipy.stats.f` rather than
against a simulated distribution. A test phrased as agreement within Monte
Carlo error would pass for any divergence smaller than its tolerance, and this
module is what every convergence claim in the package rests on.
"""

import numpy as np
import pytest
import scipy.stats

from ergmpy.convergence import batch_means_covariance
from ergmpy.hotelling import (
    ellipsoid_mahalanobis,
    nonconvergence_pvalue,
    quadratic_form,
    standardized_quadratic_form,
)


def test_the_distance_is_measured_to_the_boundary_not_to_the_point() -> None:
    """The statistic is how far inside the region the gap sits, not how far out.

    With the metric the identity and the ellipsoid the sphere of radius r, the
    nearest boundary point to `a * e1` is `r * e1`, so the squared distance is
    exactly `(r - a) ** 2`. Reporting the distance from the origin instead --
    `a ** 2` -- gives a different number for every a but is still monotone in
    the gap, so a test that only checked the direction would not see it.
    """
    radius, a = 2.0, 0.5
    metric = np.eye(3)
    ellipsoid = radius**2 * np.eye(3)
    y = np.array([a, 0.0, 0.0])

    distance, nullity = ellipsoid_mahalanobis(y, metric, ellipsoid)

    assert nullity == 0
    assert distance == pytest.approx((radius - a) ** 2, rel=1e-8)
    assert distance != pytest.approx(a**2, rel=1e-3)


def test_the_distance_shrinks_as_the_gap_approaches_the_boundary() -> None:
    """Sitting further out leaves less room, down to nothing at the boundary."""
    metric, ellipsoid = np.eye(2), 4.0 * np.eye(2)
    distances = [ellipsoid_mahalanobis(np.array([a, 0.0]), metric, ellipsoid)[0]
                 for a in (0.2, 0.8, 1.4, 1.9)]
    assert distances == sorted(distances, reverse=True)
    assert distances[-1] == pytest.approx((2.0 - 1.9) ** 2, rel=1e-6)


def test_a_point_on_or_outside_the_boundary_is_refused() -> None:
    """There is no distance-to-boundary to measure once the gap is not inside.

    `ergm` guards this with `d2e < 1` before calling, and refuses rather than
    returning a number the root find cannot produce.
    """
    metric, ellipsoid = np.eye(2), np.eye(2)
    with pytest.raises(ValueError):
        ellipsoid_mahalanobis(np.array([1.0, 0.0]), metric, ellipsoid)
    with pytest.raises(ValueError):
        ellipsoid_mahalanobis(np.array([3.0, 0.0]), metric, ellipsoid)


def test_a_direction_with_no_variance_is_dropped_and_counted() -> None:
    """A null direction is dropped from the metric, not inverted.

    The matrix here has variance in two coordinates and none in the third, so
    the form must equal the two-dimensional one and report a nullity of one.
    Inverting the null direction would divide by zero; keeping it as though it
    had unit variance would add a term the two-dimensional answer does not.
    """
    a = np.diag([4.0, 1.0, 0.0])
    x = np.array([2.0, 3.0, 0.0])

    value, nullity = quadratic_form(x, a)

    assert nullity == 1
    assert value == pytest.approx(2.0**2 / 4.0 + 3.0**2 / 1.0)


def test_a_component_outside_the_span_is_refused() -> None:
    """A gap pointing where the matrix has no variance means two spaces."""
    a = np.diag([4.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="span"):
        quadratic_form(np.array([2.0, 3.0, 1.0]), a)


def test_the_rank_cut_is_decided_on_the_correlation_scale() -> None:
    """Which directions survive must not depend on the statistics' units.

    The quadratic form itself is already unchanged by a change of units --
    `(Dx)' (DAD)^+ (Dx)` equals `x' A^+ x` whenever `A` has full rank -- so the
    rescaling earns its place only at the rank cut, which compares each
    eigenvalue against a fraction of the largest. Here the two coordinates each
    carry one unit of signal, but one is expressed in units 1e9 times smaller.
    On the correlation scale both survive and the form is 2. Without the
    rescaling the second eigenvalue falls under the threshold, is discarded as
    a null direction, and half the gap is silently dropped.

    b2star2 runs to 3e5 while the attribute sums are near 3e3, which is the
    same situation four orders of magnitude milder.
    """
    covariance = np.diag([1.0, 1e-18])
    x = np.array([1.0, 1e-9])

    value, nullity = standardized_quadratic_form(x, covariance)
    assert nullity == 0
    assert value == pytest.approx(2.0)

    raw_value, raw_nullity = quadratic_form(x, covariance)
    assert raw_nullity == 1
    assert raw_value == pytest.approx(1.0)


def test_the_pvalue_is_the_upper_tail_of_the_f_distribution() -> None:
    """Compared against scipy's F directly, with ergm's degrees of freedom.

    Ports `.ptsq(..., lower.tail = FALSE)`: the statistic is rescaled by
    `(df - p + 1) / (p * df)` and referred to `F(p, df - p + 1)`. Dropping the
    rescaling or taking the lower tail both leave a number between 0 and 1 that
    still moves with the statistic, so only the value settles it.
    """
    t_squared, n_parameters, df = 30.0, 3, 40.0
    expected = scipy.stats.f.sf(
        t_squared * (df - n_parameters + 1) / (n_parameters * df),
        n_parameters, df - n_parameters + 1,
    )
    assert nonconvergence_pvalue(t_squared, n_parameters, df) == pytest.approx(expected)


def test_a_larger_statistic_is_less_likely_to_be_exceeded() -> None:
    """The upper tail falls as the statistic grows; the lower tail would rise."""
    tail = [nonconvergence_pvalue(t, 3, 40.0) for t in (1.0, 10.0, 100.0)]
    assert tail == sorted(tail, reverse=True)
    assert tail[0] > 0.5 > tail[-1]


def test_too_few_degrees_of_freedom_cannot_declare_convergence() -> None:
    """With fewer degrees of freedom than parameters the test says nothing.

    `ergm` refuses to compute a p-value here. Returning 1 rather than a number
    keeps the caller from stopping on a test that was never run.
    """
    assert nonconvergence_pvalue(1e6, 5, 4.0) == 1.0
    assert nonconvergence_pvalue(1e6, 0, 40.0) == 1.0


def test_the_root_find_survives_a_boundary_point_outside_the_span() -> None:
    """A rank-deficient ellipsoid must not stop the search, only narrow it.

    `.ellipsoid_mahalanobis` evaluates `xTAx_seigen` at trial points along the
    way to the boundary, and wraps that evaluation in `ERRVL2(..., +Inf)`:
    a trial point that leaves the ellipsoid's span is treated as lying far
    outside it, which is what the root find needs to bracket from. Letting the
    error escape instead abandons a test that has an answer.

    The ellipsoid here is singular by construction -- its third coordinate is a
    fixed combination of the first two, which is what any linear dependence
    among the statistics produces. The gap is well inside it, so the caller has
    already decided the test is worth running.
    """
    rng = np.random.default_rng(21)
    base = rng.normal(size=(2000, 2))
    draws = np.column_stack([base, base[:, 0] + 2 * base[:, 1]])
    ellipsoid = 0.1 * np.cov(draws, rowvar=False)
    # The metric the convergence test passes: the batch-means covariance of the
    # mean, which inherits the same null direction the ellipsoid has.
    metric, _ = batch_means_covariance(draws)
    y = np.array([0.02, 0.02, 0.06])

    inside, ellipsoid_nullity = standardized_quadratic_form(y, ellipsoid)
    assert ellipsoid_nullity == 1
    assert inside < 1.0

    distance, nullity = ellipsoid_mahalanobis(y, metric, ellipsoid)

    assert nullity == 1
    assert distance > 0.0

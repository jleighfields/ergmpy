"""The Hummel step length, checked on geometry with a known answer."""

import numpy as np
from ergmpy.convex_hull import shrink_into_ch


def test_interior_point_needs_no_shrinking() -> None:
    """A point at the centroid of a square is well inside it."""
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    assert shrink_into_ch(np.array([0.0, 0.0]), square, m=np.zeros(2)) > 1.0


def test_point_on_the_boundary_scales_to_one() -> None:
    """A vertex of the square is exactly at the edge of the hull."""
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    gamma = shrink_into_ch(np.array([1.0, 0.0]), square, m=np.zeros(2))
    np.testing.assert_allclose(gamma, 1.0, rtol=1e-9)


def test_exterior_point_shrinks_by_the_reciprocal_distance() -> None:
    """A point k times too far out must shrink by 1/k to reach the boundary."""
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    for k in (2.0, 4.0, 10.0):
        gamma = shrink_into_ch(np.array([k, 0.0]), square, m=np.zeros(2))
        np.testing.assert_allclose(gamma, 1.0 / k, rtol=1e-9)


def test_matches_the_factors_ergm_produced(recorded_r) -> None:
    """Reproduces ergm's shrink_into_CH on cases saved from R."""
    expected = {1: 0.5003023501, 2: 22.1831136834, 3: 0.1876695020,
                4: 21.2571901294, 5: 0.1327143535, 6: 51.8217094913}
    cases = recorded_r / "convex_hull_cases"
    for case, gamma in expected.items():
        M = np.loadtxt(cases / f"ch_M_{case}.csv", delimiter=",", skiprows=1)
        p = np.loadtxt(cases / f"ch_p_{case}.csv", delimiter=",", skiprows=1, ndmin=1)
        np.testing.assert_allclose(shrink_into_ch(p, M), gamma, rtol=1e-8)

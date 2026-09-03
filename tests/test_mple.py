"""Pseudo-likelihood estimation, checked against derivatives computed a second way."""

import numpy as np
from ergmpy.choice import mple
from ergmpy.choice.predict import change_statistics


def test_hessian_matches_central_differences(train) -> None:
    """Standard errors come from the analytic Hessian, so it needs a second source."""
    result = mple.fit(train)
    Z = change_statistics(train)
    slot = np.argmax(train.choice_sets == train.chosen[:, None], axis=1)

    step = 1e-5
    numeric = np.empty_like(result.hessian)
    for i in range(len(result.coef)):
        up, down = result.coef.copy(), result.coef.copy()
        up[i] += step
        down[i] -= step
        _, gradient_up = mple.negative_log_pseudo_likelihood(up, Z, slot)
        _, gradient_down = mple.negative_log_pseudo_likelihood(down, Z, slot)
        numeric[i] = (gradient_up - gradient_down) / (2 * step)

    np.testing.assert_allclose(numeric, result.hessian, rtol=1e-5, atol=1e-4)


def test_gradient_matches_central_differences(train) -> None:
    """The optimizer is handed an analytic gradient; a wrong one still converges."""
    Z = change_statistics(train)
    slot = np.argmax(train.choice_sets == train.chosen[:, None], axis=1)
    theta = np.array([-1.0, 0.2, 0.8, 0.3, 0.6, 0.4, 0.5, 0.01])

    _, analytic = mple.negative_log_pseudo_likelihood(theta, Z, slot)
    step = 1e-6
    numeric = np.empty_like(theta)
    for i in range(len(theta)):
        up, down = theta.copy(), theta.copy()
        up[i] += step
        down[i] -= step
        value_up, _ = mple.negative_log_pseudo_likelihood(up, Z, slot)
        value_down, _ = mple.negative_log_pseudo_likelihood(down, Z, slot)
        numeric[i] = (value_up - value_down) / (2 * step)

    np.testing.assert_allclose(numeric, analytic, rtol=1e-4, atol=1e-3)


def test_gradient_vanishes_at_the_optimum(train) -> None:
    """A fit that converged has a stationary point, whatever else is true of it."""
    result = mple.fit(train)
    Z = change_statistics(train)
    slot = np.argmax(train.choice_sets == train.chosen[:, None], axis=1)
    _, gradient = mple.negative_log_pseudo_likelihood(result.coef, Z, slot)
    assert np.abs(gradient).max() < 1e-4


def test_standard_errors_are_positive_and_finite(train) -> None:
    """A singular information matrix would surface here rather than downstream."""
    result = mple.fit(train)
    assert np.all(np.isfinite(result.std_error))
    assert np.all(result.std_error > 0)

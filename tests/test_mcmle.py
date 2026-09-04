"""The MCMLE step and the stopping rule, checked against exact references.

The step is checked against the closed-form maximizer of the objective it
claims to maximize, so a wrong gradient is visible even though a wrong gradient
still converges. The stopping rule is checked against `ergm`'s own converged
coefficients: the test must accept them, and must refuse a parameter that is
plainly not them.
"""

import numpy as np
import polars as pl
import pytest

from ergmpy import mcmle
from ergmpy.choice.predict import TERM_NAMES
from ergmpy.control import MCMLEControl
from ergmpy.convergence import confidence_test
from ergmpy.mcmle import (
    geyer_thompson_step,
    observed_statistics,
    simulate,
)

# MCMLE.MCMC.precision from the converged reference fit, read off its saved
# control with `readRDS("results/r/fit_star.rds")$control$MCMLE.MCMC.precision`.
# It is also ergm 4.12.0's own default under MCMLE.termination = "confidence".
ERGM_MCMC_PRECISION = 0.1


def ergm_converged_coefficients(recorded_r) -> np.ndarray:
    """Reads ergm's converged estimates in the order `TERM_NAMES` uses.

    Args:
        recorded_r: Directory holding the outputs saved from R.

    Returns:
        (8,) coefficients.
    """
    star = pl.read_csv(recorded_r / "r" / "mcmle_star.csv",
                       schema_overrides={"estimate": pl.Float64})
    by_term = dict(zip(star["term"].to_list(), star["estimate"].to_list(), strict=True))
    return np.array([by_term[name] for name in TERM_NAMES], dtype=float)


def test_the_step_reaches_the_closed_form_maximizer() -> None:
    """The lognormal objective is a quadratic, so its maximizer is exact.

    Writing the objective in standardized coordinates as
    `phi . t - (m . phi + phi' S phi / 2)`, where m and S are the mean and the
    population covariance of the standardized draws, the maximizer is
    `S^-1 (t - m)`. No optimizer tolerance and no sampling enter it.

    This is what checks the analytic gradient the step hands to BFGS. A wrong
    gradient still converges -- to a point that is stationary for the wrong
    function -- so agreeing with the exact maximizer is the check that
    separates the two.
    """
    rng = np.random.default_rng(3)
    spread = np.array([[2.0, 0.5, 0.0, 0.0], [0.0, 1.0, 0.3, 0.0],
                       [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.7]])
    draws = rng.normal(size=(800, 4)) @ spread + 10.0
    theta_t = rng.normal(size=4)

    center, scale = draws.mean(axis=0), draws.std(axis=0)
    standardized = (draws - center) / scale
    target = center + scale * np.array([0.3, -0.2, 0.15, 0.1])
    target_standardized = (target - center) / scale

    covariance = np.cov(standardized, rowvar=False, ddof=0)
    expected = np.linalg.solve(covariance,
                               target_standardized - standardized.mean(axis=0))
    # Well inside the trust region, so the step is the maximizer and not a
    # rescaling of it.
    assert np.linalg.norm(expected) < mcmle.MAX_STANDARDIZED_STEP

    stepped = geyer_thompson_step(theta_t, draws, target)
    np.testing.assert_allclose((stepped - theta_t) * scale, expected, atol=1e-8)


def test_a_statistic_that_never_varies_moves_no_parameter() -> None:
    """Its coordinate carries no information, so the step must leave it alone."""
    rng = np.random.default_rng(4)
    draws = np.column_stack([rng.normal(size=(400, 2)), np.full(400, 7.0)])
    theta_t = np.array([0.5, -0.5, 2.0])
    target = draws.mean(axis=0) + np.array([0.4, -0.3, 0.0])

    stepped = geyer_thompson_step(theta_t, draws, target)

    assert stepped[2] == theta_t[2]
    assert stepped[0] != theta_t[0]


def test_the_step_stops_short_of_the_hull_boundary(monkeypatch) -> None:
    """`MCMLE.steplength.margin` keeps the target off the hull it was fitted in.

    The shrink factor is stubbed at 1, which is the boundary itself, so what
    the recorded step length shows is the margin and nothing else. Taking the
    full factor would put the target on the hull, where the importance-sampling
    approximation has no draws beyond it to be supported by.
    """
    captured = {}

    def fake_confidence_test(observed, draws, confidence, precision, n_chains):
        captured["precision"] = precision
        return False, 1.0, 1.0 - confidence

    monkeypatch.setattr(mcmle, "shrink_into_ch", lambda observed, draws: 1.0)
    monkeypatch.setattr(mcmle, "confidence_test", fake_confidence_test)
    monkeypatch.setattr(mcmle, "simulate",
                        lambda *a, **k: (np.random.default_rng(5).normal(size=(40, 8)),
                                         np.zeros((1, 3), dtype=np.int32)))
    monkeypatch.setattr(mcmle, "observed_statistics", lambda data: np.zeros(8))

    control = MCMLEControl(max_iterations=1, n_chains=1, n_draws=40, target_ess=1.0)
    result = mcmle.fit(None, np.zeros(8), control)

    assert not result.converged
    assert result.history[0]["step_length"] == pytest.approx(1.0 - control.step_margin)


def test_the_fit_tests_convergence_at_ergms_tolerance(monkeypatch) -> None:
    """`fit` must hand the stopping rule the tolerance region ergm uses.

    The tolerance region is `precision * cov(statistics)`, so this value is
    what decides how large a gap counts as converged -- not a matter of degree.
    At five times ergm's value a gap of 0.2 marginal standard deviations is
    accepted where ergm's own setting refuses it outright, which
    `test_a_looser_tolerance_region_accepts_a_larger_gap` shows directly.
    """
    captured = {}

    def fake_confidence_test(observed, draws, confidence, precision, n_chains):
        captured["precision"] = precision
        return True, 0.0, 1.0 - confidence

    monkeypatch.setattr(mcmle, "confidence_test", fake_confidence_test)
    monkeypatch.setattr(mcmle, "simulate",
                        lambda *a, **k: (np.random.default_rng(6).normal(size=(40, 8)),
                                         np.zeros((1, 3), dtype=np.int32)))
    monkeypatch.setattr(mcmle, "observed_statistics", lambda data: np.zeros(8))

    mcmle.fit(None, np.zeros(8),
              MCMLEControl(max_iterations=1, n_chains=1, n_draws=40, target_ess=1.0))

    assert captured["precision"] == pytest.approx(ERGM_MCMC_PRECISION)


def test_the_convergence_test_accepts_ergms_converged_estimates(
        train, recorded_r) -> None:
    """Simulating at ergm's answer must produce a sample the stopping rule stops on.

    This is the property the whole fit turns on, and it is checked without
    running one: ergm's converged coefficients are read from
    `results/r/mcmle_star.csv`, a sample is drawn at them, and the observed
    statistics are asked of the same test `fit` uses. A rule that refuses here
    would refuse the right answer, and a fit under it could only stop by
    running out of iterations.

    The sample is far smaller than the 4,676 draws ergm ended on, so the margin
    it passes by is smaller too; what it establishes is the direction, at a
    cost the suite can pay.
    """
    theta_star = ergm_converged_coefficients(recorded_r)
    draws, _ = simulate(train, theta_star, n_draws=3000, burn_in=50, thin=1,
                        n_chains=1, seed=7)
    g_obs = observed_statistics(train)

    gap_in_sd = np.abs(g_obs - draws.mean(axis=0)) / draws.std(axis=0)
    assert gap_in_sd.max() < 1.0

    passed, pvalue, threshold = confidence_test(
        g_obs, draws, precision=ERGM_MCMC_PRECISION
    )
    assert passed
    assert pvalue < threshold


def test_the_convergence_test_refuses_a_clearly_wrong_parameter(
        train, recorded_r) -> None:
    """Dropping the dependence term must not be mistaken for the right answer.

    `b2star2` is the parameter the whole model is about, and setting it to zero
    is the plainest wrong value there is: it removes the popularity effect and
    nothing else. The simulated statistics then sit tens of standard deviations
    from the observed ones, and a stopping rule that accepts that would accept
    anything.
    """
    wrong = ergm_converged_coefficients(recorded_r)
    wrong[TERM_NAMES.index("b2star2")] = 0.0
    draws, _ = simulate(train, wrong, n_draws=500, burn_in=50, thin=1,
                        n_chains=1, seed=7)
    g_obs = observed_statistics(train)

    star2_gap = (abs(g_obs[-1] - draws[:, -1].mean()) / draws[:, -1].std())
    assert star2_gap > 10.0

    passed, pvalue, threshold = confidence_test(
        g_obs, draws, precision=ERGM_MCMC_PRECISION
    )
    assert not passed
    assert pvalue > threshold

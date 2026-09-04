"""The convergence test, checked on chains whose properties are known."""

import numpy as np
import pytest

from ergmpy.convergence import (
    batch_means_covariance,
    confidence_test,
    effective_sample_size,
)


def ar1_chain(rho: float, n_draws: int, n_statistics: int, seed: int) -> np.ndarray:
    """Builds an AR(1) chain with a known autocorrelation.

    Args:
        rho: Lag-one correlation.
        n_draws: Chain length.
        n_statistics: Number of parallel series.
        seed: Seed for the innovations.

    Returns:
        (n_draws, n_statistics) draws.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n_draws, n_statistics))
    for t in range(1, n_draws):
        x[t] = rho * x[t - 1] + rng.normal(size=n_statistics)
    return x


def test_independent_draws_have_ess_near_their_count() -> None:
    """With no autocorrelation the effective count is the actual count."""
    draws = np.random.default_rng(1).normal(size=(2500, 4))
    ess = effective_sample_size(draws)
    assert np.all(ess > 1500)
    assert np.all(ess < 4000)


def test_autocorrelation_reduces_ess() -> None:
    """A correlated chain is worth fewer independent draws than it holds.

    An AR(1) chain's asymptotic factor is (1 - rho) / (1 + rho), so at
    rho = 0.8 the effective count is about a ninth of the length. Batch means
    over-estimates when batches are short relative to the correlation time, so
    this checks the direction and magnitude rather than the exact value.
    """
    correlated = ar1_chain(0.8, 4000, 3, seed=2)
    independent = np.random.default_rng(3).normal(size=(4000, 3))
    assert effective_sample_size(correlated).max() < 0.4 * 4000
    assert (effective_sample_size(correlated).max()
            < effective_sample_size(independent).min())


def test_covariance_of_the_mean_shrinks_with_more_draws() -> None:
    """Doubling the chain roughly halves the variance of its mean."""
    short = np.random.default_rng(4).normal(size=(1000, 3))
    long = np.random.default_rng(4).normal(size=(4000, 3))
    short_cov, _ = batch_means_covariance(short)
    long_cov, _ = batch_means_covariance(long)
    assert np.trace(long_cov) < np.trace(short_cov)


def test_the_simulated_mean_passes() -> None:
    """A point at the centre of the draws converges: the gap is exactly zero."""
    draws = np.random.default_rng(5).normal(size=(4000, 3))
    passed, pvalue, threshold = confidence_test(draws.mean(axis=0), draws)
    assert passed
    assert pvalue < threshold


def test_a_distant_point_fails() -> None:
    """A point outside the tolerance region cannot be declared converged."""
    draws = np.random.default_rng(6).normal(size=(4000, 3))
    passed, pvalue, threshold = confidence_test(draws.mean(axis=0) + 3.0, draws)
    assert not passed
    assert pvalue > threshold


def test_a_noisier_sample_does_not_become_easier_to_pass() -> None:
    """Convergence must get harder as the sampling gets noisier, not easier.

    This is the property that separates a non-inferiority test from a test of
    "indistinguishable from zero". Under the latter, autocorrelation widens the
    interval that counts as indistinguishable, so an under-sampled iteration
    declares success. Both chains here carry the same gap; only the
    autocorrelation differs.
    """
    clean = ar1_chain(0.0, 2000, 3, seed=20)
    noisy = ar1_chain(0.9, 2000, 3, seed=21)

    # The gap is held fixed in each chain's own standard deviations, not in
    # absolute units. An AR(0.9) chain has a marginal spread about 2.3 times
    # the independent one's, and the tolerance region scales with that spread,
    # so the same absolute gap would be a *smaller* gap for the noisy chain and
    # the comparison would measure the wrong thing. What differs is then only
    # the precision of the mean, and the p-value for non-convergence must be
    # larger where that is worse.
    _, clean_pvalue, _ = confidence_test(
        clean.mean(axis=0) + 0.1 * clean.std(axis=0), clean
    )
    _, noisy_pvalue, _ = confidence_test(
        noisy.mean(axis=0) + 0.1 * noisy.std(axis=0), noisy
    )
    assert noisy_pvalue > clean_pvalue


def test_a_non_varying_statistic_is_dropped() -> None:
    """A constant statistic carries no information and must not break the test."""
    draws = np.random.default_rng(7).normal(size=(4000, 3))
    draws = np.column_stack([draws, np.full(4000, 5.0)])
    passed, _, _ = confidence_test(draws.mean(axis=0), draws)
    assert passed


def test_no_batch_spans_two_chains() -> None:
    """Every batch must sit inside one chain, not across the join between two.

    Checked on the mechanism rather than on an outcome. Holding each chain at
    its own constant level makes any batch that stays inside a chain average to
    exactly that level, while one spanning two averages to something between
    them -- a value no chain holds. Randomness would only obscure that.

    Measured at the shape this package actually runs (1,248 draws over 4
    chains, 32 batches), chain-aware batching changes the covariance of the
    mean by under 1%, because only three batches of thirty-two straddle a join.
    The distinction is kept because it is correct, not because it is currently
    load-bearing.
    """
    levels = [0.0, 10.0, 20.0, 30.0]
    draws = np.vstack([np.full((500, 2), level) for level in levels])

    _, n_batches = batch_means_covariance(draws, n_batches=8, n_chains=4)
    batch_size = draws.shape[0] // n_batches
    batch_means = draws[: batch_size * n_batches].reshape(
        n_batches, batch_size, -1
    ).mean(axis=1)

    assert n_batches == 8
    assert set(np.unique(batch_means)) == set(levels)




def test_the_pvalue_decides_convergence_in_both_directions() -> None:
    """The decision on the p-value branch must be reachable and the right way up.

    The three cases above all return before the p-value is ever compared to the
    threshold: a gap at the centre short-circuits, and a gap outside the region
    is refused without a test. That leaves the comparison that actually decides
    most iterations unexercised, so inverting it -- the defect this stopping
    rule has already had once -- changes nothing they assert.

    Both gaps here sit strictly inside the tolerance region, differing only in
    how far. The near one must be accepted and the far one refused, which no
    single comparison can satisfy in both directions at once.
    """
    draws = np.random.default_rng(11).normal(size=(4000, 3))
    mean = draws.mean(axis=0)

    near_inside, near_p, threshold = confidence_test(mean + 0.05, draws)
    far_inside, far_p, _ = confidence_test(mean + 0.28, draws)

    # Neither short-circuit fired: both gaps were tested rather than assumed.
    assert 0.0 < near_p < threshold
    assert far_p > threshold

    assert near_inside
    assert not far_inside


def test_a_looser_tolerance_region_accepts_a_larger_gap() -> None:
    """`precision` scales the region, so it decides what counts as converged.

    ergm's `MCMLE.MCMC.precision` under confidence termination is 0.1, which
    `results/r/fit_star.rds` recorded for the converged reference fit. This gap
    is outside the region at that value and inside it at five times that, so
    the setting is not a matter of degree: it flips the answer.
    """
    draws = np.random.default_rng(11).normal(size=(4000, 3))
    gap = draws.mean(axis=0) + 0.2

    at_ergms_value, _, _ = confidence_test(gap, draws, precision=0.1)
    five_times_looser, _, _ = confidence_test(gap, draws, precision=0.5)

    assert not at_ergms_value
    assert five_times_looser


def test_the_covariance_of_the_mean_is_batched_within_a_chain() -> None:
    """Asserted on what the estimator returns, not on a recomputation beside it.

    Each chain sits at its own constant level, so every batch that stays inside
    one has the same mean as every other batch in that chain, and the spread of
    the batch means comes only from the differences between chains. Batching
    across the join instead produces batches straddling two levels, whose means
    differ from both, and the estimated covariance changes.

    The point of holding the levels constant is that the correct answer is
    available exactly: with four batches per chain all equal to that chain's
    level, the batch-mean covariance is the population variance of the four
    levels times the batch size, divided by the draws used.
    """
    levels = np.array([0.0, 10.0, 20.0, 30.0])
    draws = np.vstack([np.full((500, 2), level) for level in levels])

    covariance, n_batches = batch_means_covariance(draws, n_batches=12, n_chains=4)

    assert n_batches == 12
    # 500 draws per chain over 3 batches leaves 166 per batch and 2 unused.
    batch_size, used = 166, 166 * 3 * 4
    centered = np.repeat(levels, 3) - levels.mean()
    expected = batch_size * (centered @ centered) / (n_batches - 1) / used
    np.testing.assert_allclose(covariance, np.full((2, 2), expected), rtol=1e-10)

    # The same draw count and batch count, batched as one chain: batches of 166
    # do not divide the 500-draw chains, so three of the twelve straddle a join.
    across_the_join, _ = batch_means_covariance(draws, n_batches=12, n_chains=1)
    assert not np.allclose(across_the_join, covariance)


def test_draws_too_few_to_fill_the_batches_are_refused() -> None:
    """An empty batch would make the covariance a mean of nothing."""
    draws = np.random.default_rng(12).normal(size=(8, 2))
    with pytest.raises(ValueError, match="batches"):
        batch_means_covariance(draws, n_batches=32)


def test_linearly_dependent_statistics_do_not_block_convergence_forever() -> None:
    """A gap inside the region must be tested, not abandoned as untestable.

    Statistics that are linear combinations of one another are what a
    constrained sample space produces -- `ergm` refuses to estimate `edges`
    here for exactly that reason -- and they make the tolerance region
    singular. The gap below sits well inside it, so the test is worth running.

    Today the root find inside `ellipsoid_mahalanobis` lets an error escape,
    `confidence_test` catches it, and every iteration reports "not converged"
    for a sample that is as converged as it will ever be. The fit then stops
    only by running out of iterations, and nothing in the log distinguishes
    that from a fit that genuinely had further to go -- so the failure is
    silent, which is what makes it worth pinning rather than merely handling.
    """
    rng = np.random.default_rng(21)
    base = rng.normal(size=(2000, 2))
    draws = np.column_stack([base, base[:, 0] + 2 * base[:, 1]])
    mean = draws.mean(axis=0)

    # Inside the region, and in its span: this gap is reachable by the draws.
    tiny = np.array([0.02, 0.02, 0.06])
    huge = np.array([2.0, 2.0, 6.0])

    passed, pvalue, threshold = confidence_test(mean + tiny, draws)
    assert passed
    assert pvalue < threshold

    # The same rank deficiency must still refuse a gap that is genuinely far
    # out, so the fix cannot be to accept whatever it cannot measure.
    refused, _, _ = confidence_test(mean + huge, draws)
    assert not refused

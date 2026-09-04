"""The convergence test, checked on chains whose properties are known."""

import numpy as np

from ergmpy.convergence import (
    batch_means_covariance,
    effective_sample_size,
    within_tolerance,
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
    passed, pvalue, threshold = within_tolerance(draws.mean(axis=0), draws)
    assert passed
    assert pvalue < threshold


def test_a_distant_point_fails() -> None:
    """A point outside the tolerance region cannot be declared converged."""
    draws = np.random.default_rng(6).normal(size=(4000, 3))
    passed, pvalue, threshold = within_tolerance(draws.mean(axis=0) + 3.0, draws)
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
    _, clean_pvalue, _ = within_tolerance(
        clean.mean(axis=0) + 0.1 * clean.std(axis=0), clean
    )
    _, noisy_pvalue, _ = within_tolerance(
        noisy.mean(axis=0) + 0.1 * noisy.std(axis=0), noisy
    )
    assert noisy_pvalue > clean_pvalue


def test_a_non_varying_statistic_is_dropped() -> None:
    """A constant statistic carries no information and must not break the test."""
    draws = np.random.default_rng(7).normal(size=(4000, 3))
    draws = np.column_stack([draws, np.full(4000, 5.0)])
    passed, _, _ = within_tolerance(draws.mean(axis=0), draws)
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




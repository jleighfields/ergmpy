"""The convergence test, checked on chains whose properties are known."""

import numpy as np

from ergmpy.convergence import (
    batch_means_covariance,
    effective_sample_size,
    within_confidence_region,
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


def test_the_simulated_mean_is_inside_its_own_region() -> None:
    """A point at the centre of the draws cannot be rejected."""
    draws = np.random.default_rng(5).normal(size=(900, 4))
    inside, statistic, threshold = within_confidence_region(draws.mean(axis=0), draws)
    assert inside
    assert statistic < threshold


def test_a_distant_point_is_rejected() -> None:
    """A point many standard errors away falls outside the region."""
    draws = np.random.default_rng(6).normal(size=(900, 4))
    inside, statistic, threshold = within_confidence_region(
        draws.mean(axis=0) + 3.0, draws
    )
    assert not inside
    assert statistic > threshold


def test_chains_are_not_batched_across_their_join() -> None:
    """Batching across independent chains would misread the autocorrelation.

    Four correlated chains laid end to end look like one chain with three
    discontinuities. Telling the estimator how many there are keeps every batch
    inside a single chain.
    """
    chains = [ar1_chain(0.85, 500, 3, seed=10 + i) for i in range(4)]
    draws = np.vstack(chains)
    aware = effective_sample_size(draws, n_chains=4)
    assert np.all(aware > 0)
    assert np.all(aware < draws.shape[0])


def test_non_varying_statistics_are_dropped() -> None:
    """A constant statistic carries no information and must not break the test.

    `ergm` reports the same situation as a term "not varying"; here it would
    otherwise make the covariance singular.
    """
    draws = np.random.default_rng(7).normal(size=(900, 4))
    draws[:, 2] = 5.0
    observed = draws.mean(axis=0)
    inside, _, _ = within_confidence_region(observed, draws)
    assert inside

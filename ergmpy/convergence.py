"""Convergence testing for Monte Carlo maximum likelihood.

The question at each MCMLE iteration is whether the observed statistics are
far enough from the simulated mean to justify another step. Comparing the gap
to a fixed number of standard deviations answers it badly: the statistics are
correlated with one another, so a per-coordinate test ignores the shape of the
joint distribution, and the draws are autocorrelated, so the sample standard
deviation understates the uncertainty in their mean.

`ergm` terminates on a confidence statement instead -- `MCMLE.termination =
"confidence"` at `MCMLE.confidence = 0.99` -- asking whether the observed
statistics lie inside a joint confidence region around the simulated mean.
This implements the same test.

Two pieces are needed. The covariance of the *mean* must account for the
chain's autocorrelation, which multivariate batch means estimates by splitting
the draws into consecutive batches and using the spread of the batch means
rather than of the draws themselves. The resulting Mahalanobis distance is
then referred to an F distribution, whose degrees of freedom account for
estimating that covariance from a finite number of batches.

References:
    Vats, D., Flegal, J.M. and Jones, G.L. (2019). Multivariate output
        analysis for Markov chain Monte Carlo. Biometrika, 106(2), 321-337.
    Hummel, R.M., Hunter, D.R. and Handcock, M.S. (2012). Improving
        simulation-based algorithms for fitting ERGMs. Journal of
        Computational and Graphical Statistics, 21(4), 920-939.
"""

import numpy as np
import scipy.stats


def batch_means_covariance(draws: np.ndarray, n_batches: int | None = None,
                           n_chains: int = 1) -> tuple[np.ndarray, int]:
    """Estimates the covariance of the sample mean, allowing for autocorrelation.

    Consecutive draws from a Markov chain carry information about one another,
    so dividing the sample covariance by the number of draws understates the
    uncertainty in their mean. Batch means recovers it: within a batch long
    enough to span the chain's memory, the batch means are close to
    independent, so their spread estimates the asymptotic variance directly.

    Args:
        draws: (n_draws, n_statistics) sampled statistics, chain-major: the
            first n_draws // n_chains rows are one chain, and so on.
        n_batches: Number of consecutive batches; defaults to the square root
            of the draw count, the usual choice that lets both batch size and
            batch count grow with the chain.
        n_chains: How many independent chains the draws came from. Batches are
            taken within a chain, never across the join between two, where
            consecutive rows are unrelated and the batch mean would be
            meaningless.

    Returns:
        The (n_statistics, n_statistics) covariance of the mean, and the
        number of batches actually used.
    """
    n_draws = draws.shape[0]
    if n_batches is None:
        n_batches = max(2, int(np.sqrt(n_draws)))
    per_chain = n_draws // n_chains
    batches_per_chain = max(1, n_batches // n_chains)
    batch_size = per_chain // batches_per_chain
    if batch_size < 1:
        raise ValueError(f"{n_draws} draws cannot fill {n_batches} batches")

    n_batches = batches_per_chain * n_chains
    used_per_chain = batch_size * batches_per_chain
    used = used_per_chain * n_chains
    chains = draws.reshape(n_chains, per_chain, -1)[:, :used_per_chain]
    batch_means = chains.reshape(n_batches, batch_size, -1).mean(axis=1)
    centered = batch_means - batch_means.mean(axis=0)

    # batch_size scales the batch-mean covariance up to an asymptotic variance;
    # dividing by the draws used turns it into the covariance of the mean.
    asymptotic = (batch_size / (n_batches - 1)) * (centered.T @ centered)
    return asymptotic / used, n_batches


def effective_sample_size(draws: np.ndarray, n_chains: int = 1) -> np.ndarray:
    """Estimates how many independent draws each statistic's chain is worth.

    Args:
        draws: (n_draws, n_statistics) sampled statistics, chain-major.
        n_chains: How many independent chains the draws came from.

    Returns:
        (n_statistics,) effective sample sizes.
    """
    n_draws = draws.shape[0]
    covariance, _ = batch_means_covariance(draws, n_chains=n_chains)
    marginal = draws.var(axis=0, ddof=1)
    inflated = np.diag(covariance) * n_draws
    return np.where(inflated > 0, n_draws * marginal / inflated, float(n_draws))


def within_confidence_region(observed: np.ndarray, draws: np.ndarray,
                             confidence: float = 0.99,
                             n_chains: int = 1) -> tuple[bool, float, float]:
    """Tests whether the observed statistics lie inside the simulated region.

    Statistics that never vary carry no information and are dropped from the
    test rather than making the covariance singular; `ergm` reports the same
    situation as a term "not varying".

    Args:
        observed: (n_statistics,) statistics of the observed network.
        draws: (n_draws, n_statistics) sampled statistics, in chain order.
        confidence: Level of the joint region, matching `MCMLE.confidence`.
        n_chains: How many independent chains the draws came from.

    Returns:
        Whether the observed statistics are inside the region, the Mahalanobis
        statistic, and the threshold it was compared against.
    """
    varying = draws.std(axis=0) > 1e-12
    if not varying.any():
        return True, 0.0, 0.0

    gap = (observed - draws.mean(axis=0))[varying]
    covariance, n_batches = batch_means_covariance(draws[:, varying],
                                                   n_chains=n_chains)
    n_parameters = int(varying.sum())

    if n_batches - n_parameters < 1:
        raise ValueError(
            f"{n_batches} batches cannot support a joint test on "
            f"{n_parameters} statistics; take more draws"
        )

    distance = float(gap @ np.linalg.solve(covariance, gap))
    # Hotelling's T-squared, scaled to an F: the covariance is estimated from
    # n_batches batch means, so the threshold is wider than a chi-square would
    # give at this batch count.
    scale = (n_batches - n_parameters) / ((n_batches - 1) * n_parameters)
    threshold = scipy.stats.f.ppf(confidence, n_parameters, n_batches - n_parameters)
    return distance * scale < threshold, distance * scale, float(threshold)

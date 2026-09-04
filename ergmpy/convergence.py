"""Convergence testing for Monte Carlo maximum likelihood.

The question at each MCMLE iteration is whether the observed statistics are
far enough from the simulated mean to justify another step. Comparing the gap
to a fixed number of standard deviations answers it badly: the statistics are
correlated with one another, so a per-coordinate test ignores the shape of the
joint distribution, and the draws are autocorrelated, so the sample standard
deviation understates the uncertainty in their mean.

`ergm` terminates on a confidence statement instead -- `MCMLE.termination =
"confidence"` at `MCMLE.confidence = 0.99` -- declaring convergence only when
it can rule out non-convergence at that level. `confidence_test` follows that
direction, and its docstring sets out where it departs from `ergm`.

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

from ergmpy.hotelling import (
    ellipsoid_mahalanobis,
    nonconvergence_pvalue,
    standardized_quadratic_form,
)


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


def confidence_test(observed: np.ndarray, draws: np.ndarray,
                     confidence: float = 0.99, precision: float = 0.1,
                     n_chains: int = 1) -> tuple[bool, float, float]:
    """Runs `ergm`'s confidence test on the current sample.

    Ported from `ergm:::ergm.MCMLE`'s `MCMLE.termination = "confidence"` branch.
    The null is that the fit has *not* converged, and it is rejected only when
    the gap sits confidently inside a tolerance region -- so a noisier sample
    raises the p-value and refuses, where a test of "indistinguishable from
    zero" would accept.

    Three things differ from `ergm`, all in how the inputs to that test are
    estimated rather than in the test itself:

    - **Where the gap is measured.** `ergm` reweights the draws by importance
      to the parameter its optimizer has just proposed, and tests the gap
      there. This tests the gap at the parameter the draws came from, so the
      estimate it approves is the one it tested.
    - **How the covariance of the mean is estimated.** `ergm` fits a vector
      autoregression to the draws (`ergm:::spectrum0.mvar`) and reads the
      asymptotic variance off it. This uses multivariate batch means, which
      needs no order selection and no model for the autocorrelation.
    - **Where the degrees of freedom come from.** Following from that, the
      F reference here is `n_batches - 1`, the number of independent
      quantities the covariance was estimated from. `ergm` uses the effective
      sample size minus one, which its autoregression yields directly and
      which is the larger number, so this test is the more conservative of
      the two at equal sample size.

    A statistic that never varies is dropped rather than zeroed out of the
    tolerance region, which is `ergm`'s treatment; the two agree because a
    dropped coordinate contributes nothing either way.

    Args:
        observed: (n_statistics,) statistics of the observed network.
        draws: (n_draws, n_statistics) sampled statistics, chain-major. For
            this model the estimating equations are the statistics themselves,
            so this is `ergm`'s `esteq`.
        confidence: Matching `MCMLE.confidence`.
        precision: Scales the tolerance region, matching
            `MCMLE.MCMC.precision`.
        n_chains: How many independent chains the draws came from.

    Returns:
        Whether convergence can be declared, the p-value for non-convergence,
        and the threshold it must fall below.
    """
    varying = draws.std(axis=0) > 1e-12
    if not varying.any():
        return True, 0.0, 1.0 - confidence

    kept = draws[:, varying]
    gap = (observed - draws.mean(axis=0))[varying]
    n_parameters = int(varying.sum())
    threshold = 1.0 - confidence

    # Vm: the tolerance region, precision times the statistics' covariance.
    tolerance_region = precision * np.cov(kept, rowvar=False)

    # estcov: the covariance of the mean, which batch means estimates so the
    # chain's autocorrelation is carried into the test.
    mean_covariance, n_batches = batch_means_covariance(kept, n_chains=n_chains)

    # ergm tests only once the point estimate is inside the region; outside it,
    # there is no distance-to-boundary to measure and it keeps sampling.
    inside, _ = standardized_quadratic_form(gap, tolerance_region)
    if inside >= 1.0:
        return False, 1.0, threshold
    if inside <= 1e-12:
        # A gap indistinguishable from zero sits at the centre of the region,
        # where the root find below has no crossing to bracket: scaling zero
        # never reaches a boundary. It is also the most converged a fit can be.
        return True, 0.0, threshold

    try:
        t_squared, metric_nullity = ellipsoid_mahalanobis(
            gap, mean_covariance, tolerance_region
        )
    except (ValueError, np.linalg.LinAlgError):
        # ergm reports "Unable to test for convergence" here and samples more.
        return False, 1.0, threshold

    free = n_parameters - metric_nullity
    pvalue = nonconvergence_pvalue(t_squared, free, n_batches - 1)
    return pvalue < threshold, pvalue, threshold

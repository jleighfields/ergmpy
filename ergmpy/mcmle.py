"""Monte Carlo maximum likelihood estimation by importance sampling.

At the current parameter theta_t, networks are simulated with the Gibbs sweep
and their statistics collected. Geyer and Thompson's approximation gives the
log-likelihood ratio against that reference point,

    l(theta) - l(theta_t) = (theta - theta_t) . g_obs
                            - log mean_m exp((theta - theta_t) . g_m)

whose gradient is g_obs minus an importance-weighted mean of the sampled
statistics. Maximizing it directly is unreliable: when g_obs lies outside the
convex hull of the sample, the approximation is being extrapolated past its
support and the step diverges. Following Hummel, Hunter and Handcock (2012),
the observed statistics are first shrunk toward the sampled mean by the largest
factor that keeps them inside the hull, and the step targets that instead.

Standard errors come from the inverse of the sampled statistics' covariance,
which is the Fisher information for an exponential family.

References:
    Geyer, C.J. and Thompson, E.A. (1992). Constrained Monte Carlo maximum
        likelihood for dependent data. Journal of the Royal Statistical
        Society B, 54(3), 657-699.
    Hummel, R.M., Hunter, D.R. and Handcock, M.S. (2012). Improving
        simulation-based algorithms for fitting ERGMs. Journal of
        Computational and Graphical Statistics, 21(4), 920-939.
"""

import numpy as np
import scipy.optimize
from scipy.special import logsumexp

from ergmpy import sampler
from ergmpy.choice.predict import ChoiceData
from ergmpy.convex_hull import shrink_into_ch

TERM_NAMES = ("b2cov.V1", "b2cov.V2", "b2cov.V3", "b2factor.V4.2",
              "b2factor.V4.3", "b2factor.V4.4", "b2factor.V4.5", "b2star2")


class MCMLEResult:
    """Fitted Monte Carlo maximum likelihood estimates.

    Attributes:
        coef: (8,) point estimates, ordered as TERM_NAMES.
        std_error: (8,) standard errors from the inverse sample covariance.
        n_iterations: Outer iterations actually run.
        converged: Whether the convergence test passed before the cap.
        history: One dict per iteration with its step length and gradient norm.
    """

    def __init__(self, coef: np.ndarray, std_error: np.ndarray, n_iterations: int,
                 converged: bool, history: list[dict]) -> None:
        """Stores the estimates; the class docstring describes each attribute."""
        self.coef = coef
        self.std_error = std_error
        self.n_iterations = n_iterations
        self.converged = converged
        self.history = history

    def summary(self) -> str:
        """Formats the estimates the way ergm's summary does.

        Returns:
            A table of term, estimate, standard error and z value.
        """
        z = self.coef / self.std_error
        lines = [f"{'term':<16}{'Estimate':>13}{'Std. Error':>13}{'z value':>11}"]
        for name, c, s, zz in zip(TERM_NAMES, self.coef, self.std_error, z, strict=True):
            lines.append(f"{name:<16}{c:>13.6f}{s:>13.6f}{zz:>11.3f}")
        status = "converged" if self.converged else "did NOT converge"
        lines.append(f"\n{status} after {self.n_iterations} iterations")
        return "\n".join(lines)


def observed_statistics(data: ChoiceData) -> np.ndarray:
    """Computes the statistic vector of the observed purchase network.

    Args:
        data: The dataset.

    Returns:
        (8,) statistics: the seven attribute sums, then b2star2.
    """
    return sampler.network_statistics(data.choice_sets, data.chosen,
                                      data.design, data.n_products)


def simulate(data: ChoiceData, theta: np.ndarray, n_draws: int, burn_in: int,
             thin: int, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draws network statistics from the model at theta.

    Args:
        data: The dataset defining choice sets and product attributes.
        theta: (8,) parameter vector to simulate at.
        n_draws: Number of retained draws.
        burn_in: Sweeps discarded before the first draw.
        thin: Sweeps between retained draws.
        state: Optional (n_customers,) starting configuration; defaults to the
            observed purchases, which is ergm's own starting point.

    Returns:
        The (n_draws, 8) statistics and the final configuration.
    """
    choice_sets = np.ascontiguousarray(data.choice_sets)
    current = (data.chosen if state is None else state).astype(np.int32).copy()
    degree = np.bincount(current, minlength=data.n_products).astype(np.int64)
    linear = np.ascontiguousarray(data.design @ theta[:7])
    theta_star2 = float(theta[7])

    sampler.run_sweeps(choice_sets, current, degree, linear, theta_star2, burn_in)
    draws = np.empty((n_draws, 8))
    for m in range(n_draws):
        sampler.run_sweeps(choice_sets, current, degree, linear, theta_star2, thin)
        draws[m] = sampler.network_statistics(choice_sets, current, data.design,
                                              data.n_products)
    return draws, current


def geyer_thompson_step(theta_t: np.ndarray, draws: np.ndarray,
                        target: np.ndarray, max_standardized_step: float = 3.0
                        ) -> np.ndarray:
    """Maximizes the importance-sampled log-likelihood ratio.

    The statistics differ in scale by two orders of magnitude -- b2star2 runs to
    3e5 while the attribute sums are near 3e3 -- and the parameters span 0.006
    to 3, so optimizing on the raw scale is badly conditioned and the step is
    unreliable. The draws are standardized first and the parameter transformed
    back afterwards, which leaves the objective unchanged: writing phi = delta *
    sd, the delta . mean terms cancel between the two halves of the ratio.

    Args:
        theta_t: (8,) reference parameter the draws were simulated at.
        draws: (n_draws, 8) sampled statistics.
        target: (8,) statistics the step should match, already shrunk.
        max_standardized_step: Trust region on the norm of the standardized
            step, which bounds how far one iteration can move.

    Returns:
        The (8,) maximizing parameter vector.
    """
    center = draws.mean(axis=0)
    scale = draws.std(axis=0)
    # A statistic that never varies carries no information about its parameter,
    # so hold that coordinate fixed rather than dividing by zero.
    varying = scale > 1e-12
    scaled = np.ones_like(scale)
    scaled[varying] = scale[varying]

    standardized = (draws - center) / scaled
    standardized[:, ~varying] = 0.0
    target_standardized = (target - center) / scaled
    target_standardized[~varying] = 0.0

    def objective(phi: np.ndarray) -> tuple[float, np.ndarray]:
        scores = standardized @ phi
        normalizer = logsumexp(scores)
        weights = np.exp(scores - normalizer)
        value = phi @ target_standardized - (normalizer - np.log(len(draws)))
        gradient = target_standardized - weights @ standardized
        return -value, -gradient

    result = scipy.optimize.minimize(objective, x0=np.zeros(len(scale)), jac=True,
                                     method="BFGS", options={"gtol": 1e-10})
    phi = result.x
    norm = np.linalg.norm(phi)
    if norm > max_standardized_step:
        phi = phi * (max_standardized_step / norm)
    delta = np.where(varying, phi / scaled, 0.0)
    return theta_t + delta


def fit(data: ChoiceData, theta0: np.ndarray, max_iterations: int = 30,
        n_draws: int = 1250, burn_in: int = 200, thin: int = 200,
        tolerance: float = 0.05, seed: int = 123) -> MCMLEResult:
    """Fits the model by Monte Carlo maximum likelihood.

    Args:
        data: The dataset to fit.
        theta0: (8,) starting parameter, normally the pseudo-likelihood estimate.
        max_iterations: Cap on outer iterations.
        n_draws: Retained draws per iteration.
        burn_in: Sweeps discarded before the first draw of each iteration.
        thin: Sweeps between retained draws.
        tolerance: Convergence threshold on the largest standardized gap between
            observed and simulated mean statistics.
        seed: Seed for the sampler's random stream.

    Returns:
        The fitted MCMLEResult.
    """
    np.random.seed(seed)
    g_obs = observed_statistics(data)
    theta = np.asarray(theta0, dtype=float).copy()
    state = None
    history: list[dict] = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        draws, state = simulate(data, theta, n_draws, burn_in, thin, state)
        mean = draws.mean(axis=0)
        spread = draws.std(axis=0)
        # Statistics that never move carry no information and would divide by
        # zero here; ergm reports the same situation as "not varying".
        varying = spread > 1e-12
        gap = np.zeros_like(spread)
        gap[varying] = np.abs(g_obs - mean)[varying] / spread[varying]

        if gap.max() < tolerance:
            converged = True
            history.append({"iteration": iteration, "step_length": 1.0,
                            "max_standardized_gap": float(gap.max())})
            break

        gamma = min(1.0, shrink_into_ch(g_obs, draws))
        target = gamma * g_obs + (1.0 - gamma) * mean
        theta = geyer_thompson_step(theta, draws, target)
        history.append({"iteration": iteration, "step_length": float(gamma),
                        "max_standardized_gap": float(gap.max())})

    draws, _ = simulate(data, theta, n_draws, burn_in, thin, state)
    covariance = np.cov(draws, rowvar=False)
    std_error = np.sqrt(np.diag(np.linalg.pinv(covariance)))
    return MCMLEResult(theta, std_error, len(history), converged, history)

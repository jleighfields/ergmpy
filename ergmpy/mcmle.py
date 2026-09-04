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

import dataclasses
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.optimize

from ergmpy import sampler
from ergmpy.choice.predict import ChoiceData
from ergmpy.control import MCMLEControl
from ergmpy.convergence import effective_sample_size, within_tolerance
from ergmpy.convex_hull import shrink_into_ch

log = logging.getLogger(__name__)

TERM_NAMES = ("b2cov.V1", "b2cov.V2", "b2cov.V3", "b2factor.V4.2",
              "b2factor.V4.3", "b2factor.V4.4", "b2factor.V4.5", "b2star2")


class MCMLEResult:
    """Fitted Monte Carlo maximum likelihood estimates.

    Attributes:
        coef: (8,) point estimates, ordered as TERM_NAMES.
        std_error: (8,) standard errors from the inverse sample covariance.
        n_iterations: Outer iterations actually run.
        converged: Whether the convergence test passed before the cap.
        history: One dict per iteration recording its step length, the
            confidence statistic and threshold, the smallest effective sample
            size across statistics, and the interval and draw count it settled
            on.
        control: The settings the fit ran under.
        sweeps: Total sweeps drawn, counting resample attempts that were
            discarded and every chain's burn-in.
    """

    def __init__(self, coef: np.ndarray, std_error: np.ndarray, n_iterations: int,
                 converged: bool, history: list[dict],
                 control: MCMLEControl, sweeps: int) -> None:
        """Stores the estimates; the class docstring describes each attribute."""
        self.coef = coef
        self.std_error = std_error
        self.n_iterations = n_iterations
        self.converged = converged
        self.history = history
        self.control = control
        self.sweeps = sweeps

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


def simulate_chain(data: ChoiceData, theta: np.ndarray, n_draws: int,
                   burn_in: int, thin: int, seed: int,
                   state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draws network statistics from one chain at theta.

    Args:
        data: The dataset defining choice sets and product attributes.
        theta: (8,) parameter vector to simulate at.
        n_draws: Number of retained draws.
        burn_in: Sweeps discarded before the first draw.
        thin: Sweeps between retained draws.
        seed: Seed for this chain's random stream.
        state: Optional (n_customers,) starting configuration; defaults to the
            observed purchases, which is ergm's own starting point.

    Returns:
        The (n_draws, 8) statistics and the final configuration.
    """
    # Seeds numba's generator, which np.random.seed cannot reach from here.
    sampler.seed_numba(seed)
    np.random.seed(seed)
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


def simulate(data: ChoiceData, theta: np.ndarray, n_draws: int, burn_in: int,
             thin: int, state: np.ndarray | None = None, n_chains: int = 1,
             seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    """Draws network statistics from the model at theta, over one or more chains.

    Chains are independent, which is the axis `ergm` parallelises too -- the R
    script runs `parallel = 4` and splits its sample size across four workers.
    The draws are returned chain-major, so the convergence test can take batch
    means within a chain rather than across the join between two.

    Each chain is seeded from `seed` and its index, so a run reproduces
    regardless of how the work is scheduled.

    Args:
        data: The dataset defining choice sets and product attributes.
        theta: (8,) parameter vector to simulate at.
        n_draws: Total retained draws, divided among the chains.
        burn_in: Sweeps discarded before the first draw of each chain.
        thin: Sweeps between retained draws.
        state: Optional (n_chains, n_customers) starting configurations, one
            per chain. Sharing one across chains would start them all at the
            same configuration each iteration, which weakens the independence
            the batch-means estimator assumes.
        n_chains: Independent chains to run, in that many worker processes.
        seed: Base seed; chain i uses seed + i.

    Returns:
        The (n_draws, 8) statistics, chain-major, and each chain's final
        configuration as an (n_chains, n_customers) array.
    """
    if n_chains == 1:
        start = None if state is None else np.asarray(state).reshape(1, -1)[0]
        draws, end = simulate_chain(data, theta, n_draws, burn_in, thin, seed,
                                    start)
        return draws, end.reshape(1, -1)

    per_chain = n_draws // n_chains
    if per_chain < 1:
        raise ValueError(f"{n_draws} draws cannot be split across {n_chains} chains")

    starts = [None] * n_chains if state is None else list(np.asarray(state))
    with ProcessPoolExecutor(max_workers=n_chains) as pool:
        futures = [
            pool.submit(simulate_chain, data, theta, per_chain, burn_in, thin,
                        seed + index, starts[index])
            for index in range(n_chains)
        ]
        results = [f.result() for f in futures]

    draws = np.vstack([draw for draw, _ in results])
    return draws, np.vstack([end for _, end in results])


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

    The objective is `ergm`'s lognormal metric rather than the exact
    importance-sampled ratio; the comment on `objective` says why.

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
        # ergm's MCMLE.metric defaults to "lognormal": rather than the exact
        # log-mean-exp of the importance weights, it approximates
        # log E[exp(X)] by E[X] + Var(X)/2, which is exact when X is normal and
        # far less sensitive to a single dominant weight when it is not. See
        # ergm:::llik.fun.lognormal, whose varweight is 0.5.
        scores = standardized @ phi
        mean_score = scores.mean()
        variance = scores.var()
        value = phi @ target_standardized - (mean_score + 0.5 * variance)

        # d/dphi of the variance is twice the covariance between the scores and
        # the statistics that produced them.
        centred = scores - mean_score
        covariance = (centred @ standardized) / len(scores)
        gradient = target_standardized - standardized.mean(axis=0) - covariance
        return -value, -gradient

    result = scipy.optimize.minimize(objective, x0=np.zeros(len(scale)), jac=True,
                                     method="BFGS", options={"gtol": 1e-10})
    phi = result.x
    norm = np.linalg.norm(phi)
    if norm > max_standardized_step:
        phi = phi * (max_standardized_step / norm)
    delta = np.where(varying, phi / scaled, 0.0)
    return theta_t + delta


def fit(data: ChoiceData, theta0: np.ndarray,
        control: MCMLEControl | None = None, **overrides: object) -> MCMLEResult:
    """Fits the model by Monte Carlo maximum likelihood.

    Stops on the same criterion `ergm` uses -- `MCMLE.termination =
    "confidence"` -- asking whether the observed statistics lie inside a joint
    confidence region around the simulated mean, with the covariance of that
    mean estimated by batch means so the chain's autocorrelation is accounted
    for.

    Args:
        data: The dataset to fit.
        theta0: (8,) starting parameter, normally a contrastive-divergence seed.
        control: Run settings; defaults to `MCMLEControl()`, whose defaults
            match what the reference R script asks `control.ergm` for.
        **overrides: Individual `MCMLEControl` fields to replace, for a caller
            that wants one setting changed without constructing a control.

    Returns:
        The fitted MCMLEResult, carrying the control it ran under.

    Raises:
        TypeError: If an override names no field on `MCMLEControl`, rather
            than being silently ignored.
    """
    control = control or MCMLEControl()
    if overrides:
        unknown = set(overrides) - set(control.to_dict())
        if unknown:
            raise TypeError(f"not MCMLEControl fields: {sorted(unknown)}")
        control = dataclasses.replace(control, **overrides)

    g_obs = observed_statistics(data)
    theta = np.asarray(theta0, dtype=float).copy()
    state = None
    interval, draw_count = control.thin, control.n_draws
    sweeps_drawn = 0
    history: list[dict] = []
    converged = False

    for iteration in range(1, control.max_iterations + 1):
        # Resample until the draws are worth target_ess independent ones,
        # shortening the interval and taking proportionally more draws each
        # time. A convergence test on a sample that autocorrelation has made
        # smaller than it looks is the failure this prevents.
        #
        # interval and draw_count persist across iterations rather than
        # resetting, which is what ergm does: it assigns the adapted values
        # back onto its control object, and its reference fit walked the
        # interval from 1e6 down to 62,500 proposals monotonically over 34
        # iterations. Restarting each iteration would re-pay every discarded
        # attempt.
        for attempt in range(control.max_resamples):
            draws, state = simulate(
                data, theta, draw_count, control.burn_in, interval, state,
                n_chains=control.n_chains,
                # The attempt index enters the seed so a resample draws a new
                # sample rather than replaying the one that fell short.
                seed=control.seed + 1000 * iteration + 17 * attempt,
            )
            sweeps_drawn += (len(draws) * interval
                             + control.burn_in * control.n_chains)
            smallest_ess = float(
                effective_sample_size(draws, n_chains=control.n_chains).min()
            )
            if smallest_ess >= control.target_ess:
                break
            log.info(
                "iteration %d: effective sample size %.0f below target %.0f; "
                "shortening interval %d -> %d and taking %d draws",
                iteration, smallest_ess, control.target_ess, interval,
                max(1, int(interval / control.interval_drop)),
                int(draw_count * control.interval_drop),
            )
            interval = max(1, int(interval / control.interval_drop))
            draw_count = int(draw_count * control.interval_drop)

        inside, statistic, threshold = within_tolerance(
            g_obs, draws, control.confidence, control.precision,
            n_chains=control.n_chains
        )
        record = {"iteration": iteration, "statistic": statistic,
                  "threshold": threshold, "min_ess": smallest_ess,
                  "interval": interval, "n_draws": len(draws),
                  "sweeps": sweeps_drawn}

        if inside:
            log.info(
                "iteration %d of at most %d: p = %.4g. Converged with %g%% "
                "confidence.", iteration, control.max_iterations, statistic,
                control.confidence * 100,
            )
            converged = True
            history.append({**record, "step_length": 1.0})
            break

        # Stop short of the hull boundary, as MCMLE.steplength.margin does.
        gamma = min(1.0, shrink_into_ch(g_obs, draws)) * (1.0 - control.step_margin)
        target = gamma * g_obs + (1.0 - gamma) * draws.mean(axis=0)
        theta = geyer_thompson_step(theta, draws, target)
        log.info(
            "iteration %d of at most %d: p = %.4g. Not converged; step length "
            "%.3f, %d draws at interval %d, effective sample size %.0f.",
            iteration, control.max_iterations, statistic, gamma,
            len(draws), interval, smallest_ess,
        )
        history.append({**record, "step_length": float(gamma)})

    # The last iteration's draws are what the convergence test approved, so
    # they are what the covariance comes from. Drawing a fresh sample here
    # would compute standard errors from one the effective-sample-size gate
    # never saw, and would add an uncounted iteration's worth of sampling.
    if not converged:
        log.warning(
            "MCMLE did not converge in %d iterations; the estimates may not be "
            "accurate. Raise max_iterations, or start from a better seed.",
            control.max_iterations,
        )
    covariance = np.cov(draws, rowvar=False)
    std_error = np.sqrt(np.diag(np.linalg.pinv(covariance)))
    return MCMLEResult(theta, std_error, len(history), converged, history,
                       control, sweeps_drawn)

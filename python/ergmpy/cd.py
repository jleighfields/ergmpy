"""Contrastive divergence, used to seed Monte Carlo maximum likelihood.

Standard MCMLE simulates to equilibrium at the current parameter. Started far
from the truth that is unstable: at the pseudo-likelihood estimate, whose
b2star2 coefficient is roughly 2.5 times the true one, the popularity term
compounds and the simulated statistic reaches 537,000 against an observed
299,000 -- 69 standard deviations out, where the importance-sampling
approximation has no support.

Contrastive divergence (Krivitsky 2017) avoids that by never letting the chain
leave the data's neighbourhood: every draw restarts at the observed purchases
and takes only a few single-customer updates. The resulting gradient is biased,
so it is not an estimator to report, but it cannot run away, and it lands close
enough to seed MCMLE.

The step machinery is shared with `ergmpy.mcmle` rather than duplicated -- only
how the draws are generated differs.
"""

import numpy as np

from ergmpy import sampler
from ergmpy.choice.predict import ChoiceData
from ergmpy.convex_hull import shrink_into_ch
from ergmpy.mcmle import geyer_thompson_step, observed_statistics


def draw_statistics(data: ChoiceData, theta: np.ndarray, n_draws: int,
                    n_updates: int, generator: np.random.Generator) -> np.ndarray:
    """Simulates short excursions away from the observed configuration.

    Args:
        data: The dataset defining choice sets and product attributes.
        theta: (8,) parameter vector to simulate at.
        n_draws: Number of excursions.
        n_updates: Single-customer updates per excursion.
        generator: Source of the customers each excursion visits.

    Returns:
        (n_draws, 8) statistics, one row per excursion.
    """
    choice_sets = np.ascontiguousarray(data.choice_sets)
    linear = np.ascontiguousarray(data.design @ theta[:7])
    theta_star2 = float(theta[7])
    observed_degree = data.degree.astype(np.int64)

    draws = np.empty((n_draws, 8))
    for m in range(n_draws):
        current = data.chosen.astype(np.int32).copy()
        degree = observed_degree.copy()
        customers = generator.integers(0, len(current), size=n_updates,
                                       dtype=np.int32)
        sampler.updates_numba(choice_sets, current, degree, linear,
                              theta_star2, customers)
        draws[m] = sampler.network_statistics(choice_sets, current, data.design,
                                              data.n_products)
    return draws


def fit(data: ChoiceData, theta0: np.ndarray, max_iterations: int = 60,
        n_draws: int = 400, n_updates: int = 500, tolerance: float = 0.01,
        seed: int = 123) -> tuple[np.ndarray, list[dict]]:
    """Runs contrastive divergence to produce a starting parameter for MCMLE.

    Args:
        data: The dataset to fit.
        theta0: (8,) starting parameter.
        max_iterations: Cap on iterations.
        n_draws: Excursions per iteration.
        n_updates: Single-customer updates per excursion.
        tolerance: Convergence threshold on the largest standardized gap
            between observed and simulated mean statistics.
        seed: Seed for the excursion customers and the sampler.

    Returns:
        The (8,) seed parameter and one history dict per iteration.
    """
    generator = np.random.default_rng(seed)
    np.random.seed(seed)
    g_obs = observed_statistics(data)
    theta = np.asarray(theta0, dtype=float).copy()
    history: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        draws = draw_statistics(data, theta, n_draws, n_updates, generator)
        spread = draws.std(axis=0)
        varying = spread > 1e-12
        gap = np.zeros_like(spread)
        gap[varying] = np.abs(g_obs - draws.mean(axis=0))[varying] / spread[varying]
        history.append({"iteration": iteration,
                        "max_standardized_gap": float(gap.max())})
        if gap.max() < tolerance:
            break
        # Every excursion starts at the observed network, but with enough
        # updates the draws travel far enough that g_obs leaves their convex
        # hull again, and an unshrunk step then diverges. Shrink as MCMLE does.
        gamma = min(1.0, shrink_into_ch(g_obs, draws))
        target = gamma * g_obs + (1.0 - gamma) * draws.mean(axis=0)
        theta = geyer_thompson_step(theta, draws, target)

    return theta, history

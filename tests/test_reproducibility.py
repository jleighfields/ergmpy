"""The same seed must give the same answer.

This property has broken three times, in three modules, and was caught by hand
each time. The cause is always the same: inside `@njit`, numpy's random
functions draw from numba's own generator, which `np.random.seed` cannot reach
from the interpreter. Seeding numpy alone therefore looks like seeding and
does nothing, and the failure is silent -- runs still work, they just are not
reproducible, and the settings documented as matching `ergm`'s `seed = 123`
quietly do not.

Every entry point that takes a seed is checked here, in both directions: the
same seed reproduces, and a different one does not. The second half matters as
much as the first, because a function returning a constant would pass the
first on its own.
"""

import numpy as np

from ergmpy import contrastive_divergence, mcmle
from ergmpy.choice import mple


def test_a_chain_reproduces_from_the_same_seed(train) -> None:
    """`simulate_chain` is the sampler's entry point and draws through numba."""
    theta = np.array([-3.0, -0.04, 1.6, 1.2, 2.2, 1.2, 1.2, 0.0058])
    first, _ = mcmle.simulate_chain(train, theta, 4, 5, 5, seed=42)
    again, _ = mcmle.simulate_chain(train, theta, 4, 5, 5, seed=42)
    np.testing.assert_array_equal(first, again)


def test_a_chain_differs_from_a_different_seed(train) -> None:
    """Without this, a kernel returning a constant would pass the test above."""
    theta = np.array([-3.0, -0.04, 1.6, 1.2, 2.2, 1.2, 1.2, 0.0058])
    first, _ = mcmle.simulate_chain(train, theta, 4, 5, 5, seed=42)
    other, _ = mcmle.simulate_chain(train, theta, 4, 5, 5, seed=43)
    assert not np.array_equal(first, other)


def test_contrastive_divergence_reproduces_from_the_same_seed(train) -> None:
    """Its excursions run through the compiled kernel too.

    This is where the third instance of the defect lived: `fit` seeded numpy
    and not numba, so its `seed` argument had no effect at all.
    """
    start = mple.fit(train).coef
    first, _ = contrastive_divergence.fit(train, start, max_iterations=2,
                                          n_draws=20, n_updates=2000, seed=7)
    again, _ = contrastive_divergence.fit(train, start, max_iterations=2,
                                          n_draws=20, n_updates=2000, seed=7)
    np.testing.assert_allclose(first, again, rtol=0, atol=0)


def test_contrastive_divergence_differs_from_a_different_seed(train) -> None:
    """The seed has to reach the draws, not merely be accepted."""
    start = mple.fit(train).coef
    first, _ = contrastive_divergence.fit(train, start, max_iterations=2,
                                          n_draws=20, n_updates=2000, seed=7)
    other, _ = contrastive_divergence.fit(train, start, max_iterations=2,
                                          n_draws=20, n_updates=2000, seed=8)
    assert not np.allclose(first, other)


def test_the_pseudo_likelihood_fit_is_deterministic(train) -> None:
    """It draws nothing, so it must agree exactly with itself."""
    np.testing.assert_array_equal(mple.fit(train).coef, mple.fit(train).coef)

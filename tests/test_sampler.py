"""The Gibbs kernel, checked where randomness can be removed or averaged out."""

import numpy as np
from ergmpy import sampler
from ergmpy.choice.predict import softmax_utilities


def test_degree_stays_consistent_with_choices(train) -> None:
    """Degrees are maintained incrementally and must match a recount."""
    choice_sets = np.ascontiguousarray(train.choice_sets)
    current = train.chosen.astype(np.int32).copy()
    degree = np.bincount(current, minlength=train.n_products).astype(np.int64)
    linear = np.ascontiguousarray(train.design @ np.arange(7, dtype=float) * 0.1)

    np.random.seed(0)
    sampler.run_sweeps(choice_sets, current, degree, linear, 0.01, 5)

    recounted = np.bincount(current, minlength=train.n_products).astype(np.int64)
    np.testing.assert_array_equal(degree, recounted)


def test_every_choice_stays_inside_its_consideration_set(train) -> None:
    """The constraint forbids edges outside a customer's own alternatives."""
    choice_sets = np.ascontiguousarray(train.choice_sets)
    current = train.chosen.astype(np.int32).copy()
    degree = np.bincount(current, minlength=train.n_products).astype(np.int64)
    linear = np.ascontiguousarray(train.design @ np.full(7, 0.3))

    np.random.seed(1)
    sampler.run_sweeps(choice_sets, current, degree, linear, 0.02, 3)

    assert (choice_sets == current[:, None]).any(axis=1).all()


def test_marginals_match_the_closed_form_when_customers_are_independent(train) -> None:
    """With theta_star2 = 0 the customers decouple and the answer is analytic.

    Removing the dependence term makes each customer's marginal exactly the
    softmax of their attribute utilities, so the sampler can be checked against
    a formula rather than against itself.
    """
    subset = np.ascontiguousarray(train.choice_sets[:300])
    linear = np.ascontiguousarray(train.design @ np.array(
        [-2.0, 0.1, 1.2, 0.5, 1.0, 0.8, 0.9]))
    current = subset[:, 0].copy().astype(np.int32)
    degree = np.bincount(current, minlength=train.n_products).astype(np.int64)

    n_sweeps = 4000
    counts = np.zeros(subset.shape, dtype=float)
    np.random.seed(2)
    for _ in range(n_sweeps):
        sampler.run_sweeps(subset, current, degree, linear, 0.0, 1)
        counts += subset == current[:, None]

    empirical = counts / n_sweeps
    analytic = softmax_utilities(linear[subset])
    # Binomial standard error at this sweep count is ~0.008; 6 sigma leaves
    # room for the worst of 1800 cells without hiding a real bias.
    np.testing.assert_allclose(empirical, analytic, atol=0.05)


def test_compiled_and_plain_kernels_agree_when_the_choice_is_forced(train) -> None:
    """The two kernels select identically once randomness is removed.

    They cannot be compared draw for draw: inside @njit, numpy's random
    functions run against numba's own generator state, which `np.random.seed`
    called from Python does not touch. Seeding both therefore fixes two
    different streams. What is shared is the source definition, so giving one
    alternative an overwhelming utility makes the sampled choice deterministic
    and any divergence in the selection logic visible.
    """
    subset = np.ascontiguousarray(train.choice_sets[:150])
    order = np.arange(subset.shape[0], dtype=np.int32)

    # Utility strictly increasing in the product index gives every customer a
    # unique best alternative -- their highest-indexed one -- with a gap wide
    # enough that its probability rounds to 1. Marking a per-customer slot
    # instead would not work: products recur across consideration sets, so some
    # customers would end up with two dominant alternatives and a coin flip.
    linear = np.ascontiguousarray(np.arange(train.n_products, dtype=float) * 100.0)
    expected = subset.max(axis=1)

    results = []
    for kernel in (sampler.updates_python, sampler.updates_numba):
        current = subset[:, 0].copy().astype(np.int32)
        degree = np.bincount(current, minlength=train.n_products).astype(np.int64)
        kernel(subset, current, degree, linear, 0.0, order)
        results.append(current.copy())

    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], expected)


def test_both_kernels_reproduce_the_same_marginals(train) -> None:
    """What the two kernels share is a distribution, not a random stream."""
    subset = np.ascontiguousarray(train.choice_sets[:150])
    linear = np.ascontiguousarray(train.design @ np.array(
        [-2.0, 0.1, 1.2, 0.5, 1.0, 0.8, 0.9]))
    order = np.arange(subset.shape[0], dtype=np.int32)

    n_sweeps = 3000
    marginals = []
    for kernel in (sampler.updates_python, sampler.updates_numba):
        current = subset[:, 0].copy().astype(np.int32)
        degree = np.bincount(current, minlength=train.n_products).astype(np.int64)
        counts = np.zeros(subset.shape, dtype=float)
        np.random.seed(3)
        for _ in range(n_sweeps):
            kernel(subset, current, degree, linear, 0.0, order)
            counts += subset == current[:, None]
        marginals.append(counts / n_sweeps)

    np.testing.assert_allclose(marginals[0], marginals[1], atol=0.06)

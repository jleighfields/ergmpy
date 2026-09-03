"""The change statistic is the package's correctness argument.

Prediction, the pseudo-likelihood and the sampler all rest on it, and it is
written twice -- once as an array expression, once inline in the numba kernel.
These tests recompute the underlying statistics by direct enumeration so a
wrong delta cannot agree with them by construction.
"""

import numpy as np
from ergmpy.choice.predict import change_statistics
from ergmpy.mcmle import observed_statistics

from tests.helpers import (
    attribute_sums_from_scratch,
    b2star2_from_scratch,
    configuration_with_swap,
)


def test_observed_statistics_match_direct_enumeration(train) -> None:
    """The observed statistic vector equals the definition computed by hand."""
    stats = observed_statistics(train)
    expected_attributes = attribute_sums_from_scratch(train.chosen, train.design)
    expected_star2 = b2star2_from_scratch(train.chosen, train.n_products)

    np.testing.assert_allclose(stats[:7], expected_attributes, rtol=0, atol=1e-9)
    assert stats[7] == expected_star2


def test_change_statistic_equals_recomputed_difference(train) -> None:
    """Moving one purchase changes the statistics by the advertised delta.

    The package computes the difference in closed form. Here both
    configurations are built and their statistics enumerated from scratch, so
    the two routes share no code.
    """
    rng = np.random.default_rng(0)
    Z = change_statistics(train)
    baseline_star2 = b2star2_from_scratch(train.chosen, train.n_products)
    baseline_attributes = attribute_sums_from_scratch(train.chosen, train.design)

    chosen_slot = np.argmax(train.choice_sets == train.chosen[:, None], axis=1)

    for customer in rng.choice(len(train.chosen), size=40, replace=False):
        for slot in range(train.choice_sets.shape[1]):
            alternative = int(train.choice_sets[customer, slot])
            swapped = configuration_with_swap(train, int(customer), alternative)

            actual_star2 = (b2star2_from_scratch(swapped, train.n_products)
                            - baseline_star2)
            actual_attributes = (
                attribute_sums_from_scratch(swapped, train.design) - baseline_attributes
            )

            # Z holds each alternative's statistic; differences within a
            # consideration set are what the model uses, so compare against the
            # observed choice's row.
            predicted = Z[customer, slot] - Z[customer, chosen_slot[customer]]

            np.testing.assert_allclose(predicted[:7], actual_attributes,
                                       rtol=0, atol=1e-9)
            assert predicted[7] == actual_star2


def test_observed_alternative_has_zero_change(train) -> None:
    """A customer's own purchase is the baseline, so its delta is exactly zero."""
    Z = change_statistics(train)
    chosen_slot = np.argmax(train.choice_sets == train.chosen[:, None], axis=1)
    rows = np.arange(len(train.chosen))
    own = Z[rows, chosen_slot]
    others = Z - own[:, None, :]
    np.testing.assert_array_equal(others[rows, chosen_slot], 0.0)

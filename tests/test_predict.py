"""Choice probabilities, including a regression against R's own output.

The comparison file was produced by `ergm` and committed, so this needs no R
installation and runs in milliseconds -- which is why it belongs here rather
than in `benchmarks/`, where the rule about R comparisons is aimed at things
needing a live R and minutes of runtime.
"""

import numpy as np
import polars as pl

from ergmpy.choice.predict import (
    TERM_NAMES,
    choice_probabilities,
    load,
    top_n_accuracy,
)


def test_probabilities_sum_to_one(train) -> None:
    """Every customer's consideration set is a distribution."""
    theta = np.array([-2.0, 0.1, 1.2, 0.5, 1.0, 0.8, 0.9])
    probabilities = choice_probabilities(train, theta, 0.01)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_choice_sets_are_well_formed(train) -> None:
    """Each customer considers a fixed set and buys exactly one of its members."""
    assert train.choice_sets.shape[1] == 6
    assert (train.choice_sets >= 0).all()
    assert (train.choice_sets == train.chosen[:, None]).sum(axis=1).max() >= 1
    assert (train.choice_sets == train.chosen[:, None]).any(axis=1).all()


def test_degree_equals_purchase_counts(train) -> None:
    """`degree` is the product's purchase count, and nothing else."""
    recounted = np.bincount(train.chosen, minlength=train.n_products)
    np.testing.assert_array_equal(train.degree, recounted)


def test_matches_the_probability_matrix_ergm_produced(test_set, recorded_r) -> None:
    """Reproduces R's saved prob_y_star for the coefficients that made it."""
    coefficients = pl.read_csv(recorded_r / "r" / "coef_star_maxit2.csv")
    lookup = dict(zip(coefficients["term"].to_list(),
                      coefficients["estimate"].cast(pl.Float64, strict=False).to_list(),
                      strict=True))
    theta_linear = np.array([lookup[t] for t in TERM_NAMES[:7]])

    probabilities = choice_probabilities(test_set, theta_linear, lookup["b2star2"])

    r_matrix = np.loadtxt(recorded_r / "r" / "prob_star_maxit2_n200.csv",
                          delimiter=",", skiprows=1)
    compared = 200
    rows = np.arange(compared)[:, None]
    r_aligned = r_matrix[rows, test_set.choice_sets[:compared]]

    np.testing.assert_allclose(probabilities[:compared], r_aligned, rtol=0, atol=1e-11)


def test_top_n_accuracy_is_monotone(train) -> None:
    """Widening the window cannot lose a hit."""
    theta = np.array([-2.0, 0.1, 1.2, 0.5, 1.0, 0.8, 0.9])
    probabilities = choice_probabilities(train, theta, 0.01)
    scores = [top_n_accuracy(train, probabilities, n) for n in (1, 2, 3, 6)]
    assert scores == sorted(scores)
    assert scores[-1] == 1.0


def test_ragged_choice_sets_are_rejected(tmp_path) -> None:
    """A customer with fewer alternatives means the wrong file, not a variation."""
    rows = ["rspd_id,model_id,purchase,V1,V2,V3,V4"]
    for customer in range(3):
        alternatives = 6 if customer else 5
        for alternative in range(alternatives):
            bought = 1 if alternative == 0 else 0
            rows.append(f"{customer},{alternative},{bought},0.1,0.2,0.3,A")
    path = tmp_path / "ragged.csv"
    path.write_text("\n".join(rows) + "\n")

    try:
        load(str(path))
    except ValueError:
        return
    raise AssertionError("a ragged choice set was accepted")

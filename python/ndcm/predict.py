"""Choice probabilities for the bipartite-ERGM discrete choice model.

Part 4 of the reference R script computes, for every customer, the probability
of each alternative in their consideration set by moving that customer's single
purchase edge and re-evaluating the network statistics. It does so by calling
`summary(formula)` on the whole 5,300-node network once per alternative, which
recomputes every statistic from scratch.

Only two statistics change when one customer's edge moves from product k to
product j, and both change by a closed-form amount:

  b2cov / b2factor  the product's own attribute contribution, x_j - x_k
  b2star2           sum_p C(d_p, 2) over product degrees d. Moving the edge
                    sends d_k -> d_k - 1 and d_j -> d_j + 1, and since
                    C(n+1,2) - C(n,2) = n and C(n-1,2) - C(n,2) = -(n-1),
                    the total change is d_j - d_k + 1.

`edges` is constant under the b1degrees constraint and the edgecov offset is
zero on every configuration in the sample space, so neither contributes. That
makes each alternative O(1) instead of a full network traversal, and the whole
customer x alternative grid a single vectorized expression.
"""

import numpy as np
import polars as pl

# Non-baseline levels of the categorical attribute. R's relevel(factor(V4), "A")
# orders levels A,B,C,D,E, so ergm's b2factor.V4.2 through .V4.5 are B through E.
V4_LEVELS = ("B", "C", "D", "E")


class ChoiceData:
    """Choice sets, product attributes, and product degrees for one dataset.

    Attributes:
        choice_sets: (n_customers, set_size) product indices, in file order.
        chosen: (n_customers,) product index each customer actually bought.
        design: (n_products, 7) product design matrix, V1-V3 then the four
            non-baseline V4 dummies.
        degree: (n_products,) number of customers who bought each product.
        n_products: Number of distinct products.
    """

    def __init__(self, choice_sets: np.ndarray, chosen: np.ndarray,
                 design: np.ndarray, degree: np.ndarray) -> None:
        self.choice_sets = choice_sets
        self.chosen = chosen
        self.design = design
        self.degree = degree
        self.n_products = design.shape[0]


def load(path: str) -> ChoiceData:
    """Reads a reference CSV into choice sets, a design matrix, and degrees.

    Product indices follow first-appearance order in the file, matching the
    `unique(df$model_id)` ordering the R script uses to number network nodes.

    Args:
        path: Path to a `*_data_to_share.csv` file.

    Returns:
        The assembled ChoiceData.
    """
    df = pl.read_csv(path)
    products = df["model_id"].unique(maintain_order=True)
    product_index = {p: i for i, p in enumerate(products.to_list())}
    df = df.with_columns(pl.col("model_id").replace_strict(product_index).alias("pidx"))

    set_size = df.group_by("rspd_id").len()["len"].max()
    # Customers also follow first-appearance order, so rows group contiguously.
    customers = df["rspd_id"].unique(maintain_order=True).to_list()
    customer_index = {c: i for i, c in enumerate(customers)}
    df = df.with_columns(pl.col("rspd_id").replace_strict(customer_index).alias("cidx"))

    n_customers = len(customers)
    choice_sets = np.full((n_customers, set_size), -1, dtype=np.int32)
    slot = np.zeros(n_customers, dtype=np.int32)
    cidx = df["cidx"].to_numpy()
    pidx = df["pidx"].to_numpy()
    for c, p in zip(cidx, pidx):
        choice_sets[c, slot[c]] = p
        slot[c] += 1
    if (choice_sets < 0).any():
        raise ValueError("ragged choice sets: some customer has fewer rows than others")

    bought = df.filter(pl.col("purchase") == 1)
    chosen = np.empty(n_customers, dtype=np.int32)
    chosen[bought["cidx"].to_numpy()] = bought["pidx"].to_numpy()

    attrs = df.unique(subset=["pidx"], maintain_order=True).sort("pidx")
    design = np.column_stack(
        [attrs["V1"].to_numpy(), attrs["V2"].to_numpy(), attrs["V3"].to_numpy()]
        + [(attrs["V4"].to_numpy() == lvl).astype(float) for lvl in V4_LEVELS]
    )

    degree = np.bincount(chosen, minlength=len(products)).astype(np.int64)
    return ChoiceData(choice_sets, chosen, design, degree)


def change_statistics(data: ChoiceData) -> np.ndarray:
    """Builds the change statistic of every alternative, conditional on the rest.

    Holding all other customers' purchases fixed, the statistic vector for
    customer i choosing product j is the product's own attributes together with
    the b2star2 contribution. Since C(n+1,2) - C(n,2) = n, that contribution is
    the degree j would have *without* customer i's own edge:

        d_minus_i[j] = degree[j] - 1 if j is i's observed purchase else degree[j]

    Only differences within a consideration set matter to the multinomial logit
    that consumes this, so no reference level has to be chosen here.

    Args:
        data: The dataset to build statistics for.

    Returns:
        (n_customers, set_size, 8) change statistics: V1-V3, the four V4
        dummies, then the b2star2 term.
    """
    cs = data.choice_sets
    degree_minus_i = data.degree[cs] - (cs == data.chosen[:, None])
    return np.concatenate(
        [data.design[cs], degree_minus_i[:, :, None].astype(float)], axis=2
    )


def choice_probabilities(data: ChoiceData, theta_linear: np.ndarray,
                         theta_star2: float) -> np.ndarray:
    """Computes each customer's probability over their consideration set.

    Every other customer's purchase is held at its observed value, matching the
    conditional-on-the-rest quantity the R script computes.

    Args:
        data: The dataset to score.
        theta_linear: (7,) coefficients for V1-V3 and the four V4 dummies.
        theta_star2: Coefficient on the b2star2 statistic.

    Returns:
        (n_customers, set_size) probabilities, aligned with `data.choice_sets`.
    """
    theta = np.concatenate([np.asarray(theta_linear, dtype=float), [theta_star2]])
    return softmax_utilities(change_statistics(data) @ theta)


def softmax_utilities(utility: np.ndarray) -> np.ndarray:
    """Converts row-wise utilities to probabilities, shifted for stability.

    Args:
        utility: (n_customers, set_size) utilities.

    Returns:
        (n_customers, set_size) probabilities summing to 1 along each row.
    """
    utility = utility - utility.max(axis=1, keepdims=True)
    exp_u = np.exp(utility)
    return exp_u / exp_u.sum(axis=1, keepdims=True)


def top_n_accuracy(data: ChoiceData, probabilities: np.ndarray, n: int = 3) -> float:
    """Fraction of customers whose actual purchase ranks in the top n.

    Args:
        data: The dataset scored.
        probabilities: Output of `choice_probabilities`.
        n: How many ranks count as a hit.

    Returns:
        The accuracy in [0, 1].
    """
    chosen_prob = probabilities[data.choice_sets == data.chosen[:, None]]
    rank = (probabilities > chosen_prob[:, None]).sum(axis=1)
    return float((rank < n).mean())

"""Plain functions used by the tests, readable without knowing pytest."""

import numpy as np

from ergmpy.choice.predict import ChoiceData


def b2star2_from_scratch(chosen: np.ndarray, n_products: int) -> int:
    """Counts two-stars centred on products by direct enumeration.

    This is the definition -- sum of C(d, 2) over product degrees -- computed
    without any of the incremental reasoning the package relies on, so it can
    contradict the closed-form change statistic.

    Args:
        chosen: (n_customers,) product index each customer bought.
        n_products: Number of distinct products.

    Returns:
        The b2star2 statistic.
    """
    degree = np.bincount(chosen, minlength=n_products)
    return int(sum(d * (d - 1) // 2 for d in degree))


def attribute_sums_from_scratch(chosen: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Sums each product attribute over the purchased edges, one edge at a time.

    Args:
        chosen: (n_customers,) product index each customer bought.
        design: (n_products, 7) product design matrix.

    Returns:
        (7,) attribute sums.
    """
    total = np.zeros(design.shape[1])
    for product in chosen:
        total = total + design[product]
    return total


def configuration_with_swap(data: ChoiceData, customer: int, product: int) -> np.ndarray:
    """Returns the purchase vector with one customer's choice replaced.

    Args:
        data: The dataset.
        customer: Index of the customer to move.
        product: Product index they should buy instead.

    Returns:
        A copy of `data.chosen` with that one entry changed.
    """
    swapped = data.chosen.copy()
    swapped[customer] = product
    return swapped

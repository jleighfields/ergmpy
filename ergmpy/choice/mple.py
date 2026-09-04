"""Maximum pseudo-likelihood estimation for the constrained bipartite ERGM.

Under `constraints = ~b1degrees` with the consideration-set offset at -Inf, the
sample space is exactly one purchase per customer. Conditioning on every other
customer's purchase therefore leaves customer i's choice multinomial over their
own consideration set, with the change statistics as covariates -- so the
pseudo-likelihood is a conditional logit, and its gradient and Hessian are the
standard multinomial-logit ones.

This is not the estimator `ergm(..., estimate = "MPLE")` computes. ergm forms
its pseudo-likelihood dyad by dyad, as a binary logit over consideration-set
dyads, which drops the one-purchase-per-customer constraint. On this model that
version reports coefficients whose signs contradict ergm's own MCMLE fit, along
with a separability warning.
"""

import numpy as np
import scipy.optimize
from scipy.special import logsumexp

from ergmpy.choice.predict import (
    TERM_NAMES,
    ChoiceData,
    change_statistics,
    softmax_utilities,
)


class MPLEResult:
    """Fitted pseudo-likelihood estimates.

    Attributes:
        coef: (8,) point estimates, ordered as TERM_NAMES.
        std_error: (8,) asymptotic standard errors from the inverse Hessian.
        log_pseudo_likelihood: Value of the objective at the optimum.
        n_iterations: Optimizer iteration count.
        hessian: (8, 8) observed information at the optimum.
    """

    def __init__(self, coef: np.ndarray, std_error: np.ndarray, llk: float,
                 n_iterations: int, hessian: np.ndarray) -> None:
        """Stores the estimates; the class docstring describes each attribute."""
        self.coef = coef
        self.std_error = std_error
        self.log_pseudo_likelihood = llk
        self.n_iterations = n_iterations
        self.hessian = hessian

    def summary(self) -> str:
        """Formats the estimates the way ergm's summary does.

        Returns:
            A table of term, estimate, standard error and z value.
        """
        z = self.coef / self.std_error
        lines = [f"{'term':<16}{'Estimate':>13}{'Std. Error':>13}{'z value':>11}"]
        for name, c, s, zz in zip(TERM_NAMES, self.coef, self.std_error, z, strict=True):
            lines.append(f"{name:<16}{c:>13.6f}{s:>13.6f}{zz:>11.3f}")
        lines.append(f"\nlog pseudo-likelihood: {self.log_pseudo_likelihood:.6f}")
        return "\n".join(lines)


def negative_log_pseudo_likelihood(theta: np.ndarray, Z: np.ndarray,
                                   chosen_slot: np.ndarray) -> tuple[float, np.ndarray]:
    """Evaluates the objective and its gradient.

    Args:
        theta: (8,) parameter vector.
        Z: (n_customers, set_size, 8) change statistics.
        chosen_slot: (n_customers,) index of the observed choice within each set.

    Returns:
        The negative log pseudo-likelihood and its (8,) gradient.
    """
    utility = Z @ theta
    rows = np.arange(Z.shape[0])
    llk = float((utility[rows, chosen_slot] - logsumexp(utility, axis=1)).sum())

    probabilities = softmax_utilities(utility)
    expected = np.einsum("nk,nkp->np", probabilities, Z)
    gradient = (Z[rows, chosen_slot] - expected).sum(axis=0)
    return -llk, -gradient


def observed_information(theta: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Computes the negative Hessian of the log pseudo-likelihood.

    Args:
        theta: (8,) parameter vector.
        Z: (n_customers, set_size, 8) change statistics.

    Returns:
        The (8, 8) observed information matrix.
    """
    probabilities = softmax_utilities(Z @ theta)
    expected = np.einsum("nk,nkp->np", probabilities, Z)
    second = np.einsum("nk,nkp,nkq->npq", probabilities, Z, Z)
    return (second - np.einsum("np,nq->npq", expected, expected)).sum(axis=0)


def fit(data: ChoiceData) -> MPLEResult:
    """Fits the model by maximum pseudo-likelihood.

    Args:
        data: The dataset to fit.

    Returns:
        The fitted MPLEResult.
    """
    Z = change_statistics(data)
    chosen_slot = np.argmax(data.choice_sets == data.chosen[:, None], axis=1)

    result = scipy.optimize.minimize(
        negative_log_pseudo_likelihood, x0=np.zeros(Z.shape[2]),
        args=(Z, chosen_slot), jac=True, method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    information = observed_information(result.x, Z)
    std_error = np.sqrt(np.diag(np.linalg.inv(information)))
    return MPLEResult(result.x, std_error, -result.fun, result.nit, information)

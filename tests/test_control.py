"""The control's defaults, checked against what ergm recorded.

A comparison against `ergm` is only meaningful if both sides were asked for the
same thing. These pin the defaults to the settings exported from a fitted ergm
object, so a change to either surfaces here rather than quietly weakening the
comparison.
"""

import csv
from pathlib import Path

import pytest

from ergmpy.control import MCMLEControl

N_CUSTOMERS = 5000
SETTINGS = (Path(__file__).resolve().parents[1] / "results" / "r"
            / "control_settings.csv")


def recorded(setting: str) -> float:
    """Reads one setting from what a fitted ergm object recorded.

    Args:
        setting: The `control.ergm` parameter name.

    Returns:
        Its value for the converged star fit.
    """
    for row in csv.DictReader(SETTINGS.open()):
        if row["fit"] == "fit_star" and row["setting"] == setting:
            return float(row["value"])
    raise AssertionError(f"{setting} not exported; see export_control_settings.R")


@pytest.mark.skipif(not SETTINGS.exists(), reason="R settings not exported")
def test_the_tolerance_region_matches_what_ergm_recorded() -> None:
    """The tolerance scale is read from ergm's own fit, not transcribed.

    This setting was 0.5 for a while against ergm's 0.1, which made the
    stopping rule accept gaps ergm refuses. Nothing caught it because the
    value lived only in Python.
    """
    assert MCMLEControl().precision == recorded("MCMLE.MCMC.precision")


def test_defaults_match_what_the_r_script_requests() -> None:
    """The four settings the script names explicitly."""
    control = MCMLEControl()
    assert control.n_draws == 1250          # MCMC.samplesize
    assert control.thin * N_CUSTOMERS == 1_000_000  # MCMC.interval, in proposals
    assert control.n_chains == 4            # parallel
    assert control.seed == 123              # seed


def test_defaults_match_the_published_iteration_limit() -> None:
    """The authors' published output reports MCMLE.maxit = 200."""
    assert MCMLEControl().max_iterations == 200


def test_defaults_match_ergms_own_defaults() -> None:
    """Settings the script leaves alone, carried over from ergm."""
    control = MCMLEControl()
    assert control.confidence == 0.99
    assert control.interval_drop == 2.0
    assert control.max_resamples == 16
    assert control.step_margin == 0.05


def test_every_field_names_its_ergm_counterpart() -> None:
    """The mapping is the point; a field without one cannot be checked."""
    control = MCMLEControl()
    assert set(control.to_dict()) == set(MCMLEControl.ERGM_EQUIVALENT)




def test_settings_that_would_make_a_fit_do_nothing_are_refused() -> None:
    """Each of these once produced a well-formed wrong answer rather than an error.

    No resamples left the draws unbound; no iterations returned the starting
    parameter with standard errors as though a fit had run. Both are silent, so
    the guards are only worth having if they are known to fire.
    """
    for kwargs in ({"max_resamples": 0}, {"max_iterations": 0},
                   {"n_draws": 2, "n_chains": 4}):
        with pytest.raises(ValueError):
            MCMLEControl(**kwargs)

"""The control's defaults, checked against what ergm recorded.

A comparison against `ergm` is only meaningful if both sides were asked for the
same thing. These pin the defaults to the settings exported from a fitted ergm
object, so a change to either surfaces here rather than quietly weakening the
comparison.
"""

from pathlib import Path

import pytest

from ergmpy.control import CDControl, MCMLEControl

N_CUSTOMERS = 5000
SETTINGS = Path(__file__).resolve().parents[1] / "results" / "r" / "control_settings.csv"


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


@pytest.mark.skipif(not SETTINGS.exists(), reason="R settings not exported")
def test_control_can_be_built_from_ergms_recorded_settings() -> None:
    """Matching a run reads ergm's settings rather than retyping them.

    ergm adapts its sample size and interval during a fit, so the settings
    worth matching are the ones it ended with, not the ones it was asked for.
    """
    control = MCMLEControl.from_ergm_settings(SETTINGS, N_CUSTOMERS)
    assert control.n_draws > 0
    assert control.thin >= 1
    assert control.n_chains == 4
    assert control.confidence == 0.99


def test_a_missing_settings_file_is_reported_not_ignored() -> None:
    """Silently returning defaults would produce a comparison against nothing."""
    with pytest.raises(FileNotFoundError):
        MCMLEControl.from_ergm_settings("does/not/exist.csv", N_CUSTOMERS)


def test_an_unknown_fit_name_is_reported() -> None:
    """A typo in the fit name must not fall back to another fit's settings."""
    if not SETTINGS.exists():
        pytest.skip("R settings not exported")
    with pytest.raises(ValueError):
        MCMLEControl.from_ergm_settings(SETTINGS, N_CUSTOMERS, fit="no_such_fit")


def test_every_field_names_its_ergm_counterpart() -> None:
    """The mapping is the point; a field without one cannot be checked."""
    control = MCMLEControl()
    assert set(control.to_dict()) == set(MCMLEControl.ERGM_EQUIVALENT)


def test_cd_defaults_are_recorded() -> None:
    """CD settings round-trip for recording beside a run."""
    assert CDControl().to_dict()["n_updates"] == 50000

"""The control's defaults, checked against what ergm recorded.

A comparison against `ergm` is only meaningful if both sides were asked for the
same thing. These pin the defaults to the settings exported from a fitted ergm
object, so a change to either surfaces here rather than quietly weakening the
comparison.
"""

from ergmpy.control import MCMLEControl

N_CUSTOMERS = 5000


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



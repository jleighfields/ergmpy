"""Run settings for the estimators, mirroring `control.ergm`.

Every knob lives here rather than as a default buried in a function signature,
for two reasons. The settings interact -- shortening the interval while raising
the draw count is one decision, not two -- and a measurement is only
interpretable beside the settings that produced it, so they have to be
recordable as a unit.

Each field names the `control.ergm` parameter it corresponds to. That mapping
is the point: this package is checked against `ergm`, and a comparison is only
as good as the reader's ability to confirm both sides were asked for the same
thing. `results/r/control_settings.csv` holds what a fitted ergm object
recorded, for checking these against.

`ergm` counts MCMC effort in proposals and this counts it in sweeps, where one
sweep is one update per customer -- 5,000 proposals on this data.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class MCMLEControl:
    """Settings for Monte Carlo maximum likelihood.

    Attributes:
        max_iterations: Cap on outer iterations. `MCMLE.maxit`.
        n_draws: Retained draws per iteration, split across chains.
            `MCMC.samplesize`.
        thin: Sweeps between retained draws. `MCMC.interval`, in sweeps.
        burn_in: Sweeps discarded before a chain's first draw. `MCMC.burnin`,
            in sweeps.
        n_chains: Independent chains, run in that many processes. `parallel`.
        confidence: Level of the joint confidence region used to stop.
            `MCMLE.confidence`.
        precision: Scales the tolerance region the gap must sit inside,
            passed to `ergmpy.convergence.within_tolerance` as its `precision`.
            `MCMLE.MCMC.precision`, which `results/r/fit_star.rds` recorded as
            0.1 for the converged reference fit.
        target_ess: Effective sample size an iteration aims for before testing
            convergence. `MCMC.effectiveSize` -- the figure ergm computes for
            the model, not `MCMC.base.effectiveSize`, the base it scales from.
        interval_drop: Factor by which a short draw shortens the interval while
            taking proportionally more draws.
            `MCMLE.effectiveSize.interval_drop`.
        max_resamples: Resampling attempts allowed in one iteration.
            `MCMC.effectiveSize.maxruns`.
        step_margin: Fraction by which to stop short of the convex hull's
            boundary. `MCMLE.steplength.margin`.
        seed: Base seed; chain i uses seed + i. `seed`.
    """

    # Requested by the reference R script's control.ergm call.
    n_draws: int = 1250              # MCMC.samplesize = 1250
    thin: int = 200                  # MCMC.interval = 1e6 proposals
    n_chains: int = 4                # parallel = 4
    seed: int = 123                  # seed = 123

    # From the authors' published output, which reports MCMLE.maxit = 200.
    # Their committed script sets 30 for this model; at 30 ergm reports the
    # fit did not converge, and at 200 it converges after 34.
    max_iterations: int = 200

    # ergm's own defaults, carried over so both sides run the same rules.
    confidence: float = 0.99         # MCMLE.confidence
    # Multiplies the statistics' covariance to give the tolerance region the
    # gap must sit inside, so a larger value accepts a larger gap. ergm's
    # MCMLE.MCMC.precision under confidence termination, whose default is 0.1
    # and which the converged reference fit recorded as 0.1.
    precision: float = 0.1
    interval_drop: float = 2.0       # MCMLE.effectiveSize.interval_drop
    max_resamples: int = 16          # MCMC.effectiveSize.maxruns
    step_margin: float = 0.05        # MCMLE.steplength.margin

    # Computed by ergm for this model rather than fixed: it scales
    # MCMC.base.effectiveSize = 64 up, reaching 895 here. burn_in is likewise
    # ergm's choice given the requested interval, and it adapts both during a
    # fit -- the converged run ended at MCMC.burnin = 2e6, or 400 sweeps.
    target_ess: float = 895.0        # MCMC.effectiveSize
    # From the maxit = 2 fit's MCMC.burnin of 8e6 proposals. Note this is the
    # one setting not taken from the converged reference, which ended at 2e6
    # (400 sweeps) having adapted down; 1600 is the more conservative choice
    # and the fit's own adaptation shortens it when the sample allows.
    burn_in: int = 1600

    def __post_init__(self) -> None:
        """Rejects settings that would make a fit do nothing.

        Both cases produced a well-formed wrong answer rather than an error:
        no resamples left `draws` unbound, and no iterations returned the
        starting parameter with standard errors as though a fit had run.

        Raises:
            ValueError: If `max_resamples` or `max_iterations` is below 1, or
                if `n_draws` cannot be split across `n_chains`.
        """
        if self.max_resamples < 1:
            raise ValueError("max_resamples must be at least 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.n_draws < self.n_chains:
            raise ValueError(
                f"n_draws {self.n_draws} cannot be split across "
                f"{self.n_chains} chains"
            )

    # Which `control.ergm` parameter each field mirrors, and whether the value
    # needs converting from proposals to sweeps.
    ERGM_EQUIVALENT = {
        "max_iterations": ("MCMLE.maxit", False),
        "n_draws": ("MCMC.samplesize", False),
        "thin": ("MCMC.interval", True),
        "burn_in": ("MCMC.burnin", True),
        "n_chains": ("parallel", False),
        "confidence": ("MCMLE.confidence", False),
        "precision": ("MCMLE.MCMC.precision", False),
        "target_ess": ("MCMC.effectiveSize", False),
        "interval_drop": ("MCMLE.effectiveSize.interval_drop", False),
        "max_resamples": ("MCMC.effectiveSize.maxruns", False),
        "step_margin": ("MCMLE.steplength.margin", False),
        "seed": ("seed", False),
    }

    def to_dict(self) -> dict:
        """Returns the settings as plain values, for recording with a run.

        Returns:
            One entry per field.
        """
        return dataclasses.asdict(self)

    def describe(self) -> str:
        """Formats the settings against their `control.ergm` counterparts.

        Returns:
            A table naming each field, its value, and the ergm parameter it
            mirrors.
        """
        lines = [f"{'setting':<16}{'value':>12}   control.ergm"]
        for field, (ergm_name, in_proposals) in self.ERGM_EQUIVALENT.items():
            unit = " (proposals -> sweeps)" if in_proposals else ""
            counterpart = f"{ergm_name}{unit}" if ergm_name else "(no counterpart)"
            lines.append(f"{field:<16}{getattr(self, field):>12}   {counterpart}")
        return "\n".join(lines)

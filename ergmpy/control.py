"""Run settings for the estimators, mirroring `control.ergm`.

Every knob lives here rather than as a default buried in a function signature,
for two reasons. The settings interact -- shortening the interval while raising
the draw count is one decision, not two -- and a measurement is only
interpretable beside the settings that produced it, so they have to be
recordable as a unit.

Each field names the `control.ergm` parameter it corresponds to. That mapping
is the point: this package is checked against `ergm`, and a comparison is only
as good as the reader's ability to confirm both sides were asked for the same
thing. `MCMLEControl.from_ergm_settings` builds one directly from the settings
a fitted ergm object recorded, so matching a run does not depend on anyone
transcribing numbers correctly.

`ergm` counts MCMC effort in proposals and this counts it in sweeps, where one
sweep is one update per customer. The conversion is the customer count, and
`from_ergm_settings` applies it.
"""

import csv
import dataclasses
import json
from pathlib import Path


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
    interval_drop: float = 2.0       # MCMLE.effectiveSize.interval_drop
    max_resamples: int = 16          # MCMC.effectiveSize.maxruns
    step_margin: float = 0.05        # MCMLE.steplength.margin

    # Computed by ergm for this model rather than fixed: it scales
    # MCMC.base.effectiveSize = 64 up, reaching 895 here. burn_in is likewise
    # ergm's choice given the requested interval, and it adapts both during a
    # fit -- the converged run ended at MCMC.burnin = 2e6, or 400 sweeps.
    target_ess: float = 895.0        # MCMC.effectiveSize
    burn_in: int = 1600              # MCMC.burnin = 8e6 proposals

    #: Which `control.ergm` parameter each field mirrors, and whether the value
    #: needs converting from proposals to sweeps.
    ERGM_EQUIVALENT = {
        "max_iterations": ("MCMLE.maxit", False),
        "n_draws": ("MCMC.samplesize", False),
        "thin": ("MCMC.interval", True),
        "burn_in": ("MCMC.burnin", True),
        "n_chains": ("parallel", False),
        "confidence": ("MCMLE.confidence", False),
        "target_ess": ("MCMC.effectiveSize", False),
        "interval_drop": ("MCMLE.effectiveSize.interval_drop", False),
        "max_resamples": ("MCMC.effectiveSize.maxruns", False),
        "step_margin": ("MCMLE.steplength.margin", False),
        "seed": ("seed", False),
    }

    @classmethod
    def from_ergm_settings(cls, path: str | Path, n_customers: int,
                           fit: str | None = None) -> "MCMLEControl":
        """Builds a control from the settings a fitted ergm object recorded.

        `benchmarks/r/export_control_settings.R` writes that file. Reading it
        rather than retyping the numbers is what keeps a matched run matched:
        ergm adapts its sample size and interval away from what it was asked
        for, so the settings that matter are the ones the fit ended with.

        Args:
            path: CSV of fit, setting and value, as exported from R.
            n_customers: Customers in the data, used to convert proposals to
                sweeps.
            fit: Which fit's settings to read; defaults to the first in the
                file.

        Returns:
            A control carrying ergm's settings, with anything absent from the
            file left at this class's default.

        Raises:
            FileNotFoundError: If the settings file is missing, rather than
                silently returning defaults that were never ergm's.
            ValueError: If `fit` names no rows in the file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run benchmarks/r/export_control_settings.R"
            )

        rows = list(csv.DictReader(path.open()))
        if fit is not None:
            rows = [r for r in rows if r["fit"] == fit]
            if not rows:
                raise ValueError(f"no rows for fit {fit!r} in {path}")
        elif rows:
            rows = [r for r in rows if r["fit"] == rows[0]["fit"]]

        recorded = {r["setting"]: r["value"] for r in rows}
        values = {}
        for field, (ergm_name, in_proposals) in cls.ERGM_EQUIVALENT.items():
            if ergm_name not in recorded:
                continue
            raw = float(recorded[ergm_name])
            declared = cls.__dataclass_fields__[field].type
            if in_proposals:
                values[field] = max(1, int(round(raw / n_customers)))
            elif declared is int or declared == "int":
                values[field] = int(round(raw))
            else:
                values[field] = raw
        return cls(**values)

    def to_dict(self) -> dict:
        """Returns the settings as plain values, for recording with a run.

        Returns:
            One entry per field.
        """
        return dataclasses.asdict(self)

    def record(self, path: str | Path) -> None:
        """Writes the settings beside whatever they produced.

        Args:
            path: Destination JSON file.
        """
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def describe(self) -> str:
        """Formats the settings against their `control.ergm` counterparts.

        Returns:
            A table naming each field, its value, and the ergm parameter it
            mirrors.
        """
        lines = [f"{'setting':<16}{'value':>12}   control.ergm"]
        for field, (ergm_name, in_proposals) in self.ERGM_EQUIVALENT.items():
            unit = " (sweeps)" if in_proposals else ""
            lines.append(f"{field:<16}{getattr(self, field):>12}   {ergm_name}{unit}")
        return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class CDControl:
    """Settings for contrastive divergence, used to seed MCMLE.

    `ergm`'s equivalents are `CD.maxit` and `CD.nsteps`. The step counts are in
    the same unit -- under this model's constraint ergm proposes with
    `CondB1Degree`, which moves one customer's purchase, exactly what
    `n_updates` counts -- but the two objectives extract signal from an
    excursion differently, so matching the number does not match the behaviour.
    See `docs/settings-comparison.md`.

    Attributes:
        max_iterations: Cap on outer iterations. `CD.maxit`.
        n_draws: Excursions per iteration.
        n_updates: Single-customer updates per excursion. `CD.nsteps`.
        tolerance: Convergence threshold on the largest standardized gap.
        seed: Seed for the excursion customers and the sampler.
    """

    max_iterations: int = 60
    n_draws: int = 300
    n_updates: int = 50000
    tolerance: float = 0.01
    seed: int = 123

    def to_dict(self) -> dict:
        """Returns the settings as plain values, for recording with a run.

        Returns:
            One entry per field.
        """
        return dataclasses.asdict(self)

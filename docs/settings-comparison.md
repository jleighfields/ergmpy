# Settings, side by side

What `ergm` was asked for, what it used, and what `ergmpy` does. `ergm` counts
MCMC effort in **proposals**; `ergmpy` counts it in **sweeps**, where one sweep
is one update per customer — 5,000 proposals on this data.

Source of truth: `results/r/control_settings.csv`, written from the fitted
object by `benchmarks/r/export_control_settings.R`, and the signature of
`ergmpy.mcmle.fit`.

## Matched

| Setting | `ergm` | `ergmpy` |
|---|---|---|
| Outer iteration cap | `MCMLE.maxit = 200` | `max_iterations = 200` |
| Effective sample size target | `MCMC.effectiveSize = 895` | `target_ess = 895` |
| Stopping rule | `MCMLE.termination = "confidence"` | ported; see `ergmpy/hotelling.py` |
| Tolerance region | `MCMLE.MCMC.precision = 0.1` | `precision = 0.1` |
| Burn-in | `MCMC.burnin = 8e6` proposals, from the `maxit = 2` fit | `burn_in = 1600` sweeps |
| Objective | `MCMLE.metric = "lognormal"` | same approximation |
| Optimizer | `MCMLE.method = "BFGS"` | BFGS |
| Chain continuation | `MCMLE.sequential = TRUE` | each chain resumes from its own end state |
| Starting point | `init.method = "CD"` | contrastive divergence |
| Resampling on a short draw | shorten interval by `MCMLE.effectiveSize.interval_drop = 2`, take more draws | same |
| Resampling attempts | `MCMC.effectiveSize.maxruns = 16` | `max_resamples = 16` |
| Step-length margin | `MCMLE.steplength.margin = 0.05` | `step_margin = 0.05` |
| Draws retained per iteration | `MCMC.samplesize = 1250` | `n_draws = 1250` |
| Between retained draws | `MCMC.interval = 1e6` proposals | `thin = 200` sweeps |
| Discarded before first draw | `MCMC.burnin = 8e6` proposals | `burn_in = 1600` sweeps |
| Parallel chains | `parallel = 4` (PSOCK) | `n_chains = 4` (processes) |
| Stopping rule | `MCMLE.termination = "confidence"` | joint confidence region |
| Confidence level | `MCMLE.confidence = 0.99` | `confidence = 0.99` |
| Step length | Hummel, via convex hull | Hummel, via convex hull |
| Seed | `seed = 123` | `seed = 123` |

## Not matched, and why

`ergm` reaches its sample size through a tuned controller with around fifteen
interacting parameters. The targets and the direction of adaptation are
matched above; the controller itself is not, and reproducing it would mean
porting a heuristic rather than a statistical method. The unmatched parts:

| Setting | What it does |
|---|---|
| `MCMLE.confidence.boost = 2`, `.lag = 4`, `.threshold = 1` | multiplies the sample size when the confidence statistic stops improving over four iterations |
| `MCMC.effectiveSize.damp = 10` | damps the adaptation so one short draw does not swing the sample size far |
| `MCMLE.sampsize.boost.pow = 0.5` | the exponent relating a boost in sample size to the shortfall |
| `MCMC.effectiveSize.burnin.{min,max,nmin,nmax,pval,scl}` | adapts the burn-in by testing when the chain has forgotten its start |
| `MCMLE.dampening`, `CD.dampening` | off by default in both |

Each iteration records the interval, draw count and achieved effective sample
size it ended with, so a run can be compared against `ergm`'s recorded
`MCMC.interval` and `MCMC.samplesize` rather than assumed equivalent.



| Setting | `ergm` | `ergmpy` | Why |
|---|---|---|---|
| Sampling controller | fifteen-parameter adaptive scheme, see above | targets and direction only | Reproducing the controller means porting a heuristic, not a method. Both record what they ended up using. |
| Final sample boost | `MCMLE.last.boost = 4` | none | `ergm` enlarges its last sample before reporting standard errors. `ergmpy` computes them from the sample the stopping rule accepted. |
| Degeneracy guard | `MCMLE.density.guard = 20.09` | none | `ergm` abandons a fit whose simulated networks grow far denser than the observed one. The constraint here fixes the edge count, so density cannot run away. |
| Contrastive divergence | `CD.nsteps = 8` steps per draw | `n_updates = 50000` | Same unit — under this constraint `ergm` proposes with `CondB1Degree`, which moves one customer's edge, exactly what `n_updates` counts. But matching the *number* does not match the behaviour: at `n_updates = 8` this implementation is inert at any draw count (measured: 0.90 from a start of 1.19 at 300 draws, and no movement at all at 40,000). Its objective needs the draws to move appreciably before the gradient is non-zero, where `ergm`'s extracts signal from much smaller perturbations. This is a difference in how the two CD objectives are formulated, not in settings. |

CD only produces a starting point; both implementations then run the same
MCMLE under the matched settings above, so a difference in seeds does not
propagate into the estimates. The seed each produced is recorded with the run.

## What this means for a timing comparison

Under the matched settings the two do comparable work per iteration, and both
*shorten* the interval and take proportionally more draws when a sample falls
short of the effective-sample-size target. `results/r/fit_metadata.csv` records what `ergm` ended up using, and
each `ergmpy` iteration records its own interval and achieved effective sample
size in `history`, so a timing can be checked against the effort behind it
rather than assumed comparable.

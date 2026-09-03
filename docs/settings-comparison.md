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
| Effective sample size target | `MCMLE.effectiveSize = 64` | `target_ess = 64`, lengthening the interval when short |
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


| Setting | `ergm` | `ergmpy` | Why |
|---|---|---|---|
| Contrastive divergence | `CD.nsteps = 8` steps per draw | `n_updates = 50000` | Same unit — under this constraint `ergm` proposes with `CondB1Degree`, which moves one customer's edge, exactly what `n_updates` counts. But matching the *number* does not match the behaviour: at `n_updates = 8` this implementation is inert at any draw count (measured: 0.90 from a start of 1.19 at 300 draws, and no movement at all at 40,000). Its objective needs the draws to move appreciably before the gradient is non-zero, where `ergm`'s extracts signal from much smaller perturbations. This is a difference in how the two CD objectives are formulated, not in settings. |

CD only produces a starting point; both implementations then run the same
MCMLE under the matched settings above, so a difference in seeds does not
propagate into the estimates. The seed each produced is recorded with the run.

## What this means for a timing comparison

Under the matched settings the two do comparable work per iteration, and both
adapt the interval upward when a draw falls short of an effective sample size
of 64. `results/r/fit_metadata.csv` records what `ergm` ended up using, and
each `ergmpy` iteration records its own interval and achieved effective sample
size in `history`, so a timing can be checked against the effort behind it
rather than assumed comparable.

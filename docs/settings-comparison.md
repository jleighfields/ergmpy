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
| Effective sample size target | `MCMLE.effectiveSize = 64`, adapting sample size and interval upward to reach it | none; takes the requested settings as given | `ergm` therefore does more work per iteration than it was asked for — the maxit=2 fit adapted `MCMC.interval` from 1e6 down to 250,000 and the maxit=30 fit ran `samplesize = 1768`. `ergmpy` records the effective sample size it achieved in `history` rather than steering to a target. |
| Step-length margin | `MCMLE.steplength.margin = 0.05` | none | `ergm` shrinks slightly inside the hull rather than to its boundary. `ergmpy` steps to the boundary. |
| Contrastive divergence | `CD.nsteps = 8` proposals per draw | `n_updates = 50000` customer updates | The two implementations differ in what a step is: `ergm` proposes dyad toggles, `ergmpy` resamples one customer's purchase. Matching the count would not match the work. CD only seeds MCMLE in both, and both then run the same MCMLE. |

## What this means for a timing comparison

The two are doing comparable work per iteration under the matched settings,
with one asymmetry: `ergm`'s effective-sample-size target makes it sample more
than requested when the chain mixes poorly, and `ergmpy` does not. So a
per-iteration timing favours `ergmpy` by an amount that depends on how far
`ergm` had to adapt, which `results/r/fit_metadata.csv` records for each run.

Iteration counts are not comparable at all unless the effective sample sizes
are, for the same reason: a criterion evaluated on a larger sample is met
sooner in iterations and later in total sampling.

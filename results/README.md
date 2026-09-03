# results

Measurements, separated by what produced them. Nothing here is written by
hand except `r/RESULTS.md`.

## r/

What the R baseline produced: `timings*.tsv` and run logs from `bench.R`, and
the coefficient tables and probability matrix that the Python tests and
benchmarks compare against. `RESULTS.md` writes up the R timings and the two
findings that came out of them — `b2degrange(25)` failing to estimate, and the
two phases of the reference script that recompute what they could difference.

`convex_hull/` holds six saved cases from `ergm`'s `shrink_into_CH`, used to
check `ergmpy.convex_hull` without an R installation.

The `.rds` fitted objects are gitignored: they are large and regenerable.
The `.csv` files derived from them are tracked, because regenerating those
means re-running a fit that takes 13 minutes.

## python/

Logs from runs of this implementation. Each filename says what was run rather
than which script wrote it — `full_recipe_mple_cd_mcmle.log`,
`cd_excursion_sweep.log`.

`cd_sweep_before_hull_shrink.log` is kept deliberately even though it is full
of `nan`: it is the evidence for a real defect, where contrastive divergence
skipped the convex-hull shrink and diverged past about half a sweep per
excursion.

## Regenerating

See [`benchmarks/README.md`](../benchmarks/README.md) for the script order.

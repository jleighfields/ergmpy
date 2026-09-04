# results

Measurements, separated by what produced them. Only `r/RESULTS.md` is written
by hand.

## r/

`timings*.tsv` and run logs from `bench.R`, plus the coefficient tables and
probability matrix the Python tests compare against. `RESULTS.md` writes up
the R timings and two findings from them: `b2degrange(25)` failing to
estimate, and the two phases of the reference script that recompute what they
could difference. `convex_hull/` holds six saved cases from `ergm`'s
`shrink_into_CH`.

`environment.txt` records the R and package versions `setup.sh` installed;
`run_settings.txt` the `MAXIT_CAP` and `PRED_N` a `bench.R` run used. Both are
written by the scripts, so a timing can be traced to what produced it.

The `.rds` fits and `.png` plots are gitignored — large and regenerable. The
`.csv` files derived from them are tracked, because regenerating those means
re-running a 13-minute fit.

## python/

Logs from runs of this implementation, named for what was run rather than
which script wrote it.

Four of these have no script that can regenerate them, and are kept as
records rather than as reproducible outputs:

- `cd_sweep_before_hull_shrink.log` is full of `nan` on purpose. It is the
  evidence for a defect where contrastive divergence skipped the convex-hull
  shrink and diverged past about half a sweep per excursion.
- `cd_seeded_vs_mple_seeded.log`, `full_recipe_mple_cd_mcmle.log` and
  `mcmle_star_from_mple.log` were written by benchmark scripts since deleted.
  Two of them record a `max_standardized_gap` column that the stopping rule no
  longer produces, so they could not be reproduced even by rewriting the
  scripts. They hold the only measurements of MPLE-seeded against CD-seeded
  MCMLE, and of a full 31-iteration fit from a pseudo-likelihood start.

Regenerating: see [`benchmarks/README.md`](../benchmarks/README.md).

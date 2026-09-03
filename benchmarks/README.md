# benchmarks

Timing runs and comparisons against R. Nothing here is part of the suite:
these need an R installation or minutes of runtime, which is why correctness
lives in `tests/` instead. Everything writes to `results/`.

## r/

The R baseline, in dependency order.

| Script | Produces | Cost |
|---|---|---|
| `setup.sh` | `rlib/` — installs `ergm` from precompiled binaries | a few minutes, once |
| `bench.R` | `results/r/timings.tsv`, run logs, `.rds` fits | minutes to hours, see below |
| `export_fits.R` | the tracked `.csv` files the Python side compares against | seconds |
| `fit_mple.R` | `results/r/mple_{train,test}.csv` | ~1 s |
| `gen_convex_hull_cases.R` | `results/r/convex_hull/` | seconds |

`bench.R` takes three environment variables that only shorten a run — no
statistical setting changes, and the defaults reproduce the authors' script
exactly. `FITS` selects from `null,degree,star,both`; `MAXIT_CAP` caps MCMLE
iterations; `PRED_N` limits the prediction loop.

The `.rds` fits are gitignored, so `export_fits.R` needs `bench.R` to have run
first. The `.csv` files it writes are tracked, which is what lets the test
suite compare against `ergm` with no R present.

## python/

Timing runs against the same data. `verify_*` scripts also report a
correctness figure, but the assertions that actually gate the code are in
`tests/` — these print the number alongside a timing so the two can be read
together.

The root [README](../README.md) lists every command with what it does.

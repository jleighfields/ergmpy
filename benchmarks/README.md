# benchmarks

Timing runs and comparisons against R. Not part of the suite: these need an R
installation or minutes of runtime, which is why correctness lives in
`tests/`. Everything writes to `results/`.

## r/

The R baseline, in dependency order.

| Script | Produces | Cost |
|---|---|---|
| `setup.sh` | `rlib/`, `results/r/environment.txt` | minutes, once |
| `bench.R` | `results/r/timings.tsv`, run logs, `.rds` fits | minutes to hours |
| `export_fits.R` | the tracked `.csv` files Python compares against | seconds |
| `fit_mple.R` | `results/r/mple_{train,test}.csv` | ~1 s |
| `gen_convex_hull_cases.R` | `results/r/convex_hull/` | seconds |

The `.rds` fits are gitignored, so `export_fits.R` needs `bench.R` to have run
first. The `.csv` files it writes are tracked, which is what lets the suite
compare against `ergm` with no R present.

Environment variables, the snapshot pin, and the run order are in the root
[README](../README.md#regenerating-the-r-baseline).

## python/

Timing runs against the same data. The `verify_*` scripts also print a
correctness figure, but the assertions that gate the code are in `tests/` —
these report the number beside a timing so the two read together.

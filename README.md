# network-choice

A Python implementation of the bipartite-ERGM discrete choice model from
Sha et al. (2023), "A network-based discrete choice model for decision-based
design," *Design Science* 9, e7. The authors published R scripts driving the
`ergm` package; this reimplements the model so it can be fitted, simulated and
predicted from Python, and checks every component against `ergm`'s own output.

## Layout

- `python/ndcm/` — the implementation.
- `benchmarks/python/` — scripts that time it and check it against R.
- `benchmarks/r/` — the R baseline. `bench.R` is the authors'
  `Code_choice_set_6.R` with identical model and control settings, wrapped in
  per-phase timing.
- `results/` — measurements from both sides. `results/r/RESULTS.md` writes up
  the R timings.
- `reference/` — unmodified clone of the authors' tutorial repository
  (`Yaxin-Cui/network-based-discrete-choice-model`): the original script, the
  train/test CSVs, and the published output screenshots. Read-only; it is the
  specification.
- `rlib/` — R library built locally by `benchmarks/r/setup.sh`. Not tracked.

## What the reference model is

Each of 5,000 customers considers 6 products and buys 1. The model is a
bipartite ERGM on the purchase network, with two options that together
restrict its sample space to exactly one choice per customer:
`constraints = ~b1degrees` fixes each customer's degree, and
`offset(edgecov(mat_inv))` at `-Inf` forbids edges outside the consideration
set. R reports that `edges` "could not be estimated because it conflicted with
the sample space constraint" — the edge count is pinned at 5,000 — leaving
eight free parameters: `b2cov.V1`–`V3`, `b2factor.V4.2`–`.5`, and `b2star2`.

The published estimates in `reference/Plots/` are the acceptance target for
any reimplementation.

## Running it

```bash
uv sync --group dev
```

That installs `ndcm` as an editable package, so `import ndcm` works from
anywhere without a path insert.

| Command | What it does |
|---|---|
| `uv run python benchmarks/python/verify_predict.py` | Checks the choice probabilities against R's saved probability matrix. |
| `uv run python benchmarks/python/verify_ch.py` | Checks the convex-hull shrink factor against `ergm`'s `shrink_into_CH`. |
| `uv run python benchmarks/python/bench_mple.py` | Fits by pseudo-likelihood; checks the Hessian against finite differences. |
| `uv run python benchmarks/python/bench_sampler.py` | Times the Gibbs sweep, pure Python against numba. |
| `uv run python benchmarks/python/run_mcmle_full.py` | Fits by MCMLE and compares against `ergm`'s estimates. |
| `uv run python benchmarks/python/bench_cd_seeded.py` | The full recipe: pseudo-likelihood, then contrastive divergence, then MCMLE. |
| `uv run python benchmarks/python/sweep_cd.py` | Sweeps the CD excursion length. |
| `uv run ruff check python/ benchmarks/python/` | Lint. |

Set `NUMBA_DISABLE_JIT=1` to run the sampler kernel as plain Python — the same
source, uncompiled, which is how the compiled path is checked.

### The R baseline

```bash
benchmarks/r/setup.sh                       # installs ergm into ./rlib
cd results/r && FITS=star Rscript ../../benchmarks/r/bench.R
```

`setup.sh` installs from Posit Package Manager with `HTTPUserAgent` set, which
serves precompiled binaries. That matters: from CRAN source, `ergm` needs
`lpSolveAPI` and `robustbase`, both of which require a Fortran compiler.
`install.packages()` reports success and installs nothing when one is missing,
so confirm with `library(ergm)` rather than the exit code.

`bench.R` takes three environment variables, all of which only shorten a run —
no statistical setting changes. `FITS` selects which of `null,degree,star,both`
to fit; `MAXIT_CAP` caps MCMLE iterations; `PRED_N` limits the prediction loop.
The defaults reproduce the authors' script exactly.

## Results

Measured on 48 cores, R 4.6.1 with ergm 4.12.0. The R fits use
`parallel = 4`; the Python ones are single-threaded.

| Component | Agreement with `ergm` | Python | R |
|---|---|---|---|
| Prediction + top-N scoring | 2.7e-13 | 1.7 ms | 148.95 s for 200 of 5,000 customers |
| Convex-hull shrink factor | 2.2e-10 | ~2 ms | ~5 ms |
| Pseudo-likelihood fit | no `ergm` equivalent | 72 ms | — |
| Gibbs sampler | reproduces observed statistics within 0.11σ at the published θ | 15.1 M updates/s | — |
| MCMLE, star model | coefficients within 0.009, standard errors within a few percent | 94.8 s | 781.8 s |

Two cautions on reading that table. The MCMLE row is not equal sampling
effort — the Python fit used roughly 14× fewer sweeps per iteration and still
converged to matching estimates, so the time ratio mixes a real speedup with a
settings difference. And the prediction row is an *algorithmic* difference:
the R script recomputes every network statistic 25,000 times where the change
statistics differ in closed form. The same rewrite in R would capture most of
it.

`ergm`'s own `estimate = "MPLE"` returns linear coefficients whose signs
contradict its MCMLE fit on this model, and warns that the GLM may be
separable. It forms its pseudo-likelihood dyad by dyad, which drops the
one-purchase-per-customer constraint; `ndcm.mple` conditions on the other
customers instead, leaving a multinomial choice over each consideration set.

The `b2degrange(25)` specification does not estimate: `ergm` reports
`b2deg25+ not varying` and never completes an MCMLE iteration. The authors
publish output for the star model only.

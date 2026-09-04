# ergmpy

A Python recreation of the bipartite-ERGM discrete choice model from
Sha et al. (2023), checked against the `ergm` R package that produced the
original results.

## Scope

**A research and educational project.** It reproduces a published model and
makes its estimation machinery legible in Python. Not production software: no
stability guarantee, no deprecation policy, no warranty of fitness.

It originates none of the methods it implements — see [Credit](#credit). If
you use this code, cite Sha et al. (2023).

## The model

5,000 customers each consider 6 products and buy 1. It is a bipartite ERGM on
the purchase network, with two options that restrict the sample space to
exactly one purchase per customer: `constraints = ~b1degrees` fixes each
customer's degree, and `offset(edgecov(mat_inv))` at `-Inf` forbids edges
outside the consideration set.

Two things follow. `edges` becomes unidentified — the count is pinned at 5,000
in every valid configuration, and `ergm` reports it as conflicting with the
constraint — leaving eight free parameters: `b2cov.V1`–`V3`,
`b2factor.V4.2`–`.5`, and `b2star2`. And sampling becomes a per-customer
multinomial draw rather than a proposal over arbitrary tie toggles, which is
why the estimation here is cheap.

## What the recreation reproduces

Both sides run the same settings, the same stopping rule, the same objective
and four chains. `ergm`'s confidence termination is ported in
`ergmpy/hotelling.py` rather than approximated, and its lognormal likelihood
metric is used in place of the exact importance-sampled one. What matches and
what does not is set out in
[`docs/settings-comparison.md`](docs/settings-comparison.md); the settings
themselves live on one object that names each `control.ergm` parameter it
mirrors.

| Stage | Agreement with `ergm` | Source |
|---|---|---|
| Choice probabilities | 2.7e-13 against R's saved matrix | `verify_predict.py` |
| Convex-hull shrink factor | 2.2e-10 against `shrink_into_CH` | `verify_ch.py` |
| Gibbs sampler | reproduces the observed statistics to 0.121 sd when simulating at `ergm`'s published estimates | `results/python/sampler_at_published_theta.log` |
| MCMLE coefficients | 0.0077 | `results/python/matched_settings_fit.log` |
| MCMLE standard errors | within 2.8% | same |

The reference is `ergm` at `MCMLE.maxit = 200`, the setting the authors'
published output reports. It converged after 34 iterations —
`results/r/fit_metadata.csv` records that, along with the sample size and
interval it adapted to. Their committed script sets 30 instead, at which
`ergm` reports the fit did not converge.

### Cost

| | `ergm` | `ergmpy` |
|---|---|---|
| Wall clock, star fit | 1,027 s | 70 s |
| Whole pipeline | — | 138 s (MPLE 0.1 s, CD 68 s, MCMLE 70 s) |
| Cores | 4 | 4 chains |
| MCMLE iterations | 34 | 2 |
| Sweeps drawn | not recorded per iteration | 768,800 |

`ergmpy`'s sweep count is measured — `MCMLEResult.sweeps` totals every sample
actually drawn, including resample attempts the effective-sample-size gate
discarded and each chain's burn-in.

There is no comparable figure for `ergm`. Its fitted object records only the
sample size and interval it *ended* with, and it adapted throughout: its
interval fell from 1e6 proposals to 62,500 while its sample size rose from
1,250 to 4,676, so no iteration but the last ran at the recorded values.
Multiplying them by the iteration count, which an earlier version of this
README did, understates the early iterations and produces a total the run never
performed. Getting a real figure needs `ergm` instrumented per iteration.

So the honest comparison here is wall clock, at matched settings, on the same
four cores — and it should be read knowing that the two reached convergence by
different routes: 34 cheap iterations against 2 expensive ones, both under the
same stopping rule.

## What is not recreated

`ergm` catalogues 139 terms, 21 constraints, 25 proposals and 4 references
(`search.ergmTerms()` and its siblings, ergm 4.12.0). This implements three
term families — `b2cov`, `b2factor`, `b2star` — and one constraint,
`b1degrees`.

The R script's `b2degrange(25)` specification is absent because it does not
estimate: `ergm` reports `b2deg25+ not varying` and never completes an MCMLE
iteration. The authors publish output for the star model only.

The estimation core (`sampler`, `mcmle`, `contrastive_divergence`, `convex_hull`) is not specific
to this model. Adding a constraint means writing its change statistics and its
Gibbs move; the core does not change.

## Layout

- `ergmpy/` — the estimation core, with `choice/` holding this model.
- `tests/` — the suite. See [`tests/README.md`](tests/README.md).
- `notebooks/` — marimo walkthrough. See
  [`notebooks/README.md`](notebooks/README.md).
- `benchmarks/` — timing runs and the R baseline. See
  [`benchmarks/README.md`](benchmarks/README.md).
- `results/` — what each side produced. See
  [`results/README.md`](results/README.md).
- `reference/` — the authors' repository, vendored unmodified. Read-only; it
  is the specification.
- `rlib/` — R library built by `benchmarks/r/setup.sh`. Not tracked.

## Getting started

[uv](https://docs.astral.sh/uv/) manages the Python version, the environment
and the dependencies, and will fetch Python itself if you lack 3.12+.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # or brew/pipx install uv

git clone https://github.com/jleighfields/ergmpy.git
cd ergmpy
uv sync
uv run pytest
```

Nine seconds, no R required. Two of the 22 tests compare against `ergm` output
committed as CSV; the rest check internal consistency — change statistics
against direct enumeration, derivatives against central differences, sampled
marginals against the closed form.

`uv sync` installs `ergmpy` as an editable package, so `import ergmpy` works
anywhere. `uv run <cmd>` runs inside that environment, syncing first, so you
never activate a venv or call `pip`.

R is needed only to regenerate the comparison outputs, which are committed.

### The notebook

```bash
uv sync --group notebooks     # marimo is a separate group; a plain sync omits it
uv run marimo edit notebooks/01_replicate_r_script.py
```

`01_replicate_r_script.py` walks the R script's four parts with the reasoning
at each step. If marimo is new to you: notebooks are plain `.py` files,
execution is reactive rather than top-to-bottom (so a name is defined in
exactly one cell), and `marimo run` serves it read-only while
`marimo export html` executes it headless.

## Running things

| Command | What it does |
|---|---|
| `uv run pytest` | The suite. |
| `uv run ruff check ergmpy benchmarks tests notebooks` | Lint. |
| `NUMBA_DISABLE_JIT=1 uv run pytest` | The suite with the kernel uncompiled — same source, no numba. |
| `uv run python benchmarks/python/bench_cd_seeded.py` | The full fit. ~95 s. |
| `uv run python benchmarks/python/verify_predict.py` | Probabilities against R's saved matrix, with timings. |
| `uv run python benchmarks/python/verify_ch.py` | Shrink factor against `ergm`'s. |
| `uv run python benchmarks/python/bench_mple.py` | Pseudo-likelihood fit and its Hessian check. |
| `uv run python benchmarks/python/bench_sampler.py` | The Gibbs sweep, pure Python against numba. |
| `uv run python benchmarks/python/sweep_cd.py` | The CD excursion-length sweep. ~3 min. |

Four more scripts cover intermediate stages; each names what it measures in
its module docstring.

## Regenerating the R baseline

Only needed to rebuild the comparison outputs, which are already committed.

```bash
benchmarks/r/setup.sh                       # installs ergm into ./rlib, once

cd results/r
FITS=star MAXIT_CAP=2 PRED_N=200 Rscript ../../benchmarks/r/bench.R   # ~11 min
FITS=star PRED_N=1 Rscript ../../benchmarks/r/bench.R                 # ~13 min
cd ../..

Rscript benchmarks/r/export_fits.R           # fitted objects -> tracked CSVs
Rscript benchmarks/r/fit_mple.R              # ergm's own pseudo-likelihood
Rscript benchmarks/r/gen_convex_hull_cases.R # the saved shrink-factor cases
```

`bench.R` takes `FITS` (which models), `MAXIT_CAP` (MCMLE iterations) and
`PRED_N` (customers scored). All three only shorten a run; no statistical
setting changes. `PRED_N` defaults to 5,000, whose prediction loop alone takes
about an hour.

`setup.sh` installs from a dated Posit Package Manager snapshot, currently
`2026-09-03`, overridable with `SNAPSHOT=<date>`. `ergm` is the oracle every
estimate is checked against, so a version moving underneath the comparison
would invalidate it without anything failing. The snapshot also serves
precompiled binaries — from CRAN source, `ergm` needs a Fortran compiler, and
`install.packages()` reports success while installing nothing when one is
missing, so the script verifies with `library(ergm)`. It records what it
installed to `results/r/environment.txt`.

## Credit

The model, the data and the reference implementation are the authors' work,
used under the terms they set: free use for research and related projects,
with citation. Everything under `reference/` is their repository unmodified,
published by Yaxin Cui at
[`Yaxin-Cui/network-based-discrete-choice-model`](https://github.com/Yaxin-Cui/network-based-discrete-choice-model).

> Sha, Z., Cui, Y., Xiao, Y., Stathopoulos, A., Contractor, N., Fu, Y. and
> Chen, W., 2023. A network-based discrete choice model for decision-based
> design. *Design Science*, 9, p.e7.

```bibtex
@article{sha2023network,
  title={A network-based discrete choice model for decision-based design},
  author={Sha, Zhenghui and Cui, Yaxin and Xiao, Yinshuang and Stathopoulos,
          Amanda and Contractor, Noshir and Fu, Yan and Chen, Wei},
  journal={Design Science}, volume={9}, pages={e7}, year={2023},
  publisher={Cambridge University Press}
}
```

Every algorithm in `ergmpy/` is someone else's:

| Implemented here | Due to |
|---|---|
| `mcmle.py` — importance-sampled maximum likelihood | Geyer & Thompson (1992), *JRSS-B* 54(3), 657–699 |
| `convex_hull.py`, and the step control in `mcmle.py` | Hummel, Hunter & Handcock (2012), *JCGS* 21(4), 920–939 |
| `contrastive_divergence.py` — as an MCMLE seed | Krivitsky (2017), *CSDA* 107, 149–161 |
| `choice/` — the model and its constraint | Sha et al. (2023), *Design Science* 9, e7 |

`ergm` and the Statnet Project are the reference implementation checked
against, and the source of the term vocabulary used throughout (`b2star2`,
`b2degrange`, `b1degrees`); `network` (Butts, 2008, *JSS* 24(2)) builds the
bipartite objects the R baseline uses.

> Handcock, M.S., Hunter, D.R., Butts, C.T., Goodreau, S.M., Krivitsky, P.N.
> and Morris, M. (2026). *ergm: Fit, Simulate and Diagnose Exponential-Family
> Models for Networks*. The Statnet Project, <https://statnet.org>.
> R package version 4.12.0.

> Krivitsky, P.N., Hunter, D.R., Morris, M. and Klumb, C. (2023). ergm 4: New
> Features for Analyzing Exponential-Family Random Graph Models. *Journal of
> Statistical Software*, 105(6), 1–44.
> [doi:10.18637/jss.v105.i06](https://doi.org/10.18637/jss.v105.i06)

> Hunter, D.R., Handcock, M.S., Butts, C.T., Goodreau, S.M. and Morris, M.
> (2008). ergm: A Package to Fit, Simulate and Diagnose Exponential-Family
> Models for Networks. *Journal of Statistical Software*, 24(3), 1–29.
> [doi:10.18637/jss.v024.i03](https://doi.org/10.18637/jss.v024.i03)

Built with NumPy, SciPy, polars, numba, marimo, uv, ruff and pytest.

### Licence

This project's code, tests, notebooks and documentation are [MIT](LICENSE)
licensed. **`reference/` is not**: it stays under the authors' terms, and
`results/r/` is derived from running their script on their data. An MIT notice
here does not relicense their work.

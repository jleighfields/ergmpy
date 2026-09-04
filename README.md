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
multinomial draw rather than a proposal over arbitrary tie toggles.

## Agreement with `ergm`

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
| MCMLE coefficients | 0.0072 | `results/python/matched_settings_fit.log` |
| MCMLE standard errors | within 2.7% | same |

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
Multiplying them by the iteration count understates the early iterations and
produces a total the run never performed. Getting a real figure needs `ergm`
instrumented per iteration.

The comparable figure is therefore wall clock, at matched settings on the
same four cores, read knowing that the two converged by different routes: 34
cheap iterations against 2 expensive ones, under the same stopping rule.

## What is not implemented

`ergm` catalogues 139 terms, 21 constraints, 25 proposals and 4 references
(`search.ergmTerms()` and its siblings, ergm 4.12.0). This implements three
term families — `b2cov`, `b2factor`, `b2star` — and one constraint,
`b1degrees`.

The R script's `b2degrange(25)` specification is absent because it does not
estimate: `ergm` reports `b2deg25+ not varying` and never completes an MCMLE
iteration. The authors publish output for the star model only.

The estimation core (`sampler`, `mcmle`, `contrastive_divergence`,
`convex_hull`) is not specific to this model. Adding a constraint means writing
its change statistics and its Gibbs move; the core does not change.

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

### Using uv

[uv](https://docs.astral.sh/uv/) is a Python package and project manager. It
does in one tool what `pyenv`, `virtualenv`, `pip` and `pip-tools` do
separately: installs the Python interpreter, creates the environment, resolves
the dependencies, and writes a lockfile pinning exact versions and hashes.

The reason it is used here is reproducibility, not convenience. **The
environment is part of the experiment.** This package's claim is that it
reproduces `ergm`'s numbers, and those numbers depend on the whole numerical
stack — `numba` caps which `numpy` it will run against, and a different `numpy`
could move a coefficient in the last digits the comparison is measured in.
`uv.lock` freezes that stack the same way the dated Posit Package Manager
snapshot freezes the R side, so both halves of the comparison are pinned.

Two consequences:

- **`uv run <cmd>` syncs before it runs.** You cannot accidentally run against
  a stale or half-installed environment, which would silently invalidate a
  comparison.
- **You never activate a virtualenv or call `pip`.** `uv sync` builds `.venv/`
  from `pyproject.toml` and `uv.lock`; `uv run` executes inside it.

To install uv, follow
[Astral's installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

### Setup

```bash
git clone https://github.com/jleighfields/ergmpy.git
cd ergmpy
uv sync
uv run pytest
```

uv fetches Python itself if you do not have 3.12+, so uv is the only
prerequisite.

The suite runs in seconds and needs no R. Some tests compare against `ergm`
output committed as CSV; the rest check internal consistency — change
statistics against direct enumeration, derivatives against central
differences, sampled marginals against the closed form.

`uv sync` also installs `ergmpy` itself as an editable package, so
`import ergmpy` works from anywhere without a path insert.

R is needed only to regenerate the comparison outputs, which are committed.

### The notebook

```bash
uv sync                       # marimo and altair are regular dependencies
uv run marimo edit notebooks/01_replicate_r_script.py
```

`uv sync --group notebooks` additionally installs JupyterLab, which is a front
end for working on notebooks rather than something they import.

`01_replicate_r_script.py` walks the R script's four parts with the reasoning
at each step. If marimo is new to you: notebooks are plain `.py` files,
execution is reactive rather than top-to-bottom (so a name is defined in
exactly one cell), and `marimo run` serves it read-only while
`marimo export html` executes it headless.

## Tooling

| | Where it is used | Why this one |
|---|---|---|
| **NumPy** | everything below `ChoiceData` | The estimators are dense numeric code over flat arrays. |
| **numba** | the Gibbs kernel, `sampler.py` | The one place an interpreter is fatal. See below. |
| **polars** | reading the two CSVs, building choice sets | Nothing here forces pandas, so the boundary uses one library and stops. |
| **SciPy** | the step-length LP, BFGS, the F distribution | All three already needed; no separate solver. |
| **ruff, pytest** | lint and the suite | |

### numba

The Gibbs sweep is sequential, scalar and branchy: each customer's update reads
the product degrees the previous update just changed. The work per operation is
tiny, so interpreter overhead dominates — the workload a JIT helps most and
vectorisation least.

Measured on this data (rounded; repeat runs vary a few percent):

| | ms per sweep |
|---|---|
| plain Python | ~80 |
| NumPy, vectorising a customer's alternatives | ~97 |
| **numba** | **~1.4** (about 15 M customer updates/s) |
| NumPy, vectorised across 64 chains | ~2.9 |
| NumPy, vectorised across 256 chains | ~1.1 |

The second row is the version most people would write first, and it is *slower
than plain Python*: the arrays are six elements long, so NumPy's per-call
overhead exceeds what the interpreter was costing. numba beats it by about 70×.

The last two rows matter because the dependency is between customers, not
between chains — so a batch dimension over chains sidesteps it, and with enough
chains NumPy overtakes numba. At the four this package runs, matching `ergm`'s
`parallel = 4`, vectorising across them is not close; break-even is somewhere
past a hundred. And 256 chains is not free statistically: splitting 1,250 draws
that many ways leaves about five per chain, far too few for the within-chain
batch means the convergence test needs.

So numba is the right tool at this sampling design, not in general.

Two consequences show up in the code. The kernel is written as explicit scalar
loops rather than vectorised NumPy, because that form is what numba compiles.
And `updates_python` and `updates_numba` come from one source definition, so
`NUMBA_DISABLE_JIT=1` runs the compiled path as plain Python and either can
check the other. They share a definition but not a random stream: inside
`@njit`, `np.random` draws from numba's own generator, which `np.random.seed`
cannot reach from the interpreter, so `sampler.seed_numba` exists to seed it.

### polars

polars reads the CSVs and assembles the choice sets; nothing below
`ChoiceData` sees it. The split is deliberate — above the line is tabular work,
below it is flat arrays handed to a compiled kernel — and it is why polars
rather than pandas: no dependency here forces pandas, so the boundary uses one
dataframe library and the numeric core uses none.

### Deliberate omissions

**No Rust extension.** The kernel was the obvious candidate, and numba closed
the gap — at ~15 M updates/s it is not the bottleneck. A `pyo3` port would add a
build matrix, a toolchain requirement for every contributor, and a second
implementation to keep in step, for a speedup that does not change what the
package can do.

**No autodiff framework.** The gradients here are short closed-form
expressions, and they are checked against central differences in the suite,
which is a stronger guarantee than a framework computing them correctly and
nobody verifying it. JAX or PyTorch would also be poor fits for the hot loop:
it mutates state in a sequential scan, which is what array frameworks are
worst at.

## Commands

| Command | What it does |
|---|---|
| `uv run pytest` | The suite. |
| `uv run ruff check ergmpy benchmarks tests notebooks` | Lint. |
| `NUMBA_DISABLE_JIT=1 uv run pytest` | The suite with the kernel uncompiled — same source, no numba. |
| `uv run python benchmarks/python/fit_matched.py` | The full fit under settings matched to the R script. ~140 s. |
| `uv run python benchmarks/python/verify_predict.py` | Probabilities against R's saved matrix, with timings. |
| `uv run python benchmarks/python/verify_ch.py` | Shrink factor against `ergm`'s. |
| `uv run python benchmarks/python/bench_mple.py` | Pseudo-likelihood fit and its Hessian check. |
| `uv run python benchmarks/python/bench_sampler.py` | The Gibbs sweep, pure Python against numba. |
| `uv run python benchmarks/python/sweep_cd.py` | The CD excursion-length sweep. ~3 min. |
| `uv run python benchmarks/python/check_sampler_calibration.py` | Simulates at `ergm`'s published estimates, reports standardized gaps. |
| `uv run python benchmarks/python/bench_vectorization.py` | numba against the vectorised NumPy alternatives. |

Each script names what it measures in its module docstring.

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
setting changes. `PRED_N` defaults to 5,000.

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

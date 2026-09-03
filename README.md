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

| Stage | Agreement with `ergm` | Source |
|---|---|---|
| Choice probabilities | 2.7e-13 against R's saved matrix | `verify_predict.py` |
| Convex-hull shrink factor | 2.2e-10 against `shrink_into_CH` | `verify_ch.py` |
| Gibbs sampler | reproduces the observed statistics to 0.121 sd when simulating at `ergm`'s published estimates | `results/python/sampler_at_published_theta.log` |
| MCMLE coefficients, CD-seeded | 0.0043 | `results/python/full_recipe_mple_cd_mcmle.log` |
| MCMLE coefficients, MPLE-seeded | 0.0089, standard errors within 4% | `results/python/mcmle_star_from_mple.log` |

The two MCMLE rows compare against `results/r/mcmle_star_maxit30.csv` — the R
script's own setting for the star model. The authors' published output used
`MCMLE.maxit = 200` instead, so the committed script and the published figures
do not match. At 30, ergm reports that the fit did not converge; the
coefficients still land within about 0.01 of the published ones. Pass
`MAXIT=200` to `bench.R` to reproduce the published setting.

Timings, for reference rather than as a benchmark:

| | Python | R |
|---|---|---|
| Full fit (MPLE → CD → MCMLE) | 94.8 s, 1 core | 781.8 s, 4 cores |
| Choice probabilities, 5,000 customers | 1.4 ms | ~62 min (projected from 148.95 s for 200) |
| Convex-hull shrink factor, per case | 1.4–7.0 ms | 1–9 ms |

The three rows measure different things.

The first is not equal work, and neither side's stopping rule is the other's.
R drew about 250,000 sweeps per MCMLE iteration against Python's 18,100. R ran
to its iteration limit of 30 and reported "MCMLE estimation did not converge";
Python stopped after 3 CD-seeded iterations having met a tolerance of 0.15 on
the largest standardized gap. The estimates agree to 0.0043 regardless, which
is the comparison worth making — but neither number is a converged fit in
ergm's sense.

The second is an algorithm, not a language. To score one alternative, the R
script calls `summary(formula)` on the whole 5,300-node network — 25,000 times
over, once per alternative per customer — for values that differ from each
other by a single toggled edge. `change_statistics` computes those differences
directly instead. Making that same substitution in R would close most of the
gap.

The third is the control: the same linear program on both sides, at the same
speed.

## What is not recreated

`ergm` catalogues 139 terms, 21 constraints, 25 proposals and 4 references
(`search.ergmTerms()` and its siblings, ergm 4.12.0). This implements three
term families — `b2cov`, `b2factor`, `b2star` — and one constraint,
`b1degrees`.

The R script's `b2degrange(25)` specification is absent because it does not
estimate: `ergm` reports `b2deg25+ not varying` and never completes an MCMLE
iteration. The authors publish output for the star model only.

The estimation core (`sampler`, `mcmle`, `cd`, `convex_hull`) is not specific
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
| `cd.py` — contrastive divergence as an MCMLE seed | Krivitsky (2017), *CSDA* 107, 149–161 |
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

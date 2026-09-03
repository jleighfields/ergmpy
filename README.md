# ergmpy

Exponential-random-graph models on **constrained sample spaces**, in Python,
checked against the `ergm` R package.

## Scope

**This is a research and educational project.** It reproduces a published
model and makes its estimation machinery legible in Python. It is not
production software — no stability guarantee, no deprecation policy, no
warranty of fitness. Don't use it to support a decision without validating it
against `ergm` on your own data, which is what `benchmarks/` is for.

It originates none of the methods it implements. Full attribution is in
[Credit](#credit) below; if you use this code, cite Sha et al. (2023).

## What this is, and is not

Not a port of `ergm`. That package carries 176 terms, 31 constraints, 18
proposals and 4 references; this implements **three term families and one
constraint**, and has no ambition to close that gap.

What it does instead is exploit the constraint. `ergm` reaches its generality
through a proposal mechanism that toggles arbitrary ties. When the constraint
already says what a valid configuration looks like, a Gibbs move can be O(1) in
the change statistics rather than a network traversal — which is why the fit
below takes 94.8 s single-threaded against `ergm`'s 781.8 s on four cores, at
matching estimates.

The estimation core (`sampler`, `mcmle`, `cd`, `convex_hull`) is not specific
to any model: importance-sampled maximum likelihood with the Hummel step
length, seeded by contrastive divergence. `ergmpy.choice` is the first
constraint implemented — the bipartite discrete choice model of Sha et al.
(2023), "A network-based discrete choice model for decision-based design,"
*Design Science* 9, e7 — see **Scope** above for attribution.

Adding a constraint means writing its change statistics and its Gibbs move.
The estimation core does not change.

## Layout

- `ergmpy/` — the estimation core, with `choice/` holding the first
  constrained model.
- `benchmarks/python/` — timing runs. See
  [`benchmarks/README.md`](benchmarks/README.md).
- `benchmarks/r/` — the R baseline. `bench.R` is the authors'
  `Code_choice_set_6.R` with identical model and control settings, wrapped in
  per-phase timing.
- `results/r/` and `results/python/` — what each side produced. The R
  directory also holds the coefficients and probability matrix the tests
  compare against, so the suite needs no R installation.
  `results/r/RESULTS.md` writes up the R timings.
- `notebooks/` — marimo notebooks. See
  [`notebooks/README.md`](notebooks/README.md).
- `tests/` — the suite. See [`tests/README.md`](tests/README.md).
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

## Getting started

### Prerequisites

**[uv](https://docs.astral.sh/uv/)** manages the Python version, the virtual
environment and the dependencies. It is the only thing you need installed
first — it will fetch Python itself if you don't have 3.12+.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux
# or:  brew install uv
# or:  pipx install uv
```

You never create a venv or run `pip` in this project. `uv sync` reads
`pyproject.toml` and `uv.lock` and builds an exact environment in `.venv/`;
`uv run <command>` runs a command inside it, syncing first if anything drifted.
So `uv run pytest` always runs against the right dependencies, whether or not
you remembered to activate anything.

**R** is optional. It is needed only to regenerate the comparison outputs,
which are already committed — everything else runs without it.

### Setup

```bash
git clone https://github.com/jleighfields/ergmpy.git
cd ergmpy
uv sync --group dev
uv run pytest
```

The suite finishes in about nine seconds. If it passes, the implementation is
reproducing `ergm`'s numbers on committed comparison data.

`uv sync` installs `ergmpy` as an editable package, so `import ergmpy` works
from anywhere — no path juggling in scripts or notebooks.

### The notebook

```bash
uv sync --group notebooks     # marimo is a separate group; a plain sync omits it
uv run marimo edit notebooks/01_replicate_r_script.py
```

That opens the notebook in a browser. If you have not used
[marimo](https://docs.marimo.io/) before, three things differ from Jupyter:

- **Notebooks are plain `.py` files.** They diff, review and import like any
  other source, and there is no JSON or output blob in the repo.
- **Execution is reactive, not top-to-bottom.** marimo tracks which cell
  defines which variable and reruns whatever depends on a change. There is no
  stale hidden state, and a variable cannot be defined in two cells.
- **`edit` versus `run`.** `marimo edit` is the editable notebook;
  `marimo run <file>` serves it read-only as an app, with code hidden;
  `marimo export html <file> -o out.html` executes it headless and writes a
  static page, which is how a notebook is checked without a browser.

## Running things

| Command | What it does |
|---|---|
| `uv run pytest` | The suite. Nine seconds, no R needed. |
| `uv run ruff check ergmpy benchmarks tests` | Lint. |
| `uv run marimo edit notebooks/01_replicate_r_script.py` | The walkthrough notebook. Needs `--group notebooks`. |
| `uv run python benchmarks/python/bench_cd_seeded.py` | The full fit: pseudo-likelihood, then contrastive divergence, then MCMLE. ~95 s. |
| `uv run python benchmarks/python/verify_predict.py` | Prediction against R's saved probability matrix, with timings. |
| `uv run python benchmarks/python/bench_sampler.py` | The Gibbs sweep, pure Python against numba. |
| `uv run python benchmarks/python/sweep_cd.py` | The contrastive-divergence excursion-length sweep. ~3 min. |

Set `NUMBA_DISABLE_JIT=1` to run the sampler kernel as plain Python — the same
source, uncompiled, which is how the compiled path is checked.

## Rerunning the comparison against R

Only needed to regenerate the R side; the committed outputs already let the
Python checks run. Full detail in
[`benchmarks/README.md`](benchmarks/README.md).

```bash
benchmarks/r/setup.sh                    # installs ergm into ./rlib, once

cd results/r
FITS=star MAXIT_CAP=2 PRED_N=200 Rscript ../../benchmarks/r/bench.R   # ~8 min
FITS=star Rscript ../../benchmarks/r/bench.R                          # ~13 min
cd ../..

Rscript benchmarks/r/export_fits.R           # fitted objects -> tracked CSVs
Rscript benchmarks/r/fit_mple.R              # ergm's own pseudo-likelihood
Rscript benchmarks/r/gen_convex_hull_cases.R # the saved shrink-factor cases
```

`setup.sh` installs from Posit Package Manager, which serves precompiled
binaries. That matters: from CRAN source, `ergm` needs `lpSolveAPI` and
`robustbase`, both of which require a Fortran compiler. `install.packages()`
reports success and installs nothing when one is missing, so the script
verifies with `library(ergm)` rather than trusting the exit code.

`bench.R` takes three environment variables that only shorten a run — no
statistical setting changes. `FITS` selects from `null,degree,star,both`;
`MAXIT_CAP` caps MCMLE iterations; `PRED_N` limits the prediction loop. The
defaults reproduce the authors' script exactly, which takes a few hours.

## Credit

The model, the data and the reference implementation are the authors' work,
reproduced under the terms they set: free use for research and related
projects, with citation.

> Sha, Z., Cui, Y., Xiao, Y., Stathopoulos, A., Contractor, N., Fu, Y. and
> Chen, W., 2023. A network-based discrete choice model for decision-based
> design. *Design Science*, 9, p.e7.

```bibtex
@article{sha2023network,
  title={A network-based discrete choice model for decision-based design},
  author={Sha, Zhenghui and Cui, Yaxin and Xiao, Yinshuang and Stathopoulos,
          Amanda and Contractor, Noshir and Fu, Yan and Chen, Wei},
  journal={Design Science},
  volume={9},
  pages={e7},
  year={2023},
  publisher={Cambridge University Press}
}
```

If you use this code, cite the paper it implements. Everything under
`reference/` is the authors' repository, unmodified — the tutorial and its
data were published by Yaxin Cui at
[`Yaxin-Cui/network-based-discrete-choice-model`](https://github.com/Yaxin-Cui/network-based-discrete-choice-model).

### The methods

Every algorithm in `ergmpy/` is someone else's:

| Implemented here | Due to |
|---|---|
| `mcmle.py` — importance-sampled maximum likelihood | Geyer & Thompson (1992), *JRSS-B* 54(3), 657–699 |
| `convex_hull.py` and the step control in `mcmle.py` | Hummel, Hunter & Handcock (2012), *JCGS* 21(4), 920–939 |
| `cd.py` — contrastive divergence as an MCMLE seed | Krivitsky (2017), *CSDA* 107, 149–161 |
| `choice/` — the bipartite choice model and its constraint | Sha et al. (2023), *Design Science* 9, e7 |

**`ergm` and the Statnet Project** are the reference implementation this is
checked against, and the source of the term vocabulary used throughout
(`b2star2`, `b2degrange`, `b1degrees`). Its authors:

> Handcock, M.S., Hunter, D.R., Butts, C.T., Goodreau, S.M., Krivitsky, P.N.
> and Morris, M. (2026). *ergm: Fit, Simulate and Diagnose Exponential-Family
> Models for Networks*. The Statnet Project, <https://statnet.org>.
> R package version 4.12.0.

> Krivitsky, P.N., Hunter, D.R., Morris, M. and Klumb, C. (2023). ergm 4: New
> Features for Analyzing Exponential-Family Random Graph Models.
> *Journal of Statistical Software*, 105(6), 1–44.
> [doi:10.18637/jss.v105.i06](https://doi.org/10.18637/jss.v105.i06)

> Hunter, D.R., Handcock, M.S., Butts, C.T., Goodreau, S.M. and Morris, M.
> (2008). ergm: A Package to Fit, Simulate and Diagnose Exponential-Family
> Models for Networks. *Journal of Statistical Software*, 24(3), 1–29.
> [doi:10.18637/jss.v024.i03](https://doi.org/10.18637/jss.v024.i03)

The `network` package (Butts, 2008, *JSS* 24(2)) builds the bipartite objects
the R baseline uses.

The timings in this README compare a specialised implementation against a
general one. `ergm` supports 176 terms and 31 constraints through a proposal
mechanism that toggles arbitrary ties; `ergmpy` implements one constraint and
uses its structure to make each move O(1). Read the numbers as a measurement
of that specialisation on this model, not as a benchmark of the two packages.

**Tools.** NumPy, SciPy, polars, numba, marimo, uv, ruff and pytest.

### Licence

This project's own code, tests, notebooks and documentation are
[MIT](LICENSE) licensed.

**`reference/` is not covered by that grant.** It is the authors' repository,
vendored unmodified, and stays under the terms they published: free use for
research and related projects, with citation. An MIT notice on this repository
does not relicense their data, their R script, or the paper — nobody here has
the standing to do that.

`results/r/` is derived from running their script on their data, so treat it
the same way.

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
one-purchase-per-customer constraint; `ergmpy.choice.mple` conditions on the other
customers instead, leaving a multinomial choice over each consideration set.

The `b2degrange(25)` specification does not estimate: `ergm` reports
`b2deg25+ not varying` and never completes an MCMLE iteration. The authors
publish output for the star model only.

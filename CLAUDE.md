# Project instructions

Standalone project. Nothing here is synced from a template, so these
instructions are edited in place and there are no sidecar files.

## Orientation

- **What this is:** a Python implementation of the bipartite-ERGM discrete
  choice model from Sha et al. (2023), "A network-based discrete choice model
  for decision-based design," *Design Science* 9, e7. The authors published R
  scripts driving the `ergm` package; this reimplements the model so it can be
  fitted, simulated and predicted from Python.
- **R is the oracle, not a dependency.** `ergm` produces the numbers every
  Python component is checked against. Nothing in `python/ndcm/` calls R, and
  the package must stay installable without it.
- **Where things live:** `python/ndcm/` is the implementation,
  `benchmarks/python/` the scripts that time it and check it against R,
  `benchmarks/r/` the instrumented copy of the authors' script, `results/`
  the measurements both produce, and `rlib/` a locally built R library that
  is not tracked.
- **`reference/` is read only.** It is an unmodified clone of the authors'
  tutorial repository — their script, their two CSVs, their published output
  screenshots. It is the specification. Never edit anything under it; if a
  comparison needs the data reshaped, reshape it on the way out.

### What the model is

Each of 5,000 customers considers 6 products and buys 1. `constraints =
~b1degrees` fixes each customer's degree and the consideration-set offset at
`-Inf` forbids every other edge, so the sample space is exactly one purchase
per customer — `ergm` confirms this by refusing to estimate `edges`, which the
constraint holds constant. Eight parameters are free: `b2cov.V1`–`V3`,
`b2factor.V4.2`–`.5`, and `b2star2`.

Only the chosen product's attributes and `b2star2` vary, and both change in
closed form when one customer's purchase moves. That is why prediction needs
no network traversal and why the sampler is a per-customer multinomial draw
rather than general tie toggling.

## Core principles

- **Simplicity first.** Make every change as simple as possible, and touch
  only what the task requires — no features, abstractions, or error handling
  beyond it.
- **Root causes, not workarounds.** If a change only stops the symptom, ask
  "knowing everything I know now, what removes the cause?" before shipping.
  A fix that leaves the cause in place has to be made again.
- **Ask, don't assume.** When requirements are ambiguous or several
  approaches exist, ask. Don't guess at intent or decide design alone.
- **Verify by running, not by reading.** Any claim you can run is unverified
  until you have run it — yours and everyone else's. Never mark work complete
  without running the tests, executing the command, reading the output.
  When a comment, docstring or README asserts a behavior, run it before
  believing it.
- **A check you have not watched fail is not known to work.** Break what it
  guards and confirm it reports. Some operations fail by doing nothing — a
  find-and-replace that matched nothing, a loop over an empty list — and each
  returns success, letting every later step pass for the wrong reason.
  `install.packages()` in this repo's own history returned exit code 0 while
  failing to install anything; `library(ergm)` is what caught it.
- **A statistical test is the easiest kind to fool yourself with.** Estimates
  agreeing within Monte Carlo error is weak evidence: a real divergence
  smaller than the tolerance passes. Prefer checks that remove randomness —
  the analytic marginal when `theta_star2` is zero, the finite-difference
  Hessian, R's own saved probability matrix compared element by element.
- **A converging optimizer is not a correct one.** Two failures here looked
  identical from the outside and had unrelated causes: statistics differing by
  two orders of magnitude made BFGS diverge, and skipping the convex-hull
  shrink made contrastive divergence diverge. Diagnose which component is
  wrong by holding the others at known-good values before changing anything.
- **Plan non-trivial work.** Plan mode for anything spanning 3+ steps or an
  architectural decision. If work goes sideways, stop and re-plan.
- **Use subagents to keep the main context clean.** Offload research,
  exploration, and heavy read-only passes.
- **Review with the `code-reviewer` agent** before committing: `commit` over
  what is being committed, the full pass before merging a feature branch.
  Resolve or waive every Must Fix and Should Fix.

## Code style

- Prioritize simplicity and readability.
- **Docstrings**: Google style — summary line, then `Args`, `Returns`,
  `Yields`, `Raises` as appropriate. Every public function and class.
- **Type hints on every signature.** Use `X | None`, not `Optional[X]`.
- Import modules directly for type hints; no quoted forward references.
- **No `from __future__ import annotations` or `TYPE_CHECKING` guards** as
  circular-import workarounds. Fix the cycle structurally — move one side's
  import into the function that uses it.
- **No `_` prefix on function names.** Internal helpers get real names.
- Preserve inline comments when refactoring or extracting helpers.
- Avoid duplication. Two near-identical blocks is a smell; three is a
  refactor. The change-statistic formula is the exception and the thing to
  watch: `predict.change_statistics` builds it as an array expression, and
  the numba kernel writes it out again as a scalar one because a compiled
  kernel cannot call the array version. Two copies, deliberately. Change one
  and you must change the other.
- Prefer `if/elif/else` over `continue` — it makes the structure explicit.
- Comments explain *why*, not *what*. Magic numbers need a justification.
- **Never silently skip a missing file or directory.** Raise or warn, so the
  problem surfaces at its source rather than three steps downstream.
- **NumPy is the working array type.** The estimators, the sampler and the
  prediction path are all dense numeric code over flat arrays. polars reads
  the input CSVs and shapes result tables; it does not appear below
  `ChoiceData`.

### The numba kernel

- **`gibbs_updates` is decorated, not duplicated.** `updates_python` and
  `updates_numba` are built from one source definition, so the readable
  reference and the compiled version cannot drift, and either can serve as
  the oracle for the other. Run with `NUMBA_DISABLE_JIT=1` to execute the
  compiled path as plain Python. `sampler.py`'s module docstring is the
  authority on this.
- **Keep it to the numba-supported subset** — loops, flat arrays, scalars.
  A dict, a class or a ragged structure inside it will fail to compile, and
  the resulting typing error names almost anything except the offending line.
- **The kernel takes the customers to visit, not a sweep count.** A full sweep
  passes every index; a contrastive-divergence excursion passes a random
  handful. One implementation serves both.

## Configuration and parameters

- **Estimator settings are keyword arguments with defaults, not globals.**
  `n_draws`, `burn_in`, `thin`, `tolerance` and `max_iterations` are passed
  down from the caller. A benchmark that wants lighter sampling passes lighter
  sampling; it does not edit a module constant.
- **Never write a value in two places.** A constant that also appears as a
  literal elsewhere is a defect; the copies will drift. `TERM_NAMES` is
  duplicated between `mple` and `mcmle` today and should not stay that way.
- **Record the settings with the measurement.** Anything written to
  `results/` states the sampling settings that produced it. A timing without
  its `n_draws` and `thin` cannot be compared to anything.

## Logging, paths, and secrets

- **Module-scoped logger, never `basicConfig`.** `log =
  logging.getLogger(__name__)` at module scope; a library calling
  `basicConfig()` silently reconfigures its host application. Configure
  handlers only in an entry point — a benchmark or driver script.
- **Anchor paths to `__file__`, never the working directory.** The benchmark
  scripts derive the project root from `Path(__file__).resolve().parents[2]`
  so they run from anywhere.
- **Nothing here authenticates to anything, and no `.env` is expected.** All
  inputs are the two committed CSVs. A tracked `.env` appearing is a real
  finding rather than the usual false positive.
- **RNG seeds are not secrets.** They exist so a run reproduces; they are
  meant to be committed and read.

## Tests

- **Run unit tests whenever code changes**, and add tests when behavior
  changes.
- **`tests/` mirrors the package.** Every test directory needs an
  `__init__.py`, subdirectories included.
- **Pytest machinery in `conftest.py`, plain functions in `helpers.py`** — a
  helper should be readable without knowing pytest.
- **Prefer a check that cannot pass by accident.** The strongest tests here
  compare against something computed a different way: a change statistic
  against a full recomputation, an analytic Hessian against central
  differences, a sampled marginal against the closed-form softmax.
- **R comparisons belong in `benchmarks/`, not `tests/`.** They need an R
  installation and minutes of runtime. The suite must pass with no R present.
- Fixtures are generated in code from the committed CSVs, never as opaque
  blobs.

## Benchmarks

- **A speedup claim names its baseline, its settings and its core count.**
  Reporting Python against R is meaningless unless both did comparable work:
  the MCMLE comparison in `results/` is not equal sampling effort, and says so.
- **Distinguish an algorithmic win from a language win.** Prediction is
  ~2e6 times faster than the R script because it differences change statistics
  instead of recomputing every statistic 25,000 times. The same rewrite in R
  would capture most of it. Say that, rather than implying Python is fast.
- **Projected numbers are labelled projected.** The 62-minute figure for R's
  prediction loop is 200 customers measured and multiplied by 25.
- **Record what a run actually did.** Settings and machine go beside the
  timing, because the same script at different `thin` is a different
  measurement.

## Git and tooling

- **Use `uv`, not `pip`.** `uv add <pkg>` for new dependencies.
- **Never reference Claude, Anthropic, or AI in commit messages.** No
  `Co-Authored-By`, no "generated by", no attribution of any kind.
- **Feature branches** for anything non-trivial, cut from `master`. There is
  no remote, so nothing is pushed and there are no pull requests.
- Review all commits for quality and style before pushing.
- **Check for README updates** after changes affecting usage.
- **Don't assume.** Verify against the codebase and ask rather than guessing.
- `rlib/` and `results/*.rds` are deliberately untracked — the first is a
  compiled R library, the second is regenerable. `results/*.log`,
  `*.tsv` and `*.csv` are tracked, because measurements are not regenerable
  without hours of compute.

Tooling configuration lives in the file that configures it, not here —
restating it in prose creates drift.

## Security checks

- **Never baseline a real secret.** Remove it and **rotate** it — working-tree
  removal is not enough if it was ever committed.
- **Prefer removing the secret over baselining it.** Drop the secret-shaped
  value, or mark that line `# pragma: allowlist secret` with the reason
  beside it — the same rule as `# noqa: S106` for ruff's equivalent.
- **`.claude/skills/security-scan/SKILL.md` owns the rest**: which tools run
  and with what, the required baseline format, and this repo's caveat that
  seeds and model parameters are meant to be committed.

## Plans and lessons

- **Write a plan first** for non-trivial work, to
  `tasks/<description>-<YYYY>-<MM>-<DD>.md` — short, kebab-case, meaningful
  (not "todo"). Check it in with the user before implementing, mark items
  complete as you go, and add a review section when finished.
- **Move finished plans to `tasks/completed/`.** Several can coexist.
- **After any correction from the user, add the pattern to
  `docs/lessons.md`** — the rule that prevents the same mistake next time.
  Review it when starting related work.

## Minimalism (write less)

Run this *before* writing code; stop at the first step that solves the problem.

1. **Does this need to exist?** Prefer deletion, then refactoring, then
   addition.
2. **Use the standard library** before hand-rolling.
3. **Use a library already imported here** — reach for its built-in first.
4. **Use an installed dependency** before adding a new one.
5. **Prefer the smallest correct form.** No helper, class, config knob, or
   generalization until there are 2–3 real call sites.
6. **Only then** write minimal custom code.

`/simplify-audit` finds existing bloat (report-only delete-list).

## Prose is professional and factual

**Everything written here — comments, docstrings, READMEs, commit messages,
pull-request bodies, skills, agents and plans — states what is true and how the
reader can check it.** A sentence that rates something without evidence
describes the author's opinion, not the code's behavior. When the code
changes, unsupported ratings do not update with it.

Common categories to avoid: unmeasured rankings, personified programs where
the verb stands in for a mechanism, unmeasured cost or effort claims,
aesthetic verdicts like "elegant" or "hacky", aphorisms, and filler run-ups.
See `comment-docstring` for rewrites, greps, and the categories that need
manual review.

**A speedup claim is a measurement or it is nothing.** State the baseline it
was measured against, the settings, and the core count. "Python is faster
than R" is the exact sentence this section exists to prevent.

**Argument is not editorializing.** State each claim with its reason, in the
same sentence or the next one — for example, "two copies of the same value
drift apart over time." Give the reader something to check; keep the
reasoning and drop unsupported ratings.

Judge sentences in context — some individual words that look like offenders
are fine. See `comment-docstring` for details.

## Comments & docstrings are self-contained

**Every comment, docstring and doc must stand on its own for a reader who has
the repo and nothing else**, and must describe the code as it is now.
References that only make sense outside the repo, or only to people involved
in the original conversation, break for future readers.

Common categories to avoid: references to plan files, commits, tickets, "as
discussed", earlier versions of the code, and bare dates. See
`comment-docstring` for examples and greps.

Describe the thing directly — what it does, what the constraint is, why this
way rather than the obvious alternative. Test: **delete every plan, ticket and
commit message; would this sentence still teach a new reader anything?**

Point to durable references freely: a README section, another module, an
external spec, the paper. Naming a statistic the way `ergm` names it —
`b2star2`, `b2degrange` — is a durable reference and is preferred to
paraphrase, because it is what a reader will search for.

- **Pull-request and issue bodies, at a stricter bar.** Their reader has the
  diff and little else. Name the thing, not its number.
- **Directory READMEs point, never restate.** Each says what belongs in its
  directory and links to whatever owns the detail.

## Skills and agents

- **Where they live.** Skills in `.claude/skills/`, subagents in
  `.claude/agents/`. They are committed, and their names and descriptions are
  injected at session start, so a new agent needs a session restart to be seen.
- **Nothing syncs them.** This project owns its copies outright; edit them in
  place when this repo's practice differs from what they say.
- **Agents read their skill's `SKILL.md` at runtime** rather than copying the
  checklist, so it stays in step as skills evolve. Given no path,
  `code-reviewer commit` defaults to the changed files (`git diff --name-only
  HEAD` plus untracked) and its full pass to the branch diff against the base
  the pull request targets; `simplify-auditor` defaults to the whole repo.

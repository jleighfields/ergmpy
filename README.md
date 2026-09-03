# network-choice

Working area for a Python implementation of the bipartite-ERGM discrete choice
model from Sha et al. (2023), "A network-based discrete choice model for
decision-based design," *Design Science* 9, e7.

## Layout

- `reference/` — unmodified clone of the authors' tutorial repository
  (`Yaxin-Cui/network-based-discrete-choice-model`): the original
  `Code_choice_set_6.R`, the train/test CSVs, and the published output
  screenshots. Treat this as read-only; it is the specification.
- `benchmarks/r/` — the R baseline. `bench.R` is `Code_choice_set_6.R` with
  identical model and control settings, wrapped in per-phase timing.
- `results/r/` — timings and fitted models produced by that run.

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

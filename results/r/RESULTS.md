# R baseline timings

Measured on 48-core x86_64, Pop!_OS 24.04, R 4.6.1, ergm 4.12.0, network 1.20.0.
The script pins `parallel = 4`, so the other 44 cores are idle throughout.

Run settings: `FITS=star MAXIT_CAP=2 PRED_N=200`. Every statistical setting is
the original's — `MCMC.samplesize = 1250`, `MCMC.interval = 1e6`, `seed = 123`.
Only the iteration count and the prediction-loop length are shortened, both of
which scale linearly, so the projections below are arithmetic rather than guesses.

## Measured

| Phase | Seconds | Notes |
|---|---|---|
| `01_read_train_csv` | 0.05 | |
| `02_make_network_train` | 5.25 | id factorize + 2 network objects + `mat_inv` |
| `03_plot_networks` | 9.55 | |
| `04_fit_null` | 136.57 | 8 CD + 2 MCMLE iterations (separate run) |
| `06_fit_star` | 265.53 | 4 CD + 2 MCMLE iterations |
| `08_read_test_csv` | 0.05 | |
| `09_make_network_test` | 3.80 | |
| `10_fit_test_structure` | 211.50 | a full fit at `maxit = 1`, used only for its formula |
| `11_prediction_loop` | 148.95 | 200 of 5000 customers |
| `12_topn_eval` | 0.02 | |

Top-3 accuracy on the 200-customer subset: 0.955. Not comparable to the paper —
this used a star fit stopped at 2 MCMLE iterations, which has not converged.

## Per-iteration cost

CD iterations cost ~3.3 s each (they ignore `MCMC.interval`). Netting those out:

- null: ~55 s per MCMLE iteration
- star: ~126 s per MCMLE iteration

`b2star(2)` makes the model dyad-dependent, which roughly doubles the cost of
every proposal.

## Projected full script

| Phase | `maxit` | Projected |
|---|---|---|
| `04_fit_null` | 100 | ~93 min |
| `05_fit_degree` | 30 | did not complete one MCMLE iteration in >15 min |
| `06_fit_star` | 30 | ~63 min |
| `07_fit_both` | 50 | ~108 min |
| `10_fit_test_structure` | 1 | 3.5 min (measured) |
| `11_prediction_loop` | — | ~62 min |
| **Total, excluding degree** | | **~5.5 hours** |

**Superseded.** The projections above assumed a constant per-iteration cost
measured from a two-iteration run. They are wrong, because later iterations are
much cheaper and `ergm` adapts its sampling as it goes. The star fit was later
run at the published `MCMLE.maxit = 200` and took **1,027 s**, converging after
34 iterations -- against a projection here of about seven hours. Treat the
measured figures in `timings.tsv` and `fit_metadata.csv` as authoritative and
this table as a record of how far a linear extrapolation missed.

## The degree model does not estimate

`05_fit_degree` never finished a single MCMLE iteration. ergm reports:

    Warning: Model statistics 'b2deg25+' are not varying. This may indicate that
    the observed data occupies an extreme point in the sample space or that the
    estimation has reached a dead-end configuration.

There are 281 products and 5000 purchases, so mean product degree is ~17.8 and
`b2degrange(25)` counts products bought by at least 25 customers. If that count
does not move across sampled networks the term is unidentified, and ergm cannot
reach its effective-sample-size target. The authors publish output for the star
model only.

## Two phases that are algorithmically unnecessary

Together these are ~65 minutes, about a fifth of the run, and neither computes
anything that requires the time it takes.

`11_prediction_loop` calls `summary(xAlt$formula)` once per alternative per
customer — 25,000 full recomputations of every network statistic over a
5,300-node network — to obtain values that differ from each other by a single
toggled edge. Change statistics give the same numbers in O(1) per alternative.

`10_fit_test_structure` runs a complete `ergm()` fit on the test network,
including its CD phase and one MCMLE iteration, purely to obtain a fitted object
whose `$formula` can then be evaluated. Nothing from the fit is used.

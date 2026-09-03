# tests

```bash
uv run pytest
```

Runs in about nine seconds and needs no R installation. Anything requiring a
live `ergm` lives in `benchmarks/` instead.

The tests are built so that none can pass by agreeing with the code it checks:

| File | Checked against |
|---|---|
| `test_change_statistics.py` | both configurations built and their statistics enumerated one edge at a time |
| `test_mple.py` | central differences, for both the gradient and the Hessian |
| `test_convex_hull.py` | squares whose shrink factor is `1/k` by construction, plus six cases saved from `ergm` |
| `test_sampler.py` | the closed-form softmax that holds once `theta_star2` is zero and the customers decouple |
| `test_predict.py` | the probability matrix `ergm` produced, committed as CSV |

Two of these read files under `results/r/`. That does not contradict the rule
about R comparisons belonging in `benchmarks/`: the rule is aimed at things
needing a live R installation and minutes of runtime, and these read a
committed CSV in milliseconds.

Every test has been watched to fail. Dropping the `+1` from the star delta,
skipping the sampler's degree decrement, deleting a term from the Hessian and
inverting the shrink factor each turn the relevant file red.

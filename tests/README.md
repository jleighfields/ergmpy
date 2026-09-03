# tests

```bash
uv run pytest
```

About nine seconds, no R installation needed. Anything requiring a live `ergm`
lives in `benchmarks/`.

No test can pass by agreeing with the code it checks:

| File | Checked against |
|---|---|
| `test_change_statistics.py` | both configurations built and their statistics enumerated one edge at a time |
| `test_mple.py` | central differences, for the gradient and the Hessian |
| `test_convex_hull.py` | squares whose shrink factor is `1/k` by construction, plus six cases saved from `ergm` |
| `test_sampler.py` | the closed-form softmax that holds once `theta_star2` is zero and the customers decouple |
| `test_predict.py` | the probability matrix `ergm` produced, committed as CSV |

Two of these read files under `results/r/`. That does not contradict the rule
about R comparisons belonging in `benchmarks/`: the rule targets things needing
a live R and minutes of runtime, and these read a committed CSV in
milliseconds.

Four mutations confirm the tests can fail — dropping the `+1` from the star
delta, skipping the sampler's degree decrement, deleting a term from the
Hessian, inverting the shrink factor — each turning the relevant file red.
That covers one test per file, not all twenty-two. The degree-decrement case is
not hypothetical: it was committed by accident in `247833f`, and this suite
caught it.

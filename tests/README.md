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
| `test_convergence.py` | chains whose autocorrelation is known by construction, and constant-level chains where a batch mean can only take one value |
| `test_control.py` | the settings a fitted `ergm` object recorded, transcribed with the `control.ergm` name each one mirrors |
| `test_hotelling.py` | ellipsoids whose distance-to-boundary is available in closed form, and `scipy.stats.f` for the tail |
| `test_mcmle.py` | the closed-form maximizer of the lognormal objective, and `ergm`'s converged coefficients |

Three of these read files under `results/r/`. That does not contradict the rule
about R comparisons belonging in `benchmarks/`: the rule targets things needing
a live R and minutes of runtime, and these read a committed CSV in
milliseconds.

Mutations confirm the tests can fail. Dropping the `+1` from the star delta,
skipping the sampler's degree decrement, deleting a term from the Hessian and
inverting the shrink factor each turn the relevant file red; the degree
decrement is not a hypothetical case, having once been deleted by accident and
caught here. The convergence and estimation tests are pinned the same way —
measuring the ellipsoid distance from the origin rather than from the boundary,
inverting the p-value comparison, and dropping the covariance term from the
step's gradient each turn a named test red.

That is a mutation per behaviour, not per test. A test not named above has not
been watched to fail, so its passing says only that the code agrees with it.

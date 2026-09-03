"""Walks the authors' R script end to end in Python, explaining each step.

Mirrors the four parts of reference/Code_choice_set_6.R. Modeling logic lives
in the ergmpy package; this notebook imports it and explains what it does.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    ROOT = Path(__file__).resolve().parents[1]
    return ROOT, alt, mo, np, pl


@app.cell
def _(mo):
    mo.md(
        r"""
        # Replicating the reference R script

        `reference/Code_choice_set_6.R` fits a bipartite ERGM discrete choice
        model with the `ergm` package, then predicts held-out choices. This
        notebook walks the same four parts in Python and explains what each one
        is doing.

        | R script part | Here |
        |---|---|
        | 1. Load data, build networks | Choice sets and the sample space |
        | 2. Network plots | Skipped — the plots carry no result |
        | 3. ERGM estimation | Pseudo-likelihood, contrastive divergence, MCMLE |
        | 4. Prediction | Change statistics and top-N scoring |

        The model is from Sha et al. (2023), *Design Science* 9, e7, and the
        data under `reference/` is the authors', unmodified and reproduced
        under their terms — free use for research, with citation.

        This notebook, and the package it demonstrates, are for research and
        teaching. Nothing here is production software.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Part 1 — What the data is, and what the constraint does

        5,000 customers each consider 6 products and buy exactly 1.
        """
    )
    return


@app.cell
def _(ROOT):
    from ergmpy.choice.predict import load

    train = load(str(ROOT / "reference" / "Sampled_data_to_share.csv"))
    return load, train


@app.cell
def _(mo, train):
    mo.md(
        f"""
        - **{len(train.chosen):,} customers**, each with a consideration set of
          **{train.choice_sets.shape[1]}** products
        - **{train.n_products}** distinct products across all sets
        - product degree — how many customers bought it — runs from
          **{train.degree.min()}** to **{train.degree.max()}**, mean
          **{train.degree.mean():.1f}**
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### The two options that define the sample space

        The R script passes these to `ergm`:

        ```r
        constraints = ~b1degrees          # every customer's degree is fixed
        offset(edgecov(mat_inv)), -Inf    # no edge outside the consideration set
        ```

        Together they say: each customer holds exactly one purchase, drawn from
        their own six alternatives. That is the entire sample space — 6^5000
        configurations, not all bipartite graphs.

        Two consequences follow, and both matter.

        **`edges` cannot be estimated.** The edge count is pinned at 5,000 in
        every valid configuration, so the statistic never varies and its
        coefficient is unidentified. R says so itself:

        > The following terms could not be estimated because they conflicted
        > with the sample space constraint: `edges`

        Most `b1*` terms go the same way — with every customer's degree fixed at
        1, any sum over customer attributes is constant too.

        **Sampling becomes cheap.** A move is "customer *i* buys *j* instead of
        *k*", so a Gibbs sweep is one multinomial draw per customer rather than
        a proposal over arbitrary tie toggles. That is where the speed comes
        from, not from Python.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Part 3 — The statistics, and why they are cheap to update

        Eight parameters are free: `b2cov.V1`–`V3` (continuous product
        attributes), `b2factor.V4.2`–`.5` (a categorical attribute, baseline
        `A`), and `b2star2`.

        `b2star2` counts two-stars centred on products — pairs of customers who
        bought the same thing:

        $$\text{b2star2} = \sum_j \binom{d_j}{2}$$

        where $d_j$ is product $j$'s purchase count. This is the only term that
        couples customers to one another, and it is what makes the model an
        ERGM rather than a plain conditional logit.
        """
    )
    return


@app.cell
def _(np, train):
    from ergmpy.mcmle import observed_statistics

    observed = observed_statistics(train)
    degree_counts = np.bincount(train.chosen, minlength=train.n_products)
    star2_by_hand = int(sum(d * (d - 1) // 2 for d in degree_counts))
    return degree_counts, observed, observed_statistics, star2_by_hand


@app.cell
def _(mo, np, observed, star2_by_hand):
    mo.md(
        f"""
        The observed statistic vector, with `b2star2` recomputed by direct
        enumeration as an independent check:

        - attribute sums: `{np.array2string(observed[:7], precision=1)}`
        - `b2star2` from the package: **{observed[7]:,.0f}**
        - `b2star2` counted one product at a time: **{star2_by_hand:,}**

        Same number, computed two ways.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Change statistics

        Moving one customer's purchase from $k$ to $j$ changes only two things,
        and both in closed form:

        | Statistic | Change |
        |---|---|
        | attribute sums | $x_j - x_k$ |
        | `b2star2` | $d_j - d_k + 1$ |

        The second follows from $\binom{n+1}{2} - \binom{n}{2} = n$ and
        $\binom{n-1}{2} - \binom{n}{2} = -(n-1)$.

        This is the whole efficiency argument. The R script instead calls
        `summary(formula)` on the full 5,300-node network once per alternative
        per customer — 25,000 complete recomputations — to obtain values that
        differ by one toggled edge.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Estimation, in three stages

        Each stage exists because the one before it is not enough.
        """
    )
    return


@app.cell
def _(train):
    import time

    from ergmpy.choice import mple

    mple_start = time.perf_counter()
    mple_fit = mple.fit(train)
    mple_seconds = time.perf_counter() - mple_start
    return mple, mple_fit, mple_seconds, time


@app.cell
def _(mo, mple_fit, mple_seconds):
    mo.md(
        f"""
        ### 1. Pseudo-likelihood — {mple_seconds * 1000:.0f} ms

        Conditioning on every other customer's purchase leaves customer *i*'s
        choice multinomial over their own alternatives, so the pseudo-likelihood
        is a conditional logit and has a closed-form gradient and Hessian.

        This is **not** what `ergm(estimate = "MPLE")` computes. `ergm` forms its
        pseudo-likelihood dyad by dyad, discarding the one-purchase constraint,
        and on this model returns linear coefficients whose signs contradict its
        own MCMLE fit — see `results/r/mple_train.csv`.

        It is fast and biased. `b2star2` comes out at
        **{mple_fit.coef[7]:.5f}** against a true value near **0.0058** — about
        2.5× too large, which matters in the next stage.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### 2. Contrastive divergence

        Starting MCMLE from the pseudo-likelihood estimate does not work. With
        `b2star2` inflated, the popularity term compounds: simulated networks
        reach a `b2star2` around 537,000 against an observed 299,000, roughly 69
        standard deviations out, where importance sampling has no support and the
        step diverges.

        Contrastive divergence avoids that by never letting the chain leave the
        data. Every draw restarts at the observed purchases and takes a short run
        of single-customer updates. The gradient is biased, so it is not an
        estimate to report — but it cannot run away, and it lands close enough to
        seed MCMLE.

        The excursion length is the dial. Too short and the draws never leave the
        observed configuration, so the gradient carries no signal; too long and it
        approaches MCMLE and inherits the instability it exists to avoid.
        """
    )
    return


@app.cell
def _(ROOT, alt, mo, pl):
    sweep = pl.DataFrame(
        {
            "updates": [500, 1250, 2500, 5000, 10000, 25000, 50000],
            "distance": [1.0972, 0.9787, 0.8276, 0.6088, 0.3656, 0.1147, 0.0259],
        }
    ).with_columns((pl.col("updates") / 5000).alias("sweeps"))

    sweep_chart = (
        alt.Chart(sweep)
        .mark_line(point=True)
        .encode(
            x=alt.X("updates:Q", scale=alt.Scale(type="log"),
                    title="single-customer updates per excursion"),
            y=alt.Y("distance:Q", title="max |coefficient − ergm|"),
            tooltip=["updates", "sweeps", "distance"],
        )
        .properties(height=260, title="CD seed quality against excursion length")
    )

    mo.vstack([
        mo.md(
            "Measured on the training data, recorded in "
            "`results/python/cd_excursion_sweep.log`. The pseudo-likelihood "
            "start sits at 1.19; ten sweeps per excursion reaches 0.026."
        ),
        sweep_chart,
    ])
    return sweep, sweep_chart


@app.cell
def _(mo):
    mo.md(
        r"""
        ### 3. Monte Carlo maximum likelihood

        Simulate at the current parameter, then maximise Geyer and Thompson's
        importance-sampled likelihood ratio:

        $$\ell(\theta) - \ell(\theta_t) = (\theta - \theta_t)^\top g_{\text{obs}}
          - \log \frac{1}{M}\sum_m e^{(\theta - \theta_t)^\top g_m}$$

        Maximising that directly is unreliable. When $g_{\text{obs}}$ lies outside
        the convex hull of the sampled statistics, the approximation is being
        extrapolated past its support. Following Hummel, Hunter and Handcock
        (2012), the observed statistics are first shrunk toward the sampled mean
        by the largest factor that keeps them inside the hull — a small linear
        program, solved here with `scipy.optimize.linprog`.

        Two practical notes, both learned the hard way:

        - The statistics differ in scale by two orders of magnitude
          (`b2star2` ≈ 3×10⁵, attribute sums ≈ 3×10³) while the parameters span
          0.006 to 3. On the raw scale BFGS diverges to 10¹³. The step is
          computed on standardised draws and transformed back.
        - Skipping the hull shrink inside CD looks safe — every excursion starts
          at the observed network, so it must be inside the hull — but that holds
          only for short excursions. Past about half a sweep the draws travel far
          enough that it is outside again.
        """
    )
    return


@app.cell
def _(mo):
    run_full_fit = mo.ui.checkbox(
        value=False,
        label="Run the full fit now (about 95 seconds)",
    )
    run_full_fit
    return (run_full_fit,)


@app.cell
def _(mple_fit, run_full_fit, time, train):
    from ergmpy import cd, mcmle

    if run_full_fit.value:
        fit_start = time.perf_counter()
        cd_seed, _cd_history = cd.fit(train, mple_fit.coef, max_iterations=60,
                                      n_draws=300, n_updates=50000)
        mcmle_fit = mcmle.fit(train, cd_seed, max_iterations=120, n_draws=600,
                              burn_in=100, thin=30, tolerance=0.15)
        fit_seconds = time.perf_counter() - fit_start
    else:
        mcmle_fit = None
        fit_seconds = None
    return cd, cd_seed, fit_seconds, mcmle, mcmle_fit


@app.cell
def _(ROOT, mcmle_fit, mo, np, pl):
    ergm_converged = pl.read_csv(ROOT / "results" / "r" / "mcmle_star_maxit30.csv")
    terms = ["b2cov.V1", "b2cov.V2", "b2cov.V3", "b2factor.V4.2",
             "b2factor.V4.3", "b2factor.V4.4", "b2factor.V4.5", "b2star2"]
    ergm_lookup = dict(
        zip(ergm_converged["term"].to_list(),
            ergm_converged["estimate"].cast(pl.Float64, strict=False).to_list(),
            strict=True)
    )

    # Recorded in results/python/full_recipe_mple_cd_mcmle.log when the fit ran.
    recorded_python = [-3.043502, -0.039734, 1.595903, 1.229273,
                       2.209014, 1.221944, 1.183921, 0.005792]
    python_estimates = (
        list(mcmle_fit.coef) if mcmle_fit is not None else recorded_python
    )
    source = "just fitted" if mcmle_fit is not None else "recorded from an earlier run"

    comparison = pl.DataFrame({
        "term": terms,
        "ergmpy": python_estimates,
        "ergm": [ergm_lookup[t] for t in terms],
    }).with_columns((pl.col("ergmpy") - pl.col("ergm")).abs().alias("|difference|"))

    mo.vstack([
        mo.md(f"**Coefficients ({source}) against `ergm`'s fit at maxit = 30.** "
              f"Largest disagreement: "
              f"**{comparison['|difference|'].max():.5f}**."),
        comparison,
    ])
    return comparison, ergm_converged, ergm_lookup, python_estimates, terms


@app.cell
def _(mo):
    mo.md(
        r"""
        Two caveats on reading any timing next to `ergm`'s. The R fit ran with
        `parallel = 4` and drew roughly 14× more sweeps per iteration than the
        Python one, so the wall-clock ratio mixes a real speedup with a settings
        difference. And these are Monte Carlo estimates — two independent runs of
        the *same* implementation differ by a similar amount.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Part 4 — Prediction

        With coefficients in hand, each customer's probability over their own
        consideration set is a softmax of the change statistics. No sampling is
        involved; this is closed form.
        """
    )
    return


@app.cell
def _(ROOT, load, np, python_estimates, time):
    from ergmpy.choice.predict import choice_probabilities, top_n_accuracy

    test_set = load(str(ROOT / "reference" / "test_data_to_share.csv"))
    theta = np.asarray(python_estimates)

    predict_start = time.perf_counter()
    probabilities = choice_probabilities(test_set, theta[:7], float(theta[7]))
    predict_seconds = time.perf_counter() - predict_start

    accuracy = [top_n_accuracy(test_set, probabilities, n) for n in (1, 2, 3)]
    return (accuracy, choice_probabilities, predict_seconds, probabilities,
            test_set, theta, top_n_accuracy)


@app.cell
def _(accuracy, mo, predict_seconds, test_set):
    mo.md(
        f"""
        Scored all **{len(test_set.chosen):,}** held-out customers in
        **{predict_seconds * 1000:.1f} ms**.

        | | top-1 | top-2 | top-3 |
        |---|---|---|---|
        | share of customers whose actual purchase ranks here | {accuracy[0]:.3f} | {accuracy[1]:.3f} | {accuracy[2]:.3f} |

        The R script's equivalent loop takes about 62 minutes for the same
        5,000 customers. That gap is an algorithm, not a language. To score one
        alternative the R script calls `summary(formula)` on the whole
        5,300-node network, once per alternative per customer, for values that
        differ from each other by a single toggled edge. `change_statistics`
        computes those differences directly. Making that same substitution in R
        would close most of the gap.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What this leaves out

        `ergm` carries 176 terms, 31 constraints, 18 proposals and 4 references.
        This implements three term families and one constraint. The interesting
        direction is not closing that gap term by term — it is the constraint
        axis, where each new constraint reuses the estimation core and only needs
        its own change statistics and Gibbs move.

        One specification from the R script is missing on purpose:
        `b2degrange(25)` does not estimate. `ergm` reports `b2deg25+ not varying`
        and never completes an MCMLE iteration, which is presumably why the
        authors publish output for the star model only.

        Everything shown here is checked against `ergm` in `tests/`, which runs
        in about nine seconds and needs no R installation.
        """
    )
    return


if __name__ == "__main__":
    app.run()

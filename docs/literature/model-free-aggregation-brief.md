# Model-Free Signal Aggregation — Brief, Critical Read, Algothon Adaptation

> **Source:** `model-free-aggregation-brief.md` in the team's working repository (relocated here for the public release).

Source: palmbook, "Model-Free Trading Signals Aggregation", InsiderFinance
Wire, 31 March 2025. Third in a series with `novy-marx-multisignal-brief.md`.

---

## PART 1 — THE METHOD

`m` models each emit `Sᵢ ∈ [0,1]` (fraction of capital wanted). Aggregate
allocation is `Σ Wᵢ·Sᵢ`. Weights come **only** from each model's past realised
P/L over a trailing `lookback`. No model fitted, no dataset carved out.

**Why avoid a fitted ensemble.** (a) *Data budget* — bagging/boosting/stacking
need their own training set; 20 years of weekly bars is ~1,000 obs and you are
already short for the base strategies. (b) *Diagnosability* — with a fitted
aggregator on fitted strategies you cannot tell which layer broke.

**The seven methods.** `D` = lookback × m matrix of past per-period P/L. All
clip negative weights to zero and normalise to sum 1.

1. **Equal weight** — `Wᵢ = 1/m`. No lookback. Sharpe ≈ 0.5.
2. **Weighted average** — `Wᵢ ∝ max(mean(Dᵢ), 0)`. Peak 0.84 at lookback 30.
3. **Softmax** — `W = softmax(β·colsum(D))`, β = 30. Peak 1.06.
4. **Mean-variance** — μ, Σ over window, annualised ×52, max-Sharpe with L2
   (γ = 2) via PyPortfolioOpt. 0.84 at lookback 30.
5. **Binary Kelly** — `f* = p/a − (1−p)/b` with p = hit rate, b = mean positive
   P/L, a = |mean negative P/L|. Peak 0.74 at lookback 32.
6. **Non-binary Kelly** (Thorp 1992) — `f* = (μ−r)/σ²`, r omitted. ≈ binary.
7. **Running count** — `+1` per winning period, `−2` per losing period, summed,
   clipped, normalised. Sharpe 1.46, the highest.

Benchmark S&P 500 Sharpe −0.97 over the test window (a correction). Lookback
swept to ~32.

---

## PART 2 — CRITICAL READ

**2.1 The result is a best-of-~224 selection.** Seven methods × ~32 lookbacks,
each method's *maximum* reported. Within-method that is best-1-of-32 (5% crit
t ≈ 3.16); across the grid, best-1-of-224 (crit t ≈ 3.69). The 1.46-vs-0.50
spread is not measured against anything like that.

**2.2 The sample cannot resolve the spread.**
`SE(SR_ann) ≈ √52·√((1 + SR_period²/2)/n)`. At n = 20 (the plotted curves),
**SE ≈ 1.6 — larger than the entire spread across all seven methods.** At
n = 100 it is ~0.73. Resolving 0.5 vs 1.46 at 2 SE needs ~230 weekly periods
(~4.4 years), before the 224-configuration correction.

**2.3 Running count is probably de-risking, not aggregating.** Benchmark
Sharpe was −0.97; the window is a drawdown. Penalising losses 2:1 with a
clip-at-zero drives all weights to zero in a sustained decline and goes to
cash. A rule that mechanically de-risks posts a high Sharpe in a falling
market *by construction*, with no skill at ranking models. Diagnostic:
decompose into (a) average allocation level over time and (b) cross-sectional
weight distribution given that level. If (a) dominates, it is a timing rule.

**2.4 Two formula/implementation mismatches.**
- Softmax uses the column **sum** while the text says the **mean**. These
  differ by a factor of `lookback`, so the effective temperature is
  `β·lookback` — the method becomes progressively winner-take-all as lookback
  grows, confounding the Sharpe-vs-lookback curve.
- Non-binary Kelly is coded as `μ/σ` but stated as `(μ−r)/σ²`. `μ/σ` is the
  Sharpe ratio; after normalisation these are genuinely different weightings.
  What is labelled Kelly is Sharpe-proportional weighting.

**2.4b A THIRD mismatch, measured here, larger than the other two.** `beta` is
dimensional: `softmax(beta·x)` depends on the units of `x`, and `beta = 30` is
calibrated to whatever P/L scale the article used. Measured on identical data
at three scales, with `m = 5` (so equal weight = 0.200):

| P/L units | max weight at beta=30 |
|---|---:|
| fractional returns | 0.2007 (≈ equal weight) |
| unit P/L | 0.7419 |
| **dollar P/L** | **1.0000 (pure argmax)** |

The same `beta` spans the entire range from equal-weight to winner-take-all on
units alone. On this book's dollar P/L, "softmax" **is** argmax — which
mechanically guarantees it the highest turnover of any method and independently
predicts the commission inversion in PART 3. Fix: z-score the per-model scores
before the softmax, making `beta` unit-free ("cross-model standard deviations
of edge"). Standardising also collapses the sum-vs-mean confound of 2.4
exactly, since z-scoring is invariant to that factor — the cleanest way to show
how much of the article's lookback curve was the artefact.

**2.4c `kelly_continuous` and `sharpe_weighted` coincide under equal vol.**
`mu/sigma^2` and `mu/sigma` differ by `1/sigma`, so after normalisation they
are identical unless model volatilities differ across models. The 2.4 critique
is right, but only bites when there is cross-model volatility dispersion —
verify that dispersion exists before treating them as distinct methods.

**2.5 Structural mismatches with Algothon.** Long-only with weights summing to
1 (Algothon has no portfolio budget); no transaction costs anywhere;
PyPortfolioOpt unavailable in the sandbox.

**2.6 What holds up.** The motivation is correct and is the natural
counterpart to Novy-Marx: fitted weights are where overfitting bias lives, so
a parameter-free aggregator removes a layer of it. **Equal weight being
competitive is the most credible finding**, because it has no lookback and so
took no selection bias. The catalogue itself spans parameter-free to fully
fitted, which is the right axis.

---

## PART 3 — ADAPTING TO ALGOTHON 2026

**Sum-to-1 must go.** Score `μ·SR²/(SR²+1)` is homogeneous of degree 1 in size
with SR scale-invariant, and there is no portfolio budget — only per-asset
caps. Normalising caps gross exposure at one unit for no reason. Structure:
(1) aggregate to direction + conviction, unnormalised; (2) convert to dollars;
(3) scale globally until per-asset caps bind. Normalise only for diagnostics.
**See the measured caveat in PART 3b — this argument is only half right.**

**Long/short changes the clipping.** Algothon permits shorts, so a
persistently losing model is informative and can be inverted. But that is
exactly the **signing step** Novy-Marx identifies as the source of overfitting
bias. If done, it must live strictly inside the walk-forward, never on the
full sample, and the null must account for it.

**De-risking is penalised here.** Going flat risks the **\$25,000 minimum
traded volume floor** — breaching it is an automatic zero, not a small score.
The mechanism that made running count look best is the one most likely to zero
you out.

**Commission is the missing axis.** 1.0 bp on assets 1–50, 0.2 bp on ALGO,
both directions. Hypothesised turnover ordering (lowest → highest): equal
weight, weighted average, non-binary Kelly, mean-variance, binary Kelly,
softmax (high β), running count. This **inverts the article's ranking almost
exactly** — the two best gross performers are the two most likely to bleed on
commission. That is the thing to measure.

**Lookback selection is where you will overfit.** Fix it a priori on signal
half-life reasoning, or select inside a nested walk-forward; either way report
the critical value for the selection performed.

---

## PART 3b — MEASURED: leverage scaling does NOT survive daily clipping

The Part 3 "scale until the caps bind" argument is only half right. Measured
on the test window (751–1000) with a genuine conviction-weighted book (netted
pair + ridge dollar demands), scaling by `k` then clipping to caps:

| k | score | % of active positions pinned at cap | corr with pure bang-bang book |
|---:|---:|---:|---:|
| 0.25 | 444.6 | 4.3% | 0.771 |
| 1 | 818.9 | 68.9% | 0.897 |
| 4 | 843.9 | 91.1% | 0.964 |
| **8** | **845.3** | **94.4%** | **0.989** |
| 16 | 838.1 | 97.4% | 0.995 |
| 256 | 824.1 | 99.8% | 1.000 |

Two conclusions:

1. **Score is not linear in leverage.** It rises steeply, peaks near k ≈ 8,
   then *declines*. Homogeneity of degree 1 fails as soon as clipping binds,
   because saturated positions stop scaling while unsaturated ones continue.
   Scaling up is still right (k = 8 ≫ k = 0.25), but "scale until caps bind"
   overshoots — the peak is where caps *start* to bind broadly, not after.
2. **Conviction is destroyed at the optimum.** At the score-maximising k, 94%
   of active positions are pinned at their cap and the book is 0.989
   correlated with a pure sign×cap bang-bang book. The surviving conviction
   information is worth about 20 score points (845.3 at k=8 vs 824.1 at
   k=256).

**Consequence for this whole exercise:** once scaled for score, the seven
aggregators differ almost entirely in (a) the **sign** they assign and (b)
**which models they zero out**, not in their relative conviction weighting.
The study is therefore a *selection and turnover* study, not a *weighting*
study — which independently strengthens the Part 3 commission hypothesis,
since turnover is one of the few axes on which the methods still differ.

### PART 3b-i — E1b answered: the decline is volatility, NOT turnover

Re-running the sweep with commission zeroed (errata E1b's own test):

| k | score (with commission) | score (commission = 0) | turnover $m |
|---:|---:|---:|---:|
| 1 | 818.9 | 841.7 | 56.5 |
| **8** | **845.3** | **870.2** | 63.5 |
| 256 | 824.1 | 849.9 | 66.6 |

The k=8 → 256 decline is **−21.2 with commission and −20.3 without** — it
survives almost entirely. Turnover rises only 5% (63.5 → 66.6 $m) across that
range, so cost differences are immaterial.

Per E1b's stated branch: **it is a volatility effect and needs a different
explanation.** The explanation is the objective itself. Saturating
low-conviction names adds positions whose edge is weak relative to their
variance, so σ rises faster than μ, SR falls, and `μ·SR²/(SR²+1)` falls with
it. **The deadband is therefore an alpha/risk device, not a transaction-cost
device** — the opposite of E1b's alternative hypothesis, and it means deadband
width should be tuned against risk-adjusted return, not against commission.

### PART 3b-ii — E1c answered: conviction is real but non-monetisable

Scale-matched (every variant deploys identical gross dollars each day, so only
the *shape* of conviction differs; all variants keep the real signs):

| conviction | k=0.5 | k=1 | k=2 | k=4 | k=8 | k=64 |
|---|---:|---:|---:|---:|---:|---:|
| oracle (magnitude = true \|move\|) | 766.7 | 829.8 | 834.7 | 830.3 | 822.8 | 823.4 |
| real | 677.7 | 818.9 | 839.8 | 843.9 | **845.3** | 832.4 |
| shuffled (same magnitudes, wrong assets) | 498.0 | 740.8 | 779.8 | 792.6 | 801.4 | 825.6 |
| flat (equal magnitude, sign only) | 646.3 | **845.3** | 841.2 | 833.0 | 823.3 | 823.3 |

Three findings, and E1c was right to insist on the separation:

1. **Conviction carries genuine information.** real − shuffled is **+179.8 at
   k=0.5**, decaying through +78.1 (k=1) to +6.8 (k=64). Far more than the ~20
   points the naive k-sweep suggested. So the flatness was *not* simply
   evidence that conviction estimates are weak.
2. **But it is non-monetisable under this objective.** The score-optimal
   operating point (k≈8) sits where conviction is 94% saturated and worth only
   ~44 points; buying conviction back by de-levering to k=0.5 costs ~168 points
   of gross deployment. The objective pushes you to a leverage at which your
   own conviction is destroyed.
3. **Scale-matched equal weighting matches the best conviction book exactly** —
   flat at k=1 scores **845.3**, identical to real at its own optimum k=8.
   This is a direct empirical vindication of the article's most credible
   finding (§2.6) and sets the bar for Task 5: equal weight is not a weak
   baseline here, it is the incumbent-matching one.

**Caveat on the oracle row.** "Oracle" is perfect foresight of |move|, which is
*magnitude* foresight, not conviction *quality* — and it is negative from k≥2
(−22.5 at k=8). Sizing by realised |move| chases variance as much as edge, so
it is not an upper bound on conviction value; it is a demonstration that
magnitude-proportional sizing is the wrong objective under a Sharpe-like score.
A true conviction oracle would need foresight of the *predictable* component,
which is not constructible here.

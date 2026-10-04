# Synthetic-data and NULL testing — four_sigmas

> **Source:** `analysis_outputs/synthetic_null/report.md` in the team's working repository (relocated here for the public release).

Scoring window **751-1000**, eval.py accounting replicated verbatim (one-day
commission lag, truncating dollar-limit share caps, integer shares, mark-only
final day). The replica reproduces eval.py's real score to the cent, so every
synthetic number is on the leaderboard scale.

**Real data: score 986.31** (mean \$998.9/day, sd
\$1,784.2, Sharpe 8.85). 1,594 synthetic paths;
each is a full 1000-day price matrix replayed day-by-day with the book's state reset.

## 1. The null test

Each null destroys the exploitable structure a different way while holding
something real fixed. All keep day-1 price levels, so dollar limits and integer
rounding bite exactly as on real data. `t` is on mean daily PnL against zero.

| case | n | score_med | score_p95 | score_max | meanpl | meanpl_se | t | p_ge_real |
|---|---|---|---|---|---|---|---|---|
| xs_boot — common-index bootstrap | 150 | -27.5 | 79.1 | 124.0 | -26.4 | 7.5 | -3.5 | 0.0066 |
| iid_boot — per-asset bootstrap | 150 | -2.6 | 100.4 | 167.9 | -8.5 | 7.5 | -1.1 | 0.0066 |
| gauss_corr — matched Gaussian | 150 | -22.3 | 106.3 | 217.3 | -19.7 | 7.5 | -2.6 | 0.0066 |

Pooled across the endpoint-free nulls: **0 of 450** paths reached the
real score — empirical p = **0.0022**. The best single null path in the
whole study reached 217
against a real 986.

## 2. A null that was wrong, and how it showed up

The first pass used *permutation* nulls. They looked fine but returned a small,
strongly significant positive mean PnL:

| case | n | score_med | score_p95 | score_max | meanpl | meanpl_se | t | p_ge_real |
|---|---|---|---|---|---|---|---|---|
| xs_perm — common shuffle (pinned) | 200 | 9.0 | 158.0 | 295.1 | 46.5 | 6.3 | +7.4 | 0.0050 |
| iid_perm — per-asset shuffle (pinned) | 150 | 37.7 | 149.7 | 270.1 | 73.1 | 6.8 | +10.7 | 0.0066 |

That is an artifact of the null, not a bias in the book. A permutation preserves
the **sum** of log returns, so every permuted path lands on the real terminal
price exactly (verified: max relative difference 4e-16). Each path is therefore a
Brownian bridge, with `E[r_t+1 | S_t] = (total - S_t)/(n - t)` pulling toward a
pinned endpoint — real, exploitable mean reversion, which is exactly what this
book trades. A mean-reverting strategy *should* profit there.

Two checks confirm it:

- A signal-free control — fixed random long/short signs at the same full dollar
  limit, rebalanced daily, so identical sizing mechanics and no signal — earns
  nothing on the same pinned paths. The bridge gain needs the mean-reversion
  signal to harvest it, so it is not a sizing or accounting artifact.
| generator | n | meanpl | meanpl_se | score_med |
|---|---|---|---|---|
| stub:gauss_corr | 60 | 9.6 | 14.8 | 0.0 |
| stub:iid_perm | 60 | -17.5 | 12.1 | -6.1 |
| stub:real | 60 | -10.3 | 13.0 | -13.7 |
| stub:xs_perm | 60 | -22.4 | 13.4 | -18.8 |

- Resampling **with replacement** unpins the endpoint and the effect disappears —
  that is the §1 table, and it is what the verdict rests on.

The pinned nulls are kept in the figure because the contrast measures the
artifact. Note the direction: the bias made the null *stronger*, so the test was
conservative, and the conclusion survives it.

## 3. Horizon decomposition (block bootstrap)

Block resampling keeps dependence shorter than the block and destroys everything
longer, so the block length at which the score returns is the horizon the book
actually trades. Length 1 is the `xs_boot` null.

| L | n | score_med | pct_of_real | sharpe_med |
|---|---|---|---|---|
| 2 | 50 | 122.2 | 12% | 1.61 |
| 5 | 50 | 639.5 | 65% | 5.84 |
| 10 | 50 | 846.1 | 86% | 7.37 |
| 20 | 50 | 939.9 | 95% | 8.30 |
| 60 | 50 | 1,011.2 | 103% | 8.95 |

Essentially all the edge is intra-month, and about two thirds of it is already
back at a 5-day block. Nothing at horizons beyond ~20 days contributes: the book
is a short-horizon mean-reversion engine, which is what it claims to be. (These
paths are with-replacement and so endpoint-free.)

## 4. Positive control — synthetic cointegration

A one-factor market plus genuine OU-cointegrated spreads on an asset-disjoint
subset of the book's *own* configured pairs (~25 of 88; every other asset, and
all lead-lag structure the ridge sleeve wants, stays a pure random walk).
k = 0 means the spread is a random walk — no cointegration at all.

| kappa | n | half_life | score_med | pct_pos | sharpe_med |
|---|---|---|---|---|---|
| 0.00 | 25 | inf | -35.5 | 36% | -0.46 |
| 0.02 | 25 | 34.7 | 0.0 | 60% | 0.06 |
| 0.05 | 25 | 13.9 | 14.6 | 76% | 0.61 |
| 0.08 | 27 | 8.7 | 79.0 | 89% | 1.46 |
| 0.15 | 25 | 4.6 | 258.1 | 100% | 3.00 |
| 0.30 | 27 | 2.3 | 409.5 | 100% | 4.53 |

Monotone in true mean-reversion speed, and its own internal null (k=0) is
negative. The book responds to genuine cointegration rather than to its own
parameterisation. This is a validity check, **not** a calibration of the real
data's k — the control deliberately structures under a third of the pairs and
gives the ridge sleeve nothing, so its levels are not comparable to the real score.

## 5. How much of the score is this particular path?

`phase_surr` is not a null: shared-phase randomisation preserves the entire
cross-spectral density — every auto- and cross-covariance at every lag, including
the linear cointegration structure — and changes only the realised path. With the
88 pairs and their thresholds frozen, it estimates what this book scores on a
**fresh draw from the same linear structure**.

| | score |
|---|---|
| realised (real data) | **986** |
| surrogate median | 748 |
| surrogate 5-95% | 599 - 873 |
| shortfall | **238  (24% of realised)** |

Roughly 76% of the score is structure that reproduces on a
redrawn path; about 24% is specific to this realisation — frozen
parameters sitting where this particular path rewarded them. Two caveats, in
opposite directions: the surrogate preserves the DC term, so it inherits the same
endpoint pinning as §2 and is flattered by a few percent (the true shortfall is
somewhat larger); and the shortfall is a *floor* on out-of-sample decay, not a
forecast, because a real future window also brings regime drift that no surrogate
of this history contains.

## What this does and does not establish

**Does.** The machine does not manufacture profit from nothing. On structureless
data with real drift, volatility and cross-sectional correlation preserved, the
book's mean PnL is negative (commission drag) and its score distribution sits at
~0, with the real score far outside it (p = 0.0022). No look-ahead, no
accounting artifact, no long-drift bias masquerading as edge. The positive control
shows the harness detects edge when edge exists.

**Does not.** This is a *mechanism* null, not a *selection* null. The 88 pairs and
their per-pair thresholds were chosen by looking at this price history and stay
frozen across every synthetic path. Null-testing the selection itself would mean
re-running the whole pair-discovery pipeline on each synthetic path. So the
p-value says "the book extracts real structure from this data" — not "the
751-1000 score is unaffected by having selected on 501-750". §5 is the closest
this study gets to that question, and it puts 24% of the score on the
realised path.

Artifacts: `paths_*.csv` (every path), `null_summary.csv`, `block_sweep.csv`,
`control_sweep.csv`, `null_testing.png`.
Reproduce:
```
python research/synthetic_null.py --mode full --block-reps 50 --control-reps 25 --workers 7
python research/synthetic_null.py --mode null --reps 150 --workers 7
python research/synthetic_null.py --mode mechanism --reps 60 --workers 7
python research/synthetic_null_report.py
```

---

## Figures

![Null distributions vs the real-data score, horizon decomposition and positive control](../figures/synthetic-null-testing.png)

*Null distributions vs the real-data score, horizon decomposition and positive control.*

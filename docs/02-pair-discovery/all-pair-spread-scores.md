# Causal all-pair spread-score research

> **Source:** `research/cointegration/cointegration_output/all_pair_spread_scores/README.md` in the team's working repository (relocated here for the public release).

## Scope

- Data: 51 assets and 750 daily observations from `prices.txt`.
- Universe: every unordered combination, `51 choose 2 = 1,275` pairs.
- Causality: the score for decision day `t` fits the relationship using data
  strictly before `t`. Forward observations are used only to evaluate outcomes.
- Prices are analysed in logs.
- Reported forward returns are gross fixed-beta spread movements per unit gross
  pair exposure. They are not a transaction-costed portfolio backtest.

For a pair `i, j`, historical OLS estimates

`log(P_i) = alpha + beta * log(P_j) + residual`.

The daily trading score is the current residual relative to a historical
residual centre and standard deviation. A positive score means the fitted
spread is high and the contrarian direction is to short the spread.

## Estimators compared

| Hedge history / score centre | All-period 10d gross bps | Later-period 10d gross bps | Later win rate |
|---|---:|---:|---:|
| 50 / 30 | 18.1 | 8.6 | 51.0% |
| 100 / 30 | -1.0 | -11.3 | 48.7% |
| 250 / 30 | 38.6 | 21.5 | 52.4% |
| expanding / 30 | 39.6 | 14.1 | 51.9% |
| 250 / 250 | 49.5 | 30.4 | 52.0% |
| expanding / full | 0.7 | 13.6 | 50.9% |

The relationship-health gate for this comparison requires a positive hedge
ratio, conservative two-direction ADF t-stat below -2, half-life between 1.5
and 40 days, hedge-ratio half-window ratio at most 2, and `2 <= abs(z) < 4`.

Longer history stabilises beta: median daily relative beta movement falls from
6.34% with 50 observations to 1.32% with 250 observations and 0.38% with an
expanding estimate. Stability is not the same as predictive accuracy, however.
The expanding/full estimator mixes old regimes into both beta and the spread
centre. A finite 250-day memory was more reliable in the later data.

## Persistence findings

Across the 700 daily 50-day endpoints:

- raw two-direction `p <= 0.05`: 21.91 pairs per day on average;
- strict `p <= 0.001`: 0.53 pairs per day;
- Benjamini-Hochberg FDR at `q=0.05`: 0.06 pairs per day;
- `abs(z) >= 2`: 154.91 pairs per day;
- `abs(z) >= 2` plus the light health gate: 15.10 pairs per day;
- strict `p <= 0.001` plus `abs(z) >= 2`: 0.08 pairs per day.

The strict p-value dots are therefore too sparse to serve as the primary
runtime book. In days 501-750, strict-p extreme entries were negative as a
group, with only nine one-day entries available.

Under raw `p <= 0.05`, requiring five consecutive relationship-qualified days
before an extreme-score entry produced +78.6 gross bps over the next ten days
in days 501-750 (44 entries, 52.3% wins). Waiting ten days reversed this result.
This is a causal confirmation test, but raw p-values do not control the false
discovery rate across 1,275 dependent tests; persistence should be a ranking
feature rather than proof that a pair is truly cointegrated.

Extreme-score persistence has the opposite interpretation. In later data:

- entry on the first extreme day: +11.0 gross bps over ten days;
- entry on the second consecutive extreme day: +17.0 gross bps;
- entry on the third day: -2.2 gross bps;
- entry on the fifth day: -14.2 gross bps.

A spread remaining extreme for many days increasingly resembles a broken
relationship or level shift, not a safer convergence trade.

## Runtime conclusion

The empirical default should be a finite 250-day hedge estimate, not an
ever-growing estimate. Use the stationarity statistic as a relationship-health
gate and the current z-score as the entry signal. A practical candidate rule is:

1. Fit beta on the previous 250 log-price observations.
2. Require beta > 0, two-half beta ratio <= 2, residual ADF t < -2, and estimated
   half-life between 1.5 and 40 days.
3. Calculate the current spread z-score from past residuals. The 250-day centre
   was strongest in aggregate; a 30-day centre remains useful for pairs known to
   undergo level shifts.
4. Enter at `2 <= abs(z) < 4`, preferably on the second consecutive extreme day.
5. Do not enter a new trade after three or more consecutive extreme days; treat
   this as possible relationship failure.
6. Optionally reward five consecutive days with raw p <= 0.05 in ranking, but do
   not require p <= 0.001 and do not interpret raw p as a trade signal.
7. Select a small set of non-overlapping pairs, most attractive first, so a hub
   asset cannot create many duplicated directional positions.
8. Exit on spread recentering, relationship-gate failure, or a half-life-based
   maximum hold. Validate this inside the complete strategy with costs and
   position limits before changing `teamName.py`.

## Outputs

- `lookback_forward_metrics.csv`: every estimator, period, horizon and gate.
- `estimator_stability.csv`: beta stability and signal breadth.
- `pvalue_episodes.csv`: all raw, strict and FDR-qualified p-value runs.
- `pvalue_confirmation_outcomes.csv`: causal p-persistence entries.
- `zscore_episodes.csv`: all light-health extreme-score runs.
- `zscore_confirmation_outcomes.csv`: causal z-persistence entries.
- `daily_all_pair_breadth.csv`: daily cross-sectional counts and hub breadth.
- `all_pair_scores_w50.npz`: all-pair 50-day beta, p, z, ADF and half-life arrays.
- `engle_granger_validation.csv`: numerical validation against `statsmodels`.

The vectorised Engle-Granger p-values agree with direct `statsmodels.coint`
calls to a maximum checked absolute error of `2.97e-08`.

---

## Figures

![Top-40 pairs by cointegration persistence across 50-day blocks](../figures/cointegration-persistence-50d-blocks.png)

*Top-40 pairs by cointegration persistence across 50-day blocks.*

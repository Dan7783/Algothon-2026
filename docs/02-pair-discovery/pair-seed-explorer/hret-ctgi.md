# HRET/CTGI robust seed decision

> **Source:** `cusum_output/pair_seed_explorer/hret_ctgi/report.md` in the team's working repository (relocated here for the public release).

## Decision

Only the existing sleeve scale changes:

| Field | Legacy | Selected |
|---|---:|---:|
| `leg_scale` | 1.00 | 0.65 |

Entry `0.33`, exit `0.10`, stop `3.75`, ADF/kappa gates, priority, and every
Kalman/CUSUM field remain unchanged. This is a portfolio allocation change,
not a claim that the underlying HRET/CTGI alpha became stronger.

The selected `0.65` lies in a stable `0.60-0.70` portfolio plateau. Immediate
neighbors `0.64`, `0.65`, and `0.66` have nearly identical train/test results.
The middle two-decimal value is used rather than the best observed test point.

## Sleeve diagnosis

HRET/CTGI is positive overall but inconsistent. At full size its 50-day train
block means were `124.09, -41.80, 54.20, 134.34, -16.19`; test blocks were
`33.84, 56.64, -10.69, 27.74, 36.35`. The selected scale preserves those trade
decisions and reduces their dollar impact. Its corresponding block means are
`80.81, -27.16, 34.97, 87.13, -10.73` and
`22.09, 36.59, -7.14, 17.95, 23.62`.

| Window / metric | Full-size sleeve | 0.65 sleeve |
|---|---:|---:|
| Train score | 43.22 | 28.00 |
| Train mean/day | 50.93 | 33.00 |
| Train Sharpe / Sortino | 2.37 / 2.54 | 2.37 / 2.52 |
| Train max drawdown | 3,568 | 2,326 |
| Train CVaR 5% | -638 | -414 |
| Train turnover | 1.091m | 0.708m |
| Test score | 19.64 | 12.70 |
| Test mean/day | 28.77 | 18.62 |
| Test Sharpe / Sortino | 1.47 / 1.58 | 1.46 / 1.57 |
| Test max drawdown | 2,253 | 1,457 |
| Test CVaR 5% | -567 | -368 |
| Test turnover | 0.852m | 0.552m |

The lower dollar score is expected when scaling a signal down. Risk-adjusted
quality and trade timing are essentially unchanged. The reason to scale is its
interaction with higher-quality, conflicting portfolio sleeves.

## Filter and gate diagnosis

- The live filter produces zero CUSUM cuts, so the tested `h` and `cusum_k`
  variants are inert.
- `q_beta` from 0.5 to 3 times live produces the same relevant train path.
- Kappa blocks zero train and test days.
- ADF blocks 6.4% of train and zero test days. Disabling it raises standalone
  train score but reduces full-portfolio train score from `806.31` to `803.01`
  and does not change full test. ADF therefore remains enabled.
- Stops from 3 to 6 are train-inert. A stop of 3 improves only the already
  inspected test window, so selecting it would be test fitting.

## Fast-beta lag diagnostic

The dedicated explorer now overlays a diagnostic through-origin Kalman beta
using `20x` the live process noise (`q_beta = 3.436e-04`). It deliberately has
no CUSUM segmentation and is never used for positions, exits, or PnL. A
separate panel plots both `fast - sticky` and `rolling100 - sticky`, making the
amount and direction of smoothing visible without changing the strategy.

| Window | Median / p95 / max `abs(fast-sticky)` | Sticky MAE to rolling 100d | Fast MAE to rolling 100d | Estimated fast lead |
|---|---:|---:|---:|---:|
| Train | 0.00751 / 0.02353 / 0.02576 | 0.00466 | 0.01344 | 0d (change corr 0.857) |
| Test | 0.00740 / 0.01970 / 0.02775 | 0.00413 | 0.01168 | 0d (change corr 0.882) |

The live sticky beta is about three times closer to the rolling 100-day beta
than the fast diagnostic is. However, the maximum correlation between their
daily changes occurs at zero lag. The visible sluggishness is therefore mainly
attenuation--the live filter turns on the same day but moves too little--rather
than a stable multi-day time delay. This diagnostic alone does not justify
changing the production beta; intermediate responsiveness should be assessed
through train/test PnL and drawdown before deployment.

## Full robust portfolio

The deployed static seed was replayed with independent train/test state, the
current `robust_hybrid_v2` policy, conflicts, position caps, and delayed
eval-style commissions.

| Window / metric | Legacy | Selected | Change |
|---|---:|---:|---:|
| Train score | 806.31 | 818.15 | +11.85 |
| Train mean/day | 816.62 | 827.89 | +11.27 |
| Train Sharpe / Sortino | 8.844 / 10.471 | 9.169 / 11.466 | improved |
| Train max drawdown | 4,986 | 3,557 | -28.7% |
| Train CVaR 5% | -2,422 | -2,193 | 9.5% less severe |
| Train turnover | 66.734m | 66.940m | +0.31% |
| Train half scores | 774.93 / 837.64 | 777.16 / 859.10 | both improve |
| Test score | 846.48 | 852.67 | +6.19 |
| Test mean/day | 855.11 | 861.18 | +6.07 |
| Test Sharpe / Sortino | 9.904 / 13.531 | 10.011 / 13.536 | improved |
| Test max drawdown | 4,060 | 3,736 | -8.0% |
| Test CVaR 5% | -1,684 | -1,680 | slightly less severe |
| Test turnover | 59.839m | 59.680m | -0.27% |
| Test half scores | 733.58 / 959.26 | 725.64 / 979.56 | mixed |

The weaker first test half is explicit: the gain is concentrated in the second
half. The scale is retained because the same `0.60-0.70` neighborhood improves
both total windows and materially reduces drawdown, not because every subwindow
improves.

## Scale sensitivity and rejected alternatives

| Scale | Train score | Test score | Train / test max DD |
|---:|---:|---:|---:|
| 1.00 (legacy) | 806.31 | 846.48 | 4,986 / 4,060 |
| 0.60 | 819.48 | 850.77 | 3,560 / 3,736 |
| 0.625 | 818.74 | 850.96 | 3,557 / 3,736 |
| 0.64 | 818.10 | 852.62 | 3,557 / 3,736 |
| 0.65 (selected) | 818.15 | 852.67 | 3,557 / 3,736 |
| 0.66 | 818.11 | 852.78 | 3,557 / 3,736 |
| 0.675 | 812.73 | 852.89 | 3,557 / 3,736 |
| 0.70 | 814.81 | 851.31 | 3,557 / 3,736 |

- Removing HRET/CTGI entirely raises train score to `831.96` but lowers test to
  `830.92`; removal is not robust.
- Scale `0.25-0.55` emphasizes train too strongly and lowers test score.
- Scale `0.75` improves test but lowers train; the `0.85-0.95` region offers a
  smaller and less stable train gain with less drawdown improvement.
- A `0.75/0.35` threshold candidate raises full train score to `819.01` but
  lowers test to `835.62`. Combining it with `0.65` scale also lowers test.
- The broad multi-metric winner reduced both standalone train and test scores;
  it was not deployed.

The broad train-ranked search evaluated 79,920 configurations, of which 60,840
passed its eligibility guard. A focused 938-point threshold/ADF surface and a
full-portfolio scale sweep then resolved the allocation plateau. Because the
held-out window informed the final scale choice, future or walk-forward data is
the only clean out-of-sample confirmation.

Re-run the broad search with
`py tune_pair_seed_explorer.py --pair HRET/CTGI`, the focused surface with
`py screen_hret_ctgi_thresholds.py`, and the deployed comparison with
`py validate_hret_ctgi_candidate.py --only legacy selected_live`.

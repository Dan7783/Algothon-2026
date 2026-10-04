# HUXZ/ACAC robust seed decision

> **Source:** `cusum_output/pair_seed_explorer/huxz_acac/report.md` in the team's working repository (relocated here for the public release).

## Decision

Only the two existing trade thresholds change:

| Field | Legacy | Selected |
|---|---:|---:|
| `z_entry` | 0.50 | 0.60 |
| `z_exit` | 0.18 | 0.23 |

The Kalman/CUSUM filter, ADF and kappa gates, stop, sizing, priority, and every
other strategy parameter remain unchanged. The higher entry waits for a more
meaningful dislocation. The higher exit takes the reversion off before it has
to reach almost exactly zero.

The selected exit is not a one-point optimum: `0.225` and `0.230` generate the
same standalone and full-portfolio train/test paths. The two-decimal `0.23`
value is used. The filter recorded no CUSUM cuts for this sample, and broad
filter variants either generated the same trades or offered a worse robustness
trade-off, so no beta-responsiveness parameter was changed.

## Standalone pair, after 1 bp costs

| Window / metric | Legacy | Selected | Change |
|---|---:|---:|---:|
| Train score | 45.64 | 46.90 | +1.26 |
| Train mean/day | 49.12 | 50.18 | +1.06 |
| Train net | 12,281 | 12,546 | +265 |
| Train Sharpe / Sortino | 3.62 / 4.38 | 3.78 / 4.44 | improved |
| Train max drawdown | 819 | 819 | unchanged |
| Train CVaR 5% | -378 | -369 | 2.5% less severe |
| Train entries | 24 | 22 | -2 |
| Train turnover | 1.168m | 1.084m | -7.2% |
| Test score | 37.20 | 43.09 | +5.89 |
| Test mean/day | 40.82 | 46.08 | +5.26 |
| Test net | 10,204 | 11,519 | +1,315 |
| Test Sharpe / Sortino | 3.21 / 3.50 | 3.80 / 4.17 | improved |
| Test max drawdown | 1,151 | 1,151 | unchanged |
| Test CVaR 5% | -401 | -378 | 5.6% less severe |
| Test entries | 21 | 26 | +5 |
| Test turnover | 1.275m | 1.387m | +8.8% |

All five chronological 50-day blocks remain positive in both windows.
Selected train block means are `61.77, 45.26, 40.37, 35.32, 68.21`; selected
test block means are `34.05, 66.61, 45.09, 42.56, 42.07`. The legacy block
means were `61.77, 46.89, 41.29, 34.92, 60.75` and
`22.48, 60.26, 47.00, 39.99, 34.36` respectively.

## Full robust portfolio

The static production seed was replayed with independent train/test state, the
current `robust_hybrid_v2` policy, position caps, conflicts, and delayed
eval-style commissions.

| Window / metric | Legacy | Selected | Change |
|---|---:|---:|---:|
| Train score | 789.80 | 790.97 | +1.16 |
| Train mean/day | 800.30 | 801.50 | +1.20 |
| Train Sharpe / Sortino | 8.674 / 9.966 | 8.667 / 10.054 | mixed, effectively flat |
| Train max drawdown | 4,986 | 4,986 | unchanged |
| Train CVaR 5% | -2,405 | -2,427 | 0.9% more severe |
| Train turnover | 66.672m | 66.653m | -0.03% |
| Train half scores | 760.25 / 819.32 | 761.02 / 820.87 | both improve |
| Test score | 818.32 | 826.01 | +7.69 |
| Test mean/day | 827.40 | 834.99 | +7.59 |
| Test Sharpe / Sortino | 9.493 / 12.794 | 9.590 / 12.746 | mixed |
| Test max drawdown | 4,060 | 4,060 | unchanged |
| Test CVaR 5% | -1,821 | -1,791 | 1.6% less severe |
| Test turnover | 59.460m | 59.640m | +0.30% |
| Test half scores | 692.86 / 943.57 | 712.73 / 939.13 | first improves, second softens |

This is a modest train improvement and a larger test improvement, without a
drawdown change. It should not be interpreted as an expected +7.69 future
score: the test window was inspected during final candidate selection.

## Sensitivity and rejected alternatives

| Entry / exit | Train portfolio score | Test portfolio score | Train max DD |
|---:|---:|---:|---:|
| 0.50 / 0.18 (legacy) | 789.80 | 818.32 | 4,986 |
| 0.60 / 0.215 | 789.48 | 822.81 | 4,986 |
| 0.60 / 0.220 | 789.48 | 826.01 | 4,986 |
| 0.60 / 0.225 | 790.97 | 826.01 | 4,986 |
| 0.60 / 0.230 (selected) | 790.97 | 826.01 | 4,986 |
| 0.60 / 0.235 | 790.13 | 826.02 | 5,359 |

- The unconstrained train winner disabled both gates and used a `0.02` exit.
  Its standalone pair score rose to `56.13` in train but fell to `30.65` in
  test, so it was rejected as fitted and unnecessarily permissive.
- A historical faster-beta candidate changed `q_beta`, `h`, `cusum_k`, and the
  thresholds. It reduced full train score to `787.86` and was rejected.
- A two-times-`q_beta` candidate improved full train more, but delivered only
  `819.67` test score and added filter sensitivity. The threshold-only seed is
  simpler and materially stronger in test.
- `0.235` has essentially the same test score as the selection but raises train
  drawdown. `0.22` gives up train score. The flat `0.225-0.230` path is the
  cleaner compromise.

The broad train-ranked search evaluated 79,380 configurations using score,
mean, Sharpe/Sortino, chronological blocks, drawdown, CVaR, and turnover. A
focused 108-point threshold screen then checked the local neighborhood. Since
the held-out window has now informed the final choice, future or walk-forward
data is the only clean out-of-sample confirmation.

Re-run the broad surface with
`py tune_pair_seed_explorer.py --pair HUXZ/ACAC`, the local plateau with
`py screen_huxz_thresholds.py`, and the deployed old-versus-new portfolio
comparison with
`py validate_huxz_acac_candidate.py --only legacy selected_live`.

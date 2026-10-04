# CCNS/SRTX robust seed decision

> **Source:** `cusum_output/pair_seed_explorer/ccns_srtx/report.md` in the team's working repository (relocated here for the public release).

## Decision

CCNS/SRTX changes from a permanently dormant seed to a small, conflict-aware
diversifier:

| Field | Old | Selected |
|---|---:|---:|
| `z_entry` | 0.44 | 0.75 |
| `z_exit` | 0.24 | 0.15 |
| `z_stop` | 4.75 | 4.75 |
| `leg_scale` | 1.00 | 0.30 |
| `require_adf` | true | false |
| `require_kappa` | true | true |

Priority and every Kalman/CUSUM field remain unchanged.  The higher entry and
lower exit came from a broad train-ranked threshold neighborhood.  The 0.30
scale is the middle of a 0.25-0.35 portfolio plateau; it also reduces the
pair's effective allocator rank, so stronger conflicting pairs retain priority.

## Why the old seed did nothing

The old pair produced zero train and test PnL because ADF blocked 100% of both
windows.  Its median expanding ADF statistic was `-2.46` in train and `-1.92`
in test versus the `-2.90` gate.  Changing entry/exit/stop cannot rescue a
signal that never reaches the trading state.

The local OU diagnostic was materially less hostile: median kappa was `0.0626`
in train and `0.0340` in test, with kappa blocking 0% and 2% of the respective
windows.  The selected policy therefore relaxes only ADF and retains the OU
gate.  It is deliberately limited to 30% size because the through-origin
spread does not pass the strategy's expanding ADF standard.

## Standalone pair, after 1 bp costs

The selected sleeve is positive but weak; it is not being presented as a
strong standalone alpha.

| Metric | Train | Test |
|---|---:|---:|
| Score | 6.61 | 0.91 |
| Mean/day | 8.41 | 2.99 |
| Net | 2,104 | 747 |
| Sharpe / Sortino | 1.92 / 1.85 | 0.66 / 0.58 |
| Max drawdown | 521 | 837 |
| CVaR 5% | -142 | -156 |
| Entries | 7 | 7 |
| Turnover | 0.120m | 0.102m |

Four of five 50-day blocks are positive in each window.  Train block means are
`-4.26, 14.33, 10.85, 12.67, 8.48`; test block means are
`-13.06, 7.58, 12.22, 1.20, 6.99`.

## Full robust portfolio

The selected seed was replayed with independent train/test state, the current
`robust_hybrid_v2` production policy, position caps, and delayed eval-style
commissions.

| Window / metric | Dormant seed | Selected seed | Change |
|---|---:|---:|---:|
| Train score | 788.34 | 789.80 | +1.46 |
| Train mean/day | 799.10 | 800.30 | +1.20 |
| Train Sharpe / Sortino | 8.56 / 9.40 | 8.67 / 9.97 | improved |
| Train max drawdown | 6,392 | 4,986 | -22.0% |
| Train CVaR 5% | -2,510 | -2,405 | 4.2% less severe |
| Train turnover | 67.216m | 66.672m | -0.81% |
| Train half scores | 759.57 / 817.07 | 760.25 / 819.32 | both improve |
| Test score | 816.79 | 818.32 | +1.53 |
| Test mean/day | 825.92 | 827.40 | +1.48 |
| Test Sharpe / Sortino | 9.46 / 12.51 | 9.49 / 12.79 | improved |
| Test max drawdown | 4,060 | 4,060 | unchanged |
| Test CVaR 5% | -1,844 | -1,821 | 1.3% less severe |
| Test turnover | 59.406m | 59.460m | +0.09% |
| Test half scores | 695.64 / 937.70 | 692.86 / 943.57 | mixed |

The improvement is primarily diversification and conflict allocation, not the
small standalone PnL.  Removing the dormant pair improved train but reduced test
score to `802.37`, demonstrating that its conflict-graph role also matters.

## Sensitivity and rejected alternatives

| Scale | Train score | Test score | Train max DD |
|---:|---:|---:|---:|
| 0.25 | 788.78 | 821.38 | 5,034 |
| 0.30 | 789.80 | 818.32 | 4,986 |
| 0.35 | 791.77 | 818.69 | 4,933 |

All three neighboring scales improve both full windows versus `788.34 / 816.79`.
The central 0.30 setting was selected instead of the best individual score.

- Full-size activation improved train but degraded test materially.
- Disabling both ADF and kappa added risk without a necessary train benefit;
  kappa remains enabled.
- Retaining the old `0.44/0.24` thresholds at small size degraded train.
- Retaining `z_exit=0.24` after raising entry also degraded train; the 0.15 exit
  is necessary and lies on a flat 0.10-0.15 pair-level neighborhood.
- Filter changes (`h`/`cusum_k`) offered only small fitted gains and were not
  deployed.

The search evaluated 58,725 configurations.  Candidate ranking used train only
and included score/mean, Sharpe/Sortino, chronological blocks, drawdown, CVaR,
and turnover.  Test and portfolio replays were validation.  Because this test
window has now been inspected, future/walk-forward data remains the only clean
out-of-sample confirmation.

Re-run the current-seed surface with
`py tune_pair_seed_explorer.py --pair CCNS/SRTX`; reproduce the old-versus-live
portfolio comparison with
`py validate_ccns_srtx_candidate.py --only legacy_dormant selected_live`.

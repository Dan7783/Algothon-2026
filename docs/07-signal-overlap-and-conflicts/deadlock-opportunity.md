# Pair-overlap deadlock opportunity

> **Source:** `analysis_outputs/deadlock_opportunity/report.md` in the team's working repository (relocated here for the public release).

- Deadlocked asset-days: **1860**
- Calendar days containing a deadlock: **800**

| Period | Events | Oracle movement | Moneyness PnL | Moneyness total-PnL delta | Moneyness score delta |
|---|---:|---:|---:|---:|---:|
| YEAR1 | 622 | 113094.00 | -8821.28 | -9012.19 | -42.1368 |
| YEAR2 | 446 | 79118.51 | 2213.18 | 2028.10 | 8.1039 |
| VALIDATION | 413 | 73960.40 | -2988.26 | -3148.04 | -13.0415 |
| OOS | 379 | 63997.28 | -3959.91 | -4048.73 | -16.6888 |

`Oracle movement` is an impossible hindsight upper bound: it
assumes the correct direction is known before the next return.
`Moneyness total-PnL delta` is the causal counterfactual after
share rounding, position transitions, and commissions.

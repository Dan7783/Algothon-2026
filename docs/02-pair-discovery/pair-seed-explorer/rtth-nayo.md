# RTTH/NAYO robust seed decision

> **Source:** `cusum_output/pair_seed_explorer/report.md` in the team's working repository (relocated here for the public release).

## Decision

The live RTTH/NAYO seed changes only `z_exit`, from `0.02` to `0.25`.
`z_entry=0.50`, `z_stop=4.25`, priority, and every Kalman/CUSUM parameter remain
unchanged.  The new exit is deliberately interpretable (half the entry level)
and lies inside the broad `0.15-0.25` train-robust plateau.

The scalar train-only selector landed at `0.175`, while a larger unconstrained
search proposed changes to `q_beta`, `cusum_k`, `snap_overshoot`, and `z_stop`.
Those extra changes were rejected: their small improvement was train-only,
several settings produced identical trades, and they added unnecessary degrees
of freedom.  The `0.25` live choice has nearly the same balanced train quality,
better tail behavior, and a better full-portfolio trade-off.

## Standalone pair, after 1 bp costs

| Window / metric | Old exit 0.02 | Live exit 0.25 | Change |
|---|---:|---:|---:|
| Train score | 94.90 | 82.49 | -13.1% |
| Train mean/day | 100.76 | 87.76 | -12.9% |
| Train Sharpe / Sortino | 4.02 / 4.06 | 3.96 / 4.32 | tail ratio improves |
| Train max drawdown | 1,646 | 1,378 | -16.3% |
| Train CVaR 5% | -791 | -649 | 18.0% less severe |
| Train worst 50-day mean | 58.87 | 60.73 | +3.2% |
| Test score | 35.34 | 46.40 | +31.3% |
| Test mean/day | 46.30 | 54.64 | +18.0% |
| Test Sharpe / Sortino | 1.80 / 1.88 | 2.37 / 2.47 | improved |
| Test max drawdown | 2,710 | 2,181 | -19.5% |
| Test CVaR 5% | -766 | -702 | 8.4% less severe |

The old exit generated only 4 train and 2 test entries because trades were held
almost all the way to zero.  The new exit generated 22 and 17 entries.  That
raises standalone pair turnover, but it stops one trade from monopolizing the
pair and realizes the reversion before the spread reaches its noisy mean region.

All five 50-day train blocks and all five test blocks remain profitable.  The
new train block means are `64.7, 130.1, 112.8, 60.7, 70.4`; test block means are
`0.7, 70.6, 35.7, 74.9, 91.3`.

## Full robust portfolio

The portfolio replay uses the selected `robust_hybrid_v2` policy, independent
train/test state, position caps, and delayed eval-style commissions.

| Window / metric | Old exit 0.02 | Live exit 0.25 | Change |
|---|---:|---:|---:|
| Train score | 795.63 | 788.34 | -0.92% |
| Train mean/day | 806.26 | 799.10 | -0.89% |
| Train Sharpe / Sortino | 8.65 / 9.42 | 8.56 / 9.40 | essentially flat |
| Train half scores | 752.85 / 838.36 | 759.57 / 817.07 | mixed, both strong |
| Train max drawdown | 6,392 | 6,392 | unchanged |
| Train turnover | 67.278m | 67.216m | -0.09% |
| Test score | 799.59 | 816.79 | +2.15% |
| Test mean/day | 808.85 | 825.92 | +2.11% |
| Test Sharpe / Sortino | 9.29 / 12.21 | 9.46 / 12.51 | improved |
| Test half scores | 692.82 / 906.18 | 695.64 / 937.70 | both improve |
| Test max drawdown | 4,060 | 4,060 | unchanged |
| Test turnover | 59.534m | 59.406m | -0.21% |

Portfolio drawdown is unchanged because another sleeve determines the maximum.
Despite higher standalone RTTH/NAYO activity, total portfolio turnover falls
slightly because earlier exits free RTTH/NAYO capacity and alter pair conflicts.

## Overfit controls and limits

- Candidate ranking used train metrics only: score/mean, Sharpe/Sortino,
  chronological blocks, drawdown, CVaR, and turnover.  Test was validation.
- The result is supported by a neighboring threshold plateau, not one grid point.
- No new strategy parameter was added; one existing seed value changed.
- Later entries were rejected: moving `z_entry` above `0.50` reduced the balanced
  train result and generally weakened test performance for this pair.
- Faster/slower beta and CUSUM variants were not deployed because their edge did
  not survive as a clear train-and-test improvement.
- The reported test interval has already been inspected during research, so it
  is validation rather than pristine out-of-sample evidence.  Forward or
  walk-forward data remains the proper final check.

Reproduce the pair sweep with `py tune_pair_seed_explorer.py`, the portfolio
check with `py validate_pair_seed_candidate.py`, and the saved notebook plots by
executing `pair_seed_explorer.ipynb`.

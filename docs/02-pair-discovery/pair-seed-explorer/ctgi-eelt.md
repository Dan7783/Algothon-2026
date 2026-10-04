# CTGI/EELT robust tuning report

> **Source:** `cusum_output/pair_seed_explorer/ctgi_eelt/report.md` in the team's working repository (relocated here for the public release).

## Production decision

Only `z_entry` changes, from `0.30` to `0.35`.  The live beta speed, exit,
stop, CUSUM controls, ADF/kappa gates, priority, and sleeve scale remain
unchanged.

| Parameter | Legacy | Selected |
|---|---:|---:|
| `z_entry` | 0.30 | **0.35** |
| `z_exit` | 0.05 | 0.05 |
| `z_stop` | 3.25 | 3.25 |
| `q_beta` | 1.833e-07 | 1.833e-07 |
| `h / cusum_k / adapt_bars / snap` | 4.0 / 0.45 / 5 / 1.15 | unchanged |
| ADF / kappa | on / on | unchanged |
| Leg scale | 1.00 | 1.00 |

This is the smallest robust change supported by the search.  It avoids
deploying the faster beta merely because it looks better in isolation.

## Standalone pair behavior

Independent train and test state was used, with eval-style delayed
commissions.

| Window / metric | Legacy | Selected | Change |
|---|---:|---:|---:|
| Train score | 94.07 | 96.46 | +2.39 |
| Train mean/day | 99.05 | 101.32 | +2.27 |
| Train Sharpe / Sortino | 4.35 / 5.07 | 4.45 / 5.24 | improved |
| Train max drawdown | 1,987 | 1,987 | unchanged |
| Train CVaR 5% | -598.59 | -596.67 | less severe |
| Train turnover | 1.386m | 1.350m | -2.6% |
| Test score | 48.19 | 52.30 | +4.11 |
| Test mean/day | 56.63 | 60.34 | +3.71 |
| Test Sharpe / Sortino | 2.39 / 2.64 | 2.55 / 2.82 | improved |
| Test max drawdown | 1,783 | 1,783 | unchanged |
| Test CVaR 5% | -678.67 | -678.97 | essentially unchanged |
| Test turnover | 0.923m | 0.923m | unchanged |

The entry change is sparse rather than pervasive.  It improves train block 1
from `92.82` to `94.75`, train block 5 from `89.77` to `99.18`, and test block
1 from `82.47` to `101.03`; the other seven 50-day block means are unchanged.
All five train and all five test blocks remain positive.  Entry count and
exposure are unchanged at `14 / 8` entries and `92.0% / 95.6%` exposure for
train/test, so the gain is not coming from indiscriminately adding trades.

## Beta responsiveness and gate diagnosis

The live filter makes zero CUSUM cuts and both ADF and kappa block zero train
and test days.  Consequently, changing `h`, `cusum_k`, or the gate flags adds
parameters without changing the live path.

The dedicated explorer overlays a diagnostic beta at `64x` live process noise
(`q_beta = 1.173e-05`).  It is never used for trading.

| Window | Median / p95 / max `abs(fast-sticky)` | Sticky MAE to rolling 100d | Fast MAE to rolling 100d | Estimated fast lead |
|---|---:|---:|---:|---:|
| Train | 0.00443 / 0.01115 / 0.01154 | 0.00527 | 0.00233 | 0d (change corr 0.979) |
| Test | 0.00871 / 0.01608 / 0.01718 | 0.00902 | 0.00289 | 0d (change corr 0.944) |

The live beta is clearly over-smoothed relative to this diagnostic, especially
in test, but the maximum daily-change correlation is still at zero lag.  As
with HRET/CTGI, the visual lag is mainly attenuated movement rather than a
stable multi-day time shift.

Speeds from `32x` through `128x` form a broad standalone trading plateau and
track rolling 100-day beta more closely.  However, the `64x` beta plus its
train-led threshold candidate raises full-portfolio train score from `818.15`
to `821.47` while lowering full test from `852.67` to `851.63`.  It is therefore
retained as a visual diagnostic and not promoted to production.  Slower-than-
live speeds are also rejected because they trigger a single large CUSUM snap,
the brittle step-jump behavior identified in the earlier beta review.

## Full robust portfolio

The deployed seed was replayed with the current `robust_hybrid_v2` policy,
conflicts, caps, independent train/test state, and delayed commissions.

| Window / metric | Legacy entry 0.30 | Selected entry 0.35 | Change |
|---|---:|---:|---:|
| Train score | 818.15 | 819.49 | +1.33 |
| Train mean/day | 827.89 | 829.30 | +1.41 |
| Train Sharpe / Sortino | 9.169 / 11.466 | 9.139 / 11.550 | mixed, Sortino improves |
| Train max drawdown | 3,557 | 3,557 | unchanged |
| Train CVaR 5% | -2,193 | -2,193 | unchanged |
| Train turnover | 66.940m | 66.926m | -14.4k |
| Train half scores | 777.16 / 859.10 | 777.95 / 860.99 | both improve |
| Test score | 852.67 | 852.67 | unchanged |
| Test mean/day | 861.18 | 861.18 | unchanged |
| Test Sharpe / Sortino | 10.011 / 13.536 | 10.011 / 13.536 | unchanged |
| Test max drawdown / CVaR 5% | 3,736 / -1,680 | 3,736 / -1,680 | unchanged |
| Test turnover | 59.680m | 59.680m | unchanged |
| Test half scores | 725.64 / 979.56 | 725.64 / 979.56 | unchanged |

Removing CTGI/EELT is decisively harmful: full train/test scores fall to
`755.20 / 767.91`, with drawdowns rising to `4,980 / 5,230`.

## Sensitivity and rejected alternatives

- Entry `0.325` lowers full train score to `815.57`; entry `0.375` lowers it to
  `817.63` and full test to `850.84`.  The selected `0.35` is the local robust
  center, not the endpoint of a monotonic chase.
- Stops from `3.25` through `6.0` are behaviorally identical.  The legacy stop
  is retained instead of adding an inert tuned value.
- Exit `0.075` improves full test but reduces standalone train score below the
  legacy sleeve.  Exit `0.10` lowers both full train and test.  The live `0.05`
  exit is retained to avoid selecting a test-specific exit.
- The broad train-ranked screen evaluated `79,380` candidates; the focused
  beta/threshold screen evaluated `5,824`.  These grids are diagnostic.  They
  do not justify carrying dozens of pair-specific controls into production.

Re-run the broad search with
`py archive/pair_validation/tune_pair_seed_explorer.py --pair CTGI/EELT`, the focused
screen with `py archive/pair_validation/screen_ctgi_eelt.py`, and portfolio validation
with `py archive/pair_validation/validate_ctgi_eelt_candidate.py --only live legacy_entry030`.

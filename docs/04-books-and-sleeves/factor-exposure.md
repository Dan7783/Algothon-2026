# Factor-exposure economic audit

> **Source:** `analysis_outputs/factor_exposure/report.md` in the team's working repository (relocated here for the public release).

Signals and execution limits remain fixed. Pair and ridge exposure change their authority in conflicts; ALGO exposure scales the requested residual basket before the existing beta-neutral projection.

## Current economic profile

| Window | PnL | Sharpe | Max drawdown | Hit rate | Net bps/turnover | Gross | ALGO beta ratio | PC1 ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | \$242,786 | 9.01 | \$5,398 | 74.8% | 32.80 | \$456,757 | 18.65% | 2.46% |
| VALIDATION | \$267,107 | 9.90 | \$6,725 | 73.6% | 38.57 | \$448,534 | 15.03% | 2.08% |
| EXPOSED | \$249,724 | 8.85 | \$5,637 | 71.6% | 35.43 | \$457,359 | 16.28% | 2.53% |

## Development-only increased-exposure verdicts

| Family | Best increased variant | Dominates without score? | Train PnL Δ | Validation PnL Δ | Train Sharpe Δ | Validation Sharpe Δ | Train drawdown Δ | Validation drawdown Δ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pair | pair_x1.50 | False | \$+9,700 | \$-7,170 | +0.81 | +0.32 | \$+34 | \$+338 |
| ridge | ridge_x1.00 | False | \$-844 | \$+1,073 | -0.08 | -0.02 | \$+0 | \$+0 |
| algo | algo_x1.50 | False | \$-1,475 | \$+612 | -0.03 | +0.05 | \$+225 | \$+0 |

## Frozen descriptive results on days 751-1000

| Family | Variant | PnL | Sharpe | Max drawdown | Net bps/turnover | ALGO beta ratio | PC1 ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| pair | pair_x1.50 | \$227,654 | 9.12 | \$4,445 | 39.13 | 13.29% | 2.07% |
| ridge | ridge_x1.00 | \$248,895 | 8.82 | \$5,637 | 35.14 | 16.39% | 2.55% |
| algo | algo_x1.50 | \$250,908 | 8.92 | \$5,710 | 35.77 | 16.27% | 2.52% |

A variant is marked as dominating only when train and validation both improve total PnL, Sharpe, and PnL per turnover, while neither maximum drawdown rises by more than 10%. Score is not part of this decision.

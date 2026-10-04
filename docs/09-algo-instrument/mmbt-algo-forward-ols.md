# MMBT lag-1 return -> ALGO 30-day forward-return audit

> **Source:** `analysis_outputs/mmbt_algo_forward_ols/report.md` in the team's working repository (relocated here for the public release).

Primary equation: `ALGO forward log return = alpha + beta * latest completed MMBT one-day return`. All forward labels end inside their own chronological window.

## Training fit

- Observations: **410**.
- Beta for a one-standard-deviation MMBT move: **+0.1013%** ALGO over 30 days.
- HAC t-stat / p-value: **+0.42 / 0.6758**.
- In-sample R-squared: **0.049%**.

## Chronological stability

| Window | Correlation | Local beta | HAC t | Local R2 | Frozen-model OOS R2 | Direction | Baseline direction | Prediction long |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN_EARLY | +0.056 | +0.255% | +0.71 | 0.31% | +0.22% | 43.7% | 43.7% | 0.0% |
| TRAIN_LATE | -0.046 | -0.164% | -1.34 | 0.21% | -0.20% | 78.2% | 78.2% | 0.0% |
| VALIDATION | -0.034 | -0.193% | -0.54 | 0.12% | -0.11% | 43.6% | 43.6% | 0.0% |
| EXPOSED | +0.068 | +0.499% | +1.05 | 0.46% | +0.13% | 40.5% | 40.5% | 0.0% |

## 30-day overlapping-sleeve ALGO strategy

One-thirtieth of the \$100k ALGO limit is initiated daily and held for 30 days, with the supplied ALGO commission.

| Window | PnL | Sharpe | Max drawdown | Hit rate | Turnover | Mean absolute position |
|---|---:|---:|---:|---:|---:|---:|
| VALIDATION | \$-5,119 | -0.32 | \$19,642 | 50.6% | \$200,000 | \$88,000 |
| EXPOSED | \$-22,975 | -1.43 | \$31,485 | 46.2% | \$200,000 | \$88,000 |

## Robustness controls

- Late-train circular-shift p-value: **0.4062** (observed |corr| 0.046; 95th-percentile placebo 0.127).
- MMBT ranks **34/51** assets by absolute training correlation.
- MMBT HAC p-value after a 51-asset Bonferroni correction: **1.0000**.

### Horizon sensitivity

| Horizon | Train corr | HAC t | HAC p | Validation corr | Validation OOS R2 | Exposed corr | Exposed OOS R2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.007 | +0.14 | 0.8884 | +0.016 | +0.02% | +0.128 | +0.17% |
| 5 | -0.075 | -1.63 | 0.1028 | -0.084 | +0.61% | +0.121 | -2.06% |
| 10 | -0.076 | -1.73 | 0.0844 | -0.028 | -0.07% | +0.148 | -1.94% |
| 20 | -0.025 | -0.59 | 0.5554 | -0.110 | +0.35% | +0.112 | -0.35% |
| 30 | +0.022 | +0.42 | 0.6758 | -0.034 | -0.11% | +0.068 | +0.13% |
| 40 | +0.039 | +0.60 | 0.5462 | -0.008 | -0.06% | +0.010 | -0.10% |
| 60 | +0.065 | +1.21 | 0.2244 | -0.042 | -0.73% | +0.018 | -0.24% |

### Feature-lag sensitivity at 30 days

| Extra lag | Train corr | HAC t | HAC p | Validation corr | Validation OOS R2 | Exposed corr | Exposed OOS R2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.022 | +0.42 | 0.6758 | -0.034 | -0.11% | +0.068 | +0.13% |
| 1 | +0.019 | +0.34 | 0.7375 | -0.046 | -0.12% | +0.033 | +0.03% |
| 2 | +0.027 | +0.44 | 0.6627 | -0.035 | -0.15% | +0.044 | +0.06% |
| 3 | +0.039 | +0.57 | 0.5714 | -0.031 | -0.22% | +0.038 | +0.06% |
| 5 | +0.070 | +1.07 | 0.2840 | +0.015 | -0.02% | +0.021 | -0.04% |

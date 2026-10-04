# Causal rolling ridge reliability gate

> **Source:** `analysis_outputs/ridge_rolling_reliability/report.md` in the team's working repository (relocated here for the public release).

Selection uses days 251-750 only. Days 751-1000 are reported
after the policy is frozen.

Selected policy: **current_production**.

## Development ranking

| variant | train_delta | validation_delta | train_a_delta | train_b_delta | validation_a_delta | validation_b_delta | positive_halves | worst_half_delta | passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roll125_sign_risky | -35.160 | -5.577 | -47.753 | -22.350 | 23.770 | -35.023 | 1 | -47.753 | False |
| roll250_sign_risky | -23.621 | -15.420 | -21.583 | -25.723 | 32.637 | -63.557 | 1 | -63.557 | False |
| roll125_250_sign_risky | -60.928 | 12.862 | -75.859 | -45.868 | 54.734 | -29.131 | 1 | -75.859 | False |
| roll125_active_risky | -53.727 | -11.922 | -85.834 | -21.536 | 32.441 | -56.385 | 1 | -85.834 | False |
| roll250_active_risky | -33.771 | 7.378 | -25.397 | -42.150 | 35.941 | -21.210 | 1 | -42.150 | False |
| roll125_250_active_risky | -85.456 | -5.096 | -105.902 | -64.984 | 41.500 | -51.781 | 1 | -105.902 | False |
| roll125_sign_all | -36.381 | -2.499 | -38.813 | -33.670 | 33.789 | -38.905 | 1 | -38.905 | False |
| roll250_sign_all | -21.206 | -32.930 | -17.419 | -25.141 | 15.231 | -81.185 | 1 | -81.185 | False |
| roll125_250_sign_all | -61.805 | 3.576 | -67.653 | -55.838 | 54.850 | -47.832 | 1 | -67.653 | False |

## Frozen comparison

| variant | window | score | mean_pnl | std_pnl | total_pnl | turnover |
| --- | --- | --- | --- | --- | --- | --- |
| current_production | TRAIN | 919.06 | 931.78 | 1733.19 | 232945.61 | 79249831.82 |
| current_production | VALIDATION | 1042.37 | 1053.64 | 1732.53 | 263410.83 | 73413899.52 |
| current_production | EXPOSED | 982.14 | 995.84 | 1859.29 | 248958.90 | 76319594.94 |

## Gate coverage

No gate passed the predeclared development rule, so no candidate gate coverage is promoted.

## Selected 50-day blocks

| window | score | mean_pnl | std_pnl | total_pnl |
| --- | --- | --- | --- | --- |
| BLOCK_251_300 | 752.67 | 767.27 | 1689.39 | 38363.26 |
| BLOCK_301_350 | 764.64 | 781.16 | 1815.76 | 39058.24 |
| BLOCK_351_400 | 996.08 | 1009.58 | 1858.23 | 50479.14 |
| BLOCK_401_450 | 707.93 | 717.81 | 1340.55 | 35890.49 |
| BLOCK_451_500 | 1372.68 | 1382.20 | 1820.03 | 69109.96 |
| BLOCK_501_550 | 840.93 | 854.57 | 1720.75 | 42728.55 |
| BLOCK_551_600 | 910.06 | 919.99 | 1519.51 | 45999.52 |
| BLOCK_601_650 | 1383.70 | 1392.16 | 1721.76 | 69608.10 |
| BLOCK_651_700 | 1202.27 | 1213.94 | 1891.16 | 60697.08 |
| BLOCK_701_750 | 873.57 | 886.78 | 1724.20 | 44339.04 |
| BLOCK_751_800 | 1551.28 | 1559.79 | 1827.00 | 77989.73 |
| BLOCK_801_850 | 1333.94 | 1343.26 | 1775.86 | 67163.07 |
| BLOCK_851_900 | 519.35 | 543.91 | 1870.29 | 27195.53 |
| BLOCK_901_950 | 535.50 | 553.49 | 1604.05 | 27674.45 |
| BLOCK_951_1000 | 962.25 | 977.89 | 1970.77 | 48894.28 |

Production should remain unchanged when current_production is selected.

# Current deadlock accounting

> **Source:** `analysis_outputs/current_deadlocks/report.md` in the team's working repository (relocated here for the public release).

A raw deadlock is an exact zero pair vote on an asset with active
pair contributors. Ridge fills and unresolved ties are separated.

## Occurrence

| window | days | days_with_raw | raw_day_rate | raw_events | raw_events_per_day | filled_events | filled_fraction | days_with_unresolved | unresolved_events | max_events_one_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN | 250 | 239 | 0.956 | 772 | 3.088 | 743 | 0.962 | 7 | 29 | 10 |
| VALIDATION | 250 | 239 | 0.956 | 728 | 2.912 | 683 | 0.938 | 19 | 45 | 8 |
| EXPOSED | 250 | 239 | 0.956 | 726 | 2.904 | 726 | 1.000 | 0 | 0 | 8 |

## Incremental effect of current ridge deadlock fills

| window | book | score | pnl | score_delta_from_ridge_fill | pnl_delta_from_ridge_fill | turnover_delta_from_ridge_fill | fee_delta_from_ridge_fill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN | production | 919.06 | 232945.61 | 51.34 | 12896.23 | 5220091.80 | 522.01 |
| TRAIN | production_core | 928.10 | 235246.52 | 59.85 | 15028.90 | 5379471.95 | 537.95 |
| VALIDATION | production | 1042.37 | 263410.83 | 37.89 | 9401.51 | 4641177.90 | 464.12 |
| VALIDATION | production_core | 1045.36 | 264212.30 | 40.46 | 10037.85 | 4876950.20 | 487.70 |
| EXPOSED | production | 982.14 | 248958.90 | 57.32 | 14372.53 | 4641364.43 | 464.14 |
| EXPOSED | production_core | 975.10 | 247314.51 | 57.07 | 14322.24 | 4863553.81 | 486.36 |

## Filled-event directional performance

| window | resolution | events | hit_rate | gross_pnl | mean_gross_pnl |
| --- | --- | --- | --- | --- | --- |
| TRAIN | active_ridge | 546 | 0.548 | 14648.759 | 26.829 |
| TRAIN | subthreshold_ridge | 197 | 0.518 | 969.132 | 4.919 |
| VALIDATION | active_ridge | 503 | 0.533 | 8897.101 | 17.688 |
| VALIDATION | subthreshold_ridge | 180 | 0.533 | 1648.768 | 9.160 |
| EXPOSED | active_ridge | 546 | 0.566 | 12125.557 | 22.208 |
| EXPOSED | subthreshold_ridge | 180 | 0.511 | 2725.650 | 15.143 |

## Winning and losing filled events

| window | events | winning_events | losing_events | positive_gross_pnl | negative_gross_pnl | net_gross_pnl |
| --- | --- | --- | --- | --- | --- | --- |
| TRAIN | 743 | 401 | 342 | 74856.54 | -59238.65 | 15617.89 |
| VALIDATION | 683 | 364 | 319 | 67576.68 | -57030.81 | 10545.87 |
| EXPOSED | 726 | 401 | 325 | 68696.31 | -53845.10 | 14851.21 |

## Exposed 50-day occurrence

| window | days | days_with_raw | raw_day_rate | raw_events | raw_events_per_day | filled_events | filled_fraction | days_with_unresolved | unresolved_events | max_events_one_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 751-800 | 50 | 50 | 1.000 | 146 | 2.920 | 146 | 1.000 | 0 | 0 | 6 |
| 801-850 | 50 | 45 | 0.900 | 110 | 2.200 | 110 | 1.000 | 0 | 0 | 5 |
| 851-900 | 50 | 49 | 0.980 | 154 | 3.080 | 154 | 1.000 | 0 | 0 | 7 |
| 901-950 | 50 | 50 | 1.000 | 164 | 3.280 | 164 | 1.000 | 0 | 0 | 6 |
| 951-1000 | 50 | 45 | 0.900 | 152 | 3.040 | 152 | 1.000 | 0 | 0 | 8 |

## Exposed 50-day full-book ridge-fill deltas

| window | book | score | pnl | score_delta_from_ridge_fill | pnl_delta_from_ridge_fill | turnover_delta_from_ridge_fill | fee_delta_from_ridge_fill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 751-800 | production | 1551.28 | 77989.73 | 128.02 | 6402.04 | 929392.67 | 92.94 |
| 801-850 | production | 1333.94 | 67163.07 | 31.79 | 1602.29 | 758810.95 | 75.88 |
| 851-900 | production | 519.35 | 27195.53 | 56.34 | 2745.74 | 850170.51 | 85.02 |
| 901-950 | production | 535.50 | 27674.45 | 54.86 | 2771.36 | 1092710.47 | 109.27 |
| 951-1000 | production | 962.25 | 48894.28 | 16.41 | 852.98 | 991547.73 | 99.15 |

Unresolved deadlocks have no causal direction and therefore no
realized PnL. Treating their next movement as a loss would be an
oracle calculation, not a tradable counterfactual.

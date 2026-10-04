# Current execution audit

> **Source:** `analysis_outputs/current_execution_audit/report.md` in the team's working repository (relocated here for the public release).

No production rule is changed. Days 751-1000 are descriptive only.

## Exact counterfactual accounting

| window | variant | score | total_pnl | turnover | fees | score_delta | pnl_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN | production | 919.06 | 232945.61 | 79249831.82 | 7924.98 | 0.00 | 0.00 |
| VALIDATION | production | 1042.37 | 263410.83 | 73413899.52 | 7341.39 | 0.00 | 0.00 |
| EXPOSED | production | 982.14 | 248958.90 | 76319594.94 | 7631.96 | 0.00 | 0.00 |
| TRAIN | core_without_algo_overlay | 928.10 | 235246.52 | 80471602.34 | 8047.16 | 9.03 | 2300.91 |
| VALIDATION | core_without_algo_overlay | 1045.36 | 264212.30 | 75079618.08 | 7507.96 | 2.99 | 801.47 |
| EXPOSED | core_without_algo_overlay | 975.10 | 247314.51 | 77611480.18 | 7761.15 | -7.05 | -1644.39 |
| TRAIN | no_subthreshold_deadlock_fill | 915.68 | 232095.88 | 78807537.37 | 7880.75 | -3.38 | -849.73 |
| VALIDATION | no_subthreshold_deadlock_fill | 1039.36 | 262679.54 | 72818421.82 | 7281.84 | -3.01 | -731.29 |
| EXPOSED | no_subthreshold_deadlock_fill | 970.37 | 246048.44 | 75982437.70 | 7598.24 | -11.77 | -2910.46 |
| TRAIN | fractional_share_upper_bound | 921.28 | 233509.54 | 79472129.45 | 7947.21 | 2.22 | 563.93 |
| VALIDATION | fractional_share_upper_bound | 1043.84 | 263786.32 | 73572033.63 | 7357.20 | 1.47 | 375.49 |
| EXPOSED | fractional_share_upper_bound | 983.69 | 249360.82 | 76527347.25 | 7652.73 | 1.55 | 401.92 |
| TRAIN | production_zero_commission | 951.18 | 240870.59 | 79249831.82 | 0.00 | 32.11 | 7924.98 |
| VALIDATION | production_zero_commission | 1072.03 | 270752.22 | 73413899.52 | 0.00 | 29.66 | 7341.39 |
| EXPOSED | production_zero_commission | 1013.07 | 256590.86 | 76319594.94 | 0.00 | 30.93 | 7631.96 |
| TRAIN | production_5bp_cost_sensitivity | 790.30 | 201245.67 | 79249831.82 | 39624.92 | -128.77 | -31699.93 |
| VALIDATION | production_5bp_cost_sensitivity | 923.56 | 234045.27 | 73413899.52 | 36706.95 | -118.81 | -29365.56 |
| EXPOSED | production_5bp_cost_sensitivity | 858.13 | 218431.06 | 76319594.94 | 38159.80 | -124.01 | -30527.84 |

The fractional-share result is an infeasible upper-bound diagnostic;
the evaluator requires integer shares.
The 5bp row is a cost sensitivity, not a production assumption.

## Daily execution state

| window | avg_gross_dollars | avg_net_dollars | avg_active_assets | avg_saturated_assets | avg_rounding_shortfall | ridge_confirmation_same_sign_rate | avg_algo_overlay_gross | avg_algo_delivery | avg_abs_algo_beta_error | conflicts | ridge_confirmed_overrides | raw_deadlocks | subthreshold_deadlock_fills | near_zero_full_positions_005 | near_zero_full_positions_010 | crash_assets | algo_overlay_sign_flips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN | 457423.523 | -2247.217 | 48.512 | 42.672 | 1245.974 | 1.000 | 13666.237 | 1.000 | 0.000 | 3091 | 911 | 772 | 197 | 180 | 385 | 0 | 0 |
| VALIDATION | 448768.938 | -1028.275 | 48.652 | 41.912 | 1045.792 | 1.000 | 15834.512 | 1.000 | 0.000 | 2829 | 829 | 728 | 180 | 193 | 363 | 0 | 0 |
| EXPOSED | 457668.500 | -4121.780 | 48.824 | 43.340 | 1210.643 | 1.000 | 13627.758 | 1.000 | 0.000 | 3228 | 968 | 726 | 180 | 233 | 445 | 0 | 0 |

## Pair hedge integrity after asset-level arbitration

| window | active_pair_days | both_legs_match | both_legs_match_rate | one_or_both_legs_flat | one_or_both_legs_flat_rate | mixed_direction | mixed_direction_rate | both_flipped | both_flipped_rate | median_hedge_ratio_error | one_leg_crash_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN | 10170 | 6194 | 0.609 | 56 | 0.006 | 3515 | 0.346 | 405 | 0.040 | 0.140 | 0 |
| VALIDATION | 9920 | 6013 | 0.606 | 110 | 0.011 | 3428 | 0.346 | 369 | 0.037 | 0.162 | 0 |
| EXPOSED | 10015 | 5931 | 0.592 | 0 | 0.000 | 3639 | 0.363 | 445 | 0.044 | 0.160 | 0 |

## Turnover decomposition

| window | turnover | entry_turnover | exit_turnover | flip_turnover | resize_turnover | flip_turnover_fraction | entries | exits | full_sign_flips | same_sign_resizes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN | 79249831.82 | 3130935.01 | 2828011.72 | 68272112.89 | 5018772.20 | 0.86 | 363 | 313 | 3688 | 7346 |
| VALIDATION | 73413899.52 | 2823882.63 | 2352063.55 | 62223258.14 | 6014695.20 | 0.85 | 301 | 257 | 3448 | 7696 |
| EXPOSED | 76319594.94 | 2681615.57 | 2259446.40 | 65790912.87 | 5587620.10 | 0.86 | 297 | 247 | 3589 | 7567 |

A pair can be statistically valid at proposal time while ceasing to
be a hedged pair after its two legs are arbitrated independently.

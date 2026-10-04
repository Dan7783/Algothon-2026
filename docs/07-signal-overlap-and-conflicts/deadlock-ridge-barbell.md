# Ridge barbell usefulness on pair-vote deadlocks

> **Source:** `analysis_outputs/deadlock_ridge_barbell/report.md` in the team's working repository (relocated here for the public release).

## What the ridge score means

`raw_strength = |forecast| / std(in-sample fitted forecasts)`. It is a normalized forecast magnitude, not a calibrated probability. The live entry threshold is 0.75. The global gate also requires at least 35 significant lagged cross-asset links in the most recent 250 returns.

A zero live ridge strength is therefore ambiguous when the global gate is closed. This audit reconstructs the latent score so those days can be tested rather than silently grouped with no-forecast days.

## Population

- Pre-ridge exact pair cancellations: 2226
- Already resolved by the live ridge: 988
- Still flat (true deadlocks): 1238

## Direction skill by deadlock state

| window | state | n | accuracy | mean_strength | mean_gross_pnl | total_gross_pnl |
| --- | --- | --- | --- | --- | --- | --- |
| contaminated_751_1000 | current_ridge_resolved | 345 | 0.5449 | 1.3153 | 20.5005 | 7072.6699 |
| contaminated_751_1000 | sub_threshold | 378 | 0.5476 | 0.3497 | 13.2836 | 5021.1915 |
| train_251_500 | current_ridge_resolved | 328 | 0.5579 | 1.3404 | 30.6614 | 10056.9469 |
| train_251_500 | gate_closed | 22 | 0.2273 | 0.4387 | -55.2983 | -1216.5625 |
| train_251_500 | sub_threshold | 419 | 0.5155 | 0.3461 | 7.0125 | 2938.2286 |
| validation_501_750 | current_ridge_resolved | 315 | 0.5619 | 1.2827 | 23.4750 | 7394.6299 |
| validation_501_750 | gate_closed | 45 | 0.4444 | 0.8063 | -42.0627 | -1892.8225 |
| validation_501_750 | sub_threshold | 367 | 0.5259 | 0.3384 | 12.0805 | 4433.5435 |

## Direction skill by fixed strength bucket

| window | state | strength_bucket | n | accuracy | mean_gross_pnl | total_gross_pnl | mean_abs_move | mean_links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contaminated_751_1000 | sub_threshold | [0,.25) | 154 | 0.4740 | -2.7131 | -417.8145 | 0.0170 | 58.2987 |
| contaminated_751_1000 | sub_threshold | [.25,.50) | 115 | 0.6174 | 33.7971 | 3886.6709 | 0.0154 | 58.7391 |
| contaminated_751_1000 | sub_threshold | [.50,.75) | 109 | 0.5780 | 14.2416 | 1552.3351 | 0.0184 | 59.1651 |
| train_251_500 | gate_closed | [0,.25) | 9 | 0.2222 | 0.6816 | 6.1342 | 0.0111 | 14.2222 |
| train_251_500 | gate_closed | [.25,.50) | 2 | 0.0000 | -130.7895 | -261.5789 | 0.0131 | 34.0000 |
| train_251_500 | gate_closed | [.50,.75) | 8 | 0.2500 | -93.8122 | -750.4972 | 0.0148 | 34.0000 |
| train_251_500 | gate_closed | [.75,1) | 2 | 0.5000 | -31.1970 | -62.3940 | 0.0239 | 33.5000 |
| train_251_500 | gate_closed | [1,1.5) | 1 | 0.0000 | -148.2266 | -148.2266 | 0.0148 | 33.0000 |
| train_251_500 | sub_threshold | [0,.25) | 155 | 0.5097 | -0.4840 | -75.0185 | 0.0179 | 49.6258 |
| train_251_500 | sub_threshold | [.25,.50) | 143 | 0.4476 | -29.5370 | -4223.7944 | 0.0173 | 50.0699 |
| train_251_500 | sub_threshold | [.50,.75) | 121 | 0.6033 | 59.8103 | 7237.0416 | 0.0180 | 48.8926 |
| validation_501_750 | gate_closed | [0,.25) | 7 | 0.4286 | -98.4175 | -688.9226 | 0.0153 | 32.7143 |
| validation_501_750 | gate_closed | [.25,.50) | 11 | 0.2727 | -106.1886 | -1168.0746 | 0.0142 | 31.2727 |
| validation_501_750 | gate_closed | [.50,.75) | 7 | 0.7143 | 144.4392 | 1011.0741 | 0.0222 | 32.2857 |
| validation_501_750 | gate_closed | [.75,1) | 5 | 0.2000 | -149.6204 | -748.1022 | 0.0154 | 31.6000 |
| validation_501_750 | gate_closed | [1,1.5) | 9 | 0.7778 | 63.4752 | 571.2772 | 0.0160 | 31.0000 |
| validation_501_750 | gate_closed | [1.5,2) | 4 | 0.2500 | -156.6173 | -626.4690 | 0.0189 | 31.5000 |
| validation_501_750 | gate_closed | [2,3) | 2 | 0.0000 | -121.8027 | -243.6053 | 0.0122 | 32.5000 |
| validation_501_750 | sub_threshold | [0,.25) | 144 | 0.5486 | 7.3601 | 1059.8569 | 0.0180 | 48.8819 |
| validation_501_750 | sub_threshold | [.25,.50) | 118 | 0.5593 | 33.6357 | 3969.0133 | 0.0180 | 49.4576 |
| validation_501_750 | sub_threshold | [.50,.75) | 105 | 0.4571 | -5.6698 | -595.3268 | 0.0183 | 48.4190 |

## Train-only rule selection

Candidates were ranked by their worst score delta across days 251-375 and 376-500, and had to be positive in both. Days 501-750 and 751-1000 were not used for selection.

Train-locked candidate: `sub_threshold_full_50plus`

| candidate | contaminated_751_1000 | contaminated_a_751_875 | contaminated_b_876_1000 | train_251_500 | train_a_251_375 | train_b_376_500 | validation_501_750 | validation_a_501_625 | validation_b_626_750 | train_floor | train_average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sub_threshold_full_50plus | 5.9113 | 5.7004 | 6.0198 | 28.8852 | 26.5687 | 31.2505 | -2.3867 | -4.8586 | 0.0710 | 26.5687 | 28.9096 |
| sub_threshold_confidence_full | 13.5741 | 15.9212 | 11.2015 | 16.3143 | 13.7410 | 18.9302 | 6.3696 | 11.5935 | 1.1464 | 13.7410 | 16.3356 |
| confidence_full | 13.5741 | 15.9212 | 11.2015 | 12.5043 | 10.2316 | 14.8198 | 2.9955 | 4.4407 | 1.5564 | 10.2316 | 12.5257 |
| sub_threshold_confidence_25pct | 3.4273 | 4.0159 | 2.8343 | 4.0740 | 3.4618 | 4.6969 | 1.6087 | 2.9313 | 0.2863 | 3.4618 | 4.0793 |
| barbell_conf25_tail_1.5 | 3.4273 | 4.0159 | 2.8343 | 3.1294 | 2.5754 | 3.6940 | -1.8683 | -4.1199 | 0.3906 | 2.5754 | 3.1347 |
| barbell_conf25_tail_2 | 3.4273 | 4.0159 | 2.8343 | 3.1294 | 2.5754 | 3.6940 | 0.0286 | -0.3255 | 0.3906 | 2.5754 | 3.1347 |
| barbell_conf25_tail_3 | 3.4273 | 4.0159 | 2.8343 | 3.1294 | 2.5754 | 3.6940 | 0.7684 | 1.1484 | 0.3906 | 2.5754 | 3.1347 |
| confidence_small_25pct | 3.4273 | 4.0159 | 2.8343 | 3.1294 | 2.5754 | 3.6940 | 0.7684 | 1.1484 | 0.3906 | 2.5754 | 3.1347 |
| sub_threshold_small_25pct | 5.3268 | 5.5274 | 5.1346 | 2.8825 | 2.2499 | 3.5219 | 3.2664 | 4.7087 | 1.8284 | 2.2499 | 2.8859 |
| sub_threshold_all_full | 19.8304 | 29.8299 | 9.8461 | 10.8784 | 1.9113 | 19.9175 | 16.8738 | 27.4961 | 6.2591 | 1.9113 | 10.9144 |
| barbell_conf25_tail_1 | 3.4273 | 4.0159 | 2.8343 | 2.6787 | 1.6780 | 3.6940 | -0.2255 | -1.4654 | 1.0113 | 1.6780 | 2.6860 |
| barbell_conf25_tail_0.75 | 3.4273 | 4.0159 | 2.8343 | 2.4775 | 1.2777 | 3.6940 | -2.4836 | -5.9796 | 1.0113 | 1.2777 | 2.4858 |
| sub_threshold_all_25pct | 5.0260 | 7.5497 | 2.5163 | 2.7888 | 0.5083 | 5.0860 | 4.2749 | 6.9136 | 1.6393 | 0.5083 | 2.7971 |
| gate_closed_tail_1.5_full | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -3.5164 | -7.0258 | 0.0000 | 0.0000 | 0.0000 |
| gate_closed_tail_2_full | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.9859 | -1.9643 | 0.0000 | 0.0000 | 0.0000 |
| gate_closed_tail_3_full | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stay_flat | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| all_small_10pct | 2.0579 | 3.0477 | 1.0753 | 0.6175 | -0.2239 | 1.4644 | 0.9491 | 1.3005 | 0.6001 | -0.2239 | 0.6202 |
| barbell_10pct_tail_1.5 | 2.0579 | 3.0477 | 1.0753 | 0.6175 | -0.2239 | 1.4644 | -2.2185 | -5.0283 | 0.6001 | -0.2239 | 0.6202 |
| barbell_10pct_tail_2 | 2.0579 | 3.0477 | 1.0753 | 0.6175 | -0.2239 | 1.4644 | 0.0606 | -0.4697 | 0.6001 | -0.2239 | 0.6202 |
| barbell_10pct_tail_3 | 2.0579 | 3.0477 | 1.0753 | 0.6175 | -0.2239 | 1.4644 | 0.9491 | 1.3005 | 0.6001 | -0.2239 | 0.6202 |
| all_small_25pct | 5.0260 | 7.5497 | 2.5163 | 1.5503 | -0.5315 | 3.6463 | 2.3421 | 3.1132 | 1.5768 | -0.5315 | 1.5574 |
| all_half | 10.0196 | 14.9595 | 5.1022 | 3.1223 | -0.9293 | 7.2038 | 4.6856 | 6.3139 | 3.0671 | -0.9293 | 3.1372 |
| gate_closed_tail_1_full | 0.0000 | 0.0000 | 0.0000 | -0.6003 | -1.1949 | 0.0000 | -1.3235 | -3.4826 | 0.8288 | -1.1949 | -0.5975 |
| barbell_10pct_tail_1 | 2.0579 | 3.0477 | 1.0753 | 0.0773 | -1.2994 | 1.4644 | -0.2442 | -1.8389 | 1.3468 | -1.2994 | 0.0825 |
| gate_closed_tail_0.75_full | 0.0000 | 0.0000 | 0.0000 | -0.8752 | -1.7422 | 0.0000 | -4.3345 | -9.5020 | 0.8288 | -1.7422 | -0.8711 |
| barbell_10pct_tail_0.75 | 2.0579 | 3.0477 | 1.0753 | -0.1681 | -1.7878 | 1.4644 | -2.9574 | -7.2630 | 1.3468 | -1.7878 | -0.1617 |
| all_full | 19.8304 | 29.8299 | 9.8461 | 5.9429 | -2.2038 | 14.1523 | 9.1550 | 12.3200 | 6.0039 | -2.2038 | 5.9742 |

## Validation selection

Among rules positive in both training halves, the validated rule must also be positive in both 125-day validation halves. It is selected by validation floor; the final window is not consulted.

Validated candidate: `sub_threshold_all_full`

## Selected candidates versus staying flat

| candidate | window | score | score_delta | mean_pnl | active_asset_days |
| --- | --- | --- | --- | --- | --- |
| stay_flat | train_a_251_375 | 768.8619 | 0.0000 | 783.3401 | 0 |
| stay_flat | train_b_376_500 | 1028.3901 | 0.0000 | 1037.6953 | 0 |
| stay_flat | train_251_500 | 898.9753 | 0.0000 | 910.5912 | 0 |
| stay_flat | validation_a_501_625 | 916.7463 | 0.0000 | 927.2345 | 0 |
| stay_flat | validation_b_626_750 | 1075.4494 | 0.0000 | 1086.3460 | 0 |
| stay_flat | validation_501_750 | 996.1372 | 0.0000 | 1006.8689 | 0 |
| stay_flat | contaminated_a_751_875 | 1269.9352 | 0.0000 | 1279.6285 | 0 |
| stay_flat | contaminated_b_876_1000 | 631.6376 | 0.0000 | 649.2604 | 0 |
| stay_flat | contaminated_751_1000 | 951.6671 | 0.0000 | 964.4826 | 0 |
| sub_threshold_all_full | train_a_251_375 | 770.7732 | 1.9113 | 784.9020 | 194 |
| sub_threshold_all_full | train_b_376_500 | 1048.3076 | 19.9175 | 1057.9530 | 223 |
| sub_threshold_all_full | train_251_500 | 909.8537 | 10.8784 | 921.5010 | 417 |
| sub_threshold_all_full | validation_a_501_625 | 944.2424 | 27.4961 | 954.2180 | 169 |
| sub_threshold_all_full | validation_b_626_750 | 1081.7085 | 6.2591 | 1092.4651 | 199 |
| sub_threshold_all_full | validation_501_750 | 1013.0110 | 16.8738 | 1023.4202 | 368 |
| sub_threshold_all_full | contaminated_a_751_875 | 1299.7651 | 29.8299 | 1309.5024 | 161 |
| sub_threshold_all_full | contaminated_b_876_1000 | 641.4838 | 9.8461 | 659.0415 | 218 |
| sub_threshold_all_full | contaminated_751_1000 | 971.4975 | 19.8304 | 984.3220 | 379 |
| sub_threshold_full_50plus | train_a_251_375 | 795.4306 | 26.5687 | 809.3656 | 49 |
| sub_threshold_full_50plus | train_b_376_500 | 1059.6406 | 31.2505 | 1068.7208 | 72 |
| sub_threshold_full_50plus | train_251_500 | 927.8605 | 28.8852 | 939.1167 | 121 |
| sub_threshold_full_50plus | validation_a_501_625 | 911.8877 | -4.8586 | 922.3438 | 48 |
| sub_threshold_full_50plus | validation_b_626_750 | 1075.5204 | 0.0710 | 1086.1550 | 57 |
| sub_threshold_full_50plus | validation_501_750 | 993.7506 | -2.3867 | 1004.3281 | 105 |
| sub_threshold_full_50plus | contaminated_a_751_875 | 1275.6356 | 5.7004 | 1285.1795 | 49 |
| sub_threshold_full_50plus | contaminated_b_876_1000 | 637.6574 | 6.0198 | 655.5003 | 60 |
| sub_threshold_full_50plus | contaminated_751_1000 | 957.5784 | 5.9113 | 970.3781 | 109 |

## Direction controls for the validated candidate

The same entry days and sizes are replayed with the ridge sign inverted and with 64 random directions. A useful ridge forecast should beat both controls.

| candidate | window | observed_delta | inverse_delta | random_mean_delta | random_std_delta | random_95pct | randomization_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sub_threshold_all_full | train_251_500 | 10.8784 | -12.7035 | -3.7113 | 20.6736 | 30.0048 | 0.2154 |
| sub_threshold_all_full | validation_501_750 | 16.8738 | -18.4347 | -0.0218 | 17.7234 | 24.6977 | 0.1385 |
| sub_threshold_all_full | contaminated_751_1000 | 19.8304 | -21.4836 | -1.3849 | 18.4814 | 27.5288 | 0.1385 |

## Concentration of the gate-respecting full fill

These are gross ridge-leg contributions on sub-threshold deadlocks. They show whether the effect is spread across symbols and event days rather than generated by one relationship.

| window | events | symbols | profitable_symbols | positive_event_days | event_days | top_symbol_abs_share | top5_symbol_abs_share | best_symbol | best_symbol_gross_pnl | worst_symbol | worst_symbol_gross_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_251_500 | 419 | 38 | 17 | 101 | 194 | 0.1187 | 0.3726 | ANSO | 2408.3774 | RRES | -1748.6649 |
| validation_501_750 | 367 | 32 | 19 | 101 | 188 | 0.0873 | 0.3791 | AETS | 1630.2688 | MTNS | -1078.9291 |
| contaminated_751_1000 | 378 | 32 | 18 | 101 | 191 | 0.1387 | 0.5054 | CCNS | 2478.1030 | RTTH | -2115.2121 |

Days 751-1000 are labelled contaminated because this repository has already used them repeatedly. Their result is descriptive, not a fresh out-of-sample claim.

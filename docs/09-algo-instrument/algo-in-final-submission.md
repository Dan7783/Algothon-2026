# ALGO attribution inside random_strat1

> **Source:** `analysis_outputs/random_strat1_algo/report.md` in the team's working repository (relocated here for the public release).

The position path is produced by the unchanged 51-instrument model. ALGO is then isolated at execution; the model is not refitted.

| Window | ALGO PnL | Always-long PnL | Timing PnL | ALGO score | ALGO Sharpe | Full score | Score without ALGO | Full - rest | ALGO/rest corr | Turnover | Fees |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| active_year1_151_250 | \$-1,509.37 | \$6,544.21 | \$-8,053.58 | -15.094 | -0.230 | 335.804 | 373.723 | -37.920 | +0.426 | \$10,904,564 | \$218.09 |
| year2_251_500 | \$-13,006.11 | \$-18,342.18 | \$+5,336.07 | -52.024 | -0.809 | 373.434 | 443.873 | -70.440 | +0.270 | \$27,700,523 | \$554.01 |
| year3_validation_501_750 | \$14,631.41 | \$6,696.13 | \$+7,935.28 | 24.561 | 0.850 | 766.303 | 718.957 | +47.347 | +0.363 | \$28,324,790 | \$566.50 |
| year4_exposed_751_1000 | \$32,354.98 | \$25,941.65 | \$+6,413.33 | 100.357 | 1.858 | 924.181 | 801.836 | +122.344 | +0.261 | \$23,963,258 | \$479.27 |
| active_full_151_1000 | \$32,466.28 | \$20,845.77 | \$+11,620.52 | 9.256 | 0.566 | 649.141 | 622.731 | +26.410 | +0.318 | \$90,992,299 | \$1,819.85 |
| long_history_701_1000 | \$39,612.94 | \$29,080.25 | \$+10,532.69 | 103.245 | 1.893 | 1005.264 | 879.767 | +125.497 | +0.250 | \$30,367,915 | \$607.36 |
| long_history_801_1000 | \$22,729.74 | \$26,758.96 | \$-4,029.23 | 83.011 | 1.646 | 865.586 | 759.100 | +106.486 | +0.239 | \$19,362,144 | \$387.24 |

## ALGO execution diagnostics

| Window | Mean abs exposure | Long | Short | Flat | Direction changes | Next-day accuracy | Mean signed return | ALGO buy/hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| active_year1_151_250 | \$99,705 | 58.0% | 42.0% | 0.0% | 54 | 45.0% | -1.311 bp | +6.2% |
| year2_251_500 | \$99,757 | 56.0% | 44.0% | 0.0% | 138 | 48.8% | -5.048 bp | -17.9% |
| year3_validation_501_750 | \$99,770 | 49.6% | 50.4% | 0.0% | 141 | 54.4% | +6.031 bp | +5.3% |
| year4_exposed_751_1000 | \$99,768 | 50.4% | 49.6% | 0.0% | 119 | 55.6% | +13.150 bp | +27.7% |
| active_full_151_1000 | \$99,755 | 52.7% | 47.3% | 0.0% | 454 | 52.0% | +4.003 bp | +17.3% |
| long_history_701_1000 | \$99,768 | 51.3% | 48.7% | 0.0% | 151 | 56.3% | +13.416 bp | +31.3% |
| long_history_801_1000 | \$99,781 | 51.5% | 48.5% | 0.0% | 96 | 54.5% | +11.599 bp | +29.1% |

Because the competition score depends on the mean and volatility of the combined daily PnL, ALGO's standalone score does not add arithmetically to the rest-of-book score. `Full - rest` is the relevant realised portfolio score contribution.

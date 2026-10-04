# Dedicated ALGO horizon audit

> **Source:** `analysis_outputs/algo_dedicated_horizon/report.md` in the team's working repository (relocated here for the public release).

## Protocol

- Candidate selection physically read only days 1-750.
- Development check: refit through day 250, evaluate days 251-500.
- Validation selection: refit through day 500, evaluate days 501-750.
- Final evaluation: refit through day 750, evaluate days 751-1000 once.
- ALGO commission is 0.2 bp one-way and its direct cap is \$100,000.
- No production strategy code was changed.

## Frozen selection

- Specification: `market_compact_h3_l0.3_algo_round_trip`
- Selection rule: `positive_all_four_halves_then_max_floor`
- Eligible candidates: 6 / 210
- Forecast horizon: 3 day(s)
- Feature family: `market_compact`
- Ridge lambda: 0.3
- Cost hurdle: `algo_round_trip`

## Selected development evidence

| Window | Score | Total PnL | Sharpe | Forecast/target correlation |
| --- | ---: | ---: | ---: | ---: |
| development_251_500 | 11.6953 | \$10,203.56 | 0.6337 | 0.0423 |
| development_a_251_375 | 16.6800 | \$6,043.78 | 0.7257 | 0.2132 |
| development_b_376_500 | 7.4161 | \$4,157.79 | 0.5357 | -0.0492 |
| validation_501_750 | 90.0950 | \$29,925.85 | 1.7444 | 0.0937 |
| validation_a_501_625 | 142.5274 | \$20,488.96 | 2.5817 | 0.0832 |
| validation_b_626_750 | 38.8505 | \$9,434.90 | 1.0299 | 0.0968 |

## Exposed standalone result

| Variant | Window | Score | Total PnL | Sharpe | Turnover | Fees |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| algo_100k_0.2bp | exposed_751_1000 | -45.2532 | \$-11,313.31 | -0.6514 | \$10,873,361 | \$217.47 |
| algo_10k_0.2bp | exposed_751_1000 | -4.5750 | \$-1,143.75 | -0.6581 | \$1,088,602 | \$21.77 |
| algo_100k_1.0bp | exposed_751_1000 | -48.7327 | \$-12,183.18 | -0.7012 | \$10,873,361 | \$1,087.34 |

## Current book interaction

| Window | Base score | Combined score | Score delta | Base PnL | Combined PnL | PnL delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| development_251_500 | 959.3371 | 997.7827 | +38.4456 | \$242,786.14 | \$252,989.70 | \$+10,203.56 |
| validation_501_750 | 1057.6280 | 1175.2245 | +117.5966 | \$267,106.96 | \$297,032.81 | \$+29,925.85 |
| exposed_751_1000 | 986.3099 | 936.1006 | -50.2093 | \$249,724.35 | \$238,411.04 | \$-11,313.31 |

## Exposed forecast diagnostic

- Forward-return correlation: -0.1722
- Forward-return sign accuracy: 0.5081
- Mean forecast: -0.000787
- Forecast standard deviation: 0.005683

## Is the result genuinely fee-dependent?

- Validation PnL at 0.2 bp: \$29,925.85.
- Validation PnL at 1.0 bp using the identical positions: \$28,896.64.
- Validation fees rise from \$257.30 to \$1,286.51.
- Repeating the complete development-only selection at 1.0 bp still chooses `market_compact_h3_l0.3_algo_round_trip` (horizon 3).
- Development-eligible candidates remaining profitable in the exposed window: 0 / 6.
- Standardized-coefficient cosine similarity between fits through days 500 and 750: 0.6291.

The complete exposed-candidate table is diagnostic only; it was not used to replace the frozen selection.

## Largest standardized model coefficients

| Fit boundary | Feature | Coefficient |
| --- | --- | ---: |
| 500 | algo_vol20 | +0.10886 |
| 500 | algo_vol10 | -0.09270 |
| 500 | algo_relative_r10 | -0.07886 |
| 500 | market_r20 | -0.05646 |
| 500 | algo_r20 | -0.05122 |
| 750 | algo_vol20 | +0.16103 |
| 750 | algo_relative_r20 | -0.08864 |
| 750 | algo_vol10 | -0.07451 |
| 750 | algo_relative_r5 | -0.04463 |
| 750 | algo_r20 | -0.03334 |

The exposed result is descriptive because days 751-1000 have already been seen in this research process.

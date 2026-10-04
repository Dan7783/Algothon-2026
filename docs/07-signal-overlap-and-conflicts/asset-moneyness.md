# Per-asset next-move moneyness

> **Source:** `analysis_outputs/asset_moneyness/README.md` in the team's working repository (relocated here for the public release).

This is the literal per-asset interpretation:

1. For each active pair, calculate remaining spread displacement.
2. Divide it by dynamic OU half-life to obtain a next-day spread forecast.
3. Decompose `r_y - beta*r_x` into individual y/x return forecasts using the
   minimum-norm solution.
4. Average all pair forecasts for each asset so relationship count does not
   multiply the forecast.
5. Add that asset's raw ridge next-return forecast.
6. Use the sign for its final bang-bang position.

Known costs reduce each forecast's magnitude. No new fitted parameter,
probability, family weight or selection grid is used.

## Results

| Window | Current | Per-asset moneyness | Change |
|---|---:|---:|---:|
| Train 251–375 | 764.93 | 779.65 | +14.72 |
| Train 376–500 | 1033.36 | 913.51 | −119.84 |
| Validation 501–625 | 853.57 | 1012.17 | +158.60 |
| Validation 626–750 | 1049.43 | 917.03 | −132.39 |
| OOS 751–875 | 1190.27 | 951.11 | −239.16 |
| OOS 876–1000 | 580.61 | 543.41 | −37.20 |
| Year 2 | 899.42 | 846.70 | −52.72 |
| Year 3 | 951.52 | 964.74 | +13.22 |
| Year 4 OOS | 886.39 | 747.96 | **−138.43** |

Average absolute cross-sectional forecasts were 0.280 for the pair family
and 0.069 for ridge, a ratio of approximately 4.1. Each pair-traded asset had
2.02 active pair estimates on average before averaging.

The failure is not caused only by removing ridge or counting the same asset
many times. The deterministic forecast changes sign decisions beneficially
in the first validation half and harmfully in the next three fixed halves.
Dynamic OU half-life and current displacement do not reliably measure the
next realised asset move in this dataset.

The production `four_sigmas.py` remains unchanged. Days 751–1000 are
retrospective because they had already been exposed before this experiment.

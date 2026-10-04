# One-day moneyness conflict engine

> **Source:** `analysis_outputs/daily_moneyness_combiner/README.md` in the team's working repository (relocated here for the public release).

This implements the clarified deterministic rule:

```text
pair predicted next move =
    remaining spread displacement / max(dynamic OU half-life, 1 day)

ridge predicted next move =
    raw de-standardized ridge next-return forecast
```

Each prediction is converted to expected next-move dollars after known
rebalance costs and an exit reserve. Pair signals retain both legs and their
beta direction. All signed asset-level dollar opportunities are summed, then
converted to final bang-bang positions.

There are no new fitted probabilities, calibration coefficients, family
multipliers or validation-selected settings.

## Results

| Window | Current method | Daily moneyness | Change |
|---|---:|---:|---:|
| Train 251–375 | 764.93 | 832.53 | +67.60 |
| Train 376–500 | 1033.36 | 856.11 | −177.25 |
| Validation 501–625 | 853.57 | 1069.23 | +215.66 |
| Validation 626–750 | 1049.43 | 879.11 | −170.32 |
| OOS 751–875 | 1190.27 | 896.05 | −294.22 |
| OOS 876–1000 | 580.61 | 546.90 | −33.70 |
| Year 2 | 899.42 | 844.39 | −55.03 |
| Year 3 | 951.52 | 974.38 | +22.86 |
| Year 4 OOS | 886.39 | 721.90 | **−164.50** |

Average immediate opportunity across the complete sample was:

- Pair one-day money: **\$6,537**
- Ridge one-day money: **\$690**
- Pair/ridge ratio: **9.48**

Ridge is not removed. It loses most conflicts because the sum of all active
pair forecasts is usually substantially larger. Introducing a multiplier to
balance these families would add the same arbitrary conflict parameter this
experiment was intended to avoid.

An additional identification limitation remains: a pair predicts the spread
move `r_y - beta*r_x`, not the two individual asset moves uniquely. The
implementation distributes the predicted pair opportunity along its complete
hedge vector. Any alternative allocation between the two legs would be
another modelling assumption.

The current `four_sigmas.py` remains unchanged. OOS results here are
retrospective because days 751–1000 had already been exposed by earlier work.

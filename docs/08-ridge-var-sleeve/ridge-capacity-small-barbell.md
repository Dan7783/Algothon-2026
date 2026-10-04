# Ridge capacity and small-pair barbell

> **Source:** `analysis_outputs/ridge_smallbarbell_report.md` in the team's working repository (relocated here for the public release).

All figures below use days 1-750 only. Days 751-1000 were not evaluated or
used to select the strategy.

## Why ridge previously scored around 250 independently

The ridge sleeve is a one-step cross-sectional VAR forecast over the 50
tradable instruments. It:

1. waits for at least 252 price observations;
2. requires at least 35 statistically large lag-one cross-asset links;
3. fits a regularised 50-by-50 return forecast;
4. enters an instrument when forecast magnitude exceeds 1.0 times that
   instrument's fitted prediction scale.

Replaying the exact current ridge signal as a standalone sleeve reproduced:

| Sleeve | Days 251-500 | Days 501-750 |
|---|---:|---:|
| Ridge only | 226.30 | 320.78 |

Its edge disappeared inside the 310-pair implementation because capacity was
allocated before ridge. On day 750 the full tuned pair selector occupied 36
legs; subordinated ridge reached only 2 assets and about \$4.6k gross exposure,
despite having 10 valid raw forecasts representing \$100k gross exposure.

Giving every marginal ridge forecast first priority was also inefficient. A
1-sigma ridge forecast then removed an entire two-leg pair, increasing
turnover and discarding useful pair diversification.

## Implemented confidence barbell

`testbedv2_tuned_ridge_smallbarbell.py` uses six fixed, disjoint pair agents
and two ridge thresholds:

- Ridge entry: 1.0 prediction-scale units.
- Pair override: 3.0 prediction-scale units.

Between 1.0 and 3.0, ridge fills idle instruments or agreeing residual
capacity. Above 3.0, ridge receives a full \$10,000 target and any pair touching
that instrument is removed as a complete two-leg trade.

## Train-only arbitration comparison

| Allocation | Days 251-500 | Days 501-750 | Floor |
|---|---:|---:|---:|
| Ridge only | 226.30 | 320.78 | 226.30 |
| Six tuned pairs only | 261.58 | 243.60 | 243.60 |
| Override every ridge entry at 1.0 | 362.46 | 473.17 | 362.46 |
| Override above 1.5 | 425.60 | 537.79 | 425.60 |
| Override above 2.0 | 432.63 | 532.60 | 432.63 |
| Override above 2.5 | 462.53 | 553.20 | 462.53 |
| **Override above 3.0** | **478.99** | **557.85** | **478.99** |

The combined days 251-750 result is score 518.50, mean daily P&L \$529.90,
annualised Sharpe 6.75, and approximately \$114.0m turnover.

## Full-universe protected-pair hierarchy

`testbedv2_tuned_ridge_protectedbarbell.py` keeps all 310 tuned pair agents
but changes their allocation priority:

1. The six selected pairs receive first claim.
2. Ridge receives second claim and therefore overrides every ordinary pair.
3. Ordinary pair candidates may fill only instruments left idle by both.

This preserves ridge's small-pair capacity without discarding the remaining
pair universe entirely.

| Window | Score | Mean daily P&L |
|---|---:|---:|
| Days 251-500 | 805.96 | \$817.50 |
| Days 501-750 | 801.73 | \$811.60 |
| Days 251-750 | 803.86 | \$814.60 |

On day 750 the hierarchy allocated 6 protected pairs, 7 ridge assets
(approximately \$60.3k gross), and 11 ordinary residual pairs. These results
remain development-only; days 751-1000 were not accessed.

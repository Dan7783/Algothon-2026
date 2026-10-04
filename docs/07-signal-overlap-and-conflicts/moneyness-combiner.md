# Parameter-light moneyness combiner

> **Source:** `analysis_outputs/moneyness_combiner/README.md` in the team's working repository (relocated here for the public release).

This experiment uses no fitted probability, expected-PnL calibration,
identity effects, forecast variance, shrinkage setting, or validation grid.
The formula was fixed before any performance result was calculated.

## Definition

For an active pair:

```text
gross money =
    |Y dollars|
    × observed spread scale
    × max(|z| − z_exit, 0)
```

For an active ridge forecast:

```text
gross money = |ridge dollars × raw forecast return|
```

Known rebalance costs and a one-way exit reserve are deducted. Positive net
opportunity is distributed across the signal's complete position vector.
Vectors are summed and converted into final bang-bang directions.

## Results

| Window | Existing method | Moneyness | Change |
|---|---:|---:|---:|
| Train 251–375 | 764.93 | 892.68 | +127.75 |
| Train 376–500 | 1033.36 | 988.84 | −44.52 |
| Validation 501–625 | 853.57 | 938.13 | +84.56 |
| Validation 626–750 | 1049.43 | 740.61 | −308.82 |
| Year 2 | 899.42 | 940.87 | +41.45 |
| Year 3 | 951.52 | 839.78 | −111.74 |
| Year 4 OOS | 886.39 | 735.18 | **−151.21** |

## Why it fails

The quantities are expressed in dollars but are not economically comparable:

- Pair moneyness is the full displacement available over an unknown,
  potentially multi-day convergence horizon.
- Ridge moneyness is the forecast next-day price displacement.

Mean aggregate opportunity was:

- Pair relationships: **\$32,497/day**
- Ridge forecasts: **\$690/day**

Consequently the method behaves approximately like a pair-dominant conflict
rule. It removes ridge precisely when ridge became valuable in later
validation and OOS periods.

Correcting the horizon mismatch would require converting both opportunities
to a common holding period, for example multiplying pair convergence
moneyness by its one-day reversion fraction. That introduces reliance on the
OU speed estimate and constitutes a new model specification. It should not
be selected using the already-inspected OOS window.

The existing `four_sigmas.py` strategy remains unchanged.

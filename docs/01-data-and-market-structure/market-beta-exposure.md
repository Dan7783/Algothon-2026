# Are we exposed to market beta? (hedging audit)

> **Source:** `cusum_output/daniel_pairs/beta_exposure.md` in the team's working repository (relocated here for the public release).

**Answer: no, not meaningfully.** The book is effectively market-neutral at
the PnL level, and hedging would be worth ~0.03 score points.

## PnL regression on the market factor

Equal-weighted return of instruments 1-50, regressed against daily book PnL:

| window | beta ($/1.0 mkt) | t-stat | R^2 | score now | score if market component removed |
|---|---|---|---|---|---|
| train 251-500 | 5377 | 0.59 | 0.0014 | 813.09 | 813.10 |
| test 501-750 | 8218 | 0.98 | 0.0039 | 837.35 | 837.38 |

t-stats below 1: the market explains 0.1-0.4% of daily PnL variance.
Perfectly hedging it changes the score by +0.01 / +0.03.  Regression on
ALGO instead of the equal-weight index gives the same verdict (t 0.34 /
0.71).

## Why it is neutral despite EQUAL-DOLLAR legs

The atlas legs are sized \$10k / -\$10k, NOT beta-weighted — `_beta_dollar_legs`
exists in daniel_filter but is never called, so the filter's hedge ratio
drives the SIGNAL only, not the SIZING.  That would leave residual beta if
the two legs had different market betas.  They mostly do not:

- mean |beta mismatch| across the 21 atlas pairs = **0.154**
- ~\$1,541 of net beta exposure per \$10k pair
- mismatch signs are mixed (11 positive, 10 negative), so across ~11
  simultaneous pairs they largely cancel: RMS ~\$6.9k on a ~\$440k gross book

Cointegrated pairs are naturally similar-beta, which is why dollar-neutral
legging happens to be close to beta-neutral here.  Worst offenders:
HETT/ULXY (-0.38), MTNS/BENI (+0.34), HRET/CTGI (-0.32).

## Residual dollar exposure (real, but not a beta problem)

| window | mean net $ | std | max abs |
|---|---|---|---|
| train | +18,796 | 58,742 | 183,508 |
| test | +5,212 | 56,852 | 171,664 |

Net exposure spikes to ~40% of gross on individual days (driven by the
directional ridge and lead-lag sleeves, not the pairs), but it changes sign
frequently enough that it never accumulates into a persistent market
loading — hence the near-zero regression beta.

## The unused instrument (opportunity, not a risk)

Instrument 0 (ALGO) is never traded — every strategy sets `dollars[0] = 0`.
It is:

- **98.6% correlated** with the equal-weighted market, beta **1.014**
- granted a **10x position limit** (\$100k vs \$10k)
- charged **5x lower commission** (0.2bp vs 1bp)

That is precisely the specification of a designed hedging/index vehicle.
The audit shows there is currently nothing to hedge with it — but its
preferential terms mean that if a market-timing signal ever existed, ALGO
would be by far the most efficient way to express it.  Ridge explicitly
excludes it (`pos[1:] = d`).

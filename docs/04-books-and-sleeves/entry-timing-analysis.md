# Entry/exit timing analysis (16 stable pairs, days 251-750)

> **Source:** `cusum_output/fifty_pairs/timing_analysis.md` in the team's working repository (relocated here for the public release).

User hypothesis: trades enter too low, miss the large-z entries.

## What the episodes show (864 episodes, shipped band lo=0.27)

- Entries cluster at the band edge (median entry |z| 0.46, p90 0.86).
- The visible pain is real: 105 episodes reach |z|>=2 and the average
  carry from entry to the z-peak is -\$690; those episodes are net ~flat
  (-\$17 avg).  On a chart this looks like "entered much too low".
- But the engine is invisible on charts: 713 small episodes entering at
  [0.27, 0.75) win 92% of the time, +\$228 avg, +\$33 per day held; the
  [0.75, 1.5) bucket is even better per day (+\$41, 97% win).
- "Missing the large z" is mechanically almost impossible in band mode:
  only 15 of 8,000 pair-days were flat with |z|>=2 (the 4.0 stop zone).

## Counterfactuals (sleeve totals, train / test)

| policy | train | test |
|---|---|---|
| shipped band 0.27-4.0 | +457.5 | +442.6 |
| band, no stop | +455.5 | +438.7 |
| late entry 0.75 (hysteresis) | +234.7 | +208.4 |
| late entry 1.25 | +257.1 | +239.7 |
| late entry 1.75 | +196.5 | +209.1 |
| late entry 2.25 | +155.3 | +125.1 |
| graded size (0.5x below 0.75z) | +424.0 | +389.8 |
| graded size (0.7x below 0.75z) | +437.8 | +411.3 |

VERDICT: waiting for the large divergences roughly halves the sleeve in
BOTH windows, and even down-weighting small-z entries loses.  The early
"low" entries are the engine, not the flaw; the adverse carry on big
excursions is the cost of being present for the many quick reversions.
The same holds at the robust_hybrid portfolio level (discovery entry 0.4
/ 0.75 both lose on both windows).

## robust_hybrid discovery sweep (two-window Pareto)

default e0.2 b4.5 w250: train 784.5 / test 779.1 — beats entry 0.4
(776.6/769.9), entry 0.75 (754.8/761.6), break 3.5 (754.4/767.7),
break 6.0 (778.0/776.6), window 150 (717.7/736.6) on BOTH windows.
No change adopted; robust_hybrid ships unchanged.

## Flip-at-extremes proposal (enter at peak, reverse at opposite peak)

Overshoot check: of 192 zero-crossings from |z|>=1.5, only 42% continue to
the opposite 1.5 extreme before bouncing back to 1.0 — almost exactly the
gambler's-ruin ratio for a driftless diffusion (1.0/2.5 = 40%), i.e. the
spreads have NO oscillation edge past the mean; median 12 days to the
opposite extreme when it does happen.

| policy | train | test |
|---|---|---|
| shipped band (flip at mean) | +457.5 | +442.6 |
| flip at extremes Z=1.0 | +436.8 | +312.9 |
| flip at extremes Z=1.5 | +238.5 | +226.7 |
| flip at extremes Z=2.0 | +89.9 | +111.7 |
| flip at extremes Z=2.5 | +13.8 | +54.9 |

VERDICT: holding through the mean fights the OU pull 58% of the time and
gives back the reversion profit; the correct switch point is the ZERO
crossing (which the band policy already implements), not the opposite
extreme.  Entries/exits on the charts at small z are correct behavior.

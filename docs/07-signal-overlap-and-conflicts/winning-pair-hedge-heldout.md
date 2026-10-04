# Frozen held-out result: complete-pair winning-side execution

> **Source:** `analysis_outputs/winning_pair_hedge/heldout_report.md` in the team's working repository (relocated here for the public release).

The development-only selection froze **winner_lp_equal_ridge_flat** before days 751-1000 were evaluated.

| Variant | Score | Mean PnL | Total PnL | Fees | Mean gross dollars |
|---|---:|---:|---:|---:|---:|
| current_live | +986.31 | \$998.90 | \$249,724 | \$7,048 | \$458,334 |
| winner_lp_equal_ridge_flat | +796.19 | \$807.77 | \$201,944 | \$6,284 | \$412,528 |

Score delta: **-190.12**.

## What the mechanics show

- Current final execution matches both intended pair-leg signs in 61.5% of active pair-events; 0.0% have a missing leg and 38.5% have a flipped leg.
- The current isolated-pair median beta-hedge error is 11.4%.
- Only 63.0% of active pair-events have both legs on their consensus-winning sides; 53.9% of all pair-events are therefore unfilled after consensus and capacity.

The held-out result is descriptive only. The production strategy is unchanged by this audit.

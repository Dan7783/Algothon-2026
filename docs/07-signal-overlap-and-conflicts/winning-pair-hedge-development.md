# Winning-side complete-pair hedge audit

> **Source:** `analysis_outputs/winning_pair_hedge/development_report.md` in the team's working repository (relocated here for the public release).

The production pair agents already calculate exact dynamic-beta vectors. The tested allocator keeps a pair only when both legs agree with the pair-consensus winning side, then scales both legs by the same fill.

## Development scores

| Variant | Train A | Train B | Train | Validation A | Validation B | Validation |
|---|---:|---:|---:|---:|---:|---:|
| current_live | +832.61 | +1085.65 | +959.34 | +959.30 | +1155.91 | +1057.63 |
| winner_lp_equal_ridge_flat | +813.45 | +893.52 | +853.51 | +969.41 | +1024.84 | +997.17 |

Frozen development winner: **winner_lp_equal_ridge_flat**. Production eligibility: **False**.

## Current execution integrity

| Period | Layer | Events | Two winning legs | One winning leg | Both signs match | Missing | Flipped | Median hedge error | Isolated error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | current_core | 10,170 | 64.4% | 32.4% | 62.7% | 0.6% | 36.7% | 12.9% | 5.5% |
| TRAIN | current_final | 10,170 | 64.4% | 32.4% | 62.9% | 0.3% | 36.9% | 14.1% | 10.4% |
| VALIDATION | current_core | 9,920 | 64.4% | 32.1% | 62.8% | 1.1% | 36.1% | 14.6% | 4.4% |
| VALIDATION | current_final | 9,920 | 64.4% | 32.1% | 63.1% | 0.4% | 36.5% | 15.4% | 5.6% |

A pair component accepted by a complete-pair allocator has zero pre-rounding hedge error by construction. The trade-off is how many active pair-events survive both-leg consensus and capacity.

## Capacity of frozen allocator

- TRAIN: 64.4% of active pair-events have two winning legs; 30.4% are fully filled, 14.8% partially filled, and 54.8% unfilled.
- VALIDATION: 64.4% of active pair-events have two winning legs; 29.9% are fully filled, 16.5% partially filled, and 53.6% unfilled.

## Capital use

| Period | Variant | Mean gross dollars | Mean active assets |
|---|---|---:|---:|
| TRAIN | current_live | \$457,829 | 48.6 |
| TRAIN | winner_lp_equal_ridge_flat | \$410,599 | 48.2 |
| VALIDATION | current_live | \$449,354 | 48.6 |
| VALIDATION | winner_lp_equal_ridge_flat | \$398,957 | 48.3 |

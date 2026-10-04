# Robust Hybrid Audit

> **Source:** `cusum_output/robust_hybrid/report.md` in the team's working repository (relocated here for the public release).

> Superseded by the conservative adaptive update in
> `adaptive_v2_report.md`.  The figures below remain the exact v1 control used
> by the new audit.

`robust_hybrid.py` combines the tuned 21-pair Daniel adaptive atlas with the
Four Sigmas trailing-250 residual-discovery sleeve.  The source strategies are
unchanged.

## Selected ordering

1. Adaptive pair atlas (pair-local sticky through-origin beta and CUSUM state)
2. Extra lead-lag
3. Ridge VAR, entry threshold 1.0
4. Trailing-OLS discovery on residual idle capacity

Discovery excludes every adaptive-atlas pair, requires both instruments to be
idle, and preserves a common hedge scale on both legs.  It therefore cannot
evict the tuned atlas, lead-lag, or ridge positions.

## Like-for-like results

All results use independent state resets and `eval.py` accounting on train days
251-500 and test days 501-750.

| Strategy | Train mean | Train score | Train std | Test mean | Test score | Test std |
|---|---:|---:|---:|---:|---:|---:|
| Four Sigmas | 734.62 | 722.42 | 1,509.36 | 761.83 | 749.68 | 1,533.32 |
| Tuned Daniel core | 755.65 | 746.95 | 1,288.79 | 764.06 | 756.36 | 1,218.76 |
| **Robust hybrid** | **795.17** | **784.48** | **1,467.08** | **788.61** | **779.08** | **1,379.38** |

Relative to Daniel, the hybrid adds +37.53 train score and +22.71 test score.
Relative to Four Sigmas, it adds +62.06 train score and +29.39 test score while
reducing turnover from 79.03m/72.74m to 67.20m/58.87m.

## Ordered sleeve attribution

Because later sleeves depend on capacity left by earlier sleeves, contribution
is measured cumulatively in production priority order.  The incremental daily
mean is the difference from the preceding row.

| Cumulative book | Train mean | Increment | Train score | Test mean | Increment | Test score |
|---|---:|---:|---:|---:|---:|---:|
| Atlas | 556.91 | +556.91 | 550.30 | 462.26 | +462.26 | 456.13 |
| + extra lead-lag | 638.22 | +81.31 | 630.75 | 572.43 | +110.17 | 566.07 |
| + ridge | 755.65 | +117.43 | 746.95 | 764.06 | +191.62 | 756.36 |
| + discovery | 795.17 | +39.52 | 784.48 | 788.61 | +24.55 | 779.08 |

All four sleeves add positive daily PnL in both windows.  Ridge is the largest
complement to the atlas; discovery is smaller but remains positive after all
capacity interactions.

## Rejected variants

| Variant | Train score | Test score | Decision |
|---|---:|---:|---|
| Ridge entry 0.75 | 725.81 | 807.72 | Rejected: test-favoured and damages train |
| Remove extra lead-lag | 744.19 | 706.04 | Rejected: damages both, especially test |
| Discovery scale 0.25 | 756.57 | 762.40 | Valid but leaves material diversified PnL unused |
| Discovery scale 0.50 | 766.06 | 768.35 | Valid but weaker than full scale in both windows |
| Discovery scale 0.75 | 775.27 | 774.12 | Valid but weaker than full scale in both windows |

The selected book never exceeded the \$10,000 per-instrument dollar limit in
either replay.  Test data participated in this requested two-window selection,
so days 501-750 should be treated as validation rather than untouched final OOS.

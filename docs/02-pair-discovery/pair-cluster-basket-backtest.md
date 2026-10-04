# Train-selected triangle basket backtest

> **Source:** `analysis_outputs/pair_clusters/cluster_backtest.md` in the team's working repository (relocated here for the public release).

The triangle set, basket coefficients, thresholds, and hedge directions
are fixed from days 1-500. Days 751-1000 are not loaded.

- Cluster assets: `AENO, CTGI, EELT, HUXZ, ILVX, NGTE, NWIG, OTCS, SMAH`
- Pair relationships removed in the clean replacement: **27**

## Scores

| Sleeve | VAL1 | VAL2 | Full validation |
|---|---:|---:|---:|
| Current full strategy | 853.5683 | 1049.4264 | 951.5164 |
| Full cluster replacement | 691.9041 | 992.3924 | 842.2630 |
| Cluster-priority arbitration | 804.7719 | 1006.8826 | 905.8551 |
| Cluster disagreement filter | 852.2637 | 1042.5916 | 947.4623 |
| Nine internal pair edges only | 260.5750 | 110.5286 | 186.0541 |
| Three cluster baskets only | 137.9043 | 72.8567 | 105.7100 |

## Full replacement deltas

- VAL1: `-161.6641`
- VAL2: `-57.0340`
- Validation: `-109.2534`

The clean replacement removes every configured pair touching a
selected cluster asset and disables ridge on those assets, then
installs one hedge-ratio-preserving vector per triangle.

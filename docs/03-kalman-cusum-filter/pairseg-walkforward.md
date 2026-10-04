# PairSegFilter walk-forward pair strategy

> **Source:** `cusum_output/pair_strategy_wf/wf_report.md` in the team's working repository (relocated here for the public release).

- In-sample length: `500` days
- Offsets: `[0, 50, 100, 150, 200]`
- OOS length: up to 200 days after each train window
- Leg notional: `$10000` / side, commission `0.0001`

## Mean OOS Algothon score by pair

| Pair | Windows | Mean score | Median | Std | Min | Max | % positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMAH/ILVX | 5 | 15.96 | 17.04 | 3.60 | 11.58 | 20.90 | 100% |
| HUXZ/ACAC | 5 | 13.14 | 12.63 | 6.47 | 2.49 | 21.63 | 100% |
| RTTH/NAYO | 5 | 9.15 | 10.20 | 2.46 | 4.92 | 11.66 | 100% |
| AENO/NWIG | 5 | 8.85 | 7.99 | 3.68 | 4.22 | 14.08 | 100% |
| MHRM/EAFC | 5 | 8.57 | 6.86 | 6.12 | 0.01 | 17.76 | 100% |
| EORC/NGTE | 5 | 6.93 | 5.19 | 3.59 | 3.11 | 13.47 | 100% |
| FWWG/BLBT | 5 | 6.01 | 8.50 | 4.72 | 0.00 | 11.44 | 100% |
| CTGI/EELT | 5 | 3.84 | 4.79 | 3.36 | -0.78 | 8.89 | 80% |
| ACIX/ITPA | 5 | 3.78 | 3.29 | 2.61 | 0.35 | 8.34 | 100% |
| HETT/ULXY | 5 | 3.68 | 4.07 | 5.85 | -5.41 | 11.92 | 80% |
| ALUT/CCNS | 5 | 2.82 | 4.36 | 5.10 | -6.99 | 8.03 | 80% |
| MTNS/ALUT | 5 | -2.88 | -3.96 | 8.59 | -13.16 | 9.96 | 40% |
| NAYO/SRTX | 5 | -6.11 | -7.00 | 5.20 | -11.90 | 0.49 | 20% |

Detail: `wf_detail.csv`
Summary: `wf_summary.csv`

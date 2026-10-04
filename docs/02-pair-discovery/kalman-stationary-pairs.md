# Kalman residual stationarity — all pairs

> **Source:** `cusum_output/kalman_pair_discovery/kalman_stationary_pairs.md` in the team's working repository (relocated here for the public release).

Method: sticky through-origin Kalman (`PairSegFilter`) then ADF(1) / half-life on `y − β_t x`.  Fixed OLS ADF reported for comparison.

Pass requires sticky β (few CUSUM snaps) — adaptive filters alone would make almost every residual look stationary.

- Universe: 50 instruments (excluded ALGO)
- Candidates scored: 1225
- Gates: ADF_KF < -4.0, HL ∈ [2.0, 40.0], snaps ≤ 6, adapt ≤ 8%, stable-β MAD ≤ 0.01
- **Pass: 50** (stable=16, adaptive-rescue=34)

## Passing pairs

| Rank | Pair | Tier | ADF KF | ADF OLS | HL | β | snaps | adapt% | score |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | AENO/NWIG | stable | -7.80 | -7.14 | 3.4 | 0.951 | 2 | 2.1% | 9.30 |
| 2 | ILVX/SMAH | stable | -7.26 | -7.07 | 5.2 | 0.732 | 2 | 2.1% | 8.70 |
| 3 | EORC/NGTE | stable | -6.79 | -6.45 | 4.7 | 0.623 | 1 | 1.1% | 8.64 |
| 4 | ACAC/HUXZ | stable | -6.79 | -6.43 | 4.8 | 0.896 | 2 | 2.1% | 8.23 |
| 5 | HETT/ULXY | stable | -7.82 | -6.79 | 3.9 | 1.306 | 5 | 5.3% | 8.05 |
| 6 | MHRM/SRTX | stable | -5.41 | -4.59 | 7.8 | 0.058 | 0 | 0.0% | 7.56 |
| 7 | EAFC/SRTX | stable | -5.40 | -4.11 | 8.9 | 0.120 | 0 | 0.0% | 7.52 |
| 8 | CCNS/SRTX | stable | -5.28 | -3.85 | 10.6 | 0.065 | 0 | 0.0% | 7.35 |
| 9 | MSDP/ULXY | stable | -6.20 | -3.14 | 6.7 | 0.543 | 4 | 4.3% | 6.77 |
| 10 | FWWG/BLBT | stable | -5.45 | -5.14 | 8.0 | 0.732 | 3 | 3.2% | 6.40 |
| 11 | MTNS/BENI | stable | -5.34 | -5.04 | 9.1 | 0.730 | 4 | 4.3% | 5.86 |
| 12 | HRET/CTGI | stable | -5.22 | -3.25 | 7.9 | 0.395 | 5 | 5.3% | 5.35 |
| 13 | ACIX/ULXY | stable | -5.26 | -2.95 | 9.1 | 0.352 | 6 | 6.4% | 4.96 |
| 14 | DIHO/RTTH | stable | -4.89 | -3.66 | 11.0 | 0.264 | 6 | 6.4% | 4.56 |
| 15 | MHRM/RCRI | stable | -4.81 | -2.53 | 10.9 | 0.311 | 6 | 6.4% | 4.46 |
| 16 | HUXZ/NWIG | stable | -4.55 | -2.64 | 11.6 | 0.723 | 6 | 6.4% | 4.19 |
| 17 | EORC/SMAH | adaptive | -6.12 | -0.99 | 6.5 | 0.382 | 4 | 4.3% | 5.21 |
| 18 | EAFC/HRET | adaptive | -6.71 | -2.06 | 5.9 | 0.508 | 6 | 6.4% | 5.01 |
| 19 | SMAH/NGTE | adaptive | -6.35 | -0.97 | 6.0 | 1.631 | 5 | 5.3% | 5.01 |
| 20 | HETT/SRTX | adaptive | -4.41 | -1.51 | 15.3 | 0.343 | 0 | 0.0% | 4.88 |
| 21 | EAFC/EELT | adaptive | -4.82 | -2.49 | 12.0 | 0.216 | 3 | 3.2% | 4.15 |
| 22 | ULXY/ELLT | adaptive | -5.95 | -1.32 | 8.0 | 1.114 | 6 | 6.4% | 4.13 |
| 23 | FWWG/SMAH | adaptive | -5.80 | -1.57 | 7.6 | 0.363 | 6 | 6.4% | 4.05 |
| 24 | CUBO/ULXY | adaptive | -6.11 | -2.21 | 7.2 | 3.440 | 6 | 6.4% | 4.04 |
| 25 | ACAC/CTGI | adaptive | -5.38 | -2.00 | 8.7 | 0.426 | 5 | 5.3% | 3.98 |
| 26 | HUXZ/CTGI | adaptive | -5.36 | -1.90 | 9.2 | 0.454 | 5 | 5.3% | 3.93 |
| 27 | MHRM/ILVX | adaptive | -4.93 | -2.19 | 9.5 | 0.289 | 4 | 4.3% | 3.93 |
| 28 | AETS/LSST | adaptive | -5.30 | -1.96 | 8.2 | 0.240 | 5 | 5.3% | 3.92 |
| 29 | NGTE/ILVX | adaptive | -5.60 | -0.81 | 8.1 | 0.841 | 6 | 6.4% | 3.79 |
| 30 | HTRK/RTTH | adaptive | -5.48 | -2.18 | 8.0 | 0.414 | 6 | 6.4% | 3.73 |
| 31 | NGTE/CTGI | adaptive | -5.41 | -1.37 | 8.6 | 0.287 | 6 | 6.4% | 3.61 |
| 32 | SMAH/CTGI | adaptive | -5.36 | -2.49 | 8.8 | 0.646 | 6 | 5.5% | 3.59 |
| 33 | DUCT/FWWG | adaptive | -5.30 | -2.02 | 8.9 | 0.655 | 6 | 6.4% | 3.52 |
| 34 | BENI/BLBT | adaptive | -4.95 | -0.07 | 11.3 | 1.022 | 5 | 5.3% | 3.51 |
| 35 | NPCK/NAYO | adaptive | -4.92 | -2.48 | 11.5 | 0.449 | 5 | 5.3% | 3.47 |
| 36 | BLBT/CTGI | adaptive | -4.53 | -2.09 | 12.6 | 0.238 | 4 | 4.3% | 3.45 |
| 37 | FARS/BLBT | adaptive | -5.16 | -1.93 | 8.8 | 1.223 | 6 | 6.4% | 3.36 |
| 38 | DIHO/FCSG | adaptive | -4.63 | -2.34 | 12.7 | 0.313 | 5 | 5.3% | 3.17 |
| 39 | LSST/BENI | adaptive | -5.14 | -2.34 | 9.5 | 2.880 | 6 | 6.4% | 3.15 |
| 40 | FWWG/FARS | adaptive | -4.85 | -2.05 | 9.7 | 0.593 | 6 | 6.4% | 3.05 |
| 41 | SRTX/MDGI | adaptive | -5.04 | -1.70 | 11.6 | 0.807 | 6 | 6.4% | 3.02 |
| 42 | CTGI/MDGI | adaptive | -4.14 | -2.08 | 13.6 | 0.344 | 4 | 4.1% | 3.01 |
| 43 | HRND/RRES | adaptive | -4.80 | -0.93 | 12.4 | 0.195 | 6 | 6.4% | 2.91 |
| 44 | HTRK/MSDP | adaptive | -4.66 | -1.73 | 10.6 | 0.301 | 6 | 6.4% | 2.84 |
| 45 | EORC/ILVX | adaptive | -4.66 | -0.84 | 10.4 | 0.526 | 6 | 6.4% | 2.82 |
| 46 | ELLT/IHOZ | adaptive | -4.62 | -1.71 | 12.1 | 0.441 | 6 | 6.4% | 2.73 |
| 47 | MHRM/RRES | adaptive | -4.59 | -1.81 | 13.2 | 0.243 | 6 | 6.4% | 2.71 |
| 48 | ITPA/ILVX | adaptive | -4.57 | -1.82 | 12.1 | 0.551 | 6 | 6.4% | 2.71 |
| 49 | ILVX/HETT | adaptive | -4.49 | -1.40 | 14.5 | 0.594 | 6 | 6.4% | 2.58 |
| 50 | SRNA/IHOZ | adaptive | -4.43 | -1.47 | 12.7 | 0.505 | 6 | 6.4% | 2.53 |

Full table: `kalman_stationary_pairs.csv`

# Johansen all pairs — strict I(1) screen

> **Source:** `cusum_output/kalman_pair_discovery/johansen_all_pairs_i1_strict.md` in the team's working repository (relocated here for the public release).

- ADF level p ≥ **0.1** (fail to reject unit root)
- ADF diff p < **0.01** (not I(2))
- KPSS p < **0.05** (reject stationarity)
- Excluded: **ACIX, BENI, CTGI, EELT, FARS, ITPA, LSST, MDGI, MTNS, ULXY**
- I(1) universe: **40**
- Pairs tested: **780**
- Rank ≥ 1: **67**

## Top pairs

| # | Pair | Indices | Rank | Trace−crit |
|---:|---|---|---:|---:|
| 1 | SMAH/ILVX | (10,46) | 1 | 62.9 |
| 2 | AENO/NWIG | (1,20) | 1 | 35.6 |
| 3 | EORC/NGTE | (13,45) | 1 | 29.5 |
| 4 | HUXZ/ACAC | (8,27) | 1 | 29.4 |
| 5 | MHRM/EAFC | (49,50) | 1 | 22.1 |
| 6 | RTTH/NAYO | (18,35) | 1 | 18.4 |
| 7 | FWWG/BLBT | (36,41) | 1 | 12.1 |
| 8 | SRTX/MHRM | (28,49) | 1 | 11.3 |
| 9 | SRTX/EAFC | (28,50) | 1 | 9.2 |
| 10 | HRND/AETS | (38,39) | 1 | 8.5 |
| 11 | SRNA/DUCT | (3,9) | 1 | 7.5 |
| 12 | RCRI/NAYO | (30,35) | 2 | 6.6 |
| 13 | ALUT/CCNS | (26,32) | 1 | 6.3 |
| 14 | RTTH/SPLZ | (18,19) | 1 | 6.3 |
| 15 | DUCT/NWIG | (9,20) | 1 | 5.7 |
| 16 | NAYO/BLBT | (35,41) | 1 | 5.6 |
| 17 | SPLZ/NAYO | (19,35) | 1 | 5.6 |
| 18 | NAYO/FWWG | (35,36) | 1 | 4.7 |
| 19 | ANSO/IHOZ | (16,34) | 2 | 4.6 |
| 20 | SRNA/ANSO | (3,16) | 1 | 4.5 |
| 21 | RTTH/BLBT | (18,41) | 1 | 3.9 |
| 22 | IHOZ/EAFC | (34,50) | 1 | 3.8 |
| 23 | NWIG/IHOZ | (20,34) | 1 | 3.6 |
| 24 | AGVF/ILVX | (23,46) | 2 | 3.6 |
| 25 | RTTH/ALUT | (18,26) | 1 | 3.3 |
| 26 | DIHO/NAYO | (17,35) | 1 | 3.3 |
| 27 | ELLT/HRND | (4,38) | 2 | 3.2 |
| 28 | SPLZ/ALUT | (19,26) | 1 | 2.8 |
| 29 | ACAC/RCRI | (27,30) | 1 | 2.8 |
| 30 | OTCS/RCRI | (6,30) | 1 | 2.7 |

CSV: `johansen_all_pairs_i1_strict.csv`
Arrays: `johansen_all_coint_pairs_i1_strict.py`
Network: `johansen_all_pairs_network_i1_strict.png`
Screen: `integration_screen_strict.csv`

# Johansen all pairs — I(1) screened

> **Source:** `cusum_output/kalman_pair_discovery/johansen_all_pairs_i1.md` in the team's working repository (relocated here for the public release).

EELT (and other near-stationary names) were spuriously linked to every asset because Johansen assumes I(1) inputs.

- Excluded instruments: ALGO
- I(0) screened out: **CTGI, MTNS, EELT, ULXY, BENI, ITPA**
- I(1) universe: **44**
- Pairs tested: **946**
- Rank ≥ 1: **100**

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
| 9 | NPCK/FARS | (11,48) | 2 | 10.2 |
| 10 | SRTX/EAFC | (28,50) | 1 | 9.2 |
| 11 | HRND/AETS | (38,39) | 1 | 8.5 |
| 12 | SRNA/DUCT | (3,9) | 1 | 7.5 |
| 13 | RCRI/NAYO | (30,35) | 2 | 6.6 |
| 14 | ALUT/CCNS | (26,32) | 1 | 6.3 |
| 15 | RTTH/SPLZ | (18,19) | 1 | 6.3 |
| 16 | IHOZ/FARS | (34,48) | 2 | 5.8 |
| 17 | DUCT/NWIG | (9,20) | 1 | 5.7 |
| 18 | NAYO/BLBT | (35,41) | 1 | 5.6 |
| 19 | RCRI/FARS | (30,48) | 2 | 5.6 |
| 20 | SPLZ/NAYO | (19,35) | 1 | 5.6 |
| 21 | ELLT/FARS | (4,48) | 1 | 5.4 |
| 22 | ACIX/FARS | (31,48) | 2 | 5.0 |
| 23 | NAYO/FWWG | (35,36) | 1 | 4.7 |
| 24 | ANSO/FARS | (16,48) | 2 | 4.6 |
| 25 | ANSO/IHOZ | (16,34) | 2 | 4.6 |
| 26 | SRNA/ANSO | (3,16) | 1 | 4.5 |
| 27 | RTTH/BLBT | (18,41) | 1 | 3.9 |
| 28 | NWIG/FARS | (20,48) | 2 | 3.8 |
| 29 | IHOZ/EAFC | (34,50) | 1 | 3.8 |
| 30 | NWIG/IHOZ | (20,34) | 1 | 3.6 |

CSV: `johansen_all_pairs_i1.csv`
Arrays: `johansen_all_coint_pairs_i1.py`
Network: `johansen_all_pairs_network_i1.png`
Screen: `integration_screen.csv`

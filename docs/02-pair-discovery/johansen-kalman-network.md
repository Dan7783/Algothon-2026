# Johansen on Kalman-passing pair network

> **Source:** `cusum_output/kalman_pair_discovery/johansen_report.md` in the team's working repository (relocated here for the public release).

- Passing KF edges tested: **50**
- Bivariate Johansen rank≥1: **20**
- Basket groups tested: **17**
- Basket Johansen rank≥1: **15**
- Single common factor (rank = n−1): **2**

## Top bivariate pairs (by trace excess)

| # | Pair | Indices | Rank | Trace−crit |
|---:|---|---|---:|---:|
| 1 | HETT/ULXY | (7,40) | 2 | 69.9 |
| 2 | ILVX/SMAH | (46,10) | 1 | 62.9 |
| 3 | AENO/NWIG | (1,20) | 1 | 35.6 |
| 4 | EORC/NGTE | (13,45) | 1 | 29.5 |
| 5 | ACAC/HUXZ | (27,8) | 1 | 29.4 |
| 6 | MTNS/BENI | (33,42) | 2 | 23.9 |
| 7 | FWWG/BLBT | (36,41) | 1 | 12.1 |
| 8 | MHRM/SRTX | (49,28) | 1 | 11.3 |
| 9 | EAFC/SRTX | (50,28) | 1 | 9.2 |
| 10 | MSDP/ULXY | (12,40) | 2 | 7.1 |
| 11 | EAFC/EELT | (50,37) | 2 | 3.2 |
| 12 | SMAH/CTGI | (10,25) | 2 | 2.3 |
| 13 | ACIX/ULXY | (31,40) | 2 | 2.3 |
| 14 | SRNA/IHOZ | (3,34) | 1 | 1.3 |
| 15 | MHRM/RRES | (49,24) | 1 | 1.1 |
| 16 | DIHO/RTTH | (17,18) | 1 | 1.1 |
| 17 | NGTE/ILVX | (45,46) | 2 | 0.5 |
| 18 | CCNS/SRTX | (32,28) | 1 | 0.4 |
| 19 | HRET/CTGI | (15,25) | 2 | 0.3 |
| 20 | LSST/BENI | (2,42) | 2 | 0.2 |

## Cointegrated baskets

| # | Type | Assets | n | Rank | n−1? | Trace−crit | β⃗ (norm) |
|---:|---|---|---:|---:|---|---:|---|
| 1 | hub | SMAH,EORC,NGTE,ILVX | 4 | 3 | yes | 93.8 | `1.0000,-0.0922,0.0511,-1.4256` |
| 2 | triangle | EORC,NGTE,ILVX | 3 | 3 | yes | 36.2 | `1.0000,-0.6006,-0.0065` |
| 3 | hub | SMAH,EORC,CTGI,NGTE,ILVX | 5 | 3 |  | 93.5 | `1.0000,-0.0813,-0.0237,0.0370,-1.3834` |
| 4 | hub | SMAH,EORC,CTGI,FWWG,NGTE,ILVX | 6 | 3 |  | 89.9 | `1.0000,-0.0609,-0.0297,-0.0423,0.0397,-1.3901` |
| 5 | hub | HETT,SMAH,EORC,NGTE,ILVX,MHRM | 6 | 2 |  | 75.9 | `1.0000,-13.4434,3.4433,-1.2873,18.1694,-3.7086` |
| 6 | hub | ELLT,HETT,MSDP,CUBO,ACIX,ULXY | 6 | 2 |  | 73.0 | `1.0000,31.6374,-1.2465,0.3974,-3.4241,-43.0132` |
| 7 | triangle | SMAH,NGTE,ILVX | 3 | 1 |  | 60.2 | `1.0000,-0.0040,-1.4220` |
| 8 | triangle | SMAH,EORC,ILVX | 3 | 1 |  | 57.8 | `1.0000,-0.0066,-1.4294` |
| 9 | triangle | SMAH,EORC,NGTE | 3 | 1 |  | 31.9 | `1.0000,413.4412,-247.4271` |
| 10 | triangle | HUXZ,CTGI,ACAC | 3 | 1 |  | 24.6 | `1.0000,0.0121,-1.1419` |
| 11 | hub | HETT,MDGI,SRTX,CCNS,MHRM,EAFC | 6 | 1 |  | 16.2 | `1.0000,0.1711,-0.0783,6.3235,-51.7281,22.5525` |
| 12 | hub | RRES,SRTX,RCRI,ILVX,MHRM | 5 | 1 |  | 12.8 | `1.0000,-0.1947,-0.5117,-0.3443,5.3381` |
| 13 | hub | CTGI,FWWG,BLBT,BENI,FARS | 5 | 1 |  | 11.8 | `1.0000,71.7484,-56.6211,8.1922,5.3683` |
| 14 | triangle | FWWG,BLBT,FARS | 3 | 1 |  | 11.6 | `1.0000,-0.7943,0.0868` |
| 15 | hub | DUCT,SMAH,FWWG,BLBT,FARS | 5 | 1 |  | 7.7 | `1.0000,0.2262,-7.4212,5.4222,-0.4159` |

Plots: `johansen_summary.png`
CSV: `johansen_pairs.csv`, `johansen_baskets.csv`

---

## Figures

![Bivariate and basket Johansen trace statistics](../figures/johansen-summary.png)

*Bivariate and basket Johansen trace statistics.*

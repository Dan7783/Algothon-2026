# Top30 TestbedV2 OOS decay analysis

> **Source:** `analysis_outputs/top30_oos_decay_report.md` in the team's working repository (relocated here for the public release).

The approved pair set and parameters are fixed. Days 751-1000 are used only for evaluation.

## Portfolio result

- Full IS (days 31-750) score: **415.15**
- OOS (days 751-1000) score: **201.88**
- Score change: **-51.4%**
- IS mean daily P&L: **\$421.70**
- OOS mean daily P&L: **\$214.45**
- IS/OOS daily P&L volatility: **\$837.47 / \$846.36**

## Cross-sectional findings

- 4/32 pairs have non-positive standalone OOS P&L.
- 22/32 retain less than half their median IS-year score.
- 19/32 are blocked on at least half their OOS candidate days by higher-priority pairs sharing a symbol.
- 16/32 have negative leave-one-out OOS score contribution: removing them improves the actual book.
- Aggregate conflict blocking was already high and changed only from 54.8% in year 3 to 56.7% OOS.
- Pair concentration: RTTH appears in 13 pairs; NAYO appears in 6 pairs.
- ALGO-NAYO and ALGO-RCRI were reviewed with two legs, but the live wrapper zeros instrument 0 (ALGO), leaving one-legged exposure.

## Pair-by-pair results

| Pair | IS median score | OOS score | Retention | OOS P&L | OOS blocked | OOS marginal score | Book P&L change vs Y3 | Diagnosis |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HUXZ-RTTH | 17.40 | -10.66 | -61.3% | \$-2,666 | 1.1% | -15.89 | \$-6,235 | removal improves live OOS book; standalone signal reversed; more signals |
| HRET-EELT | 25.54 | -3.23 | -12.6% | \$-807 | 80.4% | +0.85 | \$-2,692 | standalone signal reversed; trade drought; slower reversion; price correlation shifted; mostly blocked by symbol conflicts |
| AGVF-NAYO | 20.38 | -1.52 | -7.5% | \$-380 | 49.5% | -10.20 | \$-4,287 | removal improves live OOS book; standalone signal reversed; book conflicts worsened; material book conflicts |
| ANSO-NAYO | 27.35 | -1.18 | -4.3% | \$-295 | 59.1% | +2.17 | \$-5,716 | adds material live-book score; standalone signal reversed; price correlation shifted; mostly blocked by symbol conflicts |
| MDGI-ULXY | 21.34 | 0.00 | 0.0% | \$8 | 0.0% | -5.06 | \$-3,757 | removal improves live OOS book; standalone edge weakened |
| HRET-CTGI | 28.38 | 0.05 | 0.2% | \$595 | 24.0% | -13.86 | \$-7,525 | removal improves live OOS book; standalone edge weakened; slower reversion |
| ALGO-RCRI | 20.52 | 0.16 | 0.8% | \$1,011 | 38.8% | +5.58 | \$-6,194 | ALGO leg disabled live; adds material live-book score; standalone edge weakened; trade drought; material book conflicts |
| RTTH-MTNS | 27.78 | 0.41 | 1.5% | \$1,172 | 93.2% | -6.33 | \$-589 | removal improves live OOS book; standalone edge weakened; more CUSUM breaks; mostly blocked by symbol conflicts |
| RTTH-CTGI | 26.68 | 0.50 | 1.9% | \$1,328 | 91.2% | -14.08 | \$-1,772 | removal improves live OOS book; standalone edge weakened; mostly blocked by symbol conflicts |
| ALGO-NAYO | 32.31 | 0.94 | 2.9% | \$1,463 | 56.2% | -2.29 | \$4,258 | ALGO leg disabled live; removal improves live OOS book; standalone edge weakened; more signals; mostly blocked by symbol conflicts |
| RTTH-IHOZ | 19.67 | 0.60 | 3.1% | \$1,453 | 97.7% | -0.98 | \$2,384 | standalone edge weakened; mostly blocked by symbol conflicts |
| RTTH-BENI | 26.47 | 0.85 | 3.2% | \$1,417 | 64.7% | -2.69 | \$-6,662 | removal improves live OOS book; standalone edge weakened; more signals; book conflicts worsened; mostly blocked by symbol conflicts |
| RTTH-BLBT | 26.57 | 1.58 | 5.9% | \$1,869 | 92.2% | -2.71 | \$-434 | removal improves live OOS book; standalone edge weakened; more signals; mostly blocked by symbol conflicts |
| RTTH-CCNS | 21.70 | 1.61 | 7.4% | \$2,383 | 96.5% | +8.88 | \$605 | adds material live-book score; standalone edge weakened; trade drought; mostly blocked by symbol conflicts |
| RTTH-SPLZ | 21.60 | 2.62 | 12.1% | \$2,597 | 94.0% | -4.64 | \$1,520 | removal improves live OOS book; standalone edge weakened; trade drought; mostly blocked by symbol conflicts |
| SRTX-RCRI | 20.00 | 3.01 | 15.0% | \$3,083 | 56.4% | +33.89 | \$1,229 | adds material live-book score; standalone edge weakened; mostly blocked by symbol conflicts |
| ANSO-EELT | 34.25 | 5.30 | 15.5% | \$3,429 | 56.4% | -7.77 | \$-6,609 | removal improves live OOS book; standalone edge weakened; trade drought; slower reversion; mostly blocked by symbol conflicts |
| SPLZ-NAYO | 21.64 | 5.17 | 23.9% | \$3,109 | 75.0% | -2.49 | \$-2,160 | removal improves live OOS book; standalone edge weakened; mostly blocked by symbol conflicts |
| LSST-RTTH | 23.39 | 6.05 | 25.9% | \$3,592 | 95.2% | +11.06 | \$1,509 | adds material live-book score; standalone edge weakened; mostly blocked by symbol conflicts |
| DUCT-HRND | 18.21 | 6.09 | 33.5% | \$2,893 | 0.0% | +11.54 | \$-4,493 | adds material live-book score; standalone edge weakened |
| SMAH-ILVX | 38.18 | 17.70 | 46.4% | \$5,248 | 0.0% | +21.62 | \$-4,877 | adds material live-book score; standalone edge broadly retained |
| CTGI-EELT | 54.05 | 26.69 | 49.4% | \$8,846 | 42.7% | +13.11 | \$-2,417 | adds material live-book score; standalone edge broadly retained; material book conflicts |
| EORC-NGTE | 37.16 | 20.13 | 54.2% | \$6,503 | 0.0% | +26.37 | \$-3,525 | adds material live-book score; standalone edge broadly retained |
| RTTH-FWWG | 22.17 | 12.46 | 56.2% | \$5,134 | 91.6% | -0.61 | \$-460 | standalone edge broadly retained; price correlation shifted; mostly blocked by symbol conflicts |
| HETT-ULXY | 45.79 | 27.03 | 59.0% | \$8,208 | 12.4% | +25.05 | \$-2,929 | adds material live-book score; standalone edge broadly retained |
| RTTH-ULXY | 25.53 | 17.06 | 66.8% | \$6,183 | 90.0% | -5.01 | \$-450 | removal improves live OOS book; standalone edge broadly retained; mostly blocked by symbol conflicts |
| HRET-RTTH | 28.90 | 20.06 | 69.4% | \$6,324 | 84.0% | +4.39 | \$1,933 | adds material live-book score; standalone edge broadly retained; mostly blocked by symbol conflicts |
| ALUT-CCNS | 25.24 | 17.83 | 70.7% | \$5,827 | 0.0% | +25.73 | \$-1,420 | adds material live-book score; standalone edge broadly retained; slower reversion |
| HUXZ-ACAC | 28.17 | 20.67 | 73.4% | \$6,057 | 33.1% | +15.72 | \$-3,299 | adds material live-book score; standalone edge broadly retained; book conflicts worsened; material book conflicts |
| SRNA-NAYO | 20.64 | 18.13 | 87.9% | \$6,477 | 78.9% | -2.78 | \$-3,850 | removal improves live OOS book; standalone edge broadly retained; more signals; mostly blocked by symbol conflicts |
| RTTH-NAYO | 37.56 | 37.93 | 101.0% | \$11,198 | 87.4% | +8.33 | \$350 | adds material live-book score; standalone edge broadly retained; more CUSUM breaks; mostly blocked by symbol conflicts |
| AENO-NWIG | 33.40 | 43.97 | 131.7% | \$11,615 | 0.0% | +48.50 | \$3,386 | adds material live-book score; standalone edge improved |

---

## Figures

![Standalone OOS retention and conflict blocking per pair](../figures/top30-oos-pair-diagnostics.png)

*Standalone OOS retention and conflict blocking per pair.*

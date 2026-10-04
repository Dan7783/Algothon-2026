# Daniel pair-specific Pareto tuning

> **Source:** `cusum_output/daniel_pairs/tuning_report.md` in the team's working repository (relocated here for the public release).

Test participates in selection by explicit request; it is not an untouched holdout.
Each replacement improves train and test score/mean, with four-block non-degradation preferred.

| Pair | Train before -> after | Test before -> after | Decision |
|---|---:|---:|---|
| RTTH/NAYO | 94.90 -> 94.90 | 35.34 -> 35.34 | kept baseline: no local train/test Pareto improvement |
| CTGI/EELT | 94.07 -> 94.07 | 48.19 -> 48.19 | kept baseline: no local train/test Pareto improvement |
| HETT/ULXY | 64.56 -> 64.56 | 29.05 -> 29.05 | kept baseline: no local train/test Pareto improvement |
| HRET/CTGI | 43.22 -> 43.22 | 19.64 -> 19.64 | kept baseline: no local train/test Pareto improvement |
| HUXZ/ACAC | 45.64 -> 46.01 | 37.20 -> 39.00 | train/test Pareto improvement |
| NAYO/SRTX | 44.35 -> 44.35 | -4.21 -> -4.21 | kept baseline: no local train/test Pareto improvement |
| AENO/NWIG | 43.47 -> 43.47 | 26.39 -> 26.39 | kept baseline: no local train/test Pareto improvement |
| MDGI/HTRK | 17.43 -> 17.43 | -5.49 -> -5.49 | kept baseline: no local train/test Pareto improvement |
| DIHO/RTTH | 0.00 -> 1.04 | -10.14 -> -8.69 | four-block Pareto improvement |
| MHRM/EAFC | 40.08 -> 41.12 | 11.14 -> 11.71 | train/test Pareto improvement |
| MTNS/ALUT | -4.49 -> -4.49 | 0.00 -> 0.00 | kept baseline: no local train/test Pareto improvement |
| MSDP/ULXY | 0.00 -> 0.00 | 0.00 -> 0.00 | kept baseline: no local train/test Pareto improvement |
| EORC/NGTE | 36.81 -> 36.81 | 60.66 -> 60.66 | kept baseline: no local train/test Pareto improvement |
| ALUT/CCNS | 28.79 -> 28.79 | 27.29 -> 27.29 | kept baseline: no local train/test Pareto improvement |
| SMAH/ILVX | 34.08 -> 36.79 | 34.44 -> 35.22 | four-block Pareto improvement |
| MTNS/BENI | 20.89 -> 20.89 | 30.75 -> 30.75 | kept baseline: no local train/test Pareto improvement |
| CCNS/SRTX | 0.00 -> 0.00 | 0.00 -> 0.00 | kept baseline: no local train/test Pareto improvement |
| FWWG/BLBT | 14.89 -> 14.89 | 15.88 -> 15.88 | kept baseline: no local train/test Pareto improvement |
| ACIX/ITPA | 17.49 -> 17.49 | 7.46 -> 7.46 | kept baseline: no local train/test Pareto improvement |
| ACIX/ULXY | 0.00 -> 0.00 | 0.00 -> 0.00 | kept baseline: no local train/test Pareto improvement |
| EAFC/SRTX | -15.43 -> -15.43 | 0.00 -> 0.00 | kept baseline: no local train/test Pareto improvement |

## Portfolio audit

```json
{
  "baseline": {
    "train": {
      "net": 188911.49443199998,
      "mean": 755.6459777279999,
      "std": 1288.7872140961674,
      "score": 746.9547728882266,
      "sharpe": 9.270585431906065,
      "turnover": 57157755.68000001
    },
    "test": {
      "net": 191014.78156799998,
      "mean": 764.059126272,
      "std": 1218.762718977065,
      "score": 756.3612022232089,
      "sharpe": 9.912376988712206,
      "turnover": 48504684.32
    }
  },
  "tuned_keep_priority": {
    "train": {
      "net": 190329.300805,
      "mean": 761.31720322,
      "std": 1295.7480402592475,
      "score": 752.5968868124224,
      "sharpe": 9.289986591694149,
      "turnover": 57178191.949999996
    },
    "test": {
      "net": 191015.673469,
      "mean": 764.062693876,
      "std": 1219.6334894605714,
      "score": 756.3539123012231,
      "sharpe": 9.90534619084948,
      "turnover": 48467465.309999995
    }
  },
  "tuned_rerank_train": {
    "train": {
      "net": 192357.700139,
      "mean": 769.430800556,
      "std": 1293.2079811466958,
      "score": 760.8337911463802,
      "sharpe": 9.407434330424628,
      "turnover": 56746998.60999999
    },
    "test": {
      "net": 187218.93290900002,
      "mean": 748.8757316360001,
      "std": 1217.0021282042967,
      "score": 741.0474013349381,
      "sharpe": 9.729452979219582,
      "turnover": 48335270.91
    }
  },
  "selected": "baseline"
}
```

## Applied outcome (2026-07-20)

Per-pair Pareto pass found 4 standalone improvements (HUXZ/ACAC, DIHO/RTTH,
MHRM/EAFC, SMAH/ILVX), but the all-in bundle failed the portfolio audit by
0.01 test score (EV-MWIS conflict interactions).  A full subset search over
the 4 changes found the best min-gain Pareto subset: SMAH/ILVX ALONE
(cusum_k 0.45 -> 0.35, z_exit 0.275 -> 0.2975 = the 0.85*entry cap).
Portfolio-level interactions invert several standalone gains (e.g.
DIHO/RTTH is +5.9 train standalone but -2.4 test in the book), which is
why seeds must be audited through the portfolio, never applied standalone.

Applied to daniel_filter.py seeds (verified byte-identical replay):

| book | before (train/test) | after (train/test) |
|---|---|---|
| Daniel core | 746.95 / 756.36 | 749.42 / 757.09 |
| robust_hybrid | 784.48 / 779.08 | 795.63 / 799.59 |

The hybrid gains more than the core (+11.2 / +20.5): the changed atlas
footprint frees/occupies capacity differently for ridge and discovery.
Both windows improve; the two-window Pareto discipline held throughout.

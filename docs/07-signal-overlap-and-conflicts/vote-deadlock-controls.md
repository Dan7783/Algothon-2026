# Vote deadlock control families

> **Source:** `analysis_outputs/vote_deadlock_deepdive/control_report.md` in the team's working repository (relocated here for the public release).

Days 751-1000 are contaminated in-sample data, not a holdout.

## Family results

| Family | Validation delta | Final-window delta | Trials | Critical z |
|---|---:|---:|---:|---:|
| Chow derived | 10.4739 | -55.2291 | 10 | 2.803 |
| Hysteresis derived | -17.7195 | -77.1432 | 30 | 3.140 |
| Joint local search | 43.4067 | -54.9646 | 1 | 1.960 |
| Continuous aim, validation-best | 15.8525 | -160.8838 | 14 | 2.910 |

## 125-day block score deltas

| Candidate | TRAIN_A | TRAIN_B | VAL_A | VAL_B | FINAL_A | FINAL_B |
|---|---:|---:|---:|---:|---:|---:|
| chow_derived | 35.48 | 1.79 | 26.74 | -5.86 | -86.55 | -24.09 |
| chow_validation_best | 18.57 | 1.28 | 48.11 | 8.91 | -63.56 | -33.97 |
| hysteresis_derived | 67.38 | 20.85 | -5.92 | -29.59 | -108.92 | -45.52 |
| hysteresis_validation_best | 10.93 | -6.94 | 17.08 | 29.50 | -13.22 | 13.53 |
| joint_local_search | 72.44 | -63.74 | 135.76 | -49.05 | -134.74 | 26.49 |
| gp_validation_best | 34.67 | -14.00 | 98.71 | -67.08 | -278.04 | -43.15 |

The overall best-of-55 Gaussian lower-bound critical z is **3.314**.
Score deltas are not t-statistics, so this is a warning threshold,
not a claim of corrected significance.

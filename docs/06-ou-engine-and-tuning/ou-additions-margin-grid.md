# Portable additions from merged_friend_ou

> **Source:** `analysis_outputs/four_sigmas_ou_additions/report_margin_grid.md` in the team's working repository (relocated here for the public release).

Only days 1-750 are loaded. No exposed result is available here.

All values below are score deltas from the unchanged current book.

| variant | TRAIN_A | TRAIN_B | TRAIN | VALIDATION_A | VALIDATION_B | VALIDATION | all_four_halves_positive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| margin_0.145 | 3.13 | 23.27 | 13.22 | 0.15 | 14.76 | 7.46 | True |
| margin_0.15 | 3.13 | 23.27 | 13.22 | 0.15 | 14.76 | 7.46 | True |
| margin_0.16 | 3.13 | 23.27 | 13.22 | 0.15 | 14.76 | 7.46 | True |
| margin_0.25 | 16.52 | 16.62 | 16.51 | 8.11 | 1.79 | 4.96 | True |
| margin_0.201 | 12.21 | 16.62 | 14.36 | 8.11 | 1.79 | 4.96 | True |
| margin_0.18 | 9.53 | 23.27 | 16.40 | -3.20 | 14.76 | 5.79 | False |
| margin_0.199 | 9.53 | 23.27 | 16.40 | -3.20 | 14.76 | 5.79 | False |
| margin_0.10 | 2.51 | 0.00 | 1.25 | 0.00 | 0.00 | 0.00 | False |
| baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | False |
| margin_0.14 | 0.59 | 8.53 | 4.57 | -6.60 | 1.45 | -2.58 | False |

A one-at-a-time OU replacement is considered structurally consistent
only if both train halves and both validation halves improve.

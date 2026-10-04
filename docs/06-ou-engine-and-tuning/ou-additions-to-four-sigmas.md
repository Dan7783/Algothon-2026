# Portable additions from merged_friend_ou

> **Source:** `analysis_outputs/four_sigmas_ou_additions/report.md` in the team's working repository (relocated here for the public release).

Only days 1-750 are loaded. No exposed result is available here.

All values below are score deltas from the unchanged current book.

| variant | TRAIN_A | TRAIN_B | TRAIN | VALIDATION_A | VALIDATION_B | VALIDATION | all_four_halves_positive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| margin_0.15 | 3.13 | 23.27 | 13.22 | 0.15 | 14.76 | 7.46 | True |
| margin_0.10 | 2.51 | 0.00 | 1.25 | 0.00 | 0.00 | 0.00 | False |
| baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | False |
| margin_0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | False |
| flip_to_flat | 113.03 | 25.00 | 68.80 | 57.79 | -66.32 | -4.22 | False |
| magnitude_weighted | 12.61 | -44.57 | -16.02 | 4.45 | -54.15 | -24.83 | False |
| all_tuned_ou | -80.70 | -190.56 | -135.95 | -322.24 | -234.14 | -278.09 | False |
| winner_default_ou | -165.08 | -334.88 | -250.28 | -304.45 | -449.57 | -376.98 | False |
| all_default_ou | -381.31 | -377.01 | -378.28 | -419.04 | -453.49 | -436.00 | False |

A one-at-a-time OU replacement is considered structurally consistent
only if both train halves and both validation halves improve.

# Vote deadlock mechanism audit

> **Source:** `analysis_outputs/vote_deadlock_deepdive/mechanism_report.md` in the team's working repository (relocated here for the public release).

- Configured mode value: `equal`
- Effective mode: **equal**
- Duplicate post-merge profiles: **0 groups**

## Failure modes

| Period | Exact ties | Near-tie flips | Flip commission | Persistent-flat days | Persistent oracle movement |
|---|---:|---:|---:|---:|---:|
| YEAR1 | 622 | 0 | \$0.00 | 430 | \$74518.43 |
| YEAR2 | 446 | 7 | \$14.00 | 126 | \$21334.46 |
| VALIDATION | 413 | 9 | \$18.00 | 118 | \$18018.08 |
| FINAL_IN_SAMPLE | 379 | 6 | \$12.00 | 93 | \$12735.86 |

## Pair-leg completion

| Period | Both legs | Exactly one | Neither | Any flat leg |
|---|---:|---:|---:|---:|
| YEAR1 | 67.0% | 30.1% | 2.9% | 16.2% |
| YEAR2 | 57.3% | 37.7% | 5.0% | 9.9% |
| VALIDATION | 57.0% | 37.9% | 5.1% | 10.0% |
| FINAL_IN_SAMPLE | 55.6% | 39.0% | 5.4% | 8.9% |

The final window is deliberately not described as out-of-sample:
`PREVIOUS_OOS_REMOVED_PAIR_NAMES` and the current catalog encode
information influenced by days 751-1000.

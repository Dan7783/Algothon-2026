# Vote deadlock resolution: final research report

> **Source:** `analysis_outputs/vote_deadlock_deepdive/final_report.md` in the team's working repository (relocated here for the public release).

## Protocol and provenance

The current catalog has no clean local holdout. Days 31-500 estimate fitted
quantities, days 501-750 are validation, and days 751-1000 are reported as a
contaminated final in-sample window. `PREVIOUS_OOS_REMOVED_PAIR_NAMES` has no
runtime effect, but the catalog itself was influenced by the final window.

The file's former mode value, `"we love the sussy q"`, entered the equal-vote
branch because only the exact value `"priority"` selects priority weighting.
It is now written explicitly as `"equal"` and invalid values raise an error.
This changes no positions.

## Tasks 1-3: mechanism

### Exact ties

No duplicate post-merge parameter profiles survive. Exact ties are created by
equal voting, not duplicate filters.

| Period | Overlap asset-days | Exact ties | Tie share | Hindsight movement |
|---|---:|---:|---:|---:|
| YEAR1 | 4,402 | 622 | 14.13% | \$113,094 |
| YEAR2 | 5,396 | 446 | 8.27% | \$79,119 |
| Validation | 5,361 | 413 | 7.70% | \$73,960 |
| Final in-sample | 5,369 | 379 | 7.06% | \$63,997 |

The movement column is not recoverable PnL. The causal OU-moneyness resolver
lost \$3,148 in validation and \$4,049 in the final window.

### Near-tie chatter

The brief's expectation that chatter dominates transaction costs is rejected.
At a fixed `|combined_vote| <= 0.10` definition, validation contains nine
opposite-sign flips costing about \$18. The final window contains six, costing
about \$12.

### Persistent flat states

Validation has 36 persistent-flat runs (118 asset-days) containing \$18,018 of
hindsight movement. The final window has 26 runs (93 asset-days) containing
\$12,736. Persistent flatness dominates chatter economically, but no tested
causal direction resolver captures it.

### Adaptation desynchronisation

The hypothesis is rejected. In validation, an exact tie occurs on 6.35% of
mixed-adaptation overlaps versus 9.25% of homogeneous overlaps
(risk ratio 0.69; correlation -0.05). Mixed recent-cut states are similarly
uninformative. Suppressing adapting votes loses 306 validation score points;
suppressing every pair touching a mixed-state asset loses 373.

### Spreads or signs?

The realised strategy trades asset signs:

| Period | Both intended legs | Exactly one leg | Neither |
|---|---:|---:|---:|
| Validation | 57.0% | 37.9% | 5.1% |
| Final in-sample | 55.6% | 39.0% | 5.4% |

In validation the intended pair book's median absolute dollar imbalance is
2.60% of gross versus 13.04% realised. Median absolute market-beta exposure is
7.03% intended versus 12.89% realised. Joint leg consistency is therefore an
alternative objective, not a description of the incumbent.

## Tasks 4-7: control families

### Chow reject option

Train calibration gives:

- mean absolute one-day movement: \$179.48 per \$10,000 position;
- round-trip commission: \$2.00;
- required correctness probability: 50.56%;
- derived raw consensus threshold: `tau = 0.17178`.

The derived rule gains 10.47 validation score, then loses 55.23 in the
contaminated final window. Validation volume remains approximately \$66 million,
far above the \$25,000 activity floor.

### Vote hysteresis

The derived `enter = 0.17178`, `exit = 0.08589` rule loses 17.72 validation and
77.14 final-window score. The best of 30 validation configurations gains
23.28 validation and is flat in aggregate in the final window, but it fails
the train-block and selection-bias tests.

### Structural balance and joint resolution

Using train-window beta signs, 659 cycles of length at most six are present and
188 are frustrated. The most frequent participants are `RTTH-ULXY`,
`ULXY-FCSG`, `SPLZ-ULXY`, `NGTE-FCSG`, and `SMAH-ULXY`; participation is also
strongly related to graph degree and is not a safe removal criterion by itself.

The joint local search:

- runs in about 2.2 seconds for the complete history;
- raises complete pair agreement from 58.86% to 70.05%;
- eliminates active-source flat positions;
- gains 43.41 validation score;
- loses 54.96 in the final window.

Its 125-day deltas alternate sharply: `+72.44, -63.74, +135.76, -49.05,
-134.74, +26.49`. It is structurally effective but economically unstable.

### Persistence/cost-aware continuous aim

Fourteen Gârleanu-Pedersen-inspired scale/adjustment variants were tested. The
best validation variant gains 15.85 but loses 160.88 in the final window.
Continuous sizing and partial adjustment do not survive the cap-dominated
competition objective.

## Selection accounting

The study evaluates 10 abstention, 30 hysteresis, one joint, and 14 continuous
aim configurations: 55 effective trials. Gaussian best-of-N lower-bound
critical z-values are 2.80, 3.14, 1.96, and 2.91 by family, and 3.31 overall.
Score deltas are not t-statistics, and no clean holdout exists, so none of the
positive validation results support a significance claim.

## Decision

Do not install Tasks 4-7 into production. Equal-vote abstention is currently
doing useful work, the apparent opportunity in flat states cannot be assigned
a reliable direction, and every more ambitious resolver is regime-unstable.
The only production edits are diagnostic state exposure and making the
already-effective equal mode explicit with validation.

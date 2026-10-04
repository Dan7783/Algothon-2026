# Is the book test-heavy because the train window under-trades?

> **Source:** `cusum_output/daniel_pairs/train_test_balance.md` in the team's working repository (relocated here for the public release).

Investigated with the pair_seed_explorer simulator (exact `PairAgent`
replica) across all seeded pairs.  Answer: **no** — the pairs sleeve is
train-heavy, participation is already balanced, and forcing more train
trading destroys the test window.

## 1. Participation is already equal (pair-days, 21 pairs x 250 days)

| window | in market | ADF-blocked | kappa-blocked | flat (\|z\| < entry) |
|---|---|---|---|---|
| train | 3386 (64.5%) | 1433 (27.3%) | 0 | 431 |
| test | 3415 (65.0%) | 1338 (25.5%) | 64 | 491 |

Test trades just +29 pair-days more than train (+0.9%).  The two windows
are not meaningfully different in how often the book is on risk.

## 2. The PAIRS sleeve is train-heavy; RIDGE makes the book test-heavy

Cumulative replay of robust_hybrid, both windows:

| cumulative book | train | test | test - train |
|---|---|---|---|
| atlas (pairs only) | 576.52 | 496.29 | **-80.23** |
| + extra lead-lag | 652.11 | 609.58 | -42.53 |
| + ridge | 770.70 | 799.41 | **+28.70** |
| + discovery | 801.24 | 831.12 | +29.88 |

The pairs sleeve earns **+80 MORE in train than in test**.  Ridge adds
+119.8/day mean in train but +191.3/day in test — that single sleeve flips
the book from train-ahead to test-ahead.  The asymmetry the user noticed is
a ridge property, not a pairs-participation problem.

Per-pair confirmation (standalone scores, train vs test): RTTH/NAYO 82 vs
46, CTGI/EELT 94 vs 48, HETT/ULXY 81 vs 47, HRET/CTGI 43 vs 20, MHRM/EAFC
40 vs 11, NAYO/SRTX 44 vs -4.  Only EORC/NGTE (37 vs 61) and MTNS/BENI
(21 vs 31) are genuinely test-favoured.

## 3. Making train trade more: possible, but it costs the test window

Seven pairs are heavily ADF-blocked in train (DIHO/RTTH 219/250 days,
MTNS/ALUT 223, EAFC/SRTX 187, MDGI/HTRK 124, ALUT/CCNS 122) and two
(MSDP/ULXY, ACIX/ULXY) are blocked 100% of BOTH windows, contributing
exactly zero.  Dropping `require_adf` on them looks excellent standalone —
all seven improve BOTH windows, e.g. MSDP/ULXY 0.00/0.00 -> 38.21/15.63,
ACIX/ULXY 0.00/0.00 -> 17.42/27.61, MTNS/ALUT -4.49/0.00 -> 40.16/12.18 —
and lifts train participation from ~0-60 days to ~235-245.

At portfolio level it collapses:

| config | core train | core test | hybrid train | hybrid test |
|---|---|---|---|---|
| baseline | 770.70 | 799.41 | 806.31 | 846.48 |
| all 7 no_adf | 775.58 (+4.88) | 739.33 (**-60.08**) | 816.81 (+10.50) | 746.30 (**-100.18**) |

Leave-one-out and singles were also tested: **no subset improves both
windows**, and greedy forward selection returns the empty set.  The newly
unblocked pairs win instrument conflicts in the EV-MWIS layer and displace
better pairs plus the ridge/discovery sleeves that were using that
capacity.  Train gains ~+5, test loses ~-60.

## Conclusion

Train does trade slightly less and can be made to trade much more, but
every route to that is negative-sum for the book.  If the goal is a more
balanced train/test profile, the lever is the **ridge sleeve** (the actual
source of the +191 vs +120 asymmetry), not pair participation.  NOTHING
WAS APPLIED from this investigation.

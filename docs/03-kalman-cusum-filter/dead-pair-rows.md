# "Why are 21 pairs hardcoded when only 14 are active?"

> **Source:** `cusum_output/DEAD_PAIR_ROWS.md` in the team's working repository (relocated here for the public release).

Short answer: there was no good reason, and the 7 disabled rows were not
inert — they were silently steering the conflict resolver.

## What a leg_scale=0 row still did

`_hub_penalty` derived each pair's EV divisor from the degree of its two
instruments **across the whole seed list**, disabled rows included.  So the
seven dead pairs inflated the divisor of every active pair that shared an
instrument with them:

| active pair | penalty without dead rows | with them | effective EV cut |
|---|---|---|---|
| ALUT/CCNS | 1.00 | 2.00 | -50% |
| HETT/ULXY | 1.00 | 1.73 | -42% |
| NAYO/SRTX | 1.41 | 2.45 | -42% |
| CTGI/EELT | 1.00 | 1.41 | -29% |
| MHRM/EAFC | 1.00 | 1.41 | -29% |
| MTNS/BENI | 1.00 | 1.41 | -29% |
| ACIX/ITPA | 1.00 | 1.41 | -29% |

They also ran a Kalman filter every day and produced nothing.

## Why they could not simply be deleted

The pair EVs were tuned WITH those inflated penalties in place, so removing
the rows re-ranks every instrument conflict away from the calibration:

| | train blocks | held-out blocks | floor |
|---|---|---|---|
| rows kept (leg_scale=0) | 868.1 / 875.5 | 765.9 / 946.6 | 765.9 |
| rows deleted | 868.6 / 877.7 | 710.4 / 931.7 | **710.4** |

That is a hidden coupling, not a feature: the seed list was doing two
unrelated jobs (which pairs trade, and how EV is normalised).

## Fix applied

Rows deleted; `HUB_PENALTY_BY_NAME` freezes the penalties that were actually
in force, with the derivation kept as a fallback for pairs added later.

| | train blocks | held-out | floor |
|---|---|---|---|
| rows deleted, hub frozen | 866.6 / 877.3 | 756.3 / 935.1 | 756.3 |

Recovers ~46 of the 55 points deletion cost.  On the TRAIN blocks — the only
legitimate basis for this decision — the change is neutral (median 871.8 ->
872.0), so it is kept: same behaviour, 14 honest rows instead of 21, and the
EV normalisation is now explicit instead of an accident of the seed list.

eval.py: train 871.97, test 845.61.

Residual ~10 points of held-out floor versus the dead-row version is not
chased, because doing so would mean selecting on the held-out window again.

Backups: /tmp/four_sigmas.pre_slim.py, /tmp/four_sigmas.pre_delete.py

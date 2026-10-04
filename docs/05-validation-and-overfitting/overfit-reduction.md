# Reducing what can be overfitted (prompted by the 599 OOS print)

> **Source:** `cusum_output/OVERFIT_REDUCTION.md` in the team's working repository (relocated here for the public release).

True OOS = 599 against 878 on our 501-750 "test" window: a 32% haircut.  That
window had been used for selection all session, so it was never a holdout.

Rolling 125-day blocks INSIDE 251-750 cannot detect this — four_sigmas scores
838 / 869 / 765 / 992 there, which looks stable, because every one of those
blocks informed the fitting.  The lever is therefore not a better search; it
is having less to search over.

## Where the overfitting lives

four_sigmas carries ~235 fitted numbers, 210 of them PER PAIR:
EV, z_entry, z_exit, z_stop, q_beta, h, cusum_k, adapt_bars, snap_overshoot,
leg_scale, x 21 pairs.

## Which of them actually earn anything

Progressively replacing per-pair values with one shared set (thresholds
chosen on the TRAIN blocks only; filter values are the modal seed values):

| variant | blocks | median | worst | fitted |
|---|---|---|---|---|
| per-pair fitted | 838/869/765/992 | 853 | 765 | ~210 |
| shared trade | 783/796/772/895 | 790 | **772** | ~150 |
| **shared trade + filter** | 763/796/765/892 | 781 | **763** | **~50** |
| + flattened EV | 682/713/689/847 | 701 | 682 | ~29 |

* Thresholds and filter knobs are NOT load-bearing: sharing both costs 2
  points of floor (765 -> 763) for a 4x cut in fitted numbers.
* EV / priority IS load-bearing: flattening it costs 80 points of floor.  It
  is not a threshold — it decides which pair wins an instrument conflict.
* The extra median the per-pair version shows (853 vs 781) lives in the two
  blocks it was fitted on; its FLOOR is no better.

## Stripping further does NOT pay

| strategy | fitted | block median | worst |
|---|---|---|---|
| four_sigmas (per-pair) | ~235 | 853 | 765 |
| shared_config | ~50 | 781 | 763 |
| simple_pairs (6 pairs, ridge) | ~10 | 479 | 383 |
| lean_pairs (6 strict pairs, ridge) | ~12 | 384 | 331 |

Cutting down to a handful of structurally-screened pairs costs far more than
overfitting ever did: 599 real OOS still beats simple_pairs' ~480 in-sample.
The sleeves and the pair count are genuine diversification; the per-pair
parameter fitting is the part that was not earning.

## Shipped

`shared_config.py` — four_sigmas' full structure (all pairs, all sleeves,
filter, conflict resolver) with ONE shared trade+filter config and per-pair
EV retained.  eval.py: train 779.15, test 826.80.

Note those two numbers are lower than four_sigmas' 853/879 by construction —
the difference is precisely the fitted component we chose to give up.  The
relevant comparison is the block FLOOR (763 vs 765) and the fitted count
(50 vs 235).

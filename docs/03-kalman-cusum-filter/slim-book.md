# Slimming the config surface

> **Source:** `cusum_output/SLIM_BOOK.md` in the team's working repository (relocated here for the public release).

## What was actually in the configs

`FiltCfg` had 20 fields, `TradeCfg` 9 — nominally per pair, 21 pairs.
Measured against the live seeds:

| finding | count |
|---|---|
| fields holding the SAME value for every pair | **18 of 29** |
| fields that vary but cost only 2 pts of floor when shared | 8 |
| fields genuinely needing to be per-pair | 3 (EV, require_adf, leg_scale) |

So 18 fields were module constants wearing a config costume, and of the 11
that varied, 8 were noise.

## Dead code found

`vol_cut` is False on every pair, so the entire **variance-CUSUM** branch —
`cv_up`, `cv_dn`, `run_v_up/dn`, `d_v`, `vol_alarm`, `vol_shift` — accumulates,
resets, and changes nothing.  Also unused: `adapt_extend`, `cut_confirm_bars`,
`adapt_p_floor`, `snap_max_rel`, `core`, `beta_mode`, `trade_while_adapting`.

Magic numbers that were buried inline are now named: `DISAGREE_ABS` (0.025),
`DISAGREE_REL` (0.03), `DRIFT_MIN` (8), `SEG_EWMA` (0.1), `H_FLOOR` (2.0).

## Three options, measured on 125-day blocks

| book | lines | params | blocks | median | **floor** |
|---|---|---|---|---|---|
| four_sigmas | 1091 | ~235 | 838/869/765/992 | 853 | 765 |
| shared_config (wrapper) | +70 | ~50 | 763/796/765/892 | 781 | 763 |
| **slim_book (rewrite)** | **445** | **~60** | 738/785/720/895 | 762 | **720** |

eval.py: slim_book train 761.78, test 805.15.

## Reading

* `shared_config` is the safest cut: SAME code as four_sigmas, 4.7x fewer
  fitted numbers, floor unchanged (763 vs 765).  Pure overfit reduction.
* `slim_book` additionally removes ~650 lines, the config classes entirely
  (everything is a module constant), and all the dead code — at a cost of
  43 points of floor (-5.6%).
* The remaining gap is implementation drift in my rewrite (lead-lag gating
  and ridge residual-capacity differ slightly from the original), not a
  fundamental limit.  It could be closed with more care.

## Recommendation

If the priority is minimum overfit for maximum score: `shared_config`.
If the priority is a codebase you can actually read and reason about:
`slim_book` — 60% fewer lines, no config classes, no dead branches, every
magic number named, for ~6% of floor.

# Applied to four_sigmas (2026-07-21): prune + config slim, per-pair KEPT

Per the plan: remove bad pairs, delete dead parameters, keep per-pair configs.

## 1. Pair prune (verdict from TRAIN blocks only)

Leave-one-out on the 15 active pairs, judged on blocks 251-375 / 376-500;
later blocks shown only as an after-the-fact check.  13 of 15 earn.  Two
flagged:

| pair | train marg | blk3 | blk4 | persistent-cointegration? |
|---|---|---|---|---|
| HRET/CTGI | -18.63 | -1.1 | +45.8 | no |
| AENO/NWIG | -1.96 | +25.5 | +42.3 | YES (ADF -4.35/-4.10) |

Structure used as tiebreaker rather than letting a -1.96 train reading
delete a structurally-real pair: AENO/NWIG kept, HRET/CTGI pruned.
Active pairs: 15 -> 14.

## 2. Config slim (behaviour-preserving)

| class | before | after |
|---|---|---|
| FiltCfg | 20 fields | **5** (q_beta, cusum_k, h, adapt_bars, snap_overshoot) |
| TradeCfg | 9 fields | **5** (z_entry, z_exit, z_stop, leg_scale, require_adf) |
| per-pair tunables | 29 | **11** (incl. EV) |

Demoted to module constants (identical value on all 21 seeds, so never
per-pair parameters): q_beta_adapt, ewma_lambda, warmup, auto_h, target_seg,
h_gain, min_seg_len, cov_inflate, snap_window, adapt_chase, adapt_p_floor,
require_kappa, trade_while_adapting.

Deleted outright as dead: vol_cut (False everywhere, so the entire variance
CUSUM branch was unreachable), adapt_extend, cut_confirm_bars, snap_max_rel,
core, beta_mode.

## Result

| | blocks | median | floor |
|---|---|---|---|
| before | 838 / 869 / 765 / 992 | 853.2 | 764.8 |
| **after** | 868 / 876 / 766 / 947 | **871.8** | **765.9** |

eval.py: **train 871.80, test 856.22.**

Train blocks (the ones selection used) improved +30 and +7; the floor is
unchanged; block 4 gave back 45 of the 992 it had before, which is expected
since nothing was selected on it any more.  Per-pair configs are intact and
ready for step 3 (tuning the survivors on train blocks only).

Backup: /tmp/four_sigmas.pre_slim.py

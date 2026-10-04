# Step 3: does per-pair tuning survive honest validation?  No.

> **Source:** `cusum_output/STEP3_PER_PAIR_VERDICT.md` in the team's working repository (relocated here for the public release).

## Protocol

14 surviving pairs, ~486 configs each over q_beta, cusum_k, h, z_entry,
z_exit, z_stop.  Selection touched ONLY blocks 251-375 and 376-500.  Three
guards, all stricter than anything used earlier in this project:

1. a candidate must improve BOTH train blocks independently (winning one and
   losing the other is noise; averaging would hide it);
2. it must also improve the neighbourhood-SMOOTHED train score, so an
   isolated grid spike cannot win;
3. survivors are portfolio-confirmed greedily on the train blocks, because
   per-pair gains do not add through the EV-MWIS conflict layer.

Blocks 501-750 were not read until after the configuration was frozen.

## What passed

| pair | train blocks | candidates passing |
|---|---|---|
| RTTH/NAYO, HETT/ULXY, NAYO/SRTX, AENO/NWIG, MHRM/EAFC, EORC/NGTE, SMAH/ILVX, FWWG/BLBT, ACIX/ITPA | — | **0 of 486** |
| CTGI/EELT | 94.2/98.7 -> 94.5/105.6 | 14 |
| HUXZ/ACAC | 49.1/44.7 -> 51.6/46.9 | 36 |
| DIHO/RTTH | 18.5/26.5 -> 25.3/41.9 | 102 |
| ALUT/CCNS | 1.7/60.6 -> 4.2/61.5 | 21 |
| MTNS/BENI | 15.3/30.3 -> 18.4/33.1 | 72 |

9 of 14 pairs had ZERO admissible candidates — the robustness filters did
most of the work.  Greedy portfolio confirmation on train then kept three:
HUXZ/ACAC, MTNS/BENI, ALUT/CCNS.

## The verdict

| | train blocks | HELD-OUT blocks |
|---|---|---|
| baseline | 868.1 / 875.5 | 765.9 / 946.6 |
| train-tuned | 884.8 / 886.8 | 738.5 / 943.1 |
| delta | **+14.0** | **-15.5** |

Per-pair tuning bought +14 on the windows it was selected on and gave back
-15.5 on the windows it had never seen.  The train gain is noise.

**NOT APPLIED.**  The seeds are unchanged.

## Why this matters beyond this experiment

This is a clean, prospective test of the method that produced the 878 vs 599
gap — and it reproduces that failure in miniature under far stricter rules.
Every earlier per-pair tuning result in this project (including the AENO/NWIG
retune, which was selected with test-window data in view) should be treated
as unsupported.

What HAS survived honest validation, and is worth keeping:
* structural pair selection (cointegration persistence across sub-windows);
* removing pairs that do not earn their slot;
* deleting dead parameters and dead code;
* sharing or hardcoding knobs that do not vary meaningfully.

What has not: choosing per-pair parameter values by score.

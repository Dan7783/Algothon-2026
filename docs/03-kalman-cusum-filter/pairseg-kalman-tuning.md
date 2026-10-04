# PairSegFilter Kalman tuning (atlas pairs, train days 1-500)

> **Source:** `cusum_output/pair_seg_tuning/tuning_report.md` in the team's working repository (relocated here for the public release).

Input: log prices.  Loss: causal prequential Gaussian score of the frozen-beta hedge error at horizons [1, 10, 20] days (burn-in 60; block-20 standard errors).  One-step levels loss is degenerate here — beta can absorb the stationary spread, and every pair pins the fastest grid speed — so the beta published at t-h must hedge the move from t-h to t.  The one-SE rule prefers sticky q in stage 1 and few cuts in stage 2.

## Stage 1 — Kalman speed per pair (segmentation off)

| pair | q_beta | lambda | loss | min loss | SE | default loss |
|---|---|---|---|---|---|---|
| SMAH/ILVX | 0.000e+00 | 0.99 | -2.7244 | -2.7270 | 0.0366 | -2.7264 |
| AENO/NWIG | 0.000e+00 | 0.99 | -2.6517 | -2.6683 | 0.0426 | -2.6663 |
| EORC/NGTE | 0.000e+00 | 0.98 | -2.5214 | -2.5461 | 0.0423 | -2.5288 |
| HUXZ/ACAC | 0.000e+00 | 0.98 | -2.4558 | -2.5058 | 0.0500 | -2.4688 |
| CTGI/EELT | 0.000e+00 | 0.98 | -2.0465 | -2.0824 | 0.0497 | -2.0586 |
| MHRM/EAFC | 0.000e+00 | 0.99 | -2.6645 | -2.6945 | 0.0501 | -2.6793 |
| HETT/ULXY | 0.000e+00 | 0.98 | -2.2153 | -2.2400 | 0.0392 | -2.2217 |
| ACIX/ITPA | 0.000e+00 | 0.99 | -2.9557 | -2.9835 | 0.0466 | -2.9728 |
| FWWG/BLBT | 0.000e+00 | 0.99 | -2.9209 | -2.9700 | 0.0519 | -2.9493 |
| RTTH/NAYO | 0.000e+00 | 0.99 | -1.9590 | -2.0127 | 0.0571 | -1.9762 |
| MTNS/ALUT | 0.000e+00 | 0.98 | -2.2733 | -2.3235 | 0.0620 | -2.2876 |
| NAYO/SRTX | 0.000e+00 | 0.96 | -2.1588 | -2.1979 | 0.0471 | -2.1494 |
| ALUT/CCNS | 0.000e+00 | 0.99 | -2.4684 | -2.5007 | 0.0464 | -2.4823 |

## Stage 2 — global segmentation policy (speeds frozen)

- cusum_k `1.0`, h `6.0`, target_seg `60`, min_seg_len `60`, snap_overshoot `1.0`
- loss with cuts `-2.4621` vs no-cut `-2.4628` (grid best `-2.4636`, SE `0.0228`, 162 of 162 combos within one SE)
- total cuts `6`

| pair | cuts | loss with cuts | loss no cuts | cut days |
|---|---|---|---|---|
| SMAH/ILVX | 0 | -2.7244 | -2.7244 | [] |
| AENO/NWIG | 0 | -2.6517 | -2.6517 | [] |
| EORC/NGTE | 0 | -2.5214 | -2.5214 | [] |
| HUXZ/ACAC | 0 | -2.4558 | -2.4558 | [] |
| CTGI/EELT | 0 | -2.0465 | -2.0465 | [] |
| MHRM/EAFC | 1 | -2.6648 | -2.6645 | [485] |
| HETT/ULXY | 0 | -2.2153 | -2.2153 | [] |
| ACIX/ITPA | 0 | -2.9557 | -2.9557 | [] |
| FWWG/BLBT | 0 | -2.9209 | -2.9209 | [] |
| RTTH/NAYO | 1 | -1.9614 | -1.9590 | [163] |
| MTNS/ALUT | 2 | -2.2603 | -2.2733 | [161, 487] |
| NAYO/SRTX | 0 | -2.1588 | -2.1588 | [] |
| ALUT/CCNS | 2 | -2.4697 | -2.4684 | [210, 258] |

Artifacts: `tuned_pair_seg.json`, `*_tuned_beta.png`, `*_default_vs_tuned.png`.
## OOS eval check (2026-07-19)

Tuned per-pair configs were wired into `second_adaptive_pairs.py` and scored
with `eval.py` against the untuned defaults:

| window | tuned | untuned defaults |
|---|---|---|
| train 251-500 | score 57.01 (mean \$62.0, Sharpe 3.39) | score 40.53 (mean \$46.3, Sharpe 2.64) |
| test 501-750 | score 38.92 (mean \$43.2, Sharpe 3.02) | score 45.60 (mean \$49.4, Sharpe 3.46) |

Paired daily-PnL test on the OOS window: mean diff -\$6.2/day, paired
t = -1.10, p = 0.274 (59/250 days differ; the gap is one divergent episode
around days 676-682).  The train-window edge overlaps the tuning window and
is discounted.  Conclusion: tuned knobs are statistically indistinguishable
from the Pine defaults, consistent with the flat in-sample loss surface, so
`second_adaptive_pairs.py` was left on the defaults.  The tuned values remain
in `tuned_pair_seg.json`.

---

## Figures

![SMAH/ILVX: sticky through-origin beta with burst adaptation](../figures/smah-ilvx-sticky-beta.png)

*SMAH/ILVX: sticky through-origin beta with burst adaptation.*

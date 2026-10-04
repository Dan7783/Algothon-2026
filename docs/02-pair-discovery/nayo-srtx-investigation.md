# NAYO/SRTX: validation, rigorous tune, and the beta-modelling question

> **Source:** `cusum_output/daniel_pairs/nayo_srtx_investigation.md` in the team's working repository (relocated here for the public release).

Live seed: train +44.35 / test -4.21 — the worst test performer among the
otherwise-healthy pairs.  **Outcome: no change applied; the live seed stands.**

## 1. The candidate was validated and REJECTED

Best robust candidate from an 87,480-point sweep (q_beta, h, cusum_k,
min_seg_len, adapt_bars x z_entry/z_exit/z_stop x require_adf), passing
Pareto + four-block + smoothed-neighbourhood robustness:

  q_beta 3e-6, z_stop 3.5 -> 4.5, require_adf False
  standalone: train +44.35 -> +45.83, test -4.21 -> +0.47

Portfolio confirmation:

| book | baseline | tuned | delta |
|---|---|---|---|
| daniel core | 793.71 / 800.99 | 793.71 / 790.00 | +0.00 / **-10.99** |
| robust_hybrid | 819.49 / 852.67 | 813.98 / 820.86 | -5.50 / **-31.81** |

REJECTED.  The gain came from `require_adf=False`, which is the
capacity-rationing mechanism for the whole book (see adf_gate_necessity.md).

## 2. With the ADF gate KEPT, no improvement exists at all

Same grid, `require_adf` fixed True: **10 of ~43,740 candidates pass Pareto,
0 pass four-block, 0 pass robustness.**  There is no configuration of this
pair that legitimately improves it.

## 3. The beta is not mis-modelled — the premise was wrong

The diagnosis showed the filter beta drifting only +0.007 across train while
rolling OLS-120 swings -0.74 to +0.88, implying 0.65 "tracking error".  That
is a definitional artifact, not a modelling failure:

`PairSegFilter` is **through-origin** (alpha frozen at 0).  As q_beta -> inf
the Kalman gain g -> 1/x, so beta -> **y/x, the log-price ratio**.  Verified:

| q_beta | filter beta range (test) | tracking err | train | test |
|---|---|---|---|---|
| 9.1e-07 (seed) | 0.506 - 0.629 | 0.650 | 44.35 | -4.21 |
| 1e-4 | 0.531 - 0.649 | 0.656 | 42.50 | -3.03 |
| 1e-2 | 0.509 - 0.665 | 0.657 | 44.35 | -0.11 |
| 1.0 | 0.506 - 0.670 | 0.658 | 44.35 | -3.17 |
| 10.0 | 0.506 - 0.670 | 0.658 | 44.35 | -3.17 |

log-price ratio y/x over test = 0.506 - 0.670 — an exact match.  The filter
**cannot** produce -0.74; that number comes from a demeaned regression, a
different quantity.  Eight orders of magnitude of q_beta move the beta range
by 0.04 and never fix test.

Across all 87,480 candidates: **corr(beta tracking error, test score) =
-0.052** — beta modelling does not drive this pair's test result.

## 4. Even a genuine modelling change does not rescue it

Rebuilding the spread with an intercept (rolling OLS-120 with alpha, a real
change to how beta is modelled) versus through-origin, same band policy:

| spread model | ADF train | ADF test | score train | score test |
|---|---|---|---|---|
| through-origin | -2.97 | -1.29 | +27.49 | -19.20 |
| with intercept | -2.99 | **-2.42** | +21.03 | **-10.99** |

The intercept genuinely improves test stationarity (-1.29 -> -2.42) but the
spread is still non-stationary at the -2.9 gate and the window is still
negative, while train gets worse.

## Conclusion

NAYO/SRTX's test weakness is a property of the RELATIONSHIP (residual sd
0.099 -> 0.140, ADF -3.63 -> -3.28, 102 gate-blocked days), not of the hedge
ratio's specification or the trade thresholds.  Tuning cannot fix it, better
beta modelling cannot fix it, and the only configuration that looked better
standalone damages the book.  The correct action is to keep the seed and let
the ADF gate stand this pair down in test — which is exactly what it does.

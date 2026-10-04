# Bare-bones rebuild, and an honest accounting of the score-mining

> **Source:** `cusum_output/SIMPLE_VS_COMPLEX.md` in the team's working repository (relocated here for the public release).

## The overfitting critique is correct

Decisions in this session were driven by "does the score go up on train and
test".  Specifically:

* the windowed-ADF window (375) and threshold (-2.4) were chosen from a
  20-cell grid scored on BOTH windows;
* the 6-pair prune set was chosen by searching combinations against BOTH
  windows (the individually-worst pairs were NOT the best set to remove);
* every per-pair "robustness" check tested whether the SCORE was stable
  across a parameter neighbourhood.  That detects fragile fits, but it does
  not undo the fact that test data informed the choice.

Test has been a selection input, not a holdout, for most of this work.  The
reported +30s therefore carry an unknown amount of mining.  The structural
findings are the parts that do not depend on score:

* only 7 of 1225 pairs hold residual ADF < -3 in BOTH halves of the sample;
* the filter is through-origin, so beta == log-price ratio and q_beta is
  inert across 8 orders of magnitude;
* 3 pairs contributed exactly 0.00 and 13 could never reach their entry
  threshold — those are facts about the code, not about PnL.

## simple_pairs.py — 161 lines vs 1091

Every parameter derived from structure, none from score:

| choice | basis |
|---|---|
| 6 pairs | residual ADF < -3 in both halves (7 qualify; SRNA/ACIX dropped: borderline -2.85 train, 22d half-life vs ~5d) |
| no conflict logic | the 6 pairs use 12 DISTINCT instruments — disjoint by construction |
| FIT_WINDOW 250 | same window as the cointegration screen |
| Z_WINDOW 60 | ~10 measured half-lives (median 5.7d) |
| entry 1.0 / exit 0.0 / stop 3.0 | textbook OU levels |

Dropped entirely: sticky Kalman, CUSUM change detection, adapt bursts, snap
overshoot, EV-MWIS + hub penalties, lead-lag health tests, 4-sleeve routing.

## Results (official eval.py, scored ONCE)

| book | lines | train | test | train/test |
|---|---|---|---|---|
| simple_pairs | 161 | 479.17 | 500.04 | 0.96 |
| four_sigmas | 1091 | 853.18 | 878.57 | 0.97 |

Sleeve decomposition of the simple book (near-additive, so the two signals
are genuinely complementary):

| sleeve | train | test |
|---|---|---|
| 6 structural pairs | 248.02 | 192.57 |
| ridge VAR | 226.30 | 320.78 |
| both | 479.17 | 500.04 |

## Interpretation

The complex book scores ~1.75x the simple one.  That gap is NOT mainly
parameter tuning — it is **more sleeves and more pairs**: four_sigmas runs
15 pairs plus extra-lead-lag plus trailing-OLS discovery, all of which are
genuinely different signals adding diversification.  What IS suspect in the
complex book is the per-pair parameter fitting layered on top.

So the two critiques land differently:
* "too complex" — largely true of the FILTER (Kalman/CUSUM/adapt/snap/MWIS
  machinery, provably inert) but not of the SLEEVE COUNT, which earns its keep.
* "overfit to score" — true, and the simple book is the control that shows
  how much survives without any score-based choices: about 57%.

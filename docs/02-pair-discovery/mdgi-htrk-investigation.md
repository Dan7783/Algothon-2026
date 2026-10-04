# MDGI/HTRK rigorous tune + "are they built in pairs?"

> **Source:** `cusum_output/daniel_pairs/mdgi_htrk_investigation.md` in the team's working repository (relocated here for the public release).

**Outcome: no change applied.  MDGI/HTRK cannot be improved by any
configuration, and the reason is structural — it is not a designed pair.**

## 1. The tune: zero candidates improve it

87,480 candidates (q_beta x h x cusum_k x min_seg_len x adapt_bars x
z_entry x z_exit x z_stop x require_adf):

- **0 pass Pareto** (both windows improve)
- best test score over the ENTIRE grid = **-5.49** = the baseline itself
- corr(beta tracking error, test score) = **+0.041** (nil, as with NAYO/SRTX)

There is no setting of this pair that improves its test window at all.

## 2. Why: the relationship dissolves out of sample

| window | resid sd | final resid | in-window ADF | cuts |
|---|---|---|---|---|
| train | 0.096 | 0.111 | **-3.34** | 0 |
| test | 0.159 | **0.491** | **-1.35** | 2 |

The test residual ends 3x its own sd away from zero and is not
mean-reverting.  (Filter beta moved 2.19 -> 2.56, tracking the log-price
ratio y/x = 2.04-2.60 exactly as a through-origin filter must.)

## 3. The panel IS built in pairs — and there are only ~7

Screened all 1,225 pairs with in-window through-origin residual ADF on both
windows.  Pairs holding ADF < -3 in BOTH:

| pair | train | test | in atlas |
|---|---|---|---|
| RTTH/NAYO | -4.76 | -3.11 | yes |
| HETT/ULXY | -4.46 | -3.65 | yes |
| AENO/NWIG | -4.35 | -4.10 | yes |
| HUXZ/ACAC | -4.28 | -4.32 | yes |
| SMAH/ILVX | -4.01 | -4.68 | yes |
| CTGI/EELT | -3.82 | -3.31 | yes |
| SRNA/ACIX | -3.08 | -3.07 | **no** |

Exactly 7 of 1,225 (0.6%).  Six are already seeded and they are precisely
the book's best earners.  **MDGI/HTRK ranks near the bottom of the seeded
set** (-3.34 / -1.35).  Its partner screen is worse still: of MDGI's 49
possible partners, NONE holds ADF < -3 in both windows; same for HTRK.

## 4. Do not remove MDGI/HTRK

Despite a negative standalone test score it earns its slot in the book:

| config | core train | core test | hybrid train | hybrid test |
|---|---|---|---|---|
| baseline | 793.71 | 800.99 | 819.49 | 852.67 |
| swap MDGI/HTRK -> SRNA/ACIX | 768.61 (**-25.10**) | 802.79 | 813.33 (-5.20) | 846.13 (-6.54) |

## 5. Open lead: SRNA/ACIX is blocked by a gate artifact

SRNA/ACIX is cointegrated *within* each window (-3.08 / -3.07) but its
EXPANDING-history ADF is -1.36 / -0.95, so the prior-gate blocks it
**250/250 days in both windows** and it contributes exactly zero.  A level
shift early in the sample permanently poisons the expanding statistic.

With `require_adf=False` for this pair only:

| book | baseline | + SRNA/ACIX | delta |
|---|---|---|---|
| daniel core | 793.71 / 800.99 | 794.95 / 806.01 | +1.24 / **+5.02** |
| robust_hybrid | 819.49 / 852.67 | 819.41 / 855.34 | **-0.08** / +2.67 |

NOT APPLIED.  The pair's own standalone test score is -9.18, so the
portfolio gain is conflict reshuffling rather than pair edge, and the
hybrid train side is marginally negative.  It fails the "improve both
windows in both books" bar.  Worth revisiting only with a windowed (not
expanding) ADF statistic, which is the real fix for the artifact.

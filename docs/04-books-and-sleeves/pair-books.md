# pair_books.py — uniform adaptive-beta books vs four_sigmas-alone

> **Source:** `cusum_output/pair_books/report.md` in the team's working repository (relocated here for the public release).

Policy (identical for every pair, ZERO score-fitted knobs): filter
defaults on LOG prices; four_sigmas constants verbatim (z window 145,
band 0.27-4.0, ADF < -2.9, OU kappa > 0, equal \$10k legs); on a CUSUM
cut the book goes flat and its spread statistics restart on the new
segment.  Universe: 13 atlas pairs + SMAH/NGTE.

| pair | core | bench train/test | book train/test | cuts | days in |
|---|---|---|---|---|---|
| SMAH/ILVX | CORE | +39.6 / +31.1 | +25.5 / +37.2 | 0 | 511 |
| AENO/NWIG | CORE | +42.7 / +29.5 | +42.7 / +29.5 | 0 | 502 |
| EORC/NGTE | CORE | +41.2 / +61.2 | +41.5 / +28.2 | 2 | 397 |
| HUXZ/ACAC | CORE | +41.5 / +35.8 | +37.1 / +41.8 | 0 | 427 |
| CTGI/EELT |  | +60.4 / +52.0 | +73.2 / +19.2 | 1 | 421 |
| MHRM/EAFC | CORE | +36.0 / +18.8 | +24.6 / +5.8 | 2 | 330 |
| HETT/ULXY |  | +70.2 / +21.1 | +62.1 / +35.9 | 0 | 522 |
| ACIX/ITPA |  | +9.8 / +10.7 | +7.0 / +7.9 | 0 | 518 |
| FWWG/BLBT |  | +9.7 / +17.6 | +10.5 / +13.6 | 0 | 410 |
| RTTH/NAYO |  | +58.6 / +27.7 | +31.8 / +38.7 | 4 | 304 |
| MTNS/ALUT |  | +14.9 / -6.8 | +5.3 / +0.0 | 8 | 103 |
| NAYO/SRTX |  | +18.2 / -12.0 | +25.0 / +1.2 | 3 | 371 |
| ALUT/CCNS |  | +21.6 / +21.0 | +1.6 / -7.9 | 7 | 70 |
| SMAH/NGTE |  | +3.0 / +0.0 | +0.1 / +39.8 | 2 | 350 |

## Combined (eval.py)

| version | train 251-500 | test 501-750 | train/test ratio |
|---|---|---|---|
| v1 per-pair score-tuned (superseded) | 692.6 | 239.6 | 2.9 |
| v2 uniform untuned | 420.3 (Sharpe 7.7) | 302.9 (Sharpe 6.5) | 1.4 |

The benchmark's own per-pair means decay ~30% train->test on these
pairs, so v2's 1.4 ratio is the structural floor, not residual fit.
Changing-beta pairs now beat their benchmark OOS (NAYO/SRTX +1.2 vs
-12.0, MTNS/ALUT 0.0 vs -6.8, SMAH/NGTE +39.8 vs untradeable, RTTH/NAYO
+38.7 vs +27.7, HETT/ULXY +35.9 vs +21.1); stable pairs track their
benchmark (AENO/NWIG identical to 0.1).  Weak spots: ALUT-heavy pairs
(7-8 cuts, mostly stood down) and EORC/NGTE + CTGI/EELT, where the
benchmark's expanding hedge had unusually strong test windows.

Per-pair diagnostics: plots/<pair>.png — prices with cut markers, hedge
beta vs rolling OLS, spread z with entries/exits and position shading,
cumulative PnL vs benchmark with the train|test divider.

## v3 — two-sleeve attribution (2026-07-20)

User direction: atlas pairs back to original four_sigmas mechanics; broad
1,212-pair CUSUM pool, un-gated (no ADF/kappa), trading larger divergences.

| configuration | train 251-500 | test 501-750 |
|---|---|---|
| atlas mechanics alone (SHIPPED default) | 496.5 (Sharpe 9.0) | 363.9 (Sharpe 7.5) |
| + CUSUM pool, late entry 1.5-4.5 sigma | 372.7 | 292.1 |
| CUSUM pool alone, late entry | +1.2 | -33.7 |
| + CUSUM pool, discovery-style 0.2-4.5 | 418.0 | (not scored - failed train gate) |
| CUSUM pool alone, discovery-style | 14.6 | (not scored) |
| v2 uniform filter books (reference) | 420.3 | 302.9 |

Decision rule (fixed before scoring): ship the pool only if full-mode train
beats atlas-alone train.  Both entry styles failed, consistently with the
shortlived_meanreversion finding that un-gated stretched spreads in this
universe diverge on average.  pair_books.py defaults to atlas mode
(train/test ratio 1.36, both windows the strongest pairs-only book of the
session); the pool stays available via PAIR_BOOKS_MODE for research.

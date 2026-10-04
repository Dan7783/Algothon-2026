# fifty_pairs.py — 16 stable (gateless atlas) + 34 adaptive (CUSUM) books

> **Source:** `cusum_output/fifty_pairs/report.md` in the team's working repository (relocated here for the public release).

User-labeled universe; NO ADF gates or OU-kappa decay logic anywhere.
Sleeve-level tuning only (train 251-500, best worst-half): stable W=145
lo=0.27 (= four_sigmas' own constants); adaptive W=145 lo=0.40 plus the
cut-frequency gate (stand down if >1 CUSUM cut in trailing 250d) — the
gate is the filter's own break count, not a stationarity statistic, and it
rescued the adaptive sleeve from a ~0 train wash.

## eval.py attribution (hedge-protecting allocation)

| mode | train 251-500 | test 501-750 |
|---|---|---|
| both (default) | 385.4 (Sharpe 7.1) | 290.0 (Sharpe 5.6) |
| stable only | 375.6 (Sharpe 7.2) | 329.0 (Sharpe 6.7) |
| adaptive only | 1.5 | 20.6 |

Decision rule (fixed pre-test): keep adaptive if both-train >= stable-train
(385.4 >= 375.6 -> kept).  Test then favored stable-only (329 vs 290): the
adaptive sleeve's big winners (SMAH/NGTE +26, NGTE/ILVX, HTRK/RTTH) share
instruments with stable books and get crowded out, leaving weaker trades.
FIFTY_MODE=stable selects the conservative book.

## Findings

- User's stable labels validated: all 16 stable pairs have ZERO CUSUM cuts
  and ALL 16 are positive on test (train 15/16), halves +411/+503.
- Allocation bug found and fixed: opposite-sign requests could net an
  earlier book's leg (capacity 2x), silently unhedging it.  Now a book only
  gets same-direction or idle capacity (stable-alone train 340 -> 376).
- Removing the ADF/OU gates was NOT free on the stable side: the gated
  13-pair atlas (pair_books.py, atlas mode) scores 496.5 / 363.9 vs the
  gateless 16-pair stable sleeve's 375.6 / 329.0.  The gates the user asked
  to remove were adding value on both windows.
- Adaptive sleeve honest read: positive standalone on test (+20.6), ~zero
  on train (+1.5), marginal +9.8 train / -39.0 test under capacity.  Only a
  handful of the 34 pairs carry weight (SMAH/NGTE, HTRK/RTTH, NGTE/ILVX,
  LSST/BENI, DIHO/FCSG); the cut gate keeps the chronic breakers
  (AETS/LSST, CTGI/MDGI, MHRM/RRES, BLBT/CTGI) mostly out of the market.

Artifacts: params.json, adaptive_gate.json, per_pair.json,
plots/<pair>.png (50 four-panel diagnostics: prices+cuts, hedge beta,
z with entries/exits/position shading, cumulative PnL with train|test line).

---

## Figures

![Four-panel diagnostic: AENO/NWIG (prices, hedge beta, spread z with trades, cumulative PnL)](../figures/fifty-pairs-aeno-nwig.png)

*Four-panel diagnostic: AENO/NWIG (prices, hedge beta, spread z with trades, cumulative PnL).*

# The OU rewrite silently killed 13 of 21 pairs — and the fix

> **Source:** `cusum_output/daniel_pairs/ou_scale_fix.md` in the team's working repository (relocated here for the public release).

## What was wrong

`PairAgent.ou_desired_side` computes an OU expected-move signal:

    kappa  = max(_ou_kappa(s), 1e-6)
    pred   = mu + exp(-kappa) * (x - mu)
    signal = (pred - x) / sigma  ==  -z * (1 - exp(-kappa))

So the signal is the old z-score multiplied by `(1 - e^-kappa)`.  Measured
kappa across the 21 pairs is **0.023 - 0.177**, i.e. the signal is
**10-40x smaller** than the z-score the seeded thresholds were written for.

The seeds still carried old-scale `z_entry` values (0.10 - 0.75), so:

| | |
|---|---|
| pooled \|signal\| | p50 **0.039**, p90 **0.163**, max 0.719 |
| seeded z_entry | 0.10 - 0.75 |
| pairs with max\|signal\| < z_entry | **13 of 21** |

Those 13 pairs could never trade under any market condition.  A census of
the live book confirmed it: only 8 of 21 pairs were ever active, and
**zero conflict losses** — the dormant pairs never even generated a signal,
so this was not an EV-MWIS crowding effect.  Three of the six genuinely
cointegrated pairs were among the dead (RTTH/NAYO entry 0.50 vs max signal
0.331; CTGI/EELT 0.35 vs 0.327; HETT/ULXY 0.75 vs 0.424).

## The fix

Thresholds recalibrated to percentiles of each pair's OWN |signal|,
measured on the **train window only** (entry p70, exit p25, stop p99.5).
No fitting to test, no per-pair search.

## Result (official eval.py, both windows)

| book | train 251-500 | test 501-750 |
|---|---|---|
| daniel_filter before | 155.63 (mean \$177.5) | 262.60 |
| **daniel_filter after** | **510.99** (mean \$519.3, Sharpe 7.86) | **521.74** (mean \$529.9, Sharpe 8.00) |

**+355 train / +259 test.**  It also fixes the train/test imbalance raised
earlier in the session: the book is now essentially balanced (ratio 0.98)
instead of test-heavy, because the pairs sleeve is finally contributing in
both windows.

Applied to the `seeds` table in daniel_filter.py (backup: /tmp/daniel_filter.bak2.py).

## Why the per-pair sweep found nothing

The rigorous per-pair sweep run alongside this reported "0 robust" for 19 of
21 pairs and +0.00 book impact for the two winners.  That is explained by
the same bug: 13 pairs were structurally dead, so no threshold inside the
old-scale grid could revive them, and the replica used for the sweep still
modelled the OLD z-fade decision rule, not `ou_desired_side`.  Per-pair
tuning should be re-run against the corrected scale before any of it is
trusted.

## Separate pre-existing breakage (not caused by this change)

- `robust_hybrid_v2.py` calls `FiltCfg._replace(cut_confirm_bars=...,
  adapt_p_floor=..., snap_max_rel=...)`, but those fields no longer exist on
  `daniel_filter.FiltCfg` -> TypeError on import path.  The hybrid cannot be
  evaluated until they are restored or the call is updated.
- `eval.py` still defaults to `--module robust_hybrid`, which was deleted.
- root `tune_daniel_pairs.py` imports `pair_book_lab`, which imports the
  deleted `pair_seg_filter`.

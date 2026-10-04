# Can cov_inflate replace the post-cut adapt machinery?

> **Source:** `cusum_output/FILTER_SIMPLIFICATION.md` in the team's working repository (relocated here for the public release).

Hypothesis (user): ~4 of the filter's knobs are unnecessary if a good
`cov_inflate` is chosen.

## Result: the knobs earn their keep, but cov_inflate itself is inert

| variant | knobs dropped | train | test |
|---|---|---|---|
| full filter | — | **853.18** | **878.57** |
| snap + cov_inflate | 4 (q_beta_adapt, adapt_chase, adapt_bars, adapt_extend) | 824.02 (-29.2) | 863.04 (-15.5) |
| cov_inflate only | 6 (+ snap_window, snap_overshoot) | 824.40 (-28.8) | 854.81 (-23.8) |

`cov_inflate` was chosen on the TRAIN window only in both variants; test is
reported as a check.

## Why inflation cannot replace the adapt burst

This is the through-origin result again.  The filter has no intercept, so
its Kalman gain is `g = P*x/(P*x^2 + R)` and inflating P drives `g -> 1/x`,
i.e. **beta -> y/x, the log-price ratio**.  But `adapt_chase=1.0` sets beta
to the short-window **OLS** estimate each day of the burst.  Those are two
different targets, so no amount of inflation reproduces the chase — it
converges somewhere else.  That is why the 4 knobs are worth ~29 train.

## What IS redundant

`cov_inflate` is effectively **inert**: sweeping it over three orders of
magnitude moves train by ~1%.

| cov_inflate | 1 | 5 | 25 | 100 | 1000 |
|---|---|---|---|---|---|
| train (snap variant) | 815.0 | 821.3 | 819.1 | 824.0 | 823.5 |
| train (pure variant) | — | 824.2 | 820.5 | 824.4 | 821.0 |

So `cov_inflate` should be a hardcoded constant, not a tunable.  Likewise
(shown earlier) `h`, `cusum_k`, `min_seg_len` and `adapt_bars` are provably
inert on any pair that never cuts, and `q_beta` is inert everywhere because
beta is pinned to the price ratio.

## The actual choice

Dropping the 4 adapt knobs costs **-3.4% train / -1.8% test** and removes
the adapt countdown, `snap_dir` ratcheting, `_accel_target`, and the
`adapt_p_floor`/`snap_max_rel` modifiers that `robust_hybrid_v2` currently
crashes on.  Whether ~16 test points is worth that reduction is a judgement
call, not something the data settles.

NOT APPLIED — four_sigmas still runs the full filter (853.18 / 878.57).

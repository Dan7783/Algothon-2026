# Staged tuning (CUSUM off -> q -> CUSUM -> inflation -> reversion)

> **Source:** `cusum_output/STAGED_TUNING.md` in the team's working repository (relocated here for the public release).

The right procedure, because each stage identifies a DIFFERENT physical
quantity against a statistical criterion, so stages 1-3 never touch PnL.

## Stage 1 — match the Kalman to the base noise (CUSUM off)

Found a genuine modelling error, not a tuning preference.

The filter hardcodes the observation variance at **R = 1** (`s_pred =
p22*x^2 + 1.0`).  The measured residual variance is **~1.4e-03**.  So
`q_beta` has always been read against a noise floor ~700x too large, and the
seeded values (3.5e-08 .. 6.8e-05) sit far below the level where the ratio
starts to matter.  That is why `q_beta` appeared inert across EIGHT orders of
magnitude earlier: every seed was effectively zero.

Identification, per pair, TRAIN only, no PnL:

| quantity | how | result |
|---|---|---|
| observation noise | residual variance around a local hedge | median 1.4e-03 |
| process noise | variance of non-overlapping local OLS beta steps, minus their own estimation variance | median 4.8e-06/day |
| q* = drift/resid | the ratio the filter should use with R=1 | median 4.8e-03 |

The raw ratio includes REGIME JUMPS, which are stage 2's job.  Validating
against how well the filter tracks the identified hedge path (CUSUM disabled
via h=1e9) puts the smooth-component answer at **q ~ 1e-04**, best or
near-best for all 14 pairs.  Note the identified values cluster within ~13x
across pairs while the seeds span 2000x — more evidence the seeds are noise.

## Stage 2 — and here the procedure pays off unexpectedly

With `q_beta` correctly scaled the CUSUM has nothing left to fire on:

| q_beta | total cuts across the book |
|---|---|
| seeded | 17 |
| 1e-05 | 15 |
| **1e-04** | **1** |
| 1e-03 | 10 |

The filter now tracks closely enough that the `disagree` gate never trips.
**The CUSUM -> adapt-burst -> snap -> cov_inflate chain exists to compensate
for a mis-scaled q_beta.**  Fix stage 1 and stages 2-3 have almost no work
to do; the filter collapses to a plain Kalman tracker.

## Book-level result

| q_beta | blocks | train median | floor |
|---|---|---|---|
| seeded (14 fitted values) | 866.6/877.3/756.3/935.1 | **872.0** | 756.3 |
| 1e-05 uniform (derived) | 867.4/849.0/785.2/917.9 | 858.2 | **785.2** |
| 1e-04 uniform (derived) | 847.3/835.0/739.7/893.9 | 841.1 | 739.7 |
| 1e-03 uniform | 835.0/822.3/746.6/891.3 | 828.7 | 746.6 |

## The honest tension — NOT resolved unilaterally

* On the TRAIN blocks, the only legitimate selection criterion, the fitted
  seeds win (872.0 vs 858.2).
* On the block FLOOR — the best available OOS proxy given the 878 -> 599
  print — the derived q=1e-05 is 29 points BETTER (785.2 vs 756.3), and it
  replaces 14 fitted numbers with one derived constant.

Train prefers the fitted values; robustness prefers the derived one.  That
is exactly the trade the 599 was warning about, but train does not support
the change on its own terms, so nothing was applied.

NOT APPLIED.  four_sigmas still runs the seeded q_beta.

## Remaining stages

Stage 3 (cov_inflate = adaptation speed) and stage 4 (reversion thresholds
from the spread's OU properties) are only worth running on top of a chosen
stage-1 value.  If q=1e-05 is adopted, stage 2/3 are near-vacuous and the
filter simplifies to a tracker with no break machinery at all.

# Fixing stage 1's error: diagnosed, fixed, and NOT applied

## The error, precisely

`R = 1` was baked into two places:

    s_pred = pp22*x*x + 1.0        innovation variance
    p22    = a*a*pp22 + g*g        Joseph update — really a^2*P + g^2*R

True observation variance is ~1.4e-03, so the gain g = P*x/(P*x^2+R) was
suppressed ~700x.  The CUSUM never revealed this because its z-score
(innov / sqrt(innov_var)) is self-normalising — only the GAIN was affected.

## Two bugs found while fixing it

1. Estimating R online as an EWMA of post-fit residuals creates a feedback
   loop (better tracking -> smaller residuals -> smaller R -> higher gain).
   NIS came out 1.44, 1.38, 27.5, 40.1, 4.0, 0.40 — nonsense.
2. Estimating R at t=0 uses a ONE-observation window, so var = 0 and R
   floored to 1e-10 — smaller than correct, not larger.  With R that tiny the
   gain saturates at 1/x and every q_beta gives an identical beta path, which
   is what exposed the bug.

Correct version: estimate R once from a real 60-day warmup, freeze it, use it
in both places, and initialise p22 = R/(x'x) — the sampling variance of the
initial beta — rather than an arbitrary 1.0.

## The corrected filter is a better ESTIMATOR

| q_beta | tracking err, R=1 | tracking err, R fixed | NIS (want ~1) |
|---|---|---|---|
| 1e-07 | 0.0074 | **0.0036** | 1.32 |
| 1e-06 | 0.0070 | 0.0082 | 0.92 |
| 1e-04 | 0.0042 | 0.0153 | 0.15 |

Best tracking with R fixed (0.0036) beats anything achievable with R=1
(0.0042), and NIS ~ 1 lands in the same q range instead of being off by
orders of magnitude.

## But it makes the STRATEGY worse

| config | blocks | train median | floor |
|---|---|---|---|
| current (R=1, seeded q) | 866.6/877.3/756.3/935.1 | **872.0** | 756.3 |
| R fixed, q=1e-08 | 833.9/838.6/724.0/907.4 | 836.3 | 724.0 |
| R fixed, q=1e-07 | 839.4/840.6/743.5/898.9 | 840.0 | 743.5 |
| R fixed, q=1e-06 | 844.1/813.7/765.6/885.1 | 828.9 | 765.6 |

Worse on TRAIN at every q.

## Why — and what it means

This is the "beta absorbs the spread" degeneracy.  A correctly-scaled gain
lets beta track short-run co-movement, which pulls the very mean-reversion
out of the residual that the strategy exists to trade.  The over-damping
caused by R=1 is doing real work: it keeps the hedge slow, so the spread
stays tradeable.

So `R = 1` is not a bug to fix — it is an accidental REGULARISER.  That is a
defensible thing to want (a cointegrating vector should be stable, not
chase), but it should be stated as a design choice rather than smuggled in
via a wrong noise model.  The honest formulations are:

* keep the filter and document that the hedge is deliberately over-damped; or
* replace it with an explicitly slow hedge — a long-window OLS, as
  `simple_pairs.py` uses — which says the same thing in one line and with no
  misleading parameters.

NOT APPLIED.  four_sigmas keeps R = 1 and its seeded q_beta.

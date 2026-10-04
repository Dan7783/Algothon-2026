# `random_strat1.py` — idea assessment vs `four_sigmas`

> **Source:** `random_strat1_assessment.md` in the team's working repository (relocated here for the public release).

**Date:** 2026-07-23
**Question:** which ideas in `random_strat1.py` are worth porting into the live book (`four_sigmas`)?

## TL;DR

The whole strategy is **not** an upgrade — it scores **691.8** on days 501–750 vs
`four_sigmas`' **856.2**, while trading *more* (\$90.7M vs \$61M volume). The value, if any, is in
individual techniques. After backtesting the standout ones and cross-checking the rest against
our own research, **only one idea is worth a test (portfolio rebalance-band hysteresis)**; the
rest are either inferior versions of what we already have or are contradicted by findings in
`notebooks/momentum_reversion_yearly.ipynb` and `seasonality_and_reversal.ipynb`.

## What `random_strat1` is

A single-file, self-contained book with weight-gated sleeves. With its shipped config only two
are active:
- `LEADLAG_WEIGHT = 1.0` (replaces everything else) — a vectorised lead-lag signal;
- `EDGEPAIRS_WEIGHT = 0.80` — correlation-selected OLS pairs reversion.

Dormant (weight 0): short-horizon momentum-fade (`LOOKBACKS`), `MPAIRS`, and the
`OLS` ALGO-vs-basket sleeve. Plus execution machinery: regime vol scaling, inverse-vol position
sizing, a rebalance band, and position/continuation caching.

## Idea-by-idea verdict

| # | Idea | What it does | Verdict | Evidence |
|---|------|--------------|---------|----------|
| 1 | **Rebalance-band hysteresis** | Only re-trade an instrument when `Δnotional ≥ band × limit` (band 0.515, widened ×2.5 in high vol) | **PORT / TEST** | Only genuinely new, low-risk idea; cuts turnover/whipsaw. four_sigmas has *no* portfolio-level hysteresis. |
| 2 | **ALGO-vs-basket sleeve** | Trade ALGO (idx 0) vs equal-weight basket on a spread z | **SKIP** (but flag capacity) | ALGO~basket residual **never cointegrates** (ADF t −1.2/+0.1/−2.3; crit −2.9); PnL erratic (test1 −13, test2 +78, full +23). |
| 3 | **Adaptive band widening in high vol** | Widen the rebalance band when short/long vol ratio ≥ 1 | **PORT with #1** | Aligns with our year-4 = higher-vol finding (regression §8): fewer trades into a noisier tape. |
| 4 | **Lead-lag on a PREFIX window** | Estimate the lead-lag corr matrix from the *oldest* `lb` days, apply to today | **SKIP** | Our lead-lag matrix is **unstable across years** (cross-year corr 0.16–0.23); a prefix-estimated matrix is stale by construction. Our ridge/extraLL is strictly better. |
| 5 | **Edge-pairs (2-timescale)** | Select pairs by corr on a long window, fit β on a short (20d) window, fade z | **SKIP** (concept noted) | Inferior to four_sigmas' adaptive atlas (Kalman β + trailing-ADF gate + EV-MWIS conflict resolution). The *select-long / trade-short* split is a good instinct we already embody. |
| 6 | **Inverse-vol position sizing** | Scale each name's limit by `median_vol / inst_vol` | **SKIP** | Realized vol is **not persistent** here (persistence ≈ 0, forecast R² < 0 — `seasonality` persistence check), so inverse-vol sizing off 20d vol adds noise, not signal. Also the `[2.75, 3.0]` clip degenerates it to a ~2.8× constant leverage that just saturates positions to the cap. |
| 7 | **Regime vol scaling** (`exposure`) | De-risk when short/long vol ratio ≥ threshold | **CONCEPT only** | Disabled in random_strat1 (`REGIME_SCALE=1.0`, no-op). Whole-book de-risking in vol spikes is defensible for the Sharpe-based score, but low-confidence and unrelated to this file's actual behaviour. |
| 8 | **Signal normalisation** (clip → `/max_abs`) | Bound each sleeve to [−1,1] and unit-normalise | **SKIP** | Clean, but four_sigmas sizes in dollars per pair already; no gain. |
| 9 | **Continuation / state caching** | Short-circuit when history is unchanged; detect day-by-day continuation | **Note (engineering)** | Sensible for a stateful book; four_sigmas' `AtlasBook` is already stateful with `reset()`. The band (idea #1) *needs* this continuation detection to be correct. |

## The one thing worth porting: rebalance-band hysteresis (#1 + #3)

**Why it might help.** four_sigmas trades \$61M over 250 days → ~\$6.1k commission (1bp) ≈ \$24/day
against ~\$856/day PnL. Commission alone is small (~3% of gross), so the *real* prize is
**whipsaw reduction** — not re-entering/exiting on noise when a signal jitters around its
threshold. A per-instrument band `Δnotional ≥ band × limit` before re-trading suppresses that.

**How to integrate rigorously.**
1. Compute the book's target dollars as today (atlas + extraLL + ridge + discovery, gates as-is).
2. Convert to target shares, but **only move a name if** `|target − current|·price ≥ BAND × limit`;
   else hold `current`. This needs four_sigmas to remember `curPos` between calls (it currently
   recomputes stateless each day) — add a small `_last_pos` state + continuation check
   (idea #9), and make `reset()` clear it so replays stay correct.
3. Widen `BAND` in high-vol windows (idea #3), which our research says year 4 will be.
4. **Validate by the block FLOOR** (per `VALIDATION_PROTOCOL.md`), not the peak: the band must
   not lower the floor, and should cut $volume. Start with `BAND ≈ 0.15–0.25` (random_strat1's
   0.515 is aggressive and, notably, still left it trading *more* than four_sigmas — so its band
   is not the reason it's cheaper; tune conservatively).

**Honest caveat:** because commission is already small and four_sigmas' pair agents have their own
z-hysteresis, expect a *modest* gain at best. It is a low-risk efficiency tweak, not an alpha
source — worth one clean backtest, not a rewrite.

## What NOT to take, and why (tie-back to our research)

- **Don't add another lead-lag/momentum sleeve.** We proved (`momentum_reversion_yearly.ipynb`)
  the cross-sectional lead-lag substrate is fragile (0.16–0.23 cross-year, per-name autocorr is
  noise), and our ridge/extraLL already exploit it with OOS gates. random_strat1's prefix-window
  lead-lag is a *staler* version of a signal we already run better.
- **Don't add vol-targeting off realized vol.** Vol isn't forecastable here at trade horizon.
- **Don't chase the ALGO sleeve via basket-reversion.** ALGO's 100k/low-comm capacity is
  genuinely untapped, but this is not the key — ALGO isn't cointegrated with the basket. If ALGO
  capacity is ever worth pursuing it needs a *different* signal, not this one.

## Bottom line

random_strat1 is a competent but weaker book (691 < 856). Its sophisticated ideas are ones we
already do better; its simple ideas are contradicted by our regime research. **Port exactly one
thing on a trial basis: a per-instrument rebalance band (+ high-vol widening), validated on the
block floor.** Everything else stays out.

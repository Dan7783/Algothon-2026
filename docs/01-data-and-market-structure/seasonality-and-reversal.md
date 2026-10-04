# Seasonality & Reversal Research — TO-DO

> **Source:** `TODO_seasonality.md` in the team's working repository (relocated here for the public release).

**Research question.** The panel is 51 instruments × 750 days = exactly **3 "years" of 250
days**. The competition scores days **751–1000 (year 4)**, which we do NOT have. Teams that
ranked well in earlier rounds reportedly **fell during 751–1000**. Is that a **calendar
(250-day) seasonal** effect, a **regime drift**, or **selection-induced edge decay**? Also:
re-examine the **reversal idea** (flip-at-extremes vs flip-at-mean). Everything must be
grounded in math and shown in a Jupyter notebook with graphs.

Deliverable: `notebooks/seasonality_and_reversal.ipynb` (executed, graphs embedded).
Builder: `research/build_seasonality_notebook.py` (re-runnable).

---

## Phase 0 — Context (DONE)
- [x] Confirm data shape: 51 × 750 = 3 × 250-day years. Scored window is 751–1000 (unseen).
- [x] Locate the "reversal idea": `research/flip_at_extremes.py` (enter at max stretch,
      reverse at the opposite extreme, hold through the mean).
- [x] Recover self-contained pair/accounting logic (Kalman β, windowed z, band/flip sims,
      faithful eval.py two-leg accounting).

## Phase 1 — Is there a 250-day seasonal cycle? (math)
- [ ] Build market index m(t), cross-sectional dispersion d(t), squared-return vol proxy.
- [ ] Lag-250 autocorrelation with ±1.96/√n white-noise bands (note trend contamination).
- [ ] Periodogram of dispersion & |r|; mark period≈250; **AR(1) red-noise surrogate null**
      (2000 sims) → is any 250-day peak significant?
- [ ] Phase-fold t mod 250 across the 3 years; overlay; **circular-shift permutation ANOVA**
      on phase bins. State the 3-cycle power caveat explicitly.

## Phase 2 — Regime by year (drift 1→2→3, extrapolate year 4)
- [ ] Per-year table: mean return, vol, Sharpe, dispersion, mean pairwise corr, lag-1
      autocorr (momentum vs reversal), variance ratio VR(5), cointegration prevalence
      (fraction of pairs ADF t < −3.34).
- [ ] Bar charts + narrative on the trajectory and what it implies for year 4.

## Phase 3 — Year boundaries (the "seasonal event")
- [ ] Event study around days 250 and 500 (±20d): |r|, dispersion, corr. Recurring shift?
      (n=2 boundaries — honest about power.)

## Phase 4 — Why successful teams fall on the next window (the real mechanism)
- [ ] Winner's-curse experiment: per-pair reversion PnL, select on days 1–500, evaluate on
      501–750. Scatter S_in vs S_out, corr, top-decile shrinkage.
- [ ] Show it is selection-induced mean reversion of edge + regime drift, NOT calendar
      seasonality. Tie to memory: per-pair tuning train +14 / held-out −15.5.

## Phase 5 — The reversal idea (flip-at-extremes)
- [ ] Overshoot statistics: after z crosses 0 from |z|≥1.5, does it reach the opposite
      extreme before bouncing? (OU spreads should not.)
- [ ] PnL: shipped band 0.27–4.0 vs flip Z∈{1.0,1.5,2.0,2.5}, train/test, 16 stable pairs.
- [ ] Seasonal-reversal test: per-instrument lag-1 autocorr sign per year — do signs flip
      year-over-year? (bridges reversal ↔ seasonality)
- [ ] Verdict: does reversal help, hurt, or neither?

## Phase 6 — Synthesis
- [ ] One-paragraph verdict on each: seasonality, regime drift, edge decay, reversal.
- [ ] Prioritized, math-backed recommendations for the 751–1000 submission.

---

## RESULTS (executed `notebooks/seasonality_and_reversal.ipynb`, seed 20260723)

- **§1 Seasonality — NOT FOUND.** Periodogram power at ~250d is *below* the AR(1) 95% null
  (disp 2.1e-6 vs 3.8e-5; |ret| 1.2e-5 vs 5.1e-5); phase-fold circular-permutation ANOVA is
  non-significant (disp F=0.88 p=0.207; |ret| F=0.81 p=0.875). 3 cycles can't establish a
  calendar season, and none is visible → the "calendar broke the leaderboard" story is unsupported.
- **§2 Regime drift — the real signal.** Monotone 1→3: vol 0.0096→0.0100→0.0103↑, mean pairwise
  corr 0.179→0.201→0.213↑, cointegration prevalence 2.04%→1.96%→1.80%↓; short-horizon sign
  *alternates* (lag1 +.016/−.022/+.019; VR5 mkt 1.10/0.83/1.18). Year 4 ⇒ higher vol, higher
  corr, thinner pairs fuel, likely reversal-leaning.
- **§3 Boundaries — mild.** dispersion/|ret| dip slightly post-boundary (0.94/0.95), n=2, not actionable.
- **§4 Winner's curse — dominant cause.** corr(selection,holdout)=+0.13; top decile by selection
  loses **70%** of edge OOS (0.0038→0.0011); more-selective picks fall harder. Reproduces the
  leaderboard fall with zero seasonality.
- **§5 Reversal idea — REJECT.** Overshoot only 42% (<50%, OU-like); flip-at-extremes loses to the
  shipped flip-at-mean band at every Z (band 457/443 vs Z=1.0 437/313); per-name momentum↔reversal
  sign flips 46%/46% ≈ coin toss.

**Bottom line:** the fall is edge decay + regime drift, not seasonality; keep flip-at-mean and a
low-parameter, dollar-neutral, reversion book sized to year-3 stats.

## Phase 7 — Regime drift applied to the PAIRS signal for OOS (DONE)
- [x] Rolling per-pair reversion gate: trade a spread only while its OU slope `b<0` over GATE_W
      days (structural, one global param, no per-pair tuning).
- [x] Faithful eval on 16 stable pairs: train 457→454, **test 442.6→460.5**, decay +14.9→**−6.1**.
- [x] Protocol check (block floor, all GATE_W): no-gate floor 361 → **+42/+34/+16/+2** for
      W=40/60/90/120; lift lands on te2 (626–750), the freshest year-4 proxy.
- [x] Honesty: inert on 1225 random pairs (0.00037→0.00036) — protects genuine cointegration only.
- [x] Notebook §7 (graphs) + `research/regime_gated_pairs.py`, `research/sweep_regime_gate.py`.
- [ ] NEXT: wire the `b<0` gate (GATE_W≈40–60) into the live pairs sleeve (four_sigmas
      `_fade_pair` already has a 120d OU-kappa gate — replace/tighten with the rolling 40–60d
      version) and re-score the full book.

_Status: Phases 0–7 complete. Notebook executed with 10 embedded figures, 0 errors._

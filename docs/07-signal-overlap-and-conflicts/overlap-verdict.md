# Pair/ridge signal overlap: diagnosis and verdict

> **Source:** `analysis_outputs/signal_overlap/OVERLAP_VERDICT.md` in the team's working repository (relocated here for the public release).

Scope: the *combination layer* of `testbedv2_tuned_ridge_protectedbarbell`.
Both sleeves are held fixed; only the merge changes. Harness verified against
`eval.py` exactly (baseline 751-1000 = **914.70**, std 1660.04, identical to
the shipped evaluator).

Split, per `cusum_output/VALIDATION_PROTOCOL.md`:

| region | days | role |
|---|---|---|
| train | 251-500 (blocks 251-375, 376-500) | selection — the only region allowed to pick |
| validation | 501-750 (501-625, 626-750) | burned; confirmation only |
| test | 751-1000 (751-875, 876-1000) | scored once per finalist |

Acceptance rule was **pre-registered before results were inspected**: beat
baseline on *both* train blocks independently, train floor not worse, plus a
structural justification. Validation/test could never promote a scheme.

Caveat on the test window: the 88-pair universe had 8 pairs removed using
751-1000 standalone P&L, so test is optimistic in *absolute* level. Every
variant shares that identical universe, so *relative* comparison remains
valid — which is all the conclusions below rely on.

---

## 1. The premise, tested

> "Both perform very well independently but when connected are limited."

Not supported. The merge is strongly synergistic, not limiting:

| sleeve | train | validation | test |
|---|---:|---:|---:|
| pairs alone (hedged, exclusive) | 594.5 | 522.7 | 565.4 |
| ridge alone | 232.2 | 406.1 | 476.4 |
| **combined (shipped)** | **789.8** | **847.0** | **914.7** |

The combination beats either sleeve by a wide margin in every region. The
overlap is load-bearing, not a defect.

## 2. What the overlap actually is

Stable across all three regions (so structural, not regime artifact):

| category | share of live signals | hit rate | mean PnL/inst-day |
|---|---:|---:|---:|
| both agree | 19-21% | **0.551-0.567** | \$31-35 |
| pair only | 48-53% | 0.528-0.532 | \$18 |
| ridge only | 11-12% | 0.525-0.554 | \$22-30 |
| **conflict** | 17-19% | **0.511-0.528** | \$5-18 |

Agreement is the best bucket, conflict the worst — a real, stable effect.

**Hedge integrity is genuinely broken.** Because arbitration is per-*asset*,
a pair's two legs are decided independently:

| region | both legs intact | one leg naked | both flipped | median hedge-ratio error |
|---|---:|---:|---:|---:|
| train | 61.4% | 34.1% | 4.5% | 14.5% |
| validation | 61.4% | 34.1% | 4.5% | 17.1% |
| test | 61.6% | 33.8% | 4.6% | 16.9% |

So ~34% of pair activity becomes a naked directional bet, and even surviving
spreads are mis-hedged ~15% because bang-bang sizes both legs at the full
limit regardless of beta. **This is real — but fixing it does not pay** (§4).

## 3. Why fixing it does not pay: the scoring function

`score = mu · SR²/(SR²+1)`. Scaling every position by k scales mu and sigma
together: SR is unchanged, mu rises, score rises. **Deployed capital is the
dominant term**, and the book is limit-constrained, not signal-constrained.

Schemes ranked almost monotonically by gross dollars deployed. Among the
three schemes with *identical* deployment (\$435,400/day, 43.5 instruments),
ordering was perfectly monotone in turnover:

| scheme | gross/day | turnover/day | train |
|---|---:|---:|---:|
| pairs win conflicts | \$435,400 | \$180,683 | 865.1 |
| baseline | \$435,400 | \$285,984 | 789.8 |
| ridge wins conflicts | \$435,400 | \$386,064 | 623.3 |

Every hedge-preserving scheme necessarily deploys *less* capital (beta-sized
legs are smaller; exclusive pair selection covers fewer instruments), which
is why correct hedging loses to incorrect-but-larger bang-bang.

## 4. All 11 schemes

| scheme | train | tr floor | val | val floor | test | test floor |
|---|---:|---:|---:|---:|---:|---:|
| s0 baseline (shipped) | 789.8 | 649.7 | 847.0 | 758.6 | **914.7** | 670.8 |
| s1 conflict → flat | 749.0 | 622.5 | 777.5 | 687.5 | 768.2 | 577.8 |
| s2 pairs win conflicts | **865.1** | **763.5** | 743.2 | 715.3 | 676.8 | 601.4 |
| s3 ridge wins conflicts | 623.3 | 468.3 | 806.1 | 653.9 | 852.3 | 544.1 |
| s4 pairs only (hedged) | 594.5 | 497.5 | 522.7 | 470.2 | 565.4 | 550.3 |
| s5 ridge only | 232.2 | 164.0 | 406.1 | 249.4 | 476.4 | 117.3 |
| s6 orthogonal (no overlap) | 718.5 | 600.9 | 793.9 | 775.3 | 786.2 | 621.4 |
| s7 orthogonal, bang-bang pairs | 744.8 | 618.3 | 837.2 | **813.2** | 822.1 | 661.8 |
| s8 orthogonal + ridge veto | 711.8 | 590.4 | 787.3 | 757.0 | 769.2 | 572.5 |
| s9 magnitude-proportional | 584.1 | 489.5 | 634.7 | 565.2 | 642.6 | 459.1 |
| s10 agreement-weighted | 498.4 | 411.0 | 517.5 | 449.6 | 533.2 | 382.1 |

**Only s2 passed the train gate — and it is the worst finalist out of sample**
(test 676.8 vs baseline 914.7, −238). The diagnostic predicted this in
advance: conflict-winner is regime-unstable (letting ridge win conflicts
*loses* \$7.9k on train but *makes* \$15.3k / \$30.8k on validation / test).
A train-fitted winner picks the wrong side for every later window.

## 5. Two capacity levers — both traps

**ALGO capacity.** Instrument 0 carries a 100k limit (10×) and 0.2bp
commission (5× cheaper) but `BANG_BANG_LIMITS[ALGO] = 0` in the working copy,
so it never trades — 100k of 590k capacity idle, with ALGO-HRET live ~18% of
days ready to drive it.

| ALGO limit | train | validation | test |
|---:|---:|---:|---:|
| 0 (current) | 789.8 | **847.0** | **914.7** |
| 10k | 791.6 | 840.8 | 912.2 |
| 25k | 794.2 | 831.4 | 908.4 |
| 50k | 798.5 | 815.6 | 902.0 |
| 100k | **806.8** | 783.6 | 888.9 |

Monotonically **up on train, down on validation *and* test**. Not market
beta (R² < 0.001 throughout) — the single-pair signal just does not support
10× capital. **ALGO = 0 is correct; keep it.** Note train-only selection
would have shipped 100k (both blocks and floor improve).

**Turnover stickiness** (hold direction unless |vote| > θ; θ=0 is baseline).
Train is jagged and structureless (778, 765, 771, 764, 789, 826, 813, 845,
761) — no plateau, so rule 3 rejects it — and every θ>0 degrades validation
and test badly (θ=0.75: val 708 vs 847, test 747 vs 915).

## 6. Knob robustness (the most actionable finding)

| RIDGE_CONFLICT_WEIGHT | train | validation | test |
|---:|---:|---:|---:|
| 0.50 | 820.0 | 735.0 | 749.7 |
| **0.75 (shipped)** | 789.8 | **847.0** | **914.7** |
| 1.00 | 748.5 | 799.5 | 846.8 |

**0.75 is a spike, not a plateau.** Its immediate neighbours cost −165 and
−68 test points. This knob means "how strong must ridge be to override the
pair consensus" (ridge wins when `0.75 × strength > |consensus|`). It is the
single largest fragility in the book, and it cannot be ensembled away:
`sign(cons + rw·X)` is monotone in `rw`, so a symmetric-grid majority vote is
mathematically identical to the median knob (verified — all ensembles
returned exactly 789.8/847.0/914.7).

| PROTECTED_VOTE_BOOST | train | validation | test |
|---:|---:|---:|---:|
| 1.0 | 785.4 | 804.8 | 917.8 |
| 4.0 (shipped) | 789.8 | 847.0 | 914.7 |
| 10.0 | 808.2 | 865.0 | 920.2 |

**Inert** — test moves <11 points across a 10× range. Per protocol rule 7 a
provably-inert knob should be hardcoded, not tuned.

## 7. Verdict

**Nothing ships.** No scheme satisfies the pre-registered rule, and the only
one that passed the train gate is the worst out-of-sample finalist. The
shipped baseline is at a local optimum for this scoring function.

The overlap is not the bottleneck. Double-coverage is precisely what lets the
book hold 43.5 instruments at full size; every attempt to de-conflict,
orthogonalise, hedge properly, or size by confidence deploys less capital and
scores less.

Three findings worth acting on, none of which is a score chase:

1. **Keep ALGO at 0** — now empirically confirmed on two independent windows.
2. **Simplify `PROTECTED_VOTE_BOOST`** — inert across 1-10; hardcode it.
3. **`RIDGE_CONFLICT_WEIGHT = 0.75` is the book's main fragility.** It sits on
   a knife-edge worth ~±150 test points. It cannot be smoothed away, so it
   should be treated as a known risk, not as a stable setting.

Real remaining headroom is *not* in arbitration: only ~6.5 of 50 instruments
sit idle on a given day, and they are idle because neither sleeve has an
opinion. Improving coverage or per-sleeve signal quality is where the
capacity is — not in re-merging two sleeves that already work together.

## 7b. How much score is overlap actually costing?

Measured by holding capital deployment constant and replacing a category's
directions with an oracle (future information — an accounting bound, not
achievable) or an anti-oracle, then reading the score span each category
spans and how much of it the shipped book captures.

| category (test) | oracle | anti-oracle | span | **capture** | net contribution |
|---|---:|---:|---:|---:|---:|
| both agree | 2260.1 | −1066.7 | 3327 | **59.6%** | +325.2 |
| ridge only | 1693.7 | −74.9 | 1769 | 56.0% | +106.0 |
| pair only | 4167.5 | −3098.3 | 7266 | 55.2% | +381.0 |
| **conflict** | 2256.8 | −709.0 | 2966 | **54.7%** | **+146.5** |

Capture % by region — conflict is the worst category in all three, but only
by a few points, and everything sits close to coin-flip:

| category | train | validation | test |
|---|---:|---:|---:|
| both agree | 58.4% | 59.7% | 59.6% |
| ridge only | 57.0% | 57.3% | 56.0% |
| pair only | 55.1% | 55.1% | 55.2% |
| **conflict** | **51.1%** | **52.3%** | **54.7%** |

Three defensible answers, depending on what "losing" means:

1. **Realistic penalty ≈ 145-210 points.** If conflicts arbitrated as well as
   the agreement bucket (the book's best category), the capture gap of
   4.9-7.3pp over a ~2700-3000 point span is worth **+145 (test), +198
   (validation), +210 (train)** — roughly 16% of the current test score.
2. **Ceiling (oracle, unachievable) ≈ +1,277 to +1,406.** Perfect conflict
   resolution more than doubles the score (test 914.7 → 2256.8). This uses
   future information and is an upper bound only.
3. **Realised, with any causal rule tested: 0.** Every scheme in §4-5 either
   failed the train gate or degraded out of sample.

**Conflicts are net positive, not a loss centre.** Not trading them at all
costs −146.5 (test), −69.6 (validation), −40.7 (train). The book gains from
conflicted assets; it just gains less efficiently than elsewhere.

**And overlap is not disproportionate.** Conflicts are 19.3% of live
instrument-days on test and account for 1342/6710 = **20.0%** of the book's
total oracle headroom (oracle-everything = 7625.2 vs baseline 914.7). Headroom
tracks share of days almost exactly, so conflicted assets are an *average*
slice of a general problem — the book hits ~55% across every category — not a
special defect created by the sleeves overlapping.

That reframes the target: the ~145-point overlap penalty is real but small
next to the ~6,700 points of total directional headroom, and no causal rule
recovered any of it. Improving base signal quality moves far more than
re-arbitrating the 19% of days where the sleeves happen to disagree.

## 7c. Learned conflict arbitration (EV, then black box) — negative result

`research/conflict_ev_model.py`. On a conflicted asset-day the sleeves point
opposite ways by construction (`pair_dir = -ridge_sign`), so
`PnL(pair) = -PnL(ridge)` exactly and arbitration reduces to one binary bet:
*is ridge right here?* The EV target is `ev = ridge_sign × forward_return`.

Five model families, EV/interpretable first as requested, all evaluated by
substitution into the real allocator and the exact eval.py simulator:

| model | train | val | **test** | Δ test | train gate |
|---|---:|---:|---:|---:|:--|
| baseline (shipped) | 789.8 | 847.0 | **914.7** | — | — |
| A online global EWMA EV (hl 60) | 771.1 | 767.2 | 869.9 | −44.8 | no |
| A online global EWMA EV (hl 120) | 783.5 | 777.9 | 847.2 | −67.5 | no |
| A online global EWMA EV (hl 250) | 815.0 | 781.4 | 821.0 | −93.7 | PASS |
| B online EV per strength bucket (hl 60) | 754.0 | 794.5 | 836.6 | −78.1 | no |
| B online EV per strength bucket (hl 120) | 734.7 | 788.6 | 868.5 | −46.2 | no |
| C static linear EV regression | 862.3 | 775.3 | 788.1 | −126.6 | PASS |
| D static logistic | 867.7 | 787.0 | 720.8 | −193.9 | PASS |
| E gradient boosting (black box) | **1488.3** | 788.0 | 801.4 | −113.3 | PASS |

**Every model loses on test.** Four pass the train gate. Gradient boosting is
the clearest illustration in the whole study: train 1488.3 against a baseline
of 789.8 — an apparent +698, nearly doubling the score — and 801.4 on test,
*worse* than doing nothing. Train AUC 0.845, test AUC 0.514.

**Why no model can work here — the label itself inverts:**

| | train | validation | test |
|---|---:|---:|---:|
| P(ridge right) | **0.4727** | 0.5003 | **0.5220** |
| mean ev (following ridge) | **−0.001367** | +0.000694 | **+0.001151** |

On train, pairs win conflicts (ridge right only 47.3%); on test, ridge wins
(52.2%). Any model that learns "trust pairs on conflicts" from train is
systematically inverted out of sample. Both deviations are only ~2σ at
n≈2000 (SE ≈ 1.1pp) and point opposite ways — consistent with noise, not a
learnable regime.

**No signal, by three independent checks:**

* **AUC**: logistic 0.541 train → **0.503 test**; boosting 0.845 train →
  **0.514 test**. Test AUC is chance.
* **Selective arbitration does not help.** Restricting to the most confident
  predictions makes accuracy *worse*, not better — 0.5186 (all) → 0.5137
  (top 80%) → 0.5048 (top 40%) → 0.5071 (top 20%). With real signal,
  confidence and accuracy would rise together.
* **No stable feature.** Largest |corr(feature, ev)| anywhere is 0.066
  (SE ≈ 0.022), and 6 of 10 features flip sign across regions.

**The online EV models degraded least** (best test 869.9 vs 720.8 for static
logistic), confirming the design intuition that a causal, regime-adaptive
estimator is the right shape for this problem. But adapting faster cannot
extract information that is not there; they still lose to the baseline
because they spend every regime turn on the wrong side of a coin flip.

**Why this is expected in hindsight.** The conflict set is *defined* by two
otherwise-informative sleeves disagreeing. Conditioning on disagreement
selects precisely the instrument-days where their information cancels — an
adversarially ambiguous subsample. The ~145-point gap in §7b is the price of
that ambiguity, and it is not addressable by a better arbitrator.

## 7d. Units: score vs PnL (read before quoting any gap)

**Score is not PnL, and it is not per-window.** With
`score = mu · SR²/(SR²+1)` and this book at annualised Sharpe ≈ 8.8, the
factor is 77.8/78.8 ≈ 0.987, so **score ≈ 98.7% of mean DAILY PnL**. A score
of 914.7 means ~\$926/day, i.e. ~\$231.6k over a 250-day window. Multiply by
~250 to get per-window dollars.

Test window (days 751-1000):

| scenario | score | $/day | $/window | Sharpe | Δ$/window |
|---|---:|---:|---:|---:|---:|
| baseline (shipped) | 914.7 | 926 | 231,611 | 8.82 | — |
| regime oracle W=250d | 852.3 | 867 | 216,766 | 7.59 | **−14,845** |
| regime oracle W=50d | 915.2 | 928 | 232,008 | 8.43 | +397 |
| regime oracle W=10d | 986.0 | 997 | 249,371 | 9.29 | +17,760 |
| regime oracle W=1d | 1379.2 | 1,385 | 346,362 | 14.88 | +114,751 |
| full oracle on conflicts | 2256.8 | 2,261 | 565,337 | 22.23 | **+333,726** |

Three benchmarks people reach for, in dollars per window:

* **Full conflict oracle**: +\$333.7k — ~2.4× the whole book's current output.
  Requires knowing every individual conflict's outcome. Unreachable.
* **Realistic penalty** (§7b: conflicts arbitrated as well as the agreement
  bucket, ~145 score): ~**\$36k**.
* **Statistically reachable** (§3b of GBM_DEEPDIVE.md: 125-250d granularity,
  the only resolution at which the regime is detectable): **−\$14.8k**, i.e.
  acting on the best knowledge you could legitimately obtain *loses* money.

The oracle gap is genuinely large and simultaneously unreachable, because the
reachable portion of it is negative.

## 7e. Contradiction as state information, and a netted portfolio process

`research/contradiction_and_netting.py`. Everything above framed conflicts as
*"which sleeve is right"*, which is zero-sum by construction (on a conflict
`pair_dir = -ridge_sign`, so exactly one is right) and therefore can only ever
produce a pick-a-winner answer. Asking each sleeve about **its own** edge
instead is not zero-sum, and gives a different answer.

### A. Both sleeves degrade when contradicted — and lose their stability

Mean return in each sleeve's own direction, bps per instrument-day:

| sleeve / context | train | validation | test | **spread** |
|---|---:|---:|---:|---:|
| pair, unopposed | 18.08 | 17.59 | 18.38 | **0.80** |
| pair, confirmed | 30.96 | 34.20 | 34.83 | 3.88 |
| pair, contradicted | 13.13 | −6.43 | −11.51 | **24.64** |
| ridge, unopposed | 27.27 | 29.91 | 22.10 | 7.81 |
| ridge, confirmed | 30.96 | 34.20 | 34.83 | 3.88 |
| ridge, contradicted | −13.13 | 6.43 | 11.51 | **24.64** |

Degradation (contradicted − unopposed, bps): pair −4.95 / −24.01 / −29.89;
ridge −40.40 / −23.49 / −10.60. **Both sleeves are worse when contradicted,
in every region.**

Two things stand out:

1. **The unopposed pair edge is extraordinarily stable** — 18.08, 17.59,
   18.38 bps across three independent 250-day windows, a spread of **0.80
   bps**. Confirmed signals are the strongest (≈31–35 bps) and also stable
   (3.88). These are the parts of the book that genuinely work.
2. **Contradiction destroys stability, not just level.** The contradicted
   buckets swing 24.64 bps and *flip sign* between windows — ~30× the
   variability of the unopposed pair edge.

The two contradicted rows are exact mirrors (+13.13/−13.13, −6.43/+6.43,
−11.51/+11.51), which is the zero-sum structure. Combined with §7c's finding
that the sign is unpredictable, this gives the economic statement that the
who-wins framing obscured: **a contradicted position has expected value ≈ 0
before costs and negative after them, while consuming full risk budget.**
Contradiction is information about *state ambiguity*, not about direction.

### B. Why that does not immediately mean "go flat"

It should follow that contradicted positions be dropped — but `flat_conflict`
scores 768.2 on test vs 914.7 (§4). The reason is the §3 capital mechanism:
per-instrument limits mean risk budget freed on one instrument **cannot be
redeployed to another** (every other active instrument is already at its cap,
and idle instruments are idle for lack of any signal). So a zero-EV position
still contributes to a score that rewards gross deployment. That is a
property of this scoring function and limit structure, not a general trading
truth — under a shared risk budget the netting conclusion would dominate.

### C. Netted / internalised portfolio construction

Rather than arbitrating per asset and sizing bang-bang, net each sleeve's own
dollar demands into one book — pair legs entering at true relative size
(`pos_x = -beta·pos_y`, so the hedge survives netting) and ridge at its own
target — then construct the portfolio from the net.

| scheme | train | tr floor | val | val floor | test | test floor | gross$ | turn$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (shipped) | 789.8 | 649.7 | 847.0 | 758.6 | **914.7** | 670.8 | 435,400 | 285,984 |
| net_clip (pure netting) | 825.5 | 719.0 | 843.1 | 801.6 | 818.9 | 643.4 | 363,659 | 230,047 |
| net_clip + band 25% | 821.9 | 715.0 | 842.2 | 802.4 | 817.2 | 644.8 | 361,635 | 224,471 |
| net_scaled (full budget) | 302.5 | 235.4 | 359.2 | 341.5 | 356.6 | 301.6 | 113,854 | 82,566 |
| **net_direction_full** | **843.6** | **712.6** | **895.0** | **812.4** | 845.5 | 650.0 | 413,800 | 253,293 |
| net ridge ×0.5 | 959.4 | 866.0 | 821.9 | 811.4 | 743.1 | 578.9 | 430,800 | 170,843 |
| net ridge ×2 | 756.2 | 600.9 | 924.7 | 827.0 | 877.1 | 611.7 | 426,200 | 328,353 |

**`net_direction_full` is the first scheme in this entire study to pass the
pre-registered gate.** Both train blocks beat baseline (712.6 > 649.7,
974.4 > 929.5), the train floor improves (+62.9), it has a structural
justification (magnitude-weighted netting preserves hedge ratios and each
sleeve's intended sizing, where the shipped vote is sign-only), **and burned
validation confirms it** (+48.0 score, +53.8 floor). It also delivers the
internalisation benefit directly: 11% less turnover at comparable gross.

**And it still loses on test: 845.5 vs 914.7 (−69.2), floor 650.0 vs 670.8.**

Two honest readings, and this study cannot separate them:

* *Against the change*: it is the same shape as every other candidate — wins
  where it was selected, loses on the untouched window.
* *For the change*: days 751–1000 were used to prune 8 pairs from this book's
  universe, so the baseline's specific pair/ridge balance is partly fitted to
  test. `net_direction_full` beats the baseline on **both** windows that were
  not used that way, on score and floor.

The ridge-weight sweep repeats the §6 knife-edge exactly — train prefers less
ridge (×0.5 → 959.4), validation and test prefer more (×2 → 924.7 / 877.1) —
so the netting weight is no more stable than `RIDGE_CONFLICT_WEIGHT` and
should not be tuned.

**Status: genuinely open, unlike everything else here.** The netted process is
the only candidate that satisfied the pre-registered rule and then confirmed
on validation. It is not recommended for the live book on this evidence
alone — it fails the one window that was never a selection input — but it is
the only line of work in this study that has not been closed out, and the
first evidence that the independent-sleeves architecture, not just its
arbitration rule, is what costs.

## 8. Reproduce

```
python research/signal_cache.py --rebuild        # one 37s replay, caches raw signals
python research/diagnose_signal_overlap.py       # §2 overlap, hit rates, hedge integrity
python research/combine_schemes.py               # §4 all 11 schemes
python research/combine_round2.py                # §5 ALGO + turnover levers
python research/conflict_ev_model.py             # §7c learned arbitration (negative)
python research/conflict_ml_deepdive.py          # GBM_DEEPDIVE.md E1-E5
python research/conflict_online_ceiling.py       # GBM_DEEPDIVE.md §3b online ceiling
python research/contradiction_and_netting.py     # §7e conditional degradation + netting
```

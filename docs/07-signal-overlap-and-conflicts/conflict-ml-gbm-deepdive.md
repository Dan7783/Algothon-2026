# Can the conflict arbitrator be regularised into working? A deep dive

> **Source:** `analysis_outputs/signal_overlap/GBM_DEEPDIVE.md` in the team's working repository (relocated here for the public release).

Reproduce: `python research/conflict_ml_deepdive.py`
Companion: `OVERLAP_VERDICT.md` §7c (the original negative result)

---

## 0. First, a correction to the premise

The question as posed was *"why did gradient boosting perform so well during
test then bad during train."* The result was the **other way round**:

| | train | validation | **test** |
|---|---:|---:|---:|
| baseline (shipped) | 789.8 | 847.0 | **914.7** |
| gradient boosting | **1488.3** | 788.0 | **801.4** |
| | *+698.5* | *−59.0* | *−113.3* |

Gradient boosting was **excellent in-sample and destructive out-of-sample**.
This matters because the two directions have opposite diagnoses:

* *Good on test, bad on train* would be genuinely anomalous — usually a
  leakage bug, a distribution shift in your favour, or a broken split. Worth
  chasing.
* *Good on train, bad on test* is the single most common failure mode in
  applied ML: **the model memorised the training window.** It is a
  well-understood phenomenon with a large literature and a standard set of
  remedies — all of which are tested below, and none of which work here.

So the honest form of the question is: *the booster memorised train — can
regularisation, better validation, or adaptive retraining recover real signal
instead?* That is what this study answers. The answer is **no, and the reason
is measurable rather than a matter of taste.**

---

## 1. What actually happened, mechanically

The booster had 150 trees × depth 3 over 10 features and **2,033 training
rows**. That is roughly enough capacity to isolate individual observations.
The target is `is_ridge_right`, a near-fair coin (train base rate 0.4727).

A model with enough capacity will fit the *realised* coin flips of the
training window perfectly. Those flips carry no information about the next
window's flips, so the fitted structure evaluates as noise out of sample —
but it is *confidently* wrong noise, and because the arbitrator overrides the
shipped vote on 100% of conflicts, it converts a 52.8%-accurate rule into a
~51%-accurate one across ~2,000 instrument-days. Hence 801.4 vs 914.7.

The learning curves make the memorisation explicit.

### E1 — Capacity / learning curves

| capacity | train AUC | val AUC | test AUC |
|---|---:|---:|---:|
| depth 1 × 25 | 0.5727 | 0.4757 | 0.5014 |
| depth 1 × 400 | 0.6457 | 0.4888 | 0.5251 |
| depth 2 × 400 | 0.8187 | 0.5087 | 0.5145 |
| depth 3 × 400 | 0.9480 | 0.5170 | 0.5179 |
| depth 5 × 100 | 0.9654 | 0.5065 | 0.5055 |
| **depth 5 × 400** | **0.9999** | 0.5130 | 0.5138 |

Train AUC climbs monotonically to **0.9999 — perfect memorisation** — while
validation and test stay pinned in the 0.50–0.52 band throughout. The
train/OOS gap is created entirely by capacity and is not accompanied by *any*
OOS improvement. That is the textbook signature of fitting noise: in a
learning-curve diagnosis, a model with recoverable signal shows OOS
performance *rising* with capacity before it turns over. Here it never rises.

Note also that several low-capacity configurations produce **validation AUC
below 0.5** (0.4757, 0.4789, 0.4812). An AUC reliably under 0.5 is not
"no signal" — it is *inverted* signal, the first sign of the label flip
quantified in §3.

---

## 2. Every standard remedy, tested

### E2 — The regularisation frontier (81 configurations)

Grid over `max_depth ∈ {1,2,3}`, `n_estimators ∈ {10,50,200}`,
`min_samples_leaf ∈ {5,50,200}`, `learning_rate ∈ {0.01,0.05,0.2}`.

| metric | value |
|---|---:|
| configurations tried | 81 |
| honest pick (selected on validation) → test AUC | **0.5140** |
| best test AUC anywhere (look-ahead, unattainable) | **0.5255** |
| median test AUC across all 81 | 0.5119 |
| **configs reaching test AUC > 0.55** | **0.0%** |

This settles "we just haven't tuned it right." Even granting the model
**perfect hyperparameter foresight** — picking the configuration that happens
to maximise *test* AUC, which is impossible in practice — the ceiling is
0.5255. Regularisation controls the train/OOS *gap* by lowering the train
side; it never raises the OOS side, because there is nothing under it.

### E3 — Purged walk-forward with embargo

The methodologically correct evaluation for overlapping financial labels
(López de Prado 2018): retrain on a trailing window, purge observations whose
label horizon overlaps the test fold, and impose an embargo gap. This is also
a genuinely better *deployment* — it tracks the regime instead of freezing
one window's answer.

| trailing window | retrain every | embargo | val | test |
|---:|---:|---:|---:|---:|
| 250d | 25d | 5d | 758.5 | 829.5 |
| 250d | 50d | 5d | 742.8 | 841.9 |
| **500d** | **25d** | 5d | 779.4 | **847.0** |
| 500d | 50d | 5d | 748.7 | 830.0 |
| *baseline* | | | *847.0* | *914.7* |

All four lose. But note the ordering: walk-forward retraining (best test
847.0) **beats the static fit (801.4)**, and the longest window with the most
frequent retraining does best. Adaptation is directionally correct — it is
just recovering some of the damage the static model does, never exceeding the
do-nothing baseline.

---

## 3. Why it cannot be fixed: the two numbers that close the question

### E4 — What accuracy would actually be required?

Injecting arbitrators of controlled accuracy on conflicts and scoring through
the real simulator gives the transfer function from conflict accuracy to book
score (mean of 3 seeds):

| conflict accuracy | train | val | test |
|---:|---:|---:|---:|
| 0.50 | 740.0 | 751.4 | 749.6 |
| 0.52 | 791.1 | 800.9 | 806.3 |
| 0.54 | 840.5 | 844.8 | 858.7 |
| **0.56** | 909.1 | 897.8 | **926.0** |
| 0.60 | 1023.2 | 1017.5 | 1059.8 |
| 0.70 | 1336.2 | 1298.3 | 1364.3 |

**Accuracy required just to match the existing baseline:**

| region | required |
|---|---:|
| train | 0.5195 |
| validation | 0.5408 |
| **test** | **0.5566** |

*Caveat, stated because it cuts against the conclusion's strength:* this
arbitrator makes **i.i.d.** errors, whereas a real model's errors correlate
with market state. Independent coin-flip noise costs more Sharpe than
structured error, so 0.5566 modestly **overstates** the true requirement.
Taking the softer validation figure of 0.5408 as a floor does not change the
verdict below.

**The gap:** models must clear roughly **54–56%**. Every model in the study —
across 81 hyperparameter configurations, 5 model families and 4 walk-forward
schemes — lands at **51–52.5%**. The requirement is not marginally out of
reach; it is on the far side of the entire achievable range.

### E5 — Statistical power: the regime turns faster than the edge can be learned

| region | n | P(ridge right) | edge | SE | **z** |
|---|---:|---:|---:|---:|---:|
| train | 2,033 | 0.4727 | −2.73pp | 0.0111 | **−2.46** |
| validation | 1,837 | 0.5003 | +0.03pp | 0.0117 | +0.02 |
| test | 2,094 | 0.5220 | +2.20pp | 0.0109 | **+2.01** |

The edge is *individually significant at 5% in both directions*: −2.46σ on
train, +2.01σ on test. A quantity that is significantly negative in one
window and significantly positive in the next is not a signal to be modelled;
it is either noise being over-read, or a real effect whose sign changes
faster than any estimator can track.

Sample size for 80% power at α=0.05 (two-sided), at ~8.2 conflicts/day:

| true edge | observations needed | trading days |
|---:|---:|---:|
| 1pp | 19,600 | **2,390** (≈9.5 years) |
| 2pp | 4,900 | **598** |
| 3pp | 2,178 | 266 |
| 5pp | 784 | 96 |

**This is the crux.** The observed edge is ~2pp, which needs **~600 trading
days** to distinguish from zero. The observed edge **flips sign between
adjacent 250-day windows**. The estimation requirement exceeds the regime's
lifetime by roughly 2.4×, so by the time an edge is measurable it has already
inverted. No model class, regularisation scheme, or retraining cadence
changes that arithmetic — it is a property of the data, not of the estimator.

---

## 3b. "It fails because you stop feeding it context" — tested to its ceiling

A natural objection: the booster fails out of sample because it stops
learning at the train boundary. Keep feeding it realised outcomes and it
should track the regime instead of freezing one window's answer.

**Partly right, and it does help** — but it was already in the study, and it
has a ceiling that can be computed exactly.

Which models already refeed:

| model | refeeds? | test |
|---|---|---:|
| static GBM (E) | no — frozen after day 500 | 801.4 |
| purged walk-forward GBM (E3) | yes, retrains every 25–50d | 847.0 |
| online EWMA EV (A) | yes, after every conflict | 869.9 |
| *baseline (no arbitrator)* | — | *914.7* |

Continuous feeding monotonically improves the result (801 → 847 → 870). The
objection identifies a real defect. It just does not close the 45-point gap.

### The ceiling of every possible online method

Every online scheme — Hedge, SGD, incremental trees, periodic retraining —
is ultimately estimating one thing: *which sleeve is better right now.* So
the strict upper bound on all of them is an oracle handed that answer for
free, with no estimation error and no lag. Partition the timeline into W-day
blocks and follow whichever sleeve actually won each block:

| regime granularity W | train | val | **test** | beats baseline? |
|---:|---:|---:|---:|:--|
| 250d | 865.1 | 806.1 | 852.3 | **no** |
| 125d | 865.1 | 806.1 | 852.3 | **no** |
| 50d | 868.0 | 832.7 | 915.2 | barely (+0.5) |
| 25d | 868.0 | 876.5 | 915.2 | barely (+0.5) |
| 10d | 953.1 | 918.3 | 986.0 | yes (+71) |
| 5d | 1025.5 | 980.4 | 1037.9 | yes |
| 1d | 1338.7 | 1280.7 | 1379.2 | yes |
| *baseline* | *789.8* | *847.0* | *914.7* | |

*(Consistency check: the 250d oracle reproduces 852.3 exactly — on test the
whole-window answer is "follow ridge", which is precisely the
`s3_conflict_ridge_wins` scheme from `OVERLAP_VERDICT.md` §4. The machinery
agrees with the independent earlier run.)*

**Perfect regime knowledge at 125–250 day granularity still LOSES to the
baseline.** Break-even needs ~50-day granularity — and +0.5 points is noise,
not a margin. A real edge needs ~10-day granularity.

### Why that closes the question

Pair the ceiling against the power analysis in E5:

| requirement | granularity |
|---|---|
| granularity at which knowing the regime is **useful** | ≤ 50 days |
| granularity at which a 2pp edge is **detectable** (80% power) | ≥ 600 days |

These are mutually exclusive by roughly **12×**. Inside a 50-day block there
are ~410 conflicts, so SE ≈ 0.025 and the observed 2pp edge is **0.8σ** —
indistinguishable from noise. To measure the edge at all you must aggregate
to ~600 days, and at that granularity the oracle *loses*.

You cannot learn fast enough to act usefully, and you cannot act usefully at
speeds where learning is reliable. This bound holds for any estimator,
because it is a statement about the information in the data, not about the
model.

### The real online learners land exactly where theory predicts

| method | train | val | test |
|---|---:|---:|---:|
| Hedge η=0.01 | 856.5 | 797.1 | 806.1 |
| Hedge η=0.05 | 767.0 | 784.2 | 813.9 |
| **Hedge η=0.2** | 623.8 | 714.9 | **857.8** |
| Hedge η=1.0 | 674.3 | 716.3 | 793.8 |
| SGD partial_fit α=1e-4 | 698.4 | 816.5 | 766.3 |
| SGD partial_fit α=1e-2 | 669.0 | 823.9 | 766.1 |
| SGD partial_fit α=1.0 | 858.3 | 743.2 | 676.8 |
| *baseline* | *789.8* | *847.0* | *914.7* |

The best online learner (Hedge η=0.2, test 857.8) lands within 6 points of
the **250-day oracle (852.3)** — i.e. genuine no-regret online learning
achieves approximately the coarse-granularity bound and no better, exactly as
the regret theory says it should. Nothing approaches the ~50-day granularity
that break-even requires.

### Is it dangerous?

Not in the ways usually feared, and the distinction matters:

* **No look-ahead leakage.** The label is next-day realised return, genuinely
  observable at t+1, and every implementation here updates *strictly after*
  acting. This is legitimate, not cheating.
* **No feedback loop.** The arbitrator's choices do not change the data it
  subsequently learns from — pair and ridge signals are computed from prices
  regardless of what was allocated. So there is no self-reinforcing
  contamination, which is the classic hazard in online trading systems.

The actual danger is subtler and worth naming: **silent regime-chasing.** The
system *looks* like it is learning — weights move, it responds to new data,
it visibly beats the frozen model — while tracking a signal too slow to be
useful. From the inside, adaptivity and progress are indistinguishable. It is
precisely because online learning is principled, and because it *does* improve
on the frozen booster, that it invites the conclusion "right track, just need
a better learner." The ceiling table above is what prevents that: the
improvement is real, bounded, and lands below the do-nothing baseline.

So: **safe to run, but futile.** The risk is not blow-up; it is spending
effort while believing progress is being made.

## 4. Grounding in the literature

The result is not an oddity of this book; it is the expected outcome under
several well-documented findings.

**The forecast combination puzzle** — the single most direct precedent. Bates
& Granger (1969) established optimal weighting of competing forecasts. Yet
Stock & Watson (2004) documented that *simple, fixed weights routinely beat
estimated optimal weights* out of sample. Smith & Wallis (2009) attributed
this specifically to **estimation error in the weights**, and Claeskens,
Magnus, Vasnev & Wang (2016) gave a theoretical account showing that
estimated combination weights are unbiased-but-noisy and the added variance
typically swamps the theoretical gain. Timmermann (2006, *Handbook of
Economic Forecasting*) surveys the same conclusion. Our fixed vote is the
"simple weights" arm; every learned arbitrator is the "estimated weights"
arm; the fixed rule wins by 45–194 score points. This is the puzzle
reproduced almost exactly.

**Meta-labelling** — López de Prado (2018, *Advances in Financial Machine
Learning*, ch. 3) proposes precisely this architecture: a primary model
generates the side, a secondary ML model decides whether to act. He is
explicit that meta-labelling is intended to improve *precision/sizing* on an
already-profitable primary signal, and that it requires the primary signal to
have genuine edge in the conditioned subsample. Our conditioning set —
instrument-days where the two sleeves *disagree* — is selected to have the
least edge available, which is why the architecture cannot help here.

**Purged CV and embargo** — the E3 design follows López de Prado (2018,
ch. 7). We applied it and it improved the static result (801.4 → 847.0)
without reaching the baseline, which is the expected behaviour when the
methodology was never the binding constraint.

**Backtest overfitting and multiple testing** — Bailey, Borwein, López de
Prado & Zhu (2014, *Notices of the AMS*) show how easily in-sample
performance is manufactured by searching configurations; Bailey & López de
Prado (2014) formalise the correction as the Deflated Sharpe Ratio. Harvey,
Liu & Zhu (2016, *RFS*) argue that after accounting for the number of
strategies tested, a t-statistic near 2 should not be treated as a discovery
— they propose a threshold nearer t > 3. Our per-window |z| ≈ 2.0–2.5, from a
study that searched 81+ configurations, sits well inside the region their
work says is indistinguishable from selection noise. The train result
(z = −2.46) and the test result (z = +2.01) are a live demonstration.

**Realistic effect sizes for ML in finance** — Gu, Kelly & Xiu (2020, *RFS*)
find that even best-in-class ML on the US cross-section delivers monthly
out-of-sample R² on the order of a fraction of one percent, with tree
ensembles and neural nets only modestly beating linear models. Israel, Kelly
& Moskowitz (2020) discuss why: finance has a low signal-to-noise ratio,
short effective samples, and a non-stationary data-generating process. An
expectation that a booster on 2,033 rows would find a 5pp directional edge is
far outside what that literature supports.

**Forecasting under structural breaks** — Pesaran & Timmermann's work on
estimation-window choice under breaks, and Rossi (2013, *Handbook of Economic
Forecasting*, "Advances in Forecasting under Instability"), formalise the
trade-off we hit in E3: a longer window reduces estimation variance but
increases bias if a break occurred inside it. When the break frequency is
high relative to the data needed for estimation, *no* window choice is good.
Our §3 arithmetic (600 days needed, ~250-day regime) is that condition in its
degenerate form.

**Ensemble ambiguity** — Krogh & Vedelsby's (1995) decomposition writes
ensemble error as average member error minus an "ambiguity" (diversity) term.
Conflicts are, by definition, the maximum-ambiguity subsample of our two-model
ensemble. The decomposition says this is where the *ensemble gains most over
its members* — which is consistent with what we measured (the vote's synergy)
— but it says nothing about individual members being separable there. In
fact, conditioning on maximum disagreement selects the region where the two
information sets most nearly cancel, which is the mechanism behind the ~50%
base rate.

**Post-publication decay** — McLean & Pontiff (2016, *J. Finance*) find
predictor returns decay ~58% after publication, evidence that much apparent
in-sample edge is selection. The general lesson — that in-sample effect sizes
in finance are systematically inflated relative to what survives — is exactly
the 1488.3 → 801.4 pattern.

---

## 5. What the literature says to do instead

Every source above converges on the same prescriptions, and they are the ones
already followed by the shipped book:

1. **Prefer simple, fixed combination weights** to estimated ones when the
   sample is short and the SNR low (Stock & Watson; Smith & Wallis;
   Claeskens et al.). The shipped vote is that rule. Keep it.
2. **Do not model the maximum-ambiguity subsample.** Meta-labelling works on
   subsamples where the primary signal retains edge, not where two signals
   cancel. If ML is to be applied here, apply it to *sizing or abstention on
   agreement days* (the 0.55–0.57 hit-rate bucket), not to arbitration on
   conflicts.
3. **Raise the significance bar.** Harvey/Liu/Zhu's t > 3 applied to this
   problem means the edge would need |z| ≳ 3 in *the same direction* across
   independent windows. Ours is ±2 in *opposite* directions.
4. **Spend the modelling budget where the headroom is.** §7b of the verdict
   shows conflicts hold ~20% of total oracle headroom, proportional to their
   19% share of instrument-days — they are not a special loss centre. Roughly
   6,700 points of directional headroom sit in the *other* 81%.

---

## 6. Verdict

**No — the gradient booster cannot be optimised into working, and the reason
is not hyperparameters.**

Four independent measurements close it:

* **Ceiling**: across 81 configurations, with look-ahead hyperparameter
  selection, the best test AUC is **0.5255**; 0% of configurations exceed
  0.55.
* **Requirement**: the book needs roughly **54–56%** conflict accuracy just
  to match doing nothing. Models deliver **51–52.5%**.
* **Learnability**: the edge is ~2pp and needs **~600 trading days** to
  detect, but **inverts between adjacent 250-day windows** (train z = −2.46,
  test z = +2.01).
* **Online ceiling** (§3b): continuously refeeding outcomes helps
  (801 → 847 → 870) but cannot close the gap. Perfect, free regime knowledge
  at 125–250d granularity still **loses** (852.3); break-even needs ~50d, at
  which granularity a 2pp edge is only **0.8σ**. Useful granularity (≤50d)
  and detectable granularity (≥600d) are mutually exclusive by ~12×.

The requirement sits above the achievable ceiling, and the target moves
faster than it can be estimated. Those are properties of the data, so they
are invariant to model class — a neural network, a transformer, or a larger
ensemble faces the identical arithmetic.

The correct engineering decision is the one already in place: **keep the
fixed vote**. That conclusion is not a failure of the modelling effort; it
is the forecast combination puzzle behaving exactly as forty years of
forecasting literature says it should.

---

### References

* Bailey, D., Borwein, J., López de Prado, M., & Zhu, Q. (2014). Pseudo-Mathematics and Financial Charlatanism. *Notices of the AMS*, 61(5).
* Bailey, D., & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*, 40(5).
* Bates, J. M., & Granger, C. W. J. (1969). The Combination of Forecasts. *Operational Research Quarterly*, 20(4).
* Claeskens, G., Magnus, J., Vasnev, A., & Wang, W. (2016). The forecast combination puzzle: A simple theoretical explanation. *International Journal of Forecasting*, 32(3).
* Dietterich, T. (2000). Ensemble Methods in Machine Learning. *MCS*.
* Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*, 33(5).
* Harvey, C., Liu, Y., & Zhu, H. (2016). …and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1).
* Israel, R., Kelly, B., & Moskowitz, T. (2020). Can Machines "Learn" Finance? *Journal of Investment Management*, 18(2).
* Krogh, A., & Vedelsby, J. (1995). Neural Network Ensembles, Cross Validation, and Active Learning. *NIPS*.
* López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (chs. 3 meta-labelling, 7 purged CV/embargo)
* McLean, R. D., & Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? *Journal of Finance*, 71(1).
* Rossi, B. (2013). Advances in Forecasting under Instability. *Handbook of Economic Forecasting*, vol. 2B.
* Smith, J., & Wallis, K. (2009). A Simple Explanation of the Forecast Combination Puzzle. *Oxford Bulletin of Economics and Statistics*, 71(3).
* Stock, J., & Watson, M. (2004). Combination forecasts of output growth in a seven-country data set. *Journal of Forecasting*, 23(6).
* Timmermann, A. (2006). Forecast Combinations. *Handbook of Economic Forecasting*, vol. 1, ch. 4.

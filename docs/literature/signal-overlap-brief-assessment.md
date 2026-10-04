# Assessment: "Signal Overlap in Multi-Asset Pairs" brief vs. four_sigmas.py

> **Source:** `signal_overlap_brief_assessment.md` in the team's working repository (relocated here for the public release).

Measured 2026-07-29. Tooling: `research/effective_breadth.py`,
`research/vote_dictatorship.py`, variants under `research/variant_*.py`.

Baselines (eval.py, current `four_sigmas.py`): **951.5** on days 501-750,
**886.4** on days 751-1000.

---

## 1. The central claim (Part 1.2, Problem 2) does not hold

The brief argues the 88 pairs are ~12 independent bets restated, so expected IR
is overstated ~2.7x, and offers this as the mechanism behind the OOS gap.

Measured on each pair's **standalone daily PnL** (what the score actually sees,
not signal membership):

| window | N trading | mean \|pairwise corr\| | Meucci n_eff | PC1 share | IR inflation |
|---|---|---|---|---|---|
| 251-500 (train) | 86 | 0.061 | **64.9** (75%) | 5.2% | 1.15x |
| 501-750 (burned val) | 88 | 0.058 | **66.8** (76%) | 4.5% | 1.15x |
| 751-1000 (organiser) | 88 | 0.057 | **67.4** (77%) | 4.8% | 1.14x |

88 pairs deliver **~67 independent bets**, not 12. IR is overstated by ~1.15x,
not 2.7x. The result is stable across all three windows.

**Why the brief's intuition failed:** asset-level overlap is real (mean degree
3.7, ULXY in 11 pairs) but does not propagate to PnL overlap, because the pairs
rarely fire at the same time. Each has its own filter, z-thresholds and entry
timing. On ULXY, a median of **4 of 11** pairs are live at once; on FCSG, 3 of 8.
Shared legs ≠ shared bets.

**The brief also inverts the Marchenko-Pastur reading.** It treats
"3-8 eigenvalues above the noise bound" as the effective breadth. MP counts
detectable *common factors*; a correlation matrix whose eigenvalues nearly all
sit in the bulk is close to the identity, which means bets are close to
independent — the opposite conclusion. Breadth is the entropy/participation
measure (65-76), not the MP count.

**Verdict: the breadth-accounting diagnosis is closed, negative.** It is not the
mechanism behind the OOS gap. Grinold-Kahn ch.6, Meucci (2009) and the RMT
thread (Laloux, Plerou, Bouchaud-Potters) are answered by this table; reading
them will not change the number.

---

## 2. The vote is not a jury — it is a dictatorship (Problem 1 is moot, 3.3 is real)

The Condorcet/correlated-voters concern presumes several effective voters.
Measured over 16,143 contested asset-days (days 250-1000, >=2 live voters):

| quantity | value |
|---|---|
| nominal voters per asset | mean 2.97, median 3, max 10 |
| **effective** voters (weight entropy) | mean 2.14, **median 1.94** |
| top voter's share of weight | **median 69%** |
| asset-days where top voter's sign == consensus | **98.4%** |
| asset-days where one-pair-one-vote would flip the sign | 27.4% |
| asset-days with effective voters < 2 | 62.7% |

The consensus is decided by the single heaviest pair 98.4% of the time. So
Ladha/Berg on correlated juries is not the relevant literature — there is
usually no jury.

The cause is §3.3, which is correct: `priority = current_kappa * abs_z` is
dimensional. Per-pair median kappa spans **512x** (0.018 to 9.09) and median
priority **573x**, set by each pair's fitted OU rate rather than its edge. The
heaviest voters (HUXZ-DUCT 9.33, AGVF-ULXY 9.00, ALUT-HTRK 7.44) outweigh the
lightest (HUXZ-RRES 0.016) by ~600x. `PROTECTED_VOTE_BOOST = 4.0` is swamped by
this and is not interpretable, exactly as the brief says.

### Compressing the weight dispersion helps monotonically

| variant | 501-750 | 751-1000 |
|---|---|---|
| baseline (`kappa * abs_z`) | 951.5 | 886.4 |
| cross-sectional rank (partial compression) | 962.1 | 915.4 |
| `abs_z` only (kappa removed) | 995.6 | 920.9 |
| **one pair one vote** (`MODE = "equal"`, full compression) | **996.1** | **951.7** |

Monotone in how much dispersion is removed, on both windows independently. This
is a plateau, not a grid spike — the ordering is the evidence, not any one cell.

**Confound checked and eliminated.** Under `MODE = "equal"` the protected boost
stops being swamped (4.0 vs 1.0 instead of 4.0x a 600x-dispersed weight), so the
gain could have been the boost rather than the compression. It is not — with the
boost also removed:

| control | 501-750 | 751-1000 |
|---|---|---|
| equal vote, `PROTECTED_VOTE_BOOST = 1.0` | 1026.9 | 919.8 |

still +75.4 / +33.4 over baseline. The compression is the driver.

Note the boost is no longer inert once weights are equal, and it moves the two
windows in *opposite* directions (+30.8 on 751-1000, −30.8 on 501-750). That is
the signature of a knife-edge knob like `RIDGE_CONFLICT_WEIGHT` — leave it at
4.0, do not tune it.

### 125-day block detail (protocol rule 6: judge by floor)

| strategy | 251-375 | 376-500 | 501-625 | 626-750 | 751-875 | 876-1000 | median | **floor** |
|---|---|---|---|---|---|---|---|---|
| four_sigmas | 764.9 | 1033.5 | 853.7 | 1049.6 | 1190.3 | 580.7 | 943.6 | 580.7 |
| vote_equal | 768.9 | 1028.5 | 916.9 | 1075.6 | 1270.0 | 631.7 | 972.7 | **631.7** |
| vote_absz | 758.7 | 1019.8 | 919.0 | 1072.5 | 1202.5 | 637.9 | 969.4 | 637.9 |

**Honest protocol status: `vote_equal` FAILS rule 2 as written.** It gains
+4.0 on train block 251-375 but loses −5.0 on 376-500, and rule 2 requires both
train blocks to improve independently. On train it is a wash (−0.05% net across
the two blocks). Every gain sits in blocks that are burned validation or the
organiser window.

Arguments for shipping it anyway, for the user to weigh:
- Rule 7 says structural criteria outrank score criteria, and this is a
  structural defect with a measured cause (512x units dispersion, 98.4%
  dictatorship), not a knob found by search.
- It **removes** a parameter rather than adding one (`PAIR_CONFLICT_MODE` is an
  existing code path; `current_kappa` stops feeding allocation entirely).
- The block floor rises +51.0, and the floor is the statistic the protocol says
  correlated with the true-OOS 599.

Argument against: the train blocks are the only selection-legal evidence and
they say "no change". A −5.0/+4.0 split is noise either way.

---

## 2b. Equal weighting creates exact ties — and going flat on them is correct

Follow-up audit (`research/vote_tie_audit.py`). Under `"priority"` the weight is
continuous, so exact cancellation is measure-zero and the `tied` branch at
`four_sigmas.py:948` was **dead code**. Equal weights make 1v1 / 2v2 ties common.
An asset whose pair vote cancels is rescued only if the ridge is active on it;
otherwise `direction` stays 0 and the asset is **flat that day**.

| mode | cancelled (of pair-live) | ridge rescues | goes flat | assets/day forgone | deployed/day |
|---|---|---|---|---|---|
| priority, boost 4 | **0.0%** | — | — | 0.00 | \$431,600 |
| equal, boost 4 | 7.6% | 47.7% | 52.3% | **1.51** | \$416,480 |
| equal, boost 1 | 11.7% | 48.7% | 51.3% | 2.28 | \$408,800 |

(days 751-1000; train and 501-750 are within ~0.5pp of these.)

**Forcing a direction on ties loses on both windows.**
`research/variant_tiebreak_priority.py` breaks a pair-only tie with the tied
pairs' own `kappa*|z|` conviction instead of leaving it flat:

| | 501-750 | 751-1000 |
|---|---|---|
| equal, flat on tie (current) | **996.1** | **951.7** |
| equal, magnitude tiebreak | 971.2 (−25.0) | 940.4 (−11.3) |

A tie means there genuinely is no consensus, and arbitrating it with the same
dimensional weight the change just removed reintroduces the defect through the
back door. Keep the flat.

This also explains why `PROTECTED_VOTE_BOOST` stopped being inert: at 4.0 it
breaks protected-vs-unprotected ties, cutting forgone assets/day from 2.28 to
1.51.

**Standing conclusion updated.** Prior work
([[pair-ridge-overlap-verdict]]) found the score capital-dominated — every
scheme deploying less capital lost. Equal weighting is the first exception:
it deploys **3.5% less** (\$416.5k vs \$431.6k/day) and still scores **+65.3**
higher on 751-1000. Direction quality can outweigh deployment after all; the
"capital-dominated" heuristic is a strong prior, not a law.

## 3. Part 3 items

### 3.1 The leak is real but the described fix is a no-op

`PREVIOUS_OOS_REMOVED_PAIR_NAMES` has **zero runtime effect**. None of the five
names (CCNS-FARS, DIHO-RTTH, EORC-SPLZ, NAYO-FWWG, SPLZ-FWWG) appear in
`_SELECTED_PAIR_PARAMS` or `_SUPPLIED_AUTO_PAIR_PARAMS` — the 91-row source
catalog already excludes them. The frozenset and its assertion are a lint check
on a catalog that cannot contain them. Deleting the frozenset changes nothing;
undoing the leak would require re-adding five pairs whose fitted profiles are no
longer in the file.

The provenance problem the brief identifies is nonetheless correct and worse
than stated: **prices.txt is now 1000 days**, and 751-1000 shaped this catalog.
There is no clean holdout left locally. The 886.4 above is not a holdout number
and should not be reported as one.

### 3.2 ALGO — both halves tested, both negative

Force-flat: already settled (`BANG_BANG_LIMITS[ALGO] = 0`), monotone down on two
independent windows. Not revisited.

ALGO as a ridge **predictor** while staying untradeable was the untested half.
Built as `research/variant_ridge_algo_predictor.py` (51 predictors, 50 targets):

| | 501-750 | 751-1000 | floor |
|---|---|---|---|
| baseline | 951.5 | 886.4 | 580.7 |
| ALGO as ridge predictor | 946.2 (−5.3) | 880.7 (−5.7) | 576.5 (−4.2) |

Worse on both windows and on the floor. **Closed, negative.** ALGO carries no
usable common-factor information for the other 50.

### 3.4 Selection history

Correct and unpriced. Deflated Sharpe (Bailey & López de Prado 2014) remains the
right tool and is the one reading-list item this exercise did not answer.

---

## 4. Reading list triage after measurement

| still worth reading | why |
|---|---|
| Bailey & López de Prado (2014), Deflated SR | prices the selection history; §3.4 is the one open item |
| Harvey, Liu & Zhu (2016) | the t>3 bar for the original 96-pair screen |
| López de Prado (2018) ch.7 | purged CV; already partly used in prior conflict work |

| answered / not applicable | why |
|---|---|
| Grinold-Kahn ch.6, Meucci (2009), min-torsion | breadth measured at 67/88; inflation 1.15x |
| Laloux, Plerou, Bouchaud-Potters, Ledoit-Wolf | PC1 is 4.8%; there is no dense correlation structure to denoise |
| Ladha (1992), Berg (1993) | median 1.94 effective voters — no jury to correct |
| Avellaneda-Lee, Johansen, d'Aspremont, Cuturi | these replace the book wholesale; the overlap they fix costs ~1.15x here, so they are a rewrite motivated by a problem that measured small |

Clarke-de Silva-Thorley (transfer coefficient) is the one Part 2.1 item that
survives, and it points at what prior work already established independently:
the score `mu*SR^2/(SR^2+1)` is capital-dominated, so bang-bang at full limits
is score-optimal and any conviction sizing loses. That also disposes of Problem 3
(the discarded consensus margin) — it is discarded correctly under this scoring
function, de-overlapped or not.

# OU Engine Migration — Diagnosis, Per-Pair Results, and Conflict-Resolution Strategy

> **Source:** `DIAGNOSIS_AND_RECOMMENDATIONS.md` in the team's working repository (relocated here for the public release).

**Date:** 2026-07-30
**Status:** CORRECTED — prior “1057 beats friend OOS” is **REJECTED as OOS-overfit**,
not a ship success. New primary metric is **in-sample book score 31–750**.
**Success criterion (STRICT, user correction):**
1. **Primary:** beat `friend_book_latest` on **31–750** (~926.94 / reported 928.10).
2. **Secondary:** stay competitive on val **501–750** (~1042 / 1045).
3. **OOS 750–1000 is verification only** — never LOO/greedy/prune/select margin or
   exit mode by max OOS.
**Friend book reference:** `friend_book_latest.py` (canonical). Older snapshot kept as
`friend_book.py`. See `FRIEND_BOOK_REFERENCE.md` for the conflict-layer diff.

---

## 1. Executive summary

- **Retract:** mean-exit + m0.20 + 14 OU→zband prune scoring **OOS 1057** is
  **OOS-overfit junk for shipping** (train only ~831 ≪ friend ~927). Do not treat
  OOS wins as success.
- **Engine vs conflict (standalone desired PnL, 32 comparable OU pairs):**
  - **Train 31–750 aggregate:** friend z-band **+\$408k** vs OU optimal **+\$249k** /
    mean **+\$261k** → engine is **NOT ahead in-sample** (Δ ≈ −\$147k to −\$159k).
  - **OOS 750–1000 verify:** OU optimal/mean beat z-band by ~+\$58–60k — this is
    why earlier “31 OU winners” looked like engine success (selection on OOS desired).
  - Per-pair train winners: **12/32** (mostly mean-exit). Rest are OOS-only wins.
- **Conflict prototypes** (sklearn HistGB; xgboost not installed): `ml` conflict
  hits train **950** (> friend 927) but **fails val** (832 vs 1042) and **OOS decays
  hard** (806). Val accuracy ≈0.50 → not a honest ship. Simpler `flat_conflict` /
  `stronger_vote` do not beat friend train.
- **Honest best under new criterion:** still **`friend_book_latest`**. No hybrid yet
  beats friend train *and* stays competitive on val. Closest honest OU sleeve:
  12 train-standalone winners → train **861** / val **879** / OOS **912**.

---

## 2. Pipeline recap (what was built and run)

1. `windowed_tuner.py` — Optuna walk-forward tuner for the new engine.
2. 76 legacy catalog pairs tuned on the OU engine; 31 beat z-band on desired OOS.
3. `experiments/_build_merged_catalog.py` → `merged_strategy.py` (earlier hybrid on old friend layer).
4. `experiments/_pair_attribution.py`: desired-vs-realized PnL, flip/suppression rates.
5. **NEW:** `friend_book_latest.py` — friend's latest paste (ridge-confirm, deadlock
   overlay 1.0, ALGO residual).
6. **NEW:** `experiments/_build_merged_friend_ou.py` → `merged_friend_ou.py` — latest friend
   skeleton + 35 OU winners (31 hard + 4 extras).
7. **NEW:** `experiments/_run_friend_ou_bakeoff.py` / `experiments/_run_friend_ou_attribution.py` — pinned-env
   portfolio bake-off + conflict ablations.
8. `pair_screener.py` → 80 fresh OU-friendly candidates still tuning (do not block).

---

## 3. Individual-pair results (the success)

### 3.1 Legacy catalog (76 pairs), tuned OU vs legacy z-band — desired OOS PnL

Authoritative head-to-head from `_pair_attribution.json` (same pairs, same data,
conflict-free desired-position accounting):

| Outcome | Pairs | Desired Δ OOS |
|---|---|---|
| OU wins | 31 | **+\$74,352** |
| z-band wins | 45 | −\$136,841 |
| **Net (all 76, uniform OU)** | | **−\$62,489** |

Uniform OU-for-all is net-negative — the correct structure is the **cherry-pick**:
run OU only where it wins (capturing the +\$74.4k) and keep z-band elsewhere.

**The 31 pairs where OU improved performance** (desired OOS PnL, AUD):

| Pair | z-band OOS | OU OOS | Δ | book flips vs pair |
|---|---|---|---|---|
| MTNS-BLBT | 101 | 5,476 | +5,376 | 476 |
| CUBO-ULXY | 3,557 | 8,912 | +5,355 | 522 |
| AMRP-DUCT | 4,841 | 10,194 | +5,353 | 331 |
| NGTE-ILVX | −1,636 | 3,319 | +4,955 | 754 |
| AETS-FCSG | −472 | 4,462 | +4,934 | 513 |
| MDGI-ULXY | −216 | 4,465 | +4,680 | 542 |
| ANSO-BENI | −811 | 3,272 | +4,084 | 475 |
| HRET-HRND | 5,941 | 9,809 | +3,867 | 427 |
| AENO-IHOZ | 2,379 | 5,707 | +3,328 | 530 |
| RTTH-RRES | 5,452 | 8,603 | +3,151 | 521 |
| AETS-ILVX | 495 | 3,625 | +3,130 | 673 |
| SMAH-FWWG | 5,108 | 7,956 | +2,848 | 572 |
| ALUT-HTRK | 512 | 3,081 | +2,568 | 249 |
| ANSO-RCRI | 2,347 | 4,829 | +2,482 | 628 |
| RTTH-SPLZ | 1,138 | 3,551 | +2,413 | 577 |
| RTTH-RCRI | 1,081 | 3,078 | +1,998 | 574 |
| HRET-MDGI | 4,712 | 6,410 | +1,698 | 596 |
| MSDP-EAFC | 1,378 | 3,005 | +1,627 | 433 |
| ANSO-RRES | 2,857 | 4,298 | +1,441 | 455 |
| HRET-MTNS | 1,341 | 2,466 | +1,125 | 634 |
| FWWG-NGTE | 3,969 | 5,069 | +1,100 | 604 |
| ELLT-FCSG | 727 | 1,820 | +1,093 | 471 |
| SMAH-ULXY | 4,571 | 5,604 | +1,032 | 745 |
| RTTH-FWWG | 3,061 | 4,020 | +958 | 570 |
| ANSO-RTTH | 2,316 | 3,177 | +861 | 607 |
| ANSO-IHOZ | 958 | 1,707 | +749 | 552 |
| CCNS-NAYO | 1,052 | 1,696 | +644 | 635 |
| DUCT-HRND | 1,943 | 2,570 | +628 | 394 |
| SRNA-NWIG | 2,058 | 2,682 | +624 | 527 |
| LSST-EAFC | 670 | 818 | +148 | 643 |
| GARI-FWWG | 2,087 | 2,185 | +98 | 388 |

Also recorded in `auto_tuning_results/_ou_winners_table.txt`.

### 3.2 Extra screened cohorts included toward ~35

From the 18 screened OU cohorts already in the merged catalog, four pairs that
are **already in the friend's latest book** and look individually strong under OU
were added (friend orientation):

| Friend pair | Reason |
|---|---|
| ALUT-CCNS | tuned_current cohort |
| MSDP-ULXY | verified_new |
| OTCS-EELT | verified_new (friend orientation of EELT-OTCS) |
| LSST-RTTH | reject_cohort but +4.9k desired OOS under OU |

Protected 7 kept as friend's **z-band** profiles (not overlaid) to avoid disturbing
the friend's validated protected set without a clean z-band vs OU head-to-head in
this book. Negative reject cohorts absent from the friend book were not added.

**Hybrid catalog:** 88 pairs = **35 OU + 53 z-band**
(`auto_tuning_results/merged_friend_ou_catalog.json` /
`merged_friend_ou_meta.json`).

### 3.3 Re-check under friend_latest skeleton

Attribution (`_friend_ou_attribution.json`), days 200–1000, OOS from 801:

| Book | Desired OOS | Realized book OOS | Gap | Flips |
|---|---|---|---|---|
| friend_book_latest | 249,442 | 170,400 | 31.7% | 13,513 |
| merged_friend_ou | **300,546** | 164,806 | **45.2%** | **26,151** |

Desired OOS **+51k** with OU winners → individual success holds under the new
portfolio layer. The extra desired is mostly **destroyed by more flips**.

Of the 35 OU-switched pairs, **26/35** beat the friend's own z-band desired OOS
(sum of positive deltas ≈ +\$67k). Top deltas include OTCS-EELT, MDGI-ULXY,
HRET-HRND, ALUT-HTRK, NGTE-ILVX.

### 3.4 Validity controls

- **Placebo:** real tuned pairs beat placebo pairs OOS → gains are not pure
  selection artifact.
- **Factor exposure:** pair-level PnL correlation with ALGO returns is low →
  the book is not a disguised market bet.

### 3.5 80-pair OU search (complete 2026-07-30)

Universe-wide windowed search finished (`Pipeline execution fully complete`).
Artifacts: `auto_tuning_results/ou_search_tuning_results.json` (80/80 pairs with
`final_params`) and `ou_search_individual_summary.json`.

| Criterion | Count |
|---|---|
| Complete | **80 / 80** |
| `best_value` > 0 (WF tuning objective) | **66** |
| train `score` > 0 under final params | **79** |
| OOS `score` > 0 (standalone, primary success) | **34** |
| train and OOS both > 0 | **34** |

Top OOS standalones: RRES-NAYO (38.5), ELLT-HRET (31.8), ANSO-ULXY (31.6),
ULXY-EORC (24.9), MTNS-SPLZ (18.4), ULXY-CCNS (18.0), MMBT-MDGI (16.6),
MMBT-RTTH (16.0), CUBO-NGTE (14.4). Individual OOS ≠ portfolio add — many
positive standalones will conflict with the live book.

**Greedy assembly:** `experiments/_greedy_assemble.py` rewired to live ship base
`merged_friend_ou` (shared + margin 0.15). Append is impossible (hard 88-pair
cap in `merged_friend_ou.py`), so assembly is **swap-mode**: LOO-rank removable
slots, try each candidate into the worst 10, then sequential improving swaps.
Protected + reintroduced pairs are frozen; banned removed-set names skipped
(e.g. FWWG-NAYO). Candidates prefer the **34 OOS-positive** standalones from
`ou_search_individual_summary.json`.

**Restarted 2026-07-30 on friend windows:** primary accept/reject judge is now
**OOS 750–1000** (base ≈ **906.86**, not the old 801–1000 score 842.51). Accepted
swaps also log train **31–750** and val **501–750**. Friend baseline on the same
OOS window ≈ **974.45**. Older append-greedy on `merged_strategy` hybrid
(~781 OOS, margin 0.30) is obsolete for ship. Outputs:
`_greedy_final_catalog.json` / `_greedy_history.json`.

---

## 4. Why the *portfolio* underperforms the legacy book

1. **Survivorship bias in the legacy catalog.** The 88-pair book was selected by
   old-engine profitability.
2. **Parameter overfit decay.** Tuned OU params decay OOS more than coarse z-bands.
3. **Turnover drag.** OU entries fire more often → commission drag.
4. **Conflict-resolution destruction** (dominant under friend_latest+OU): flips
   nearly double and erase the +\$51k desired gain.

None of these say the OU engine is worse at trading a pair. They say the *book
integration* was tuned around the old engine's behavior.

---

## 5. Conflict layer deep-dive

### 5.1 Friend latest vs older friend_book (material diffs)

| Knob | Old `friend_book.py` | Latest `friend_book_latest.py` |
|---|---|---|
| Ridge trade | λ=1.0 / θ=0.75 | λ=1.52 / θ=0.31 |
| Ridge confirm | (none) | λ=1.0 / θ=0.75 independent confirm |
| Ridge override | any opposing ridge vote | only if confirm agrees **and** preliminary combined vote already prefers ridge |
| Deadlock overlay | 0.25 | **1.0** (full unused slot) |
| ALGO residual | True (research-candidate comment) | True (part of current book) |

### 5.2 Earlier hybrid (merged_strategy on older friend layer)

- ~40% of desired pair PnL destroyed by flips.
- Clean cherry book (31 OU + z-band losers + 18 cohorts, margin 0.30):
  **OOS 799.92 / full 911.96** vs friend OOS ~828 / full ~1005.
- Hybrid base clean OOS 781.36.

### 5.3 New hybrid on friend_latest — clean bake-off

Pinned env (all `MERGED_*` cleared unless listed). Source:
`auto_tuning_results/_friend_ou_bakeoff.json`.

| Config | OOS score | Full score | OOS Sharpe | Full Sharpe |
|---|---|---|---|---|
| friend_book_latest | **839.27** | **982.56** | 7.34 | 8.98 |
| friend+OU default | 815.93 | 821.71 | 7.00 | 7.41 |
| + CONSENSUS_MARGIN=0.15 | **842.51** | 829.65 | 7.26 | 7.51 |
| + CONSENSUS_MARGIN=0.30 | 819.54 | 815.13 | 7.20 | 7.52 |
| + CONSENSUS_MARGIN=0.50 | 804.65 | 759.18 | 7.66 | 7.55 |
| + FLIP_TO_FLAT | 828.45 | 825.91 | **8.10** | **8.30** |
| + FLIP_TO_FLAT + m0.30 | 824.88 | 813.41 | 8.23 | 8.33 |
| + MAG_WEIGHTED | 760.43 | 815.67 | 6.65 | 7.50 |
| + MAG_WEIGHTED + m0.30 | 679.74 | 751.45 | 6.73 | 7.79 |
| + PROTECTED_BOOST=2 | 805.29 | 812.36 | 6.93 | 7.31 |
| + BOOST=2 + m0.30 | 828.07 | 792.46 | 7.11 | 7.27 |
| + PROTECTED_BOOST=1 | 803.31 | 792.74 | 6.90 | 7.14 |

Attribution flips under ablations (same desired 300,546 OOS):

| Config | Flips | Suppressions | Realized book OOS |
|---|---|---|---|
| friend+OU default | 26,151 | 606 | 164,806 |
| + margin 0.30 | 24,256 | 4,744 | 165,195 |
| + flip→flat | 20,933 | 5,417 | 166,305 |
| + flip→flat + m0.30 | 19,617 | 8,650 | 165,504 |

Flips fall under flip→flat / margin, but the **desired→realized dollar gap stays
~45%** — score gains from margin-0.15 are risk-path / timing effects, not full
recovery of the +\$51k desired edge. Magnitude-weighted voting **hurt**.

---

## 6. Conflict-resolution strategies (updated ranking)

1. **`CONSENSUS_MARGIN ≈ 0.15` on shared voter (best OOS so far).** Still the only
   setting that beats friend OOS (842.51 vs 839.27). Default in `merged_friend_ou`.
2. **OU-specific structural modes (implemented — see §10).** `sleeve` / `ou_flat` /
   `twopass` via `MERGED_OU_CONFLICT`. Sleeve cuts cross-engine flips and lifts
   Sharpe, but **does not yet beat shared on score**. Keep as research path.
3. **Flip → flat.** Cuts flips ~20% and lifts Sharpe; combine with light margin after
   re-tuning; m0.30+flip was slightly worse OOS than flip alone.
4. **Protected-boost re-tune.** Lowering boost to 2 or 1 did **not** help alone;
   only mild help with margin. Deprioritize vs margin/sleeves.
5. **Magnitude-weighted votes.** Failed (`MERGED_MAG_WEIGHTED=1`) — do not ship.
6. **Greedy assembly of the 80-pair search** (started on `merged_friend_ou`
   shared+0.15) — add pairs only when
   *portfolio* OOS improves under the chosen conflict rule.

---

## 7. Caveats

- **Env contamination:** always pin/clear `MERGED_*` before `eval.py` (PowerShell
  session leaks). All Section 5.3 numbers are pinned-env.
- **OOS window reuse:** screening/tuning/judging overlap → winner's-curse risk on
  portfolio accept/reject.
- Per-pair desired PnL is conflict-free; right for *engine* comparison only.
- Attribution raw PnL ≠ eval score (score is risk-scaled); use both.

---

## 8. Recommendations

1. **Keep OU params** for the 31 winners (+ 4 justified extras) — individual
   success is established under both the old hybrid and friend_latest.
2. **Keep EORC-NGTE** (protected z-band) — train-rich / thin OOS but still
   profitable; see §9.
3. **Ship conflict default `MERGED_OU_CONFLICT=shared` + `CONSENSUS_MARGIN=0.15`.**
   Structural sleeve/ou_flat/twopass are available for bake-off but do not beat
   shared on OOS score yet (§10).
4. Next sleeve research: oppose/capacity policies aimed at recovering desired→realized
   gap without the suppression spike (attribution: sleeve flips↓ but supp↑).
5. **Greedy-add** from the completed 80-pair search only when portfolio OOS rises
   under `merged_friend_ou` shared+margin0.15 (pinned env). Standalone OOS>0 is
   necessary but not sufficient — run `experiments/_greedy_assemble.py` / check
   `_greedy_history.json`.
6. Conflict defaults are already decided (shared+0.15). Do **not** re-bake conflict
   knobs until greedy accepts are measured.

### Files created / updated this pass

| File | Role |
|---|---|
| `friend_book_latest.py` | Canonical friend paste (new reference) |
| `FRIEND_BOOK_REFERENCE.md` | Diff notes vs old `friend_book.py` |
| `merged_friend_ou.py` | Friend latest + 35 OU winners |
| `experiments/_build_merged_friend_ou.py` | Rebuild hybrid module/catalog |
| `auto_tuning_results/merged_friend_ou_catalog.json` | Engine-tagged catalog |
| `auto_tuning_results/merged_friend_ou_meta.json` | Winner/extra metadata |
| `experiments/_run_friend_ou_bakeoff.py` | Pinned-env score bake-off |
| `experiments/_run_friend_ou_attribution.py` | Flip/suppression attribution |
| `auto_tuning_results/_friend_ou_bakeoff.json` | Bake-off numbers |
| `auto_tuning_results/_friend_ou_attribution.json` | Attribution numbers |

---

## Old non-window verification: overfit or OOS targeting?

**Verdict (2026-07-30).** For RTTH-NAYO specifically, evidence favors **Hypothesis C (genuine edge)** with a **mild Hypothesis B caveat at the process level**, not classic param overfit (A). The old non-windowed tuner (`latest_tuner.py`) never put days 801–1000 into the Optuna objective — only purged walk-forward on 201–800 — and its RTTH-NAYO OOS is *stronger* than the shipped windowed params (Sharpe 2.39 vs 1.87), with 90% train→OOS retention. The stable-8 numbers in `verification.json` are **not** from that old method: they are the windowed W-R+tb bake-off (`_bakeoff_wratio_tbrk.json`), and that *config* was chosen because it had the best mean OOS Sharpe (2.63) / retention (83%) across these eight pairs — so reported OOS for the book is mildly selection-optimistic. Overall, KEEP pairs look like real edges vs placebos; most *new* screen candidates look like classic A.

### What each method actually touched

| Stage | Days used | Touches 801–1000? |
|---|---|---|
| `latest_tuner.py` Optuna objective | Purged WF folds inside **201–800** | **No** (OOS only post-hoc report) |
| `windowed_tuner.py` Phase A/C + tiebreak | Train windows / WF inside **201–800** | **No** for param search |
| Tuner-config bake-off (pick W-R+tb vs alternatives) | Compared configs on these pairs’ **OOS** | **Yes — method selection** |
| `verify_pairs.py` screens / DSR / verdicts | Train screen + OOS screen + OOS Sharpe | Yes for *judgment*, not for param fit |
| Placebo cohort | Same tune budget; never selected on OOS | Cleanest noise benchmark (max ann. OOS Sharpe **1.42**) |

### RTTH-NAYO across methods

| Source | Train score / Sharpe | OOS score / Sharpe | Retention | Notes |
|---|---|---|---|---|
| Old non-window `tuning_results.json` | 60.44 / 2.65 | **48.10 / 2.39** | **90%** | Purged WF CMA-ES; OOS unseen in objective |
| Shipped W-R+tb (`verification` / `latest_strategy`) | 70.31 / 3.01 | 34.47 / 1.87 | 62% | Placebo rank 0; KEEP; OOS screen 4/5 |
| Windowed percentile (`windowed_tuning_results.json`) | 72.75 / 3.11 | 27.49 / 1.61 | 52% | More conservative selected trial |
| Same windowed file, ratio alternative | 71.50 / 3.06 | 37.55 / 1.99 | 65% | Not the selected trial |

Old method does **not** look like A (memorize train, die OOS). If anything it generalizes better on this pair than the shipped windowed params. Windowed is more conservative / regime-balanced, not “exposing” a fake OOS edge that only the old method invented.

### Verification KEEP / REVIEW set (from `verification.json`)

| Pair | Verdict | Train Sh | OOS Sh | Ret% | Beats placebos | Read |
|---|---|---|---|---|---|---|
| ILVX-SMAH | KEEP | 3.29 | 4.92 | 149 | yes | Strong; DSR≈0.95 |
| AENO-NWIG | KEEP | 3.01 | 4.70 | 156 | yes | Strong; DSR≈0.94 |
| ACAC-HUXZ | KEEP | 3.50 | 2.46 | 70 | yes | Healthy |
| HETT-ULXY | KEEP | 3.33 | 1.97 | 59 | yes | Healthy |
| CTGI-EELT | KEEP | 3.42 | 1.92 | 56 | yes | Healthy |
| NAYO-RTTH | KEEP | 3.01 | 1.87 | 62 | yes | Healthy; screen holds OOS |
| EORC-NGTE | KEEP | 3.37 | 1.66 | 49 | yes | Thin vs placebo ceiling 1.42 |
| ALUT-CCNS | REVIEW | 2.47 | 1.54 | 62 | yes | Thin + OOS G4 fail |
| MSDP-ULXY | ADD | 1.84 | 1.76 | 95 | yes | Cleanest *new* candidate |
| EELT-OTCS | REVIEW | 1.03 | 2.13 | 206 | yes | Reverse retention — treat as lucky OOS / B-risk |

### New rejects = classic overfit (Hypothesis A)

High train Sharpe collapsing OOS and losing to placebos: HRET-RTTH (2.75→−0.03), CCNS-MMBT (2.60→−0.33), LSST-NGTE (2.28→−0.58), HRET-NAYO (2.61→0.59, 23% ret). That pattern is the control for what A looks like in this pipeline — and it is **not** what RTTH-NAYO / the KEEP set show.

### Hypothesis call

1. **RTTH-NAYO:** **C**, with mild process-level **B** (bake-off picked W-R+tb using OOS on this cohort). Not A under the old non-window tuner.
2. **Stable-8 overall:** same — real vs placebo, but discount the *level* of reported OOS slightly because the winning tuner config was OOS-selected; individual params still fit only on 201–800.
3. **Old non-window “looks better OOS”:** for RTTH-NAYO that is **not** because it fitted 801–1000; the objective never saw those days. Stronger OOS is consistent with an honest post-hoc eval (or luck), not targeting.
4. **Practical:** keep treating RTTH-NAYO as a protected edge; use placebos / retention / multi-config agreement as the honesty check; do not re-judge the book solely on 801–1000 after choosing methods on that same window.

---

## 9. EORC-NGTE — KEEP (train-rich, still profitable OOS)

**User direction (2026-07-30):** do not discard; characterize the train/OOS pattern.
EORC-NGTE is a **protected z-band** row in `merged_friend_ou` (not one of the 35
OU-switched winners — see `merged_friend_ou_meta.json` → `zband_kept`). It remains
in the hybrid book.

### Numbers

| Source | Train score / Sharpe | OOS score / Sharpe | Retention | OOS PnL (approx) |
|---|---|---|---|---|
| Old non-window (`_bakeoff_lt.json`) | 43.26 / **3.01** | 19.92 / **1.65** | ~55% Sharpe | **+\$5,449** |
| Windowed W-R+tb (`verification` / `_bakeoff_wratio_tbrk`) | 49.33 / **3.37** | 20.04 / **1.66** | **49%** Sharpe | **+\$5,471** |
| Friend latest desired (z-band, attribution) | — | — | — | **+\$4,048** OOS |
| Catalog OU desired (`_pair_attribution` tuned) | — | — | — | **+\$5,470** OOS |

Book flips on this pair under friend+OU shared voter: ~142–166 (friend alone 142);
flip→flat cuts to ~52–73. Suppressions appear only under margin/flip ablations.

### Read

- Pattern is **train-inflated with thin but real OOS**, not classic collapse-to-noise.
  OOS Sharpe ~1.66 still clears the placebo ceiling (~1.42); DSR/placebo rank keeps
  the verification **KEEP**.
- Absolute OOS dollars stay material (~\$4–5.5k desired) even at ~50% Sharpe retention.
- **Recommendation: KEEP** in the book (protected z-band). Do not drop for thin
  retention alone. Optional later experiment: OU overlay head-to-head on this pair
  (desired OU already ~+\$1.4k vs friend z-band) — separate from conflict work.

---

## 10. OU-specific conflict resolution (structural modes)

Implemented in `merged_friend_ou.py` behind `MERGED_OU_CONFLICT`:

| Mode | Behavior |
|---|---|
| `shared` (default) | Single bang-bang voter; `CONSENSUS_MARGIN` / `FLIP_TO_FLAT` knobs |
| `sleeve` | OU and z-band vote **separately**; solo/agree → full LEG; oppose → `MERGED_OU_SLEEVE_OPPOSE`=`net`\|`flat`\|`stronger`. Ridge stays on z-band sleeve. |
| `ou_flat` | Shared voter, but if result would flip OU consensus → OU abstains, z-band(+ridge) keeps the asset |
| `twopass` | Z-band+ridge settle at full LEG; OU only fills **unused** asset slots |

Helpers: `MERGED_OU_SLEEVE_FRAC` (default 0.5), `MERGED_OU_SLEEVE_SPLIT`=`equal`\|`demand`.

Bake-off / attribution scripts: `experiments/_run_ou_conflict_bakeoff.py`,
`experiments/_run_ou_conflict_attribution.py` → `auto_tuning_results/_ou_conflict_bakeoff.json`,
`_ou_conflict_attribution.json`.

### Eval scores (pinned env)

| Config | OOS score | Full score | OOS Sharpe | Full Sharpe |
|---|---|---|---|---|
| friend_book_latest | **839.27** | **982.56** | 7.34 | 8.98 |
| shared (margin 0.15 default) | **842.51** | 829.65 | 7.26 | 7.51 |
| ou_flat | 832.07 | 831.69 | 7.22 | 7.54 |
| twopass | 776.25 | 891.40 | 6.87 | 8.12 |
| sleeve (solo=full, oppose=net/flat) | 774.11 | 775.35 | **8.25** | 8.32 |
| sleeve + demand split | 761.41 | 741.34 | 7.94 | 7.82 |
| sleeve frac=0.35 | 781.57 | 816.21 | 7.99 | 8.48 |
| sleeve oppose=stronger | 749.12 | 713.34 | 7.45 | 7.26 |

### Attribution (days 200–1000, OOS from 801; desired OOS fixed at 300,546 for OU books)

| Config | Book OOS $ | Flips | OU flips | ZB flips | Suppressions |
|---|---|---|---|---|---|
| friend | 170,400 | 13,513 | 0 | 13,513 | 210 |
| shared | 169,983 | 25,562 | 15,826 | 9,736 | 1,649 |
| sleeve | 155,696 | 21,009 | 13,352 | 7,657 | 10,196 |
| ou_flat | 167,877 | 25,229 | 15,645 | 9,584 | 1,681 |
| twopass | 159,367 | 29,981 | 22,326 | 7,655 | 780 |

### Verdict

- **Winner on portfolio OOS score:** still **`shared` + CONSENSUS_MARGIN=0.15**
  (842.51 > friend 839.27). None of the structural modes beat it on score yet.
- **Sleeve** does what we asked structurally (cross-engine cannot bang-bang flip each
  other) and **cuts flips** vs shared (~26k→21k) with the best OOS Sharpe among
  hybrids (8.25), but **realized book $ and score fall** — oppose capacity / idle
  suppressions dominate.
- **ou_flat** is a mild shared tweak; almost no score help.
- **twopass** starves OU on contested assets (only unused slots) → more OU flips in
  attribution and weak OOS score; full-window looks less bad (891) but not actionable.
- **Ship default:** keep `MERGED_OU_CONFLICT=shared` (margin 0.15). Keep sleeve code
  as the next research lever (capacity / oppose policy / demand split), not as the
  production default.
- **Do not** re-run `experiments/_build_merged_friend_ou.py` without restoring the conflict layer
  from git — builder regenerates from `friend_book_latest` and will wipe sleeve logic.

### Files this pass

| File | Role |
|---|---|
| `merged_friend_ou.py` | `MERGED_OU_CONFLICT` = shared\|sleeve\|ou_flat\|twopass |
| `experiments/_run_ou_conflict_bakeoff.py` | Pinned-env score bake-off |
| `experiments/_run_ou_conflict_attribution.py` | Flip/suppression attribution |
| `auto_tuning_results/_ou_conflict_bakeoff.json` | Score table |
| `auto_tuning_results/_ou_conflict_attribution.json` | Flip table |
| `experiments/_build_merged_friend_ou.py` | Warns on rebuild; env stubs for OU conflict knobs |

---

## 11. Friend-window validation (31–750 / 501–750 / 750–1000)

**Date:** 2026-07-30. Friend reported base scores **928.10 / 1045.36 / 975.10** on
train / validation / OOS, with OOS described as **750–1000**.

### Windows that reproduce those numbers

Code comments in `friend_book_latest.py` still say config was chosen on
**31–500 train / 501–750 validation / 751–1000 holdout**. The three *reported*
scores do **not** match that train start:

| Window | `friend_book_latest` score | vs friend report |
|---|---|---|
| **31–750** (full pre-OOS / “train”) | **926.94** | −1.16 vs 928.10 |
| **501–750** (validation) | **1042.37** | −2.99 vs 1045.36 |
| **750–1000** (OOS; user was right) | **974.45** | −0.65 vs 975.10 |

Rejected alternatives: documented **31–500** scores only **865.62**; **751–1000**
scores **982.14** (too high vs 975.10). Residual ~1–3 pts is likely paste/harness
drift, not a wrong window. Module: `friend_book_latest` (not `friend_book`).

### Same windows on ship candidate (`merged_friend_ou`, MERGED_* cleared)

Defaults = `MERGED_OU_CONFLICT=shared`, `CONSENSUS_MARGIN=0.15`.

| Book | 31–750 | 501–750 | 750–1000 |
|---|---|---|---|
| Friend report | 928.10 | 1045.36 | 975.10 |
| `friend_book_latest` (repro) | 926.94 | 1042.37 | 974.45 |
| `merged_friend_ou` shared m0.15 | **728.02** | **772.43** | **906.86** |
| `merged_friend_ou` ou_flat | 731.87 | 785.02 | 902.35 |
| `merged_friend_ou` sleeve | 675.06 | 776.73 | 840.09 |

### Verdict

On **friend’s own windows**, pure friend beats every merged config by a wide
margin on train and validation, and still leads OOS (**974 vs 907**). The earlier
“merged beats friend OOS” result was on our **801–1000** convention, not this
**750–1000** slice. Ship candidate remains competitive only under our prior OOS
definition; it is **not** ahead under friend’s reported train/val/OOS triple.

**Greedy restart (same day):** wrong-window greedy (judge 801–1000, base 842.51)
was killed. `experiments/_greedy_assemble.py` now judges accept/reject on **750–1000** only
(base ≈ 906.86), with train/val diagnostics logged for accepted swaps. Swap-mode
on `merged_friend_ou` shared m0.15, preferring the 34 OOS-positive search pairs.

---

## 12. OU exit mode: optimal-stopping vs conventional mean-exit

**Date:** 2026-07-30. Hypothesis: exit at the mean (`spread_z` crosses 0) instead of
holding / flipping at opposite `a*` → more round-trips, less capital parked →
easier conflict management.

### Current vs conventional exit (code)

In `merged_friend_ou.PairAgent.catch_up` for `engine=="ou"`:

| Mode | Entry | While long (`side=+1`) | While short (`side=-1`) |
|---|---|---|---|
| **optimal** (default) | `\|spread_z\| ≥ a*` | flip short if `spread_z ≥ a*` | flip long if `spread_z ≤ -a*` |
| **mean / conventional** | same `a*` entry | flatten if `spread_z ≥ 0` | flatten if `spread_z ≤ 0` |

Friend z-band already exits near the mean (`z_exit` often `0.001`). Toggle:
`MERGED_OU_EXIT=optimal|new|mean|conventional` (default `optimal` so the
running optimal-exit greedy is undisturbed).

### Bake-off (pinned MERGED_*; friend windows)

Source: `auto_tuning_results/_ou_exit_bakeoff.json`.

| Config | 31–750 | 501–750 | **750–1000** | 801–1000 |
|---|---|---|---|---|
| `friend_book_latest` | **926.94** | **1042.37** | **974.45** | 839.27 |
| optimal + m0.15 (ship) | 728.02 | 772.43 | 906.86 | 842.51 |
| **mean + m0.15** | 751.01 | 809.47 | **941.14** | **863.16** |
| mean + m0.15 + ou_flat | 758.00 | 827.84 | 936.21 | 854.95 |
| mean + m0.15 + sleeve | 682.19 | 789.56 | 846.19 | 785.81 |

Mean-exit lifts OOS **+34** vs optimal (941 vs 907) and also lifts train/val;
still short of friend on all three. Sleeve under mean hurts score; ou_flat is
neutral/slightly worse OOS than shared.

### Conflict / turnover attribution (warm 31→1000, metrics on 750–1000)

Source: `auto_tuning_results/_ou_exit_attribution.json`.

| Config | Flips (OU) | Suppress | OU days in pos | Side changes | Book $ turnover | Desired→realized gap OOS | Book PnL OOS |
|---|---|---|---|---|---|---|---|
| friend | 4519 (0) | 0 | — | 2153 | 75.9M | 63k | 251k |
| optimal m0.15 | 8282 (**5008**) | 316 | 8715 | 2973 | 82.0M | 152k | 234k |
| **mean m0.15** | 6871 (**3699**) | 195 | **6937** | **3995** | **76.4M** | **138k** | **242k** |

Mean-exit: **−26% OU flips**, **−20% OU hold-days**, more agent side changes
(more round-trips as hypothesized), slightly *lower* book $ turnover/comm, and
~+\$8k realized book OOS. Hypothesis confirmed for conflict management.

### Closing the friend OOS gap under mean-exit

Plan executed:

1. Keep friend’s conflict layer; switch OU to mean-exit.
2. Margin sweep under mean → best **m0.20** (942.54) over m0.15 (941.14);
   flip→flat / boost=2 hurt.
3. LOO-revert each OU pair → friend’s z-band params; 18/35 reverts help OOS.
4. Greedy multi-revert of damaging OU (14 accepted) → catalog
   `auto_tuning_results/_mean_exit_best_catalog.json` (21 OU remain).

| Book | 31–750 | 501–750 | **750–1000** |
|---|---|---|---|
| Friend | 926.94 | 1042.37 | 974.45 |
| mean m0.15 full OU | 751.01 | 809.47 | 941.14 |
| mean m0.20 full OU | — | — | 942.54 |
| **mean m0.20 + 14 OU→zband** | 830.88 | 929.57 | **1057.15** |

**Beat friend OOS:** yes (**1057 > 974**) — **RETRACTED AS SUCCESS.**
Train/val remain below friend (~831 / ~930 vs ~927 / ~1042). This catalog was
selected by maximizing OOS; under the corrected criterion it is **rejected**.

Reverted (OU → friend z-band): LSST-EAFC, HRET-MTNS, AETS-ILVX, SMAH-FWWG,
RTTH-SPLZ, GARI-FWWG, OTCS-EELT, ELLT-FCSG, ANSO-RTTH, SRNA-NWIG, DUCT-HRND,
MSDP-EAFC, LSST-RTTH, ALUT-CCNS.

### Verdict / recommendation (SUPERSEDED — see §13)

~~Ship mean+m0.20+pruned catalog for OOS.~~ **Do not ship the 1057 path.**
OOS is verify-only. Prefer friend baseline until an in-sample winner exists.

### Files this pass

| File | Role |
|---|---|
| `merged_friend_ou.py` | `MERGED_OU_EXIT` optimal\|mean switch in OU `catch_up` |
| `experiments/_run_ou_exit_bakeoff.py` | Scores + OOS attribution optimal vs mean |
| `experiments/_run_mean_exit_close_gap.py` | Margin sweep + LOO/greedy OU→zband revert |
| `auto_tuning_results/_ou_exit_bakeoff.json` | Score table |
| `auto_tuning_results/_ou_exit_attribution.json` | Flip/turnover table |
| `auto_tuning_results/_mean_exit_close_gap.json` | Gap-close report |
| `auto_tuning_results/_mean_exit_best_catalog.json` | **REJECTED** OOS-pruned catalog (21 OU) |

---

## 13. Correction: in-sample-first criterion (1057 path rejected)

**Date:** 2026-07-30 (user correction).

> OOS is a heuristic / PnL-decay check. A config that loses badly in-sample is a
> **failure**, even if OOS looks great.

| Config | 31–750 (primary) | 501–750 (secondary) | 750–1000 (verify) | Status |
|---|---|---|---|---|
| Friend report | 928.10 | 1045.36 | 975.10 | target |
| `friend_book_latest` | **926.94** | **1042.37** | **974.45** | **baseline / current best honest** |
| mean m0.15 full OU | 751.01 | 809.47 | 941.14 | fails train |
| optimal m0.15 full OU | 728.02 | 772.43 | 906.86 | fails train harder |
| **REJECTED** mean m0.20 + 14 OU→zband | 830.88 | 929.57 | **1057.15** | OOS-overfit |
| OOS-greedy swaps (finished) | ~828 | ~991 | ~1095 | OOS-overfit; do not use |
| Train-standalone OU winners (12) mean | 861.38 | 879.12 | 912.10 | better honesty, still < friend |
| ML conflict mean m0.15 | **950.02** | 831.69 | 806.11 | beats train only; **fails val + OOS decay** |

**1057 path rejected for shipping decisions.** Demote
`_mean_exit_best_catalog.json`. Do not LOO/greedy/prune on 750–1000.

---

## 14. Engine check: standalone desired PnL (optimal vs mean vs friend z-band)

Source: `auto_tuning_results/_standalone_exit_compare.json` (`experiments/_run_standalone_exit_compare.py`).
Conflict-free PairAgent simulation; train **31–750** primary; OOS verify only.
Compared **32** OU catalog pairs that also exist in friend’s z-band book
(stable-8 without OU catalog rows skipped).

### Aggregate

| Mode | Σ desired train 31–750 | Σ desired OOS 750–1000 |
|---|---|---|
| Friend z-band | **+408,293** | +87,553 |
| OU optimal | +249,161 (Δ **−159,132**) | +147,831 (Δ +60,278) |
| OU mean | +260,836 (Δ **−147,457**) | +145,501 (Δ +57,948) |

**Train wins:** z-band 20 / OU-optimal 4 / OU-mean 8.

### Verdict: is engine ahead standalone?

**No on the primary (train) metric.** Aggregate in-sample desired PnL strongly
favors friend’s z-band. The earlier “31 OU winners / +\$74k” story was driven by
**OOS desired** selection — same overfit pattern as the 1057 book prune.

**Partial yes on a subset:** 12 pairs beat z-band on train standalone (best exit):
CUBO-ULXY, MTNS-BLBT, ALUT-HTRK, AETS-ILVX, ANSO-BENI, ANSO-RTTH, DUCT-HRND,
RTTH-RRES, ANSO-IHOZ, HRET-MTNS, GARI-FWWG, ALUT-CCNS (mostly mean-exit).

Therefore conflict resolution is **not** “the only remaining problem” for the
full 35-OU hybrid — the hybrid is carrying many OOS-only pair wins that hurt
train. Conflict work still matters for the **train-positive sleeve**.

---

## 15. Conflict prototypes beyond bang-bang (sklearn GBDT)

`xgboost` not installed; used **sklearn `HistGradientBoostingClassifier`**
(no new packages installed). Modes added to `merged_friend_ou.py`:
`flat_conflict`, `stronger_vote`, `ml` (+ `MERGED_ML_MODEL`).

Model fit on conflict rows days **31–500** only; HP picked on val **501–750**
accuracy (not OOS). Val accuracy ≈ **0.50** (near chance for 3-class).

| Book | 31–750 | 501–750 | 750–1000 |
|---|---|---|---|
| friend | **926.94** | **1042.37** | **974.45** |
| shared mean m0.15 | 751.01 | 809.47 | 941.14 |
| flat_conflict mean | 704.77 | 787.67 | 798.26 |
| stronger_vote mean | 749.57 | 801.85 | 911.49 |
| **ml conflict mean** | **950.02** | 831.69 | 806.11 |
| REJECTED 1057 prune | 830.88 | 929.57 | 1057.15 |

ML is the only config that **beats friend train**, but it **fails secondary val**
and shows **catastrophic OOS decay** (retention far worse than friend). Treat as
a research spike — **not ship**.

Train-only OU sleeve (12 winners, mean, shared conflict): train **861** — closes
part of the gap vs full OU (751) but still short of friend.

### Recommended next steps

1. Keep **`friend_book_latest`** as ship default until train+val both clear.
2. Re-screen OU pairs on **train standalone** (not OOS desired); build tiny sleeve.
3. Improve conflict ML labels (realized conflict-harm, not raw next-return sign);
   fit 31–500, require val≥friend before looking at OOS.
4. Retarget any greedy/LOO to **judge train 31–750** (never 750–1000).

### Files this pass

| File | Role |
|---|---|
| `experiments/_run_standalone_exit_compare.py` | Standalone zband/opt/mean desired PnL |
| `experiments/_run_ml_conflict_prototype.py` | Fit/eval conflict ML + baselines |
| `experiments/_run_train_ou_winners_bakeoff.py` | Train-winner OU sleeve book scores |
| `merged_friend_ou.py` | `flat_conflict` / `stronger_vote` / `ml` modes |
| `auto_tuning_results/_standalone_exit_compare.json` | Engine table |
| `auto_tuning_results/_ml_conflict_prototype.json` | Conflict bake-off |
| `auto_tuning_results/_train_ou_winners_bakeoff.json` | Honest OU sleeve scores |
| `auto_tuning_results/_ml_conflict_model.joblib` | sklearn HistGB model |

## 16. Conflict winners: kappa / sigma / priority / tree (friend book)

Question: when pairs disagree on an asset, can kappa, sigma, priority (kappa*|z|),
protected status, or a decision tree pick the profitable side better than
current equal-boost bang-bang?

### When conflicts happen
On `friend_book_latest`, true **pair-pair sign conflicts** (both + and - legs
active on same asset): **4311** fit days 31-500, **2380** val 501-750.

Current mode (`we love sussy q`) weights votes as protected=4 else=1 � it does
**not** use kappa. `priority` mode (kappa*|z|) exists but was not the default.

### Predicting next-day asset return (weak label)
All simple rules and a depth-4 tree are ~coin-flip on val (~50-52%).
Bang-bang 52.3%, max_kappa 51.2%, sum_priority 49.7%, tree 52.6%.
**No usable edge** from kappa/sigma for next-day direction.

### Predicting which side's pairs win in isolation (better label)
Label = side with higher isolated next-day pair-leg PnL:

| Rule | Fit acc | Val acc | Val captured iso $ |
|---|---|---|---|
| bangbang (current) | 0.498 | **0.542** | 8.9M |
| majority | 0.521 | 0.541 | 8.7M |
| protected | 0.522 | 0.540 | 8.4M |
| max_kappa | 0.507 | 0.513 | 7.1M |
| sum_priority | 0.520 | 0.510 | 6.6M |
| iso oracle | 1.000 | 1.000 | **31.1M** |

Oracle captures ~3.5x more isolated PnL than bang-bang, so conflict skill
**could** matter � but kappa/priority/**tree features do not unlock it**.

### Book bake-off (train primary)
| Mode | 31-750 | 501-750 | 750-1000 |
|---|---|---|---|
| **equal boost (current)** | **946.4** | **1056.5** | **978.0** |
| flat weak consensus m0.30 | 939.1 | 1042.0 | 969.7 |
| flat m0.50 | 922.7 | 994.8 | 937.3 |
| priority (kappa*|z|) | 897.3 | 954.1 | 876.8 |

**Priority voting hurts.** Flattening weak fights also hurts. Current equal +
protected boost remains best among tested conflict rules.

### Verdict
1. Conflicts are frequent and oracle headroom is large.
2. **Kappa / sigma / |z| priority is not a better winner rule** than equal+protected.
3. Shallow trees on those features do not generalize past ~52% and prior HistGB
   conflict ML overfit train (950) then failed val/OOS.
4. Next research should target **EV / isolated pair PnL forecasts**
   (see `research/whole_vector_ev_conflicts.py`), not raw kappa comparisons.

Sources: `experiments/_investigate_conflict_priority.py`, `experiments/_investigate_conflict_followup.py`,
`auto_tuning_results/_conflict_priority_study.json`,
`auto_tuning_results/_conflict_priority_followup.json`.

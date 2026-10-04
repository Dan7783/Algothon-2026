# Validation protocol (pre-registered)

> **Source:** `cusum_output/VALIDATION_PROTOCOL.md` in the team's working repository (relocated here for the public release).

Ground truth that motivates this document: the book scored **878** on the
501-750 window we had been selecting on and **599** on the organiser's true
out-of-sample feed — a 32% haircut. Every rule below exists to prevent a
repeat. This protocol is *pre-registered*: the rules are written down before
the next experiment, and an experiment that deviates from them does not count.

## 1. Status of each data region (be honest about contamination)

| days | status | may be used for |
|---|---|---|
| 1-250 | warmup | filter/hedge state only, never scoring |
| 251-375, 376-500 | **train blocks** | selection, tuning, pruning decisions |
| 501-625, 626-750 | **burned validation** | reporting and sanity checks ONLY — these blocks were selection inputs for months of prior work; a gain here is *not evidence* |
| organiser OOS feed | **true holdout** | final verdicts; arrives rarely; never iterate against it |

The single most important rule: **days 501-750 are not a holdout.** Anything
that "improves test" was probably mined from it. Treat improvements there as
noise unless they were locked in from train blocks alone.

## 2. Decision rules for any proposed change

1. **State the hypothesis and the acceptance rule before running.** If the
   rule is invented after seeing results, the experiment is exploratory and
   cannot ship.
2. Selection may use **train blocks only**, and a candidate must improve
   **both** train blocks independently — not their average. (A candidate
   that wins one and loses the other is noise; averaging hides it.)
3. **Neighbourhood-smoothed** score must also improve, so an isolated grid
   spike cannot win. Prefer plateaus; a peak whose one-step neighbours fail
   is rejected regardless of its height.
4. Per-pair effects must be **portfolio-confirmed** (leave-one-out /
   greedy on the book), because the EV-MWIS conflict layer makes pair
   contributions non-additive. Measured examples: pruning all 7 "bad" pairs
   lost test −6.0; deleting dead rows moved the floor 766→710 via the hub
   penalty.
5. Score selected changes on 501-750 **once**, after freezing. Report the
   result whatever it is. If it degrades, the change does not ship, and you
   do NOT search for a variant that passes — that restarts the mining.
6. Judge robustness by **block floor and spread, not median**. The 125-day
   block scores inside 251-750 look stable for a fitted book (they were all
   selection inputs); the floor is the only number that correlated with the
   599.
7. **Structural criteria outrank score criteria.** A pair with persistent
   cointegration (ADF < −3 in all sub-windows) is not deleted on a marginal
   train reading; a knob that is provably inert is hardcoded, not tuned.

## 3. Tooling

- `research/block_consistency.py <modules...>` — per-block scores, median,
  floor for any strategy module.
- `research/pair_audit_trainonly.py` — leave-one-out pair marginals, verdict
  from train blocks only.
- `research/tune_survivors.py` — the reference implementation of rules 1-5
  (per-pair sweep, both-blocks + smoothed gates, greedy portfolio confirm,
  held-out scored once at the end).

## 4. Empirical verdicts already established (do not re-run these hoping for
a different answer)

| question | verdict | evidence |
|---|---|---|
| Does per-pair score tuning generalise? | **No.** Train +14.0 → held-out −15.5 under the strictest protocol | `STEP3_PER_PAIR_VERDICT.md` |
| Are the CUSUM/adapt/snap knobs worth tuning? | No — they compensate for the deliberately damped gain (R=1); with q correctly scaled the CUSUM goes inert (17 cuts → 1) | `STAGED_TUNING.md` |
| Is R=1 a bug? | It is a deliberate regulariser; "fixing" it improves tracking but costs ~35 train median at every q | `STAGED_TUNING.md` |
| Which config fields matter per-pair? | Only EV (conflict priority), require_adf, leg_scale; 18 of 29 fields were constants, thresholds share at ~2 pts of floor | `SLIM_BOOK.md` |
| Does stripping the book to few pairs help? | No — real OOS 599 beats every stripped variant (~480, ~400); sleeves and pair count are genuine diversification | `OVERFIT_REDUCTION.md` |
| Is the book market-neutral? | Yes — PnL-on-market t < 1, R² < 0.004; hedging worth +0.03 score | `beta_exposure.md` |

## 5. What is still legitimately open

- Sleeve-level (not per-pair) structural choices validated by rules 1-5.
- Efficiency work (no behaviour change) — always allowed.
- Anything testable against the organiser's next true-OOS print, treated as
  a single pre-registered bet, not a search.

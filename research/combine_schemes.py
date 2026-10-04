#!/usr/bin/env python
"""Structurally distinct ways of merging the pair and ridge sleeves.

PRE-REGISTERED ACCEPTANCE RULE (written before results were inspected, per
cusum_output/VALIDATION_PROTOCOL.md):

  A scheme may be recommended only if ALL of the following hold.
  1. It beats the shipped baseline on BOTH train blocks independently
     (251-375 and 376-500) -- not on their average.
  2. Its train block floor is >= the baseline's train block floor.
  3. It has a structural justification that does not reference any score,
     i.e. it fixes a mechanism identified by the diagnostic.
  Days 501-750 are BURNED validation: used as a confirmation/sanity read
  only, never to pick between schemes. Days 751-1000 are scored ONCE for
  the finalists and reported whatever the outcome.

  A scheme that wins on validation or test but fails rule 1 or 2 does NOT
  ship -- that is the mining pattern that produced the previous 878 -> 599
  out-of-sample haircut.

Diagnostic findings these schemes are designed against (see
diagnose_signal_overlap.py; all three regions agree, so these are structural
rather than regime artifacts):

  * Only ~61% of active pair-days keep BOTH legs pointing the way the pair
    asked. ~34% end up with one leg naked -- a market-neutral spread turned
    into a directional bet -- because the merge votes per ASSET, so a pair's
    two legs are arbitrated independently.
  * Even when both legs survive, the delivered hedge ratio is off by a
    median ~15%, because bang-bang sizes every winning leg at the full
    instrument limit regardless of the pair's beta.
  * Conflicted assets are the worst bucket by hit rate (0.51-0.53 vs 0.55-0.57
    for agreement) and by mean PnL.
  * Which sleeve *should* win a conflict is NOT stable: on train, letting
    ridge win conflicts loses money (-7.9k); on validation and test it makes
    money (+15.3k, +30.8k). Any fixed conflict-winner rule fitted on train
    would have picked the wrong side for the later windows. This is why the
    conflict-winner knob is treated as unlearnable rather than tuned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.combine_lab import (
    DLR_LIMIT,
    N_INST,
    TEST,
    TEST_BLOCKS,
    TRAIN,
    TRAIN_BLOCKS,
    VAL_BLOCKS,
    VALIDATION,
    crash_mask,
    evaluate,
    pair_consensus,
    region_report,
    score,
)
from research.signal_cache import load_or_build, load_prices

import testbedv2_tuned_ridge_protectedbarbell as strategy

OUTPUT_DIR = ROOT / "analysis_outputs" / "signal_overlap"
BB = strategy.BANG_BANG_LIMITS
BOOST = strategy.PROTECTED_VOTE_BOOST
RIDGE_W = strategy.RIDGE_CONFLICT_WEIGHT


def _sleeve_state(cache, row, prices, t):
    """Everything the schemes share: votes, signs, activity, crash mask."""
    consensus, pair_active = pair_consensus(cache, row, BOOST)
    ridge_raw = cache["ridge_raw"][row]
    ridge_strength = cache["ridge_strength"][row]
    ridge_sign = np.sign(ridge_raw)
    ridge_active = ridge_sign != 0.0
    ridge_vote = RIDGE_W * ridge_sign * ridge_strength
    pair_dir = np.sign(consensus)
    return {
        "consensus": consensus,
        "pair_active": pair_active & (pair_dir != 0.0),
        "pair_dir": pair_dir,
        "ridge_sign": ridge_sign,
        "ridge_active": ridge_active,
        "ridge_strength": ridge_strength,
        "ridge_vote": ridge_vote,
        "crash": crash_mask(prices, t),
    }


def _bangbang(direction, crash):
    direction = direction.copy()
    direction[crash] = 0.0
    return np.clip(BB * direction, -DLR_LIMIT, DLR_LIMIT)


def _resolve(state):
    """The shipped arbitration: signed vote sum, ties to ridge."""
    combined = state["consensus"] + state["ridge_vote"]
    direction = np.sign(combined)
    active = state["pair_active"] | state["ridge_active"]
    tied = active & (direction == 0.0) & state["ridge_active"]
    direction[tied] = state["ridge_sign"][tied]
    return direction


# --------------------------------------------------------------------------
# Pair allocation that PRESERVES each pair's hedge
# --------------------------------------------------------------------------

def pair_dollars_exclusive(cache, row):
    """Highest-priority non-overlapping pairs, each keeping its beta hedge.

    This is the base strategy's own conflict rule: sort by kappa*|z| and take
    a pair only if neither leg is already owned. Each surviving pair gets the
    exact leg sizes it asked for (pos_x = -beta * pos_y), so the spread stays
    market-neutral and correctly hedged. Instruments no pair claimed are left
    free for another sleeve to use.
    """
    dollars = np.zeros(N_INST)
    owned = np.zeros(N_INST, dtype=bool)
    y_idx = cache["pair_y_index"]
    x_idx = cache["pair_x_index"]
    pos_y = cache["pair_pos_y"][row]
    pos_x = cache["pair_pos_x"][row]
    priority = cache["pair_priority"][row]
    protected = cache["pair_protected"]

    order = np.argsort(-(priority * np.where(protected, BOOST, 1.0)))
    for pair_idx in order:
        if abs(pos_y[pair_idx]) <= 1.0e-9:
            continue
        yi, xi = int(y_idx[pair_idx]), int(x_idx[pair_idx])
        if owned[yi] or owned[xi]:
            continue
        owned[yi] = owned[xi] = True
        dollars[yi] += pos_y[pair_idx]
        dollars[xi] += pos_x[pair_idx]
    return dollars, owned


# --------------------------------------------------------------------------
# Schemes
# --------------------------------------------------------------------------

def s0_baseline(cache, row, prices, t):
    """Shipped: asset-level bang-bang vote, ridge weighted 0.75 by strength."""
    state = _sleeve_state(cache, row, prices, t)
    return _bangbang(_resolve(state), state["crash"])


def s1_conflict_flat(cache, row, prices, t):
    """Abstain on contested assets.

    Structural claim: the diagnostic shows conflicts are the lowest-hit-rate
    bucket AND that which sleeve should win is not stable across windows.
    When arbitration is unlearnable, not trading is the honest response.
    """
    state = _sleeve_state(cache, row, prices, t)
    direction = _resolve(state)
    conflict = (
        state["pair_active"]
        & state["ridge_active"]
        & (state["pair_dir"] != state["ridge_sign"])
    )
    direction[conflict] = 0.0
    return _bangbang(direction, state["crash"])


def s2_conflict_pair_wins(cache, row, prices, t):
    """Pairs always win contested assets (ridge yields)."""
    state = _sleeve_state(cache, row, prices, t)
    direction = _resolve(state)
    conflict = (
        state["pair_active"]
        & state["ridge_active"]
        & (state["pair_dir"] != state["ridge_sign"])
    )
    direction[conflict] = state["pair_dir"][conflict]
    return _bangbang(direction, state["crash"])


def s3_conflict_ridge_wins(cache, row, prices, t):
    """Ridge always wins contested assets (pairs yield)."""
    state = _sleeve_state(cache, row, prices, t)
    direction = _resolve(state)
    conflict = (
        state["pair_active"]
        & state["ridge_active"]
        & (state["pair_dir"] != state["ridge_sign"])
    )
    direction[conflict] = state["ridge_sign"][conflict]
    return _bangbang(direction, state["crash"])


def s4_pairs_only_exclusive(cache, row, prices, t):
    """Pairs alone, hedge intact, no ridge. Standalone reference."""
    dollars, _ = pair_dollars_exclusive(cache, row)
    dollars[crash_mask(prices, t)] = 0.0
    return np.clip(dollars, -DLR_LIMIT, DLR_LIMIT)


def s5_ridge_only(cache, row, prices, t):
    """Ridge alone, bang-bang. Standalone reference."""
    state = _sleeve_state(cache, row, prices, t)
    return _bangbang(state["ridge_sign"], state["crash"])


def s6_orthogonal(cache, row, prices, t):
    """Full separation: pairs own their instruments, ridge takes the rest.

    Structural claim: this removes the overlap entirely rather than
    arbitrating it. Each instrument is driven by exactly one sleeve, so a
    pair's hedge can never be broken by a ridge opinion, and ridge still
    harvests every instrument no pair is using. Both sleeves keep their own
    native sizing (pairs keep beta, ridge stays bang-bang).
    """
    pair_dollars, owned = pair_dollars_exclusive(cache, row)
    state = _sleeve_state(cache, row, prices, t)
    ridge_dollars = BB * state["ridge_sign"]
    dollars = pair_dollars + np.where(owned, 0.0, ridge_dollars)
    dollars[state["crash"]] = 0.0
    return np.clip(dollars, -DLR_LIMIT, DLR_LIMIT)


def s7_orthogonal_bangbang_pairs(cache, row, prices, t):
    """Separation, but pairs are sized bang-bang instead of by beta.

    Isolates which half of the hedge problem matters: leg integrity (kept
    here, since both legs of a taken pair always trade together) versus the
    beta hedge ratio (discarded here, both legs at full limit).
    """
    pair_dollars, owned = pair_dollars_exclusive(cache, row)
    state = _sleeve_state(cache, row, prices, t)
    dollars = np.where(owned, BB * np.sign(pair_dollars), BB * state["ridge_sign"])
    dollars[state["crash"]] = 0.0
    return np.clip(dollars, -DLR_LIMIT, DLR_LIMIT)


def s8_orthogonal_ridge_confirms(cache, row, prices, t):
    """Separation, plus ridge may veto a pair it directly contradicts.

    Pairs keep their hedge and their instruments, except that a pair whose
    leg ridge contradicts with high confidence is dropped WHOLE (both legs),
    never half. Tests whether ridge carries usable information about pairs
    without ever creating a naked leg.
    """
    state = _sleeve_state(cache, row, prices, t)
    dollars = np.zeros(N_INST)
    owned = np.zeros(N_INST, dtype=bool)
    y_idx = cache["pair_y_index"]
    x_idx = cache["pair_x_index"]
    pos_y = cache["pair_pos_y"][row]
    pos_x = cache["pair_pos_x"][row]
    priority = cache["pair_priority"][row]
    protected = cache["pair_protected"]
    ridge_sign = state["ridge_sign"]
    strength = state["ridge_strength"]

    order = np.argsort(-(priority * np.where(protected, BOOST, 1.0)))
    for pair_idx in order:
        if abs(pos_y[pair_idx]) <= 1.0e-9:
            continue
        yi, xi = int(y_idx[pair_idx]), int(x_idx[pair_idx])
        if owned[yi] or owned[xi]:
            continue
        if protected[pair_idx]:
            veto = False
        else:
            veto = (
                (ridge_sign[yi] != 0.0 and ridge_sign[yi] != np.sign(pos_y[pair_idx])
                 and strength[yi] > 2.0)
                or (ridge_sign[xi] != 0.0 and ridge_sign[xi] != np.sign(pos_x[pair_idx])
                    and strength[xi] > 2.0)
            )
        owned[yi] = owned[xi] = True
        if veto:
            continue
        dollars[yi] += pos_y[pair_idx]
        dollars[xi] += pos_x[pair_idx]

    ridge_dollars = BB * ridge_sign
    dollars = dollars + np.where(owned, 0.0, ridge_dollars)
    dollars[state["crash"]] = 0.0
    return np.clip(dollars, -DLR_LIMIT, DLR_LIMIT)


def s9_proportional(cache, row, prices, t):
    """Size by vote magnitude rather than bang-bang.

    Structural claim: bang-bang discards all confidence information -- a
    marginal 0.01 vote gets the same capital as an overwhelming one. Scaling
    by |vote| should concentrate risk where the evidence is strongest.
    """
    state = _sleeve_state(cache, row, prices, t)
    combined = state["consensus"] + state["ridge_vote"]
    magnitude = np.clip(np.abs(combined) / 1.5, 0.0, 1.0)
    direction = np.sign(combined) * magnitude
    direction[state["crash"]] = 0.0
    return np.clip(BB * direction, -DLR_LIMIT, DLR_LIMIT)


def s10_agreement_weighted(cache, row, prices, t):
    """Full size when the sleeves agree, half when only one speaks, flat on conflict.

    Structural claim: the hit-rate ordering (agree > single sleeve > conflict)
    is stable in all three regions, so allocating capital in that order is a
    structural bet on signal quality, not a fitted parameter.
    """
    state = _sleeve_state(cache, row, prices, t)
    direction = _resolve(state)
    agree = state["pair_active"] & state["ridge_active"] & (state["pair_dir"] == state["ridge_sign"])
    conflict = state["pair_active"] & state["ridge_active"] & (state["pair_dir"] != state["ridge_sign"])
    weight = np.full(N_INST, 0.5)
    weight[agree] = 1.0
    weight[conflict] = 0.0
    scaled = direction * weight
    scaled[state["crash"]] = 0.0
    return np.clip(BB * scaled, -DLR_LIMIT, DLR_LIMIT)


SCHEMES = {
    "s0_baseline": s0_baseline,
    "s1_conflict_flat": s1_conflict_flat,
    "s2_conflict_pair_wins": s2_conflict_pair_wins,
    "s3_conflict_ridge_wins": s3_conflict_ridge_wins,
    "s4_pairs_only_exclusive": s4_pairs_only_exclusive,
    "s5_ridge_only": s5_ridge_only,
    "s6_orthogonal": s6_orthogonal,
    "s7_orthogonal_bangbang_pairs": s7_orthogonal_bangbang_pairs,
    "s8_orthogonal_ridge_confirms": s8_orthogonal_ridge_confirms,
    "s9_proportional": s9_proportional,
    "s10_agreement_weighted": s10_agreement_weighted,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    cache = load_or_build()

    rows = []
    for name, scheme in SCHEMES.items():
        record = {"scheme": name}
        for label, (region, start, end), blocks in (
            ("train", TRAIN, TRAIN_BLOCKS),
            ("val", VALIDATION, VAL_BLOCKS),
            ("test", TEST, TEST_BLOCKS),
        ):
            pnl = evaluate(scheme, cache, prices, start, end)
            report = region_report(pnl, start, end, blocks)
            record[f"{label}_score"] = report["score"]
            record[f"{label}_floor"] = report["floor"]
            for block_idx, block_score in enumerate(report["blocks"]):
                record[f"{label}_b{block_idx + 1}"] = block_score
        rows.append(record)
        print(f"  ran {name}")

    frame = pd.DataFrame(rows).set_index("scheme")
    frame.to_csv(OUTPUT_DIR / "scheme_comparison.csv")

    base = frame.loc["s0_baseline"]
    print("\n" + "=" * 100)
    print("TRAIN-ONLY SELECTION (days 251-500). Both blocks must beat baseline.")
    print("=" * 100)
    print(f"{'scheme':<32} {'tr_b1':>9} {'tr_b2':>9} {'tr_score':>9} {'floor':>9}  both_blocks_beat_baseline")
    print("-" * 100)
    for name, row in frame.iterrows():
        passes = (row["train_b1"] > base["train_b1"]) and (row["train_b2"] > base["train_b2"])
        floor_ok = row["train_floor"] >= base["train_floor"]
        mark = "PASS" if (passes and floor_ok) else ("blocks" if passes else "no")
        print(f"{name:<32} {row['train_b1']:9.1f} {row['train_b2']:9.1f} "
              f"{row['train_score']:9.1f} {row['train_floor']:9.1f}  {mark}")

    print("\n" + "=" * 100)
    print("ALL REGIONS (validation = burned/confirmation, test = reported once)")
    print("=" * 100)
    cols = ["train_score", "train_floor", "val_score", "val_floor", "test_score", "test_floor"]
    print(frame[cols].round(1).to_string())
    print(f"\nwrote {OUTPUT_DIR / 'scheme_comparison.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Two questions the who-wins framing skipped.

PART A -- CONDITIONAL DEGRADATION.
Earlier work asked "which sleeve is right when they disagree", which is
zero-sum by construction (on a conflict pair_dir = -ridge_sign, so one is
necessarily right) and therefore forces a pick-a-winner answer. The better
question is asked of each sleeve SEPARATELY and is not zero-sum:

    does a sleeve's OWN edge degrade when the other contradicts it,
    relative to that same sleeve's edge when unopposed?

If both sleeves degrade under contradiction, then contradiction is
information about *state ambiguity* rather than about direction, and the
correct response is to reduce size or net the exposure -- not to crown a
winner. That is a different action than anything tested so far.

PART B -- INTERNALISED PORTFOLIO PROCESS.
The shipped design keeps the sleeves independent and arbitrates per asset,
then sizes bang-bang: every winning direction gets the instrument's full
limit. That discards each sleeve's own intended size, discards the pair's
beta hedge ratio, and executes the winner's full notional even when the two
sleeves' flow largely offsets.

A unified process instead nets the sleeves' dollar demands first and treats
the result as one portfolio:

    net_demand[i] = sum over pairs of that pair's leg demand on i
                  + ridge's own target on i

Pair legs enter at their true relative sizes (pos_x = -beta * pos_y), so the
hedge survives netting, and offsetting flow is internalised rather than
crossed at full size. Variants differ in how the netted book is scaled and
in whether small adjustments are suppressed (a rebalance band -- the
execution analogue of internalisation, avoiding round-trips on flow that
would net out anyway).
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
    region_report,
    score,
    simulate,
)
from research.combine_schemes import _resolve, _sleeve_state
from research.signal_cache import load_or_build, load_prices

import testbedv2_tuned_ridge_protectedbarbell as strategy

OUTPUT_DIR = ROOT / "analysis_outputs" / "signal_overlap"
BB = strategy.BANG_BANG_LIMITS
REGIONS = (("train", 251, 500), ("validation", 501, 750), ("test", 751, 1000))
BASELINE = {"train": 789.8, "validation": 847.0, "test": 914.7}


# ==========================================================================
# PART A
# ==========================================================================

def conditional_degradation(cache, prices):
    lookup = {int(d): i for i, d in enumerate(cache["days"])}
    records = []
    for region, start, end in REGIONS:
        for t in range(start, end):
            row = lookup[t]
            state = _sleeve_state(cache, row, prices, t)
            pair_active = state["pair_active"]
            ridge_active = state["ridge_active"]
            pair_dir = state["pair_dir"]
            ridge_sign = state["ridge_sign"]
            contradicted = pair_active & ridge_active & (pair_dir != ridge_sign)
            agreed = pair_active & ridge_active & (pair_dir == ridge_sign)
            forward = prices[:, t] / prices[:, t - 1] - 1.0
            for inst in range(1, N_INST):
                if state["crash"][inst]:
                    continue
                if pair_active[inst]:
                    ctx = ("contradicted" if contradicted[inst]
                           else "confirmed" if agreed[inst] else "unopposed")
                    records.append({"region": region, "sleeve": "pair", "context": ctx,
                                    "ret": float(pair_dir[inst] * forward[inst]),
                                    "hit": float(np.sign(pair_dir[inst]) == np.sign(forward[inst]))})
                if ridge_active[inst]:
                    ctx = ("contradicted" if contradicted[inst]
                           else "confirmed" if agreed[inst] else "unopposed")
                    records.append({"region": region, "sleeve": "ridge", "context": ctx,
                                    "ret": float(ridge_sign[inst] * forward[inst]),
                                    "hit": float(np.sign(ridge_sign[inst]) == np.sign(forward[inst]))})
    return pd.DataFrame.from_records(records)


def report_part_a(frame):
    print("=" * 92)
    print("PART A -- does each sleeve's OWN edge survive being contradicted?")
    print("=" * 92)
    print("  (bps = mean return in the sleeve's own direction, per instrument-day)")
    for sleeve in ("pair", "ridge"):
        print(f"\n  {sleeve.upper()} sleeve")
        print(f"    {'region':<12} {'context':<14} {'n':>6} {'hit rate':>9} {'bps':>9}")
        print("    " + "-" * 56)
        for region, _, _ in REGIONS:
            sub = frame[(frame.sleeve == sleeve) & (frame.region == region)]
            for ctx in ("unopposed", "confirmed", "contradicted"):
                s = sub[sub.context == ctx]
                if not len(s):
                    continue
                print(f"    {region:<12} {ctx:<14} {len(s):>6} {s.hit.mean():>9.4f} "
                      f"{1e4*s.ret.mean():>9.2f}")
    print("\n  DEGRADATION (contradicted minus unopposed, in bps):")
    print(f"    {'sleeve':<8} " + " ".join(f"{r:>14}" for r, _, _ in REGIONS))
    for sleeve in ("pair", "ridge"):
        cells = []
        for region, _, _ in REGIONS:
            sub = frame[(frame.sleeve == sleeve) & (frame.region == region)]
            d = (1e4 * sub[sub.context == "contradicted"].ret.mean()
                 - 1e4 * sub[sub.context == "unopposed"].ret.mean())
            cells.append(f"{d:>14.2f}")
        print(f"    {sleeve:<8} " + " ".join(cells))


# ==========================================================================
# PART B
# ==========================================================================

def raw_demands(cache, row):
    """Each sleeve's own intended dollar positions, un-arbitrated."""
    pair = np.zeros(N_INST)
    y_idx, x_idx = cache["pair_y_index"], cache["pair_x_index"]
    pos_y, pos_x = cache["pair_pos_y"][row], cache["pair_pos_x"][row]
    for k in range(len(y_idx)):
        if abs(pos_y[k]) <= 1e-9:
            continue
        pair[int(y_idx[k])] += pos_y[k]
        pair[int(x_idx[k])] += pos_x[k]
    return pair, cache["ridge_raw"][row].copy()


def make_netting_scheme(mode, ridge_scale=1.0, band=0.0):
    """Portfolio construction over netted demands."""
    state_prev = {"pos": None}

    def scheme(cache, row, prices, t):
        pair, ridge = raw_demands(cache, row)
        net = pair + ridge_scale * ridge
        crash = crash_mask(prices, t)

        if mode == "net_clip":
            dollars = np.clip(net, -DLR_LIMIT, DLR_LIMIT)
        elif mode == "net_direction_full":
            dollars = BB * np.sign(net)
        elif mode == "net_scaled":
            peak = np.max(np.abs(net[1:])) if np.any(net[1:]) else 0.0
            factor = (DLR_LIMIT[1] / peak) if peak > 1e-9 else 0.0
            dollars = np.clip(net * factor, -DLR_LIMIT, DLR_LIMIT)
        else:
            raise ValueError(mode)

        dollars[crash] = 0.0
        dollars = np.clip(dollars, -DLR_LIMIT, DLR_LIMIT)

        if band > 0.0:
            prev = state_prev["pos"]
            if prev is None:
                state_prev["pos"] = dollars.copy()
            else:
                move = np.abs(dollars - prev)
                hold = move < band * DLR_LIMIT
                dollars = np.where(hold, prev, dollars)
                state_prev["pos"] = dollars.copy()
        return dollars

    def reset():
        state_prev["pos"] = None

    scheme.reset = reset
    return scheme


def evaluate_scheme(scheme, cache, prices, start, end):
    if hasattr(scheme, "reset"):
        scheme.reset()
    lookup = {int(d): i for i, d in enumerate(cache["days"])}
    dollars = {}
    for t in range(start - 1, end):
        dollars[t] = scheme(cache, lookup[t], prices, t)
    return simulate(dollars, prices, start, end)


def gross_and_turnover(scheme, cache, prices, start, end):
    if hasattr(scheme, "reset"):
        scheme.reset()
    lookup = {int(d): i for i, d in enumerate(cache["days"])}
    gross, turn, prev = [], [], None
    for t in range(start - 1, end):
        d = scheme(cache, lookup[t], prices, t)
        gross.append(np.abs(d).sum())
        if prev is not None:
            turn.append(np.abs(d - prev).sum())
        prev = d
    return float(np.mean(gross)), float(np.mean(turn))


def report_part_b(cache, prices):
    print("\n" + "=" * 92)
    print("PART B -- unified/internalised portfolio process vs independent arbitration")
    print("=" * 92)

    schemes = {
        "net_clip (pure netting)": make_netting_scheme("net_clip"),
        "net_clip + band 25%": make_netting_scheme("net_clip", band=0.25),
        "net_clip + band 50%": make_netting_scheme("net_clip", band=0.50),
        "net_scaled (full budget)": make_netting_scheme("net_scaled"),
        "net_scaled + band 25%": make_netting_scheme("net_scaled", band=0.25),
        "net_direction_full": make_netting_scheme("net_direction_full"),
        "net_direction_full + band 25%": make_netting_scheme("net_direction_full", band=0.25),
        "net ridge x0.5, dir full": make_netting_scheme("net_direction_full", ridge_scale=0.5),
        "net ridge x2, dir full": make_netting_scheme("net_direction_full", ridge_scale=2.0),
    }

    rows = []
    print(f"{'scheme':<32} {'tr_b1':>8} {'tr_b2':>8} {'train':>8} {'val':>8} {'test':>8} "
          f"{'gross$':>9} {'turn$':>9}")
    print("-" * 92)
    print(f"{'baseline (shipped)':<32} {649.7:8.1f} {929.5:8.1f} {789.8:8.1f} "
          f"{847.0:8.1f} {914.7:8.1f} {435400:9,.0f} {285984:9,.0f}")
    for name, scheme in schemes.items():
        rec = {"scheme": name}
        for label, (_, s, e), blocks in (("train", TRAIN, TRAIN_BLOCKS),
                                         ("validation", VALIDATION, VAL_BLOCKS),
                                         ("test", TEST, TEST_BLOCKS)):
            pnl = evaluate_scheme(scheme, cache, prices, s, e)
            rep = region_report(pnl, s, e, blocks)
            rec[f"{label}_score"] = rep["score"]
            rec[f"{label}_floor"] = rep["floor"]
            rec[f"{label}_b1"], rec[f"{label}_b2"] = rep["blocks"]
        g, tn = gross_and_turnover(scheme, cache, prices, 251, 500)
        rec["gross"], rec["turnover"] = g, tn
        rows.append(rec)
        print(f"{name:<32} {rec['train_b1']:8.1f} {rec['train_b2']:8.1f} "
              f"{rec['train_score']:8.1f} {rec['validation_score']:8.1f} "
              f"{rec['test_score']:8.1f} {g:9,.0f} {tn:9,.0f}")
    return pd.DataFrame(rows).set_index("scheme")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    cache = load_or_build()

    frame = conditional_degradation(cache, prices)
    frame.to_csv(OUTPUT_DIR / "contradiction_conditional.csv", index=False)
    report_part_a(frame)

    results = report_part_b(cache, prices)
    results.to_csv(OUTPUT_DIR / "netting_schemes.csv")
    print(f"\nwrote {OUTPUT_DIR / 'netting_schemes.csv'}")


if __name__ == "__main__":
    main()

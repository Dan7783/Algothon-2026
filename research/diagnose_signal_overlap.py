#!/usr/bin/env python
"""Measure what the pair/ridge merge actually does to the two sleeves.

Three questions, in order of how much they can explain a "both sleeves work
alone but are limited together" symptom:

1. OVERLAP -- how often do the sleeves both have an opinion on the same
   instrument, and how often do those opinions disagree?
2. HIT RATE -- conditional on each overlap category, how often is the
   resulting direction right, and what does it earn? This says whether the
   arbitration is picking the better sleeve or the worse one.
3. HEDGE INTEGRITY -- the bang-bang layer votes per ASSET, so a pair's two
   legs are arbitrated independently. That can (a) flip or flatten one leg
   while keeping the other, leaving a naked directional bet where a
   market-neutral spread was intended, and (b) discard the beta hedge ratio
   entirely, since every winning direction is sized at the instrument's full
   dollar limit regardless of what the pair asked for.

Everything is reported per region so train-only conclusions stay separable
from the burned-validation and held-out windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.combine_lab import DLR_LIMIT, N_INST, crash_mask, pair_consensus
from research.signal_cache import load_or_build, load_prices

import testbedv2_tuned_ridge_protectedbarbell as strategy

OUTPUT_DIR = ROOT / "analysis_outputs" / "signal_overlap"
REGIONS = (("train", 251, 500), ("validation", 501, 750), ("test", 751, 1000))
BB = strategy.BANG_BANG_LIMITS


def collect(cache: dict, prices: np.ndarray) -> pd.DataFrame:
    days = cache["days"]
    lookup = {int(day): row for row, day in enumerate(days)}
    records = []
    y_idx = cache["pair_y_index"]
    x_idx = cache["pair_x_index"]

    for region, start, end in REGIONS:
        for t in range(start, end):
            row = lookup[t]
            consensus, pair_active = pair_consensus(
                cache, row, strategy.PROTECTED_VOTE_BOOST
            )
            ridge_raw = cache["ridge_raw"][row]
            ridge_strength = cache["ridge_strength"][row]
            ridge_sign = np.sign(ridge_raw)
            ridge_active = ridge_sign != 0.0
            ridge_vote = strategy.RIDGE_CONFLICT_WEIGHT * ridge_sign * ridge_strength
            combined = consensus + ridge_vote
            active = pair_active | ridge_active
            direction = np.sign(combined)
            tied = active & (direction == 0.0) & ridge_active
            direction[tied] = ridge_sign[tied]
            crash = crash_mask(prices, t)
            direction[crash] = 0.0

            # Position is taken at price index t-1; it earns the move into t.
            forward = prices[:, t] / prices[:, t - 1] - 1.0
            dollars = np.clip(BB * direction, -DLR_LIMIT, DLR_LIMIT)
            pnl = dollars * forward

            pair_dir = np.sign(consensus)
            for inst in range(1, N_INST):
                if not active[inst] or crash[inst]:
                    continue
                p_on = bool(pair_active[inst]) and pair_dir[inst] != 0.0
                r_on = bool(ridge_active[inst])
                if p_on and r_on:
                    category = "agree" if pair_dir[inst] == ridge_sign[inst] else "conflict"
                elif p_on:
                    category = "pair_only"
                else:
                    category = "ridge_only"
                winner = ""
                if category == "conflict":
                    winner = "ridge" if direction[inst] == ridge_sign[inst] else "pair"
                records.append(
                    {
                        "region": region,
                        "day": t,
                        "inst": inst,
                        "category": category,
                        "conflict_winner": winner,
                        "direction": float(direction[inst]),
                        "pair_dir": float(pair_dir[inst]),
                        "ridge_dir": float(ridge_sign[inst]),
                        "ridge_strength": float(ridge_strength[inst]),
                        "pnl": float(pnl[inst]),
                        "correct": float(np.sign(direction[inst]) == np.sign(forward[inst])),
                    }
                )
    return pd.DataFrame.from_records(records)


def hedge_integrity(cache: dict, prices: np.ndarray) -> pd.DataFrame:
    """How often does asset-level voting break a pair's intended hedge?"""
    days = cache["days"]
    lookup = {int(day): row for row, day in enumerate(days)}
    y_idx = cache["pair_y_index"]
    x_idx = cache["pair_x_index"]
    records = []

    for region, start, end in REGIONS:
        intact = flipped = half = 0
        beta_err = []
        for t in range(start, end):
            row = lookup[t]
            consensus, pair_active = pair_consensus(
                cache, row, strategy.PROTECTED_VOTE_BOOST
            )
            ridge_raw = cache["ridge_raw"][row]
            ridge_strength = cache["ridge_strength"][row]
            ridge_sign = np.sign(ridge_raw)
            ridge_active = ridge_sign != 0.0
            combined = consensus + strategy.RIDGE_CONFLICT_WEIGHT * ridge_sign * ridge_strength
            active = pair_active | ridge_active
            direction = np.sign(combined)
            tied = active & (direction == 0.0) & ridge_active
            direction[tied] = ridge_sign[tied]
            direction[crash_mask(prices, t)] = 0.0

            pos_y = cache["pair_pos_y"][row]
            pos_x = cache["pair_pos_x"][row]
            for pair_idx in range(len(y_idx)):
                if abs(pos_y[pair_idx]) <= 1.0e-9:
                    continue
                yi, xi = int(y_idx[pair_idx]), int(x_idx[pair_idx])
                want_y, want_x = np.sign(pos_y[pair_idx]), np.sign(pos_x[pair_idx])
                got_y, got_x = direction[yi], direction[xi]
                y_ok = got_y == want_y
                x_ok = got_x == want_x
                if y_ok and x_ok:
                    intact += 1
                    # Intended hedge ratio vs what bang-bang actually delivers.
                    want_ratio = abs(pos_x[pair_idx]) / max(abs(pos_y[pair_idx]), 1e-12)
                    got_ratio = abs(BB[xi]) / max(abs(BB[yi]), 1e-12)
                    beta_err.append(abs(got_ratio - want_ratio) / max(want_ratio, 1e-12))
                elif y_ok or x_ok:
                    half += 1
                else:
                    flipped += 1
        total = intact + half + flipped
        records.append(
            {
                "region": region,
                "pair_days": total,
                "both_legs_intact": intact,
                "one_leg_naked": half,
                "both_legs_flipped": flipped,
                "pct_intact": 100.0 * intact / max(total, 1),
                "pct_one_leg_naked": 100.0 * half / max(total, 1),
                "pct_both_flipped": 100.0 * flipped / max(total, 1),
                "median_hedge_ratio_error_pct": 100.0 * float(np.median(beta_err)) if beta_err else float("nan"),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    cache = load_or_build()

    frame = collect(cache, prices)
    frame.to_csv(OUTPUT_DIR / "overlap_observations.csv", index=False)

    print("=" * 78)
    print("1. OVERLAP AND HIT RATE BY CATEGORY")
    print("=" * 78)
    for region, _, _ in REGIONS:
        sub = frame[frame["region"] == region]
        total = len(sub)
        print(f"\n--- {region} ({total} instrument-days with a live signal) ---")
        summary = sub.groupby("category").agg(
            n=("pnl", "size"),
            hit_rate=("correct", "mean"),
            mean_pnl=("pnl", "mean"),
            total_pnl=("pnl", "sum"),
        )
        summary["share_pct"] = 100.0 * summary["n"] / total
        print(summary.round({"hit_rate": 4, "mean_pnl": 2, "total_pnl": 0, "share_pct": 1}))

        conflicts = sub[sub["category"] == "conflict"]
        if len(conflicts):
            print(f"\n  conflict resolution ({len(conflicts)} instrument-days, "
                  f"{100.0*len(conflicts)/total:.1f}% of live signals):")
            win = conflicts.groupby("conflict_winner").agg(
                n=("pnl", "size"), hit_rate=("correct", "mean"),
                mean_pnl=("pnl", "mean"), total_pnl=("pnl", "sum"))
            print("   ", win.round({"hit_rate": 4, "mean_pnl": 2, "total_pnl": 0}).to_string().replace("\n", "\n    "))

    print("\n" + "=" * 78)
    print("2. HEDGE INTEGRITY (does asset voting preserve each pair's spread?)")
    print("=" * 78)
    hedge = hedge_integrity(cache, prices)
    hedge.to_csv(OUTPUT_DIR / "hedge_integrity.csv", index=False)
    print(hedge.round(2).to_string(index=False))
    print("\nfiles written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()

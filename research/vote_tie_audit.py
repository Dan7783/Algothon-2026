#!/usr/bin/env python
"""How often does the asset vote cancel to exactly zero, and what does it cost?

Under PAIR_CONFLICT_MODE="priority" the vote weight is the continuous
kappa*|z|, so an exact cancellation is measure-zero -- the `tied` branch in
sleeve_breakdown is effectively dead code. Under "equal" every live pair
contributes +/-1 (or +/-PROTECTED_VOTE_BOOST), so 1v1 / 2v2 ties are common.

An asset whose pair vote cancels is only rescued if the ridge is active on it;
otherwise `direction` stays 0 and the asset goes FLAT for that day -- no
capital deployed. Since the score is capital-dominated, that is a real cost.

This counts, per mode:
  * asset-days where the pair vote is live but sums to exactly zero
  * how many of those the ridge rescues vs. how many go flat
  * the deployment actually lost, in asset-days and in dollars

Run from the repo root:  python research/vote_tie_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as fs

FIRST_DAY, LAST_DAY = 250, 1000
WINDOWS = {"train 251-500": (251, 500),
           "val   501-750": (501, 750),
           "org   751-1000": (751, 1000)}


def run(mode: str, boost: float) -> pd.DataFrame:
    fs.PAIR_CONFLICT_MODE = mode
    fs.PROTECTED_VOTE_BOOST = boost
    fs.reset()

    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    rows = []
    for t in range(FIRST_DAY, LAST_DAY + 1):
        s = fs.sleeve_breakdown(prices[:, :t])
        if s is None:
            continue
        pair_live = s["pair_active_mask"]
        contested = pair_live & (s["pair_contributors"] > 1)
        cancelled = pair_live & (np.abs(s["pair_vote"]) < 1e-12)
        rescued = cancelled & s["ridge_active_mask"]
        went_flat = cancelled & ~s["ridge_active_mask"]
        rows.append({
            "day": t,
            "pair_live": int(pair_live.sum()),
            "contested": int(contested.sum()),
            "cancelled": int(cancelled.sum()),
            "rescued_by_ridge": int(rescued.sum()),
            "went_flat": int(went_flat.sum()),
            "deployed": float(np.abs(s["dollars"]).sum()),
        })
    return pd.DataFrame(rows)


def report(label: str, df: pd.DataFrame) -> None:
    print(f"\n=== {label} ===")
    for wname, (lo, hi) in WINDOWS.items():
        w = df[(df.day >= lo) & (df.day <= hi)]
        live, cont = w.pair_live.sum(), w.contested.sum()
        canc, resc, flat = w.cancelled.sum(), w.rescued_by_ridge.sum(), w.went_flat.sum()
        print(f"  {wname}: pair-live asset-days {live:6d} | contested {cont:6d}"
              f" | cancelled {canc:5d} ({canc/max(live,1)*100:4.1f}% of live,"
              f" {canc/max(cont,1)*100:4.1f}% of contested)")
        print(f"{'':17s}  -> ridge rescues {resc:5d} ({resc/max(canc,1)*100:4.1f}%)"
              f" | goes FLAT {flat:5d} ({flat/max(canc,1)*100:4.1f}%)"
              f" = {flat/len(w):.2f} assets/day forgone")
        print(f"{'':17s}  mean deployed ${w.deployed.mean():,.0f}/day"
              f" of ${50*10_000:,} max")


def main() -> None:
    eq = run("equal", 4.0)
    pr = run("priority", 4.0)
    eq_nb = run("equal", 1.0)
    report("EQUAL mode, boost 4.0  (what you are running)", eq)
    report("PRIORITY mode, boost 4.0  (the old baseline)", pr)
    report("EQUAL mode, boost 1.0  (pure one-pair-one-vote)", eq_nb)

    out = ROOT / "analysis_outputs" / "effective_breadth"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("equal", eq), ("priority", pr), ("equal_noboost", eq_nb)):
        df.to_csv(out / f"tie_audit_{name}.csv", index=False)
    print(f"\nwrote tie_audit_*.csv to {out}")


if __name__ == "__main__":
    main()

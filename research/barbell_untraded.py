#!/usr/bin/env python
"""Barbell: add SMALL confidence-scaled positions where the book currently sits flat.

Distinct from every sizing scheme tested before. Those REPLACED full bang-bang
positions with scaled ones, so they deployed less capital and lost under a
capital-dominated score. This one is purely ADDITIVE -- every existing position
keeps its full limit, and small positions are opened only where the book is
currently flat. Deployed capital strictly increases.

Why it can work where full-size arbitration failed: score = mu*SR^2/(SR^2+1),
and at the book's SR~8 that fraction is 0.985 -- nearly saturated -- so
score ~ mean PnL. Adding a position of relative size w on a day with direction
accuracy p contributes:
    mean gain    ~ w * (2p - 1)      (LINEAR in w)
    variance cost ~ w^2              (QUADRATIC in w)
so break-even accuracy falls toward 50% as w shrinks. A 52.6% signal that is
marginal at full size may be clearly positive at w=0.1.

Two populations:
  DEADLOCK  -- pair vote cancels exactly and the ridge is inactive (1,238 days)
  UNTRADED  -- every asset-day the book leaves flat for any reason (larger)

Two sizings, both using the ungated sub-threshold ridge direction:
  flat       -- w * LIMIT on every qualifying day
  confidence -- w * LIMIT * (strength / RIDGE_THETA), so size scales with the
                ridge's own conviction, capped at w * LIMIT

Control at every w: a coin-flip direction. That isolates whether any gain is
direction skill or merely deployed capital.

Run from the repo root:  python research/barbell_untraded.py
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
from research.deadlock_ceiling import LIMIT, run_accounting, score
from research.deadlock_subthreshold_ridge import raw_prediction

FIRST, LAST = 250, 1000
WINDOWS = {"train 251-500": (251, 500),
           "val   501-750": (501, 750),
           "org   751-1000": (751, 1000)}
CACHE = ROOT / "analysis_outputs" / "deadlock" / "barbell_cache.npz"
WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00)


def build_cache(prices: np.ndarray) -> dict:
    fs.reset()
    days = np.arange(FIRST, LAST + 1)
    n = prices.shape[0]
    base = np.zeros((len(days), n))
    deadlock = np.zeros((len(days), n), dtype=bool)
    untraded = np.zeros((len(days), n), dtype=bool)
    pred = np.zeros((len(days), n))
    strength = np.zeros((len(days), n))

    for r, t in enumerate(days):
        s = fs.sleeve_breakdown(prices[:, :t])
        if s is None:
            continue
        base[r] = s["dollars"]
        deadlock[r] = (s["pair_active_mask"]
                       & (np.abs(s["pair_vote"]) < 1e-12)
                       & ~s["ridge_active_mask"]
                       & ~s["crash_mask"])
        flat = (np.abs(s["dollars"]) < 1e-9) & ~s["crash_mask"]
        flat[fs.I["ALGO"]] = False          # ALGO limit is 0 by decision
        untraded[r] = flat
        strength[r] = s["ridge_strength"]
        pred[r] = raw_prediction(np.diff(np.log(prices[:, :t]), axis=1))

    return {"days": days, "base": base, "deadlock": deadlock,
            "untraded": untraded, "pred": pred, "strength": strength}


def overlay(base, mask, direction, w, conf=None):
    """Add w*LIMIT (optionally conviction-scaled) where mask is set."""
    out = base.copy()
    size = np.broadcast_to(LIMIT[None, :], base.shape) * w
    if conf is not None:
        size = size * np.clip(conf / fs.RIDGE_THETA, 0.0, 1.0)
    out[mask] = (size * direction)[mask]
    return out


def main() -> None:
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    if CACHE.exists():
        with np.load(CACHE) as d:
            c = {k: d[k] for k in d.files}
    else:
        print("replaying (one pass) ...")
        c = build_cache(prices)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **c)

    days, base = c["days"], c["base"]
    dl, ut = c["deadlock"], c["untraded"]
    sgn = np.sign(c["pred"])
    conf = c["strength"]

    print(f"deadlock asset-days: {int(dl.sum()):6d} ({dl.sum()/len(days):.2f}/day)")
    print(f"untraded asset-days: {int(ut.sum()):6d} ({ut.sum()/len(days):.2f}/day)")
    print(f"   of which deadlock: {int((dl & ut).sum())}")

    base_s = {}
    for wn, (lo, hi) in WINDOWS.items():
        p = run_accounting(prices, days, base, lo, hi)
        base_s[wn] = score(p.mean(), p.std())
    print("\nbaseline: " + "  ".join(f"{k} {v:.1f}" for k, v in base_s.items()))

    rng = np.random.default_rng(7)
    coin = np.where(rng.random(base.shape) < 0.5, -1.0, 1.0)

    for pop_name, mask in (("DEADLOCK", dl), ("UNTRADED (all flat)", ut)):
        for sizing, cf in (("flat size", None), ("confidence-scaled", conf)):
            print("\n" + "=" * 78)
            print(f"{pop_name}  |  {sizing}  |  sub-threshold ridge direction")
            print("=" * 78)
            print(f"{'w':>6} " + "".join(f"{k:>22}" for k in WINDOWS)
                  + f"{'  [coinflip control]':>22}")
            for w in WEIGHTS:
                cells = []
                for wn, (lo, hi) in WINDOWS.items():
                    d = overlay(base, mask, sgn, w, cf)
                    p = run_accounting(prices, days, d, lo, hi)
                    cells.append(score(p.mean(), p.std()) - base_s[wn])
                # control: same size, random direction, org window only
                dc = overlay(base, mask, coin, w, cf)
                pc = run_accounting(prices, days, dc, 751, 1000)
                ctrl = score(pc.mean(), pc.std()) - base_s["org   751-1000"]
                print(f"{w:6.2f} " + "".join(f"{v:>+22.1f}" for v in cells)
                      + f"{ctrl:>+22.1f}")

    print("\nCells are score DELTA vs the current book (positive = better).")
    print("The coinflip column is the same capital with no direction skill:")
    print("if the signal column does not clearly beat it, the gain is not skill.")


if __name__ == "__main__":
    main()

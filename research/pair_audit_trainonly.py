"""Leave-one-out pair audit judged on TRAIN BLOCKS ONLY.

The earlier audit searched prune combinations against both windows, which is
how test leaked into selection.  Here the verdict uses only blocks 251-375
and 376-500; blocks 501-625 and 626-750 are printed alongside purely as an
after-the-fact check and are never used to choose anything.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import json
from math import sqrt

import numpy as np
import pandas as pd

import four_sigmas as fs

COMM = np.full(51, 1e-4)
COMM[0] = 2e-5
LIM = np.full(51, 10_000.0)
LIM[0] = 100_000.0
BASE = tuple(fs.PAIR_CFGS)


def sc(m, s):
    if m <= 0 or s < 1e-10:
        return m
    r = sqrt(250.0) * m / s
    return m * r * r / (r * r + 1.0)


def blocks(cfgs):
    fs.apply_pair_cfgs(cfgs)
    fs.reset()
    cash = val = comm = 0.0
    pos = np.zeros(51)
    pnl = []
    prices = blocks.prices
    for t in range(250, 751):
        h = prices[:, :t]
        cur = h[:, -1]
        if t < 750:
            req = fs.getMyPosition(h)
            lim = (LIM / cur).astype(int)
            new = np.clip(req, -lim, lim).astype(float)
        else:
            new = pos.copy()
        d = new - pos
        cash -= cur.dot(d) + comm
        comm = float(np.sum(cur * np.abs(d) * COMM))
        pos = new
        v = cash + pos.dot(cur)
        pnl.append(v - val)
        val = v
    a = np.asarray(pnl[1:])
    return [sc(float(a[i:i + 125].mean()), float(a[i:i + 125].std()))
            for i in (0, 125, 250, 375)]


def main():
    blocks.prices = pd.read_csv("prices.txt", sep=r"\s+",
                                header=0).to_numpy(float).T
    b0 = blocks(BASE)
    train0 = float(np.median(b0[:2]))
    print(f"current four_sigmas blocks: {[round(x) for x in b0]}")
    print(f"  TRAIN median {train0:.1f}   floor {min(b0):.1f}")
    active = [s.name for s in BASE if s.trade.leg_scale > 0]
    print(f"  {len(active)} active pairs\n")
    print("leave-one-out; verdict from TRAIN blocks only (later blocks = check)")
    print(f"{'pair':>11} | {'train marg':>10} | {'blk3':>7} {'blk4':>7} | verdict")
    print("-" * 60)
    rows = []
    for s in BASE:
        if s.trade.leg_scale <= 0:
            continue
        off = tuple(t._replace(trade=t.trade._replace(leg_scale=0.0))
                    if t.name == s.name else t for t in BASE)
        b = blocks(off)
        dtr = train0 - float(np.median(b[:2]))
        d3, d4 = b0[2] - b[2], b0[3] - b[3]
        verdict = "KEEP" if dtr > 1.0 else ("DEAD" if abs(dtr) <= 1.0
                                            else "HARMFUL on train")
        rows.append({"name": s.name, "train_marg": dtr, "blk3": d3, "blk4": d4,
                     "verdict": verdict})
        print(f"  {s.name:>11} | {dtr:+10.2f} | {d3:+7.1f} {d4:+7.1f} | {verdict}")
    fs.apply_pair_cfgs(BASE)
    out = _ROOT / "cusum_output" / "pair_audit_trainonly.json"
    out.write_text(json.dumps({"baseline_blocks": b0, "pairs": rows}, indent=2),
                   encoding="utf-8")
    bad = [r["name"] for r in rows if r["verdict"] != "KEEP"]
    print(f"\nnon-earning on TRAIN: {bad}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

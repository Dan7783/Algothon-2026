"""Which pairs actually earn their slot in four_sigmas?

Standalone score is misleading (a pair can look good alone yet never win its
EV-MWIS conflicts, or vice versa).  The decisive metric is the MARGINAL
contribution: switch the pair off (leg_scale=0) and see what the book loses,
on each window separately.

Classification:
  KEEP        marginal >= 0 on both windows
  TRAIN-ONLY  helps train, hurts test   <- the overfit pattern
  TEST-ONLY   hurts train, helps test
  DEAD        |marginal| < 1 on both     <- occupies a slot for nothing
  HARMFUL     marginal < 0 on both       <- actively costs money

Then tests pruning the HARMFUL + TRAIN-ONLY pairs as a set.
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

TRAIN = (251, 500)
TEST = (501, 750)
OUT = _ROOT / "cusum_output" / "daniel_pairs"
_COMM = np.full(51, 1.0e-4)
_COMM[0] = 2.0e-5
_LIMIT = np.full(51, 10_000.0)
_LIMIT[0] = 100_000.0
BASE = tuple(fs.PAIR_CFGS)


def score(mu, sd):
    if mu <= 0.0 or sd < 1e-10:
        return mu
    sr = sqrt(250.0) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def replay(prices, cfgs, start, end):
    fs.apply_pair_cfgs(cfgs)
    fs.reset()
    cash = value = comm = 0.0
    pos = np.zeros(51)
    pnl = []
    for t in range(start - 1, end + 1):
        hist = prices[:, :t]
        cur = hist[:, -1]
        if t < end:
            req = fs.getMyPosition(hist)
            lim = (_LIMIT / cur).astype(int)
            new = np.clip(req, -lim, lim).astype(float)
        else:
            new = pos.copy()
        d = new - pos
        cash -= cur.dot(d) + comm
        comm = float(np.sum(cur * np.abs(d) * _COMM))
        pos = new
        val = cash + pos.dot(cur)
        pnl.append(val - value)
        value = val
    a = np.asarray(pnl[1:])
    return score(float(a.mean()), float(a.std()))


def without(names):
    return tuple(
        s._replace(trade=s.trade._replace(leg_scale=0.0)) if s.name in names else s
        for s in BASE
    )


def main():
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    b_tr = replay(prices, BASE, *TRAIN)
    b_te = replay(prices, BASE, *TEST)
    print(f"four_sigmas baseline: train {b_tr:.2f}  test {b_te:.2f}\n")
    print(f"{'pair':>11} | {'marg train':>11} {'marg test':>10} | verdict")
    print("-" * 60)

    rows = []
    for s in BASE:
        a = replay(prices, without({s.name}), *TRAIN)
        b = replay(prices, without({s.name}), *TEST)
        m_tr, m_te = b_tr - a, b_te - b
        if abs(m_tr) < 1.0 and abs(m_te) < 1.0:
            v = "DEAD"
        elif m_tr >= 0 and m_te >= 0:
            v = "KEEP"
        elif m_tr > 0 and m_te < 0:
            v = "TRAIN-ONLY (overfit)"
        elif m_tr < 0 and m_te > 0:
            v = "TEST-ONLY"
        else:
            v = "HARMFUL"
        rows.append({"name": s.name, "marg_train": m_tr, "marg_test": m_te,
                     "verdict": v})
        print(f"  {s.name:>11} | {m_tr:+11.2f} {m_te:+10.2f} | {v}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pair_contribution.json").write_text(
        json.dumps({"baseline": [b_tr, b_te], "pairs": rows}, indent=2),
        encoding="utf-8")

    groups = {}
    for r in rows:
        groups.setdefault(r["verdict"], []).append(r["name"])
    print("\nsummary:")
    for k in ("KEEP", "TRAIN-ONLY (overfit)", "TEST-ONLY", "DEAD", "HARMFUL"):
        if k in groups:
            print(f"  {k:22s} {len(groups[k]):2d}  {groups[k]}")

    prune = groups.get("HARMFUL", []) + groups.get("TRAIN-ONLY (overfit)", [])
    if prune:
        a = replay(prices, without(set(prune)), *TRAIN)
        b = replay(prices, without(set(prune)), *TEST)
        print(f"\nprune HARMFUL+TRAIN-ONLY ({len(prune)} pairs): "
              f"train {a:.2f} ({a-b_tr:+.2f})  test {b:.2f} ({b-b_te:+.2f})")
    dead = groups.get("DEAD", [])
    if dead:
        a = replay(prices, without(set(dead)), *TRAIN)
        b = replay(prices, without(set(dead)), *TEST)
        print(f"prune DEAD ({len(dead)} pairs): "
              f"train {a:.2f} ({a-b_tr:+.2f})  test {b:.2f} ({b-b_te:+.2f})")
    if prune or dead:
        allp = set(prune) | set(dead)
        a = replay(prices, without(allp), *TRAIN)
        b = replay(prices, without(allp), *TEST)
        print(f"prune ALL non-KEEP ({len(allp)} pairs): "
              f"train {a:.2f} ({a-b_tr:+.2f})  test {b:.2f} ({b-b_te:+.2f})")
    fs.apply_pair_cfgs(BASE)
    print(f"\nwrote {OUT / 'pair_contribution.json'}")


if __name__ == "__main__":
    main()

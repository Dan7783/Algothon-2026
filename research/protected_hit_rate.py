#!/usr/bin/env python
"""Hit rate of the 7 protected pairs, and whether their 96.6% veto is earned.

Two different questions, both called "hit rate":

A. TRADING ACCURACY -- of the days each pair holds a position, what fraction
   make money on its own two legs (and what fraction of complete round trips
   close green)? This says whether the pair is a good pair.

B. VETO QUALITY -- PROTECTED_VOTE_BOOST=4.0 lets a protected pair overrule
   disagreeing normal pairs on 96.6% of contested asset-days. On exactly those
   asset-days, was the protected direction right more often than the normal-pair
   direction it overruled? This says whether the veto is EARNED. A pair can be
   excellent standalone (A) and still be the wrong tiebreaker (B).

Run from the repo root:  python research/protected_hit_rate.py
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
           "val 501-750": (501, 750),
           "org 751-1000": (751, 1000)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- an honest CI for a hit rate."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main() -> None:
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    n_pairs = len(fs.PAIR_CFGS)
    names = np.array([fs._pair_name(s) for s in fs.PAIR_CFGS])
    prot = np.array([n in fs.PROTECTED_PAIR_NAMES for n in names])
    yi = np.array([int(s.y) for s in fs.PAIR_CFGS])
    xi = np.array([int(s.x) for s in fs.PAIR_CFGS])

    days = np.arange(FIRST_DAY, LAST_DAY + 1)
    pos_y = np.zeros((len(days), n_pairs))
    pos_x = np.zeros((len(days), n_pairs))
    side = np.zeros((len(days), n_pairs), dtype=int)
    # per-asset veto ledger
    veto_rows = []

    book = fs.AtlasBook()
    for row, t in enumerate(days):
        hist = prices[:, :t]
        cands = book.candidates(hist)
        for j, agent in enumerate(book.agents):
            py, px, _ = agent.dollar_positions()
            pos_y[row, j] = py
            pos_x[row, j] = px
            side[row, j] = agent.side

        # rebuild the asset ledger to find protected-vs-normal disagreements
        legs: dict[int, list] = {}
        for priority, agent, py, px in cands:
            p = fs._pair_name(agent.spec) in fs.PROTECTED_PAIR_NAMES
            w = fs.PROTECTED_VOTE_BOOST if p else 1.0
            for idx, pos in ((int(agent.spec.y), py), (int(agent.spec.x), px)):
                if abs(pos) > 1e-9:
                    legs.setdefault(idx, []).append((w, float(np.sign(pos)), p))
        for idx, e in legs.items():
            ps = [s for _, s, p in e if p]
            ns = [s for _, s, p in e if not p]
            if not ps or not ns:
                continue
            pdir = ps[0]
            ndir = np.sign(sum(ns))
            if ndir == 0 or ndir == pdir:
                continue  # no disagreement to arbitrate
            veto_rows.append((t, idx, pdir, ndir))

    # realised return on the day the position is held
    rets = np.zeros((len(days), prices.shape[0]))
    for row, t in enumerate(days):
        if t < prices.shape[1]:
            rets[row] = prices[:, t] / prices[:, t - 1] - 1.0

    pnl = pos_y * rets[:, yi] + pos_x * rets[:, xi]

    # ---------- A. trading accuracy ----------
    print("=" * 74)
    print("A. TRADING ACCURACY -- per-pair standalone PnL (days 251-1000)")
    print("=" * 74)
    print(f"{'pair':>12} {'prot':>5} {'live days':>10} {'day hit%':>9}"
          f" {'trades':>7} {'trade win%':>11} {'total $':>12}")
    rows = []
    for j in range(n_pairs):
        m = np.abs(pos_y[:, j]) > 1e-9
        live = int(m.sum())
        if live == 0:
            continue
        dayhit = float((pnl[m, j] > 0).mean())
        # round trips: side goes non-zero -> zero
        s = side[:, j]
        trades, cur = [], None
        for r in range(len(days)):
            if s[r] != 0 and cur is None:
                cur = r
            elif s[r] == 0 and cur is not None:
                trades.append(pnl[cur:r, j].sum())
                cur = None
        if cur is not None:
            trades.append(pnl[cur:, j].sum())
        tw = float(np.mean([x > 0 for x in trades])) if trades else float("nan")
        rows.append((names[j], bool(prot[j]), live, dayhit, len(trades), tw,
                     float(pnl[:, j].sum())))

    rows.sort(key=lambda r: (not r[1], -r[3]))
    for nm, p, live, dh, nt, tw, tot in rows:
        if p:
            print(f"{nm:>12} {'YES':>5} {live:10d} {dh*100:8.1f}%"
                  f" {nt:7d} {tw*100:10.1f}% {tot:12,.0f}")
    pm = np.array([r[1] for r in rows])
    dh_all = np.array([r[3] for r in rows])
    tw_all = np.array([r[5] for r in rows])
    print("-" * 74)
    print(f"{'PROTECTED (7)':>12} {'':>5} {'':>10} {dh_all[pm].mean()*100:8.1f}%"
          f" {'':>7} {np.nanmean(tw_all[pm])*100:10.1f}%"
          f" {sum(r[6] for r in rows if r[1]):12,.0f}")
    print(f"{'OTHERS (81)':>12} {'':>5} {'':>10} {dh_all[~pm].mean()*100:8.1f}%"
          f" {'':>7} {np.nanmean(tw_all[~pm])*100:10.1f}%"
          f" {sum(r[6] for r in rows if not r[1]):12,.0f}")

    # ---------- B. veto quality ----------
    print()
    print("=" * 74)
    print("B. VETO QUALITY -- on asset-days where a protected pair overrules")
    print("   normal pairs, which direction was actually right?")
    print("=" * 74)
    v = pd.DataFrame(veto_rows, columns=["day", "asset", "prot_dir", "norm_dir"])
    day_pos = {d: i for i, d in enumerate(days)}
    v["ret"] = [rets[day_pos[d], a] for d, a in zip(v.day, v.asset)]
    v = v[np.abs(v.ret) > 0]

    print(f"{'window':>16} {'n':>7} {'protected right':>17} {'normal right':>15}"
          f"  {'edge':>8}")
    for wname, (lo, hi) in WINDOWS.items():
        w = v[(v.day >= lo) & (v.day <= hi)]
        n = len(w)
        if n == 0:
            continue
        ph = float((np.sign(w.ret) == w.prot_dir).mean())
        nh = float((np.sign(w.ret) == w.norm_dir).mean())
        lo_ci, hi_ci = wilson(int(ph * n), n)
        print(f"{wname:>16} {n:7d} {ph*100:9.1f}% [{lo_ci*100:4.1f},{hi_ci*100:4.1f}]"
              f" {nh*100:14.1f}% {(ph-nh)*100:+7.1f}pp")
    n = len(v)
    ph = float((np.sign(v.ret) == v.prot_dir).mean())
    nh = float((np.sign(v.ret) == v.norm_dir).mean())
    lo_ci, hi_ci = wilson(int(ph * n), n)
    print("-" * 74)
    print(f"{'ALL 251-1000':>16} {n:7d} {ph*100:9.1f}% [{lo_ci*100:4.1f},{hi_ci*100:4.1f}]"
          f" {nh*100:14.1f}% {(ph-nh)*100:+7.1f}pp")
    print(f"\n(directions are exclusive, so normal% ~ 100 - protected%;"
          f" 50% = the veto is a coin flip)")

    # dollar-weighted: does being right happen on the big moves?
    pw = float(np.sum(np.abs(v.ret) * (np.sign(v.ret) == v.prot_dir))
               / np.sum(np.abs(v.ret)))
    print(f"move-weighted protected accuracy: {pw*100:.1f}%"
          f"  (>50% = right on the days that matter)")

    out = ROOT / "analysis_outputs" / "effective_breadth"
    out.mkdir(parents=True, exist_ok=True)
    v.to_csv(out / "veto_ledger.csv", index=False)
    print(f"\nwrote {out / 'veto_ledger.csv'}")


if __name__ == "__main__":
    main()

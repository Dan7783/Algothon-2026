"""Judge strategies by consistency across MANY windows, not one test split.

True OOS came in at 599 while our 501-750 "test" score was 878 — a ~32% gap,
which is what mining a single window looks like.  A single split cannot
detect that; rolling blocks can.  A strategy that is genuinely robust scores
similarly in every block.  One that is fitted has a few great blocks (the
ones its parameters were chosen on) and mediocre ones elsewhere.

Reports per-strategy: score in each 125-day block from day 251 onward, the
median, the worst block, and the spread.  Median and worst-block are far
better OOS predictors than the mean of a mined window.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import importlib
import json
from math import sqrt

import numpy as np
import pandas as pd

_COMM = np.full(51, 1.0e-4)
_COMM[0] = 2.0e-5
_LIMIT = np.full(51, 10_000.0)
_LIMIT[0] = 100_000.0
BLOCK = 125
FIRST = 251
LAST = 750


def score(m, s):
    if m <= 0 or s < 1e-10:
        return m
    sr = sqrt(250.0) * m / s
    return m * sr * sr / (sr * sr + 1.0)


def daily_pnl(mod, prices, start, end):
    if hasattr(mod, "reset"):
        mod.reset()
    cash = val = comm = 0.0
    pos = np.zeros(51)
    out = []
    for t in range(start - 1, end + 1):
        h = prices[:, :t]
        cur = h[:, -1]
        if t < end:
            req = mod.getMyPosition(h)
            lim = (_LIMIT / cur).astype(int)
            new = np.clip(req, -lim, lim).astype(float)
        else:
            new = pos.copy()
        d = new - pos
        cash -= cur.dot(d) + comm
        comm = float(np.sum(cur * np.abs(d) * _COMM))
        pos = new
        v = cash + pos.dot(cur)
        out.append(v - val)
        val = v
    return np.asarray(out[1:])


def main():
    names = _sys.argv[1:] or ["four_sigmas", "simple_pairs"]
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    blocks = [(s, min(s + BLOCK - 1, LAST))
              for s in range(FIRST, LAST + 1, BLOCK)]
    blocks = [b for b in blocks if b[1] - b[0] > 60]

    report = {}
    print(f"{'strategy':>16} | " + " ".join(f"{a}-{b}".rjust(9) for a, b in blocks)
          + " |    median     worst      mean")
    print("-" * (18 + 10 * len(blocks) + 32))
    for name in names:
        mod = importlib.import_module(name)
        pnl = daily_pnl(mod, prices, FIRST, LAST)
        cells = []
        for a, b in blocks:
            seg = pnl[a - FIRST:b - FIRST + 1]
            cells.append(score(float(seg.mean()), float(seg.std())))
        med, worst, mean = float(np.median(cells)), float(min(cells)), float(np.mean(cells))
        report[name] = {"blocks": cells, "median": med, "worst": worst, "mean": mean}
        print(f"{name:>16} | " + " ".join(f"{c:9.1f}" for c in cells)
              + f" | {med:9.1f} {worst:9.1f} {mean:9.1f}")

    out = _ROOT / "cusum_output" / "block_consistency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("\nmedian / worst block are the OOS-relevant numbers; a big gap between")
    print("mean and worst is the signature of a strategy fitted to one window.")


if __name__ == "__main__":
    main()

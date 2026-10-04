"""Backtest four_sigmas vs four_sigmas_pkg (+ ablations), eval.py-faithful.

Reports per-125d-block eval score, the block FLOOR (the protocol metric), the full
501-750 score, and traded $volume (commission proxy). Replays each config once with a
clean reset so the stateful atlas is correct.

Usage:  python research/backtest_package.py [config ...]
        configs: baseline all atlas regime pc1   (default: baseline all)
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import four_sigmas as fs
import four_sigmas_pkg as pkg

prcAll = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
N, T = prcAll.shape
COMM = np.full(N, 0.0001); COMM[0] = 0.00002
DLIM = np.full(N, 10_000.0); DLIM[0] = 100_000.0


def eval_score(mu, sd):
    if mu <= 0 or sd < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def replay(getpos, do_reset):
    do_reset()
    shares = {}
    for t in range(130, T + 1):
        prc = prcAll[:, :t]
        shares[t] = np.asarray(getpos(prc), dtype=float)
    return shares


def account(share_map, start, end):
    cash = value = comm = 0.0
    pos = np.zeros(N)
    pll = []; dvol_tot = 0.0
    for t in range(start - 1, end + 1):
        cur = prcAll[:, t - 1]
        if t < end and t in share_map:
            lim = (DLIM / cur).astype(int)
            newPos = np.clip(share_map[t], -lim, lim).astype(int)
        else:
            newPos = pos
        dpos = newPos - pos
        cash -= cur.dot(dpos) + comm
        dvol = cur * np.abs(dpos)
        comm = float(np.sum(dvol * COMM)); dvol_tot += float(dvol.sum())
        pos = newPos
        today = cash + pos.dot(cur) - value
        value = cash + pos.dot(cur)
        if t >= start:
            pll.append(today)
    return np.asarray(pll), dvol_tot


BLOCKS = [("train1", 251, 375), ("train2", 376, 500),
          ("test1", 501, 625), ("test2", 626, 750)]


def score_config(name, getpos, do_reset):
    sh = replay(getpos, do_reset)
    row = {}
    for bn, s, e in BLOCKS:
        pll, _ = account(sh, s, e)
        row[bn] = eval_score(pll.mean(), pll.std())
    full, dvol = account(sh, 501, 750)
    row["floor"] = min(row[b[0]] for b in BLOCKS)
    row["full_501_750"] = eval_score(full.mean(), full.std())
    row["dvol_501_750_M"] = dvol / 1e6
    return row


def set_baseline():
    fs.ATLAS_OU_WINDOW = 120
    fs.reset()

def make_pkg_config(atlas, regime, pc1):
    def setup():
        pkg.USE_ATLAS_GATE = atlas
        pkg.USE_REGIME_GATE = regime
        pkg.USE_PC1_HEDGE = pc1
        pkg.reset()
    return setup


CONFIGS = {
    "baseline": (fs.getMyPosition, set_baseline),
    "all":      (pkg.getMyPosition, make_pkg_config(True, True, True)),
    "atlas":    (pkg.getMyPosition, make_pkg_config(True, False, False)),
    "regime":   (pkg.getMyPosition, make_pkg_config(False, True, False)),
    "pc1":      (pkg.getMyPosition, make_pkg_config(False, False, True)),
}

want = _sys.argv[1:] or ["baseline", "all"]
rows = {}
for name in want:
    getpos, setup = CONFIGS[name]
    rows[name] = score_config(name, getpos, setup)
    print(f"done {name}", flush=True)

cols = ["train1", "train2", "test1", "test2", "floor", "full_501_750", "dvol_501_750_M"]
print(f"\n{'config':9s} | " + " ".join(f"{c:>12s}" for c in cols))
print("-" * 100)
for name in want:
    r = rows[name]
    print(f"{name:9s} | " + " ".join(f"{r[c]:12.1f}" if c != 'dvol_501_750_M'
          else f"{r[c]:12.2f}" for c in cols))
if "baseline" in rows:
    b = rows["baseline"]
    print("\ndelta vs baseline:")
    for name in want:
        if name == "baseline":
            continue
        r = rows[name]
        print(f"  {name:9s} floor {r['floor']-b['floor']:+7.1f}   "
              f"full {r['full_501_750']-b['full_501_750']:+7.1f}   "
              f"test2 {r['test2']-b['test2']:+7.1f}   "
              f"dvol {r['dvol_501_750_M']-b['dvol_501_750_M']:+6.2f}M")

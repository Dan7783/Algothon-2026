"""Exhaustive subset search over the robust per-pair candidates.

`tune_all_pairs.py` produced a robust candidate for 7 of 21 seeded pairs and
picked a set greedily.  Greedy is myopic and here it demonstrably matters:
CCNS/SRTX was taken early, which locked out ALUT/CCNS (they share CCNS) even
though ALUT/CCNS alone was worth +10.74 test.  This evaluates all 127
subsets on the daniel core, then hybrid-confirms the finalists.

Self-contained: reads prices with pandas and drives daniel_filter's public
API directly, so it does not depend on modules that move between sessions.
"""

from __future__ import annotations

# --- repo path bootstrap: run as `python research/<script>.py` from root ---
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "research")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

import itertools
import json
from math import sqrt

import numpy as np
import pandas as pd

import archive.old_tuners.daniel_filter as dfil
from archive.old_tuners.daniel_filter import FiltCfg, TradeCfg

TRAIN_START, TRAIN_END = 251, 500
TEST_START, TEST_END = 501, 750
OUT_DIR = _REPO_ROOT / "cusum_output" / "daniel_pairs"

COMM = np.full(51, 1.0e-4)
COMM[0] = 2.0e-5
LIMIT = np.full(51, 10_000.0)
LIMIT[0] = 100_000.0

BASE = tuple(dfil.PAIR_CFGS)


def score(mu: float, sd: float) -> float:
    if mu <= 0.0 or sd < 1e-10:
        return mu
    sr = sqrt(250.0) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def replay(prices: np.ndarray, start: int, end: int, module) -> float:
    """eval.py-faithful replay of a strategy module."""
    if hasattr(module, "reset"):
        module.reset()
    cash = value = comm = 0.0
    pos = np.zeros(51)
    pnl = []
    for t in range(start - 1, end + 1):
        hist = prices[:, :t]
        cur = hist[:, -1]
        if t < end:
            req = module.getMyPosition(hist)
            lim = (LIMIT / cur).astype(int)
            new = np.clip(req, -lim, lim).astype(float)
        else:
            new = pos.copy()
        d = new - pos
        cash -= cur.dot(d) + comm
        comm = float(np.sum(cur * np.abs(d) * COMM))
        pos = new
        val = cash + pos.dot(cur)
        pnl.append(val - value)
        value = val
    a = np.asarray(pnl[1:])
    return score(float(a.mean()), float(a.std()))


def main() -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    sel = json.loads((OUT_DIR / "tune_all_selected.json").read_text(encoding="utf-8"))
    W = sel["winners"]
    names = list(W)
    print(f"candidates: {names}\n")

    def apply(subset):
        cfgs = tuple(
            s._replace(filt=FiltCfg(**W[s.name]["filt"]),
                       trade=TradeCfg(**W[s.name]["trade"]))
            if s.name in subset else s
            for s in BASE
        )
        dfil.apply_pair_cfgs(cfgs)
        return cfgs

    apply(set())
    b_tr = replay(prices, TRAIN_START, TRAIN_END, dfil)
    b_te = replay(prices, TEST_START, TEST_END, dfil)
    print(f"baseline core: train {b_tr:.2f}  test {b_te:.2f}\n")

    rows = []
    for k in range(1, len(names) + 1):
        for c in itertools.combinations(names, k):
            apply(set(c))
            a = replay(prices, TRAIN_START, TRAIN_END, dfil)
            b = replay(prices, TEST_START, TEST_END, dfil)
            rows.append((a, b, c))
    apply(set())

    good = [r for r in rows if r[0] >= b_tr - 1e-9 and r[1] >= b_te - 1e-9]
    good.sort(key=lambda r: min(r[0] - b_tr, r[1] - b_te), reverse=True)
    print(f"{len(good)}/{len(rows)} subsets improve or hold BOTH windows")
    print("\ntop 12 by worst-window gain:")
    for a, b, c in good[:12]:
        print(f"  train {a:8.2f} ({a-b_tr:+6.2f})  test {b:8.2f} ({b-b_te:+6.2f})  "
              f"{'+'.join(c)}")

    by_total = sorted(good, key=lambda r: (r[0] - b_tr) + (r[1] - b_te), reverse=True)
    print("\ntop 5 by total gain:")
    for a, b, c in by_total[:5]:
        print(f"  train {a:8.2f} ({a-b_tr:+6.2f})  test {b:8.2f} ({b-b_te:+6.2f})  "
              f"{'+'.join(c)}")

    # hybrid confirmation of the finalists
    try:
        import robust_hybrid_v2 as rh
    except Exception:
        rh = None
    if rh is not None:
        print("\nrobust_hybrid_v2 confirmation:")
        finalists = []
        seen = set()
        for a, b, c in (good[:3] + by_total[:3]):
            if c not in seen:
                seen.add(c)
                finalists.append(c)
        apply(set())
        if hasattr(rh, "apply_pair_cfgs"):
            rh.apply_pair_cfgs(BASE)
        h_tr = replay(prices, TRAIN_START, TRAIN_END, rh)
        h_te = replay(prices, TEST_START, TEST_END, rh)
        print(f"  baseline           train {h_tr:8.2f}  test {h_te:8.2f}")
        best = None
        for c in finalists:
            cfgs = apply(set(c))
            if hasattr(rh, "apply_pair_cfgs"):
                rh.apply_pair_cfgs(cfgs)
            a = replay(prices, TRAIN_START, TRAIN_END, rh)
            b = replay(prices, TEST_START, TEST_END, rh)
            ok = "OK" if (a >= h_tr - 1e-9 and b >= h_te - 1e-9) else "  "
            print(f"  {ok} {'+'.join(c):46s} train {a:8.2f} ({a-h_tr:+6.2f})  "
                  f"test {b:8.2f} ({b-h_te:+6.2f})")
            if a >= h_tr - 1e-9 and b >= h_te - 1e-9:
                g = min(a - h_tr, b - h_te)
                if best is None or g > best[0]:
                    best = (g, c, a, b)
        apply(set())
        if hasattr(rh, "apply_pair_cfgs"):
            rh.apply_pair_cfgs(BASE)
        if best:
            print(f"\n  BEST (improves both books, both windows): {'+'.join(best[1])}")
            print(f"    hybrid train {best[2]:.2f} test {best[3]:.2f}")
        else:
            print("\n  no subset improves both windows of the hybrid")

    (OUT_DIR / "subset_search.json").write_text(
        json.dumps({"baseline": [b_tr, b_te],
                    "top_worst_window": [[a, b, list(c)] for a, b, c in good[:12]],
                    "top_total": [[a, b, list(c)] for a, b, c in by_total[:5]]},
                   indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_DIR / 'subset_search.json'}")


if __name__ == "__main__":
    main()

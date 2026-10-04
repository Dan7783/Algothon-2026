#!/usr/bin/env python
"""Properly test the ridge's SUB-THRESHOLD opinion on deadlock asset-days.

`sleeve_breakdown` exposes ridge_raw = RIDGE_LEG * sign(prediction) * active,
which is identically zero on any asset where the ridge is inactive -- and every
deadlock day is inactive by construction. So the signed sub-threshold view is
NOT recoverable from the sleeve output; it has to be recomputed.

This recomputes the ridge VAR(1) prediction without the RIDGE_THETA gate and
asks whether its sign predicts the deadlock day's move.

Run from the repo root:  python research/deadlock_subthreshold_ridge.py
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

FIRST, LAST = 250, 1000
WINDOWS = {"train 251-500": (251, 500),
           "val   501-750": (501, 750),
           "org   751-1000": (751, 1000)}


def raw_prediction(returns: np.ndarray) -> np.ndarray:
    """fs._ridge_forecast's prediction, UNGATED (no RIDGE_THETA, no gate_n)."""
    nins, n_returns = returns.shape
    pred = np.zeros(nins)
    if n_returns + 1 < fs.RIDGE_MIN_T:
        return pred
    history = returns[1:, :].T
    scale = history.std(axis=0) + 1.0e-12
    predictors = history[:-1] / scale
    targets = history[1:] / scale
    coef = np.linalg.solve(
        predictors.T @ predictors
        + fs.RIDGE_LAM * len(predictors) * np.eye(fs.N_INST - 1),
        predictors.T @ targets,
    )
    pred[1:] = (history[-1] / scale) @ coef
    return pred


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main() -> None:
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    fs.reset()
    rows = []
    for t in range(FIRST, LAST + 1):
        hist = prices[:, :t]
        s = fs.sleeve_breakdown(hist)
        if s is None or t >= prices.shape[1]:
            continue
        dl = (s["pair_active_mask"]
              & (np.abs(s["pair_vote"]) < 1e-12)
              & ~s["ridge_active_mask"]
              & ~s["crash_mask"])
        if not dl.any():
            continue
        pred = raw_prediction(np.diff(np.log(hist), axis=1))
        ret = prices[:, t] / prices[:, t - 1] - 1.0
        for a in np.flatnonzero(dl):
            rows.append({"day": t, "asset": int(a),
                         "pred": float(pred[a]),
                         "strength": float(s["ridge_strength"][a]),
                         "ret": float(ret[a])})

    df = pd.DataFrame(rows)
    df = df[(df.ret != 0) & (df.pred != 0)]
    print(f"{len(df)} deadlock asset-days with a non-zero sub-threshold view\n")

    print("=" * 70)
    print("Does sign(sub-threshold ridge prediction) call the deadlock day?")
    print("=" * 70)
    print(f"{'window':>16} {'n':>6} {'accuracy':>20} {'move-wtd':>10}")
    for wn, (lo, hi) in WINDOWS.items():
        w = df[(df.day >= lo) & (df.day <= hi)]
        if not len(w):
            continue
        hit = np.sign(w.pred) == np.sign(w.ret)
        acc = float(hit.mean())
        lo_ci, hi_ci = wilson(int(hit.sum()), len(w))
        mw = float(np.sum(np.abs(w.ret) * hit) / np.sum(np.abs(w.ret)))
        print(f"{wn:>16} {len(w):6d} {acc*100:8.1f}%"
              f" [{lo_ci*100:4.1f},{hi_ci*100:4.1f}] {mw*100:9.1f}%")

    n = len(df)
    hit = np.sign(df.pred) == np.sign(df.ret)
    lo_ci, hi_ci = wilson(int(hit.sum()), n)
    print("-" * 70)
    print(f"{'ALL':>16} {n:6d} {hit.mean()*100:8.1f}%"
          f" [{lo_ci*100:4.1f},{hi_ci*100:4.1f}]")

    # does it get better on the strongest sub-threshold views?
    print("\naccuracy by sub-threshold strength decile (does conviction help?)")
    for wn, (lo, hi) in WINDOWS.items():
        w = df[(df.day >= lo) & (df.day <= hi)].copy()
        if len(w) < 50:
            continue
        w["q"] = pd.qcut(w.strength, 4, labels=False, duplicates="drop")
        accs = w.groupby("q").apply(
            lambda g: float((np.sign(g.pred) == np.sign(g.ret)).mean()),
            include_groups=False)
        print(f"  {wn:>16}: " + "  ".join(f"Q{i+1} {v*100:.1f}%"
                                          for i, v in enumerate(accs)))
    print("\nbreak-even ~52%; must clear it on BOTH held-out windows.")


if __name__ == "__main__":
    main()

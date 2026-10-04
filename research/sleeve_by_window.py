"""Which sleeve makes the book test-heavy?

Replays robust_hybrid cumulatively (atlas -> +extraLL -> +ridge ->
+discovery) on each window, so the train/test balance can be attributed to
a sleeve rather than guessed at.
"""

from __future__ import annotations

# --- repo path bootstrap: run as `python research/<script>.py` from root ---
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "research"), str(_REPO_ROOT / "archive")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

import json
from math import sqrt

import numpy as np
import pandas as pd

import four_sigmas as rh
from tune_daniel_pairs import OUT_DIR, TEST_END, TEST_START, TRAIN_END, TRAIN_START

COMM = np.full(51, 1.0e-4)
COMM[0] = 2.0e-5
LIMIT = np.full(51, 10_000.0)
LIMIT[0] = 100_000.0
SLEEVES = ["atlas", "extraLL", "ridge", "discovery"]


def score(mean: float, std: float) -> float:
    if mean <= 0.0 or std < 1e-10:
        return mean
    sr = sqrt(250.0) * mean / std
    return mean * sr * sr / (sr * sr + 1.0)


def replay(prices: np.ndarray, start: int, end: int, keep: list[str]) -> dict:
    rh.reset()
    cash = value = comm = 0.0
    pos = np.zeros(51)
    pnl = []
    for t in range(start - 1, end + 1):
        hist = prices[:, :t]
        cur = hist[:, -1]
        if t < end:
            sl = rh.sleeve_breakdown(hist)
            if sl is None:
                dollars = np.zeros(51)
            else:
                dollars = np.zeros(51)
                for name in keep:
                    dollars = dollars + sl[name]
                dollars[0] = 0.0
                dollars[1:] = np.clip(dollars[1:], -10_000.0, 10_000.0)
            new = (dollars / cur).astype(int)
            lim = (LIMIT / cur).astype(int)
            new = np.clip(new, -lim, lim)
        else:
            new = pos.astype(int)
        delta = new - pos
        cash -= cur.dot(delta) + comm
        comm = float(np.sum(cur * np.abs(delta) * COMM))
        pos = new.astype(float)
        val = cash + pos.dot(cur)
        pnl.append(val - value)
        value = val
    arr = np.asarray(pnl[1:])
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "score": score(float(arr.mean()), float(arr.std()))}


def main() -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    rows = []
    print(f"{'cumulative book':>28} | {'train':>8} {'incr':>7} | "
          f"{'test':>8} {'incr':>7} | {'test-train':>10}")
    print("-" * 84)
    prev_tr = prev_te = 0.0
    for k in range(1, len(SLEEVES) + 1):
        keep = SLEEVES[:k]
        tr = replay(prices, TRAIN_START, TRAIN_END, keep)
        te = replay(prices, TEST_START, TEST_END, keep)
        print(
            f"{' + '.join(keep):>28} | {tr['score']:8.2f} "
            f"{tr['mean'] - prev_tr:+7.2f} | {te['score']:8.2f} "
            f"{te['mean'] - prev_te:+7.2f} | {te['score'] - tr['score']:+10.2f}"
        )
        rows.append({"sleeves": keep, "train": tr, "test": te,
                     "incr_train_mean": tr["mean"] - prev_tr,
                     "incr_test_mean": te["mean"] - prev_te})
        prev_tr, prev_te = tr["mean"], te["mean"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sleeve_by_window.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'sleeve_by_window.json'}")


if __name__ == "__main__":
    main()

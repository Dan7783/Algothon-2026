"""Adaptive-sleeve cut-frequency gate: train-only sleeve-level selection.

The ungated adaptive sleeve nets ~0 on train: pairs whose filter cuts
frequently (>= 7 breaks) lose persistently.  The gate is the CUSUM's own
break count — NOT an ADF/decay statistic: a book may hold or enter only if
its filter cut at most ``max_cuts`` times over the trailing 250 days.
Grid over (W, lo, max_cuts), scored on summed train pll, chosen by best
worst-half (251-375 vs 376-500).
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

import numpy as np

from research.fifty_pairs_lab import (
    ADAPTIVE_NAMES,
    BAND_HI,
    LEG,
    MIN_HISTORY,
    adaptive_z_path,
    filter_paths,
    ticker_indices,
)
from research.pair_book_lab import (
    TRAIN_END,
    TRAIN_MID,
    TRAIN_START,
    TEST_START,
    TEST_END,
    account_two_leg,
    eval_score,
    load_raw_prices,
)

CUT_LOOKBACK = 250
K_GRID = [1, 2, 3, 10_000]
W_GRID = [60, 100, 145]
LO_GRID = [0.2, 0.27, 0.4]


def gated_band_sim(z, py, px, lo, cut, adapting, recent_cuts, max_cuts):
    T = len(py)
    sy = np.zeros(T)
    sx = np.zeros(T)
    side = 0
    for d in range(MIN_HISTORY - 1, T):
        if cut[d] or recent_cuts[d] > max_cuts:
            side = 0
        elif adapting[d]:
            pass
        elif lo <= abs(z[d]) <= BAND_HI:
            side = -int(np.sign(z[d]))
        else:
            side = 0
        if side != 0:
            sy[d] = int(side * LEG / py[d])
            sx[d] = int(-side * LEG / px[d])
    return sy, sx


def main() -> None:
    prices = load_raw_prices()
    idx = ticker_indices()
    pairs = [(idx[a], idx[b]) for a, b in ADAPTIVE_NAMES]
    log_prices = np.log(prices)

    paths = {p: filter_paths(log_prices[p[0]], log_prices[p[1]]) for p in pairs}
    recent = {}
    for p in pairs:
        c = paths[p]["cut"].astype(int)
        cs = np.concatenate(([0], np.cumsum(c)))
        recent[p] = np.array(
            [cs[t + 1] - cs[max(0, t + 1 - CUT_LOOKBACK)] for t in range(len(c))]
        )
    zs = {
        (p, w): adaptive_z_path(paths[p], w) for p in pairs for w in W_GRID
    }

    table = []
    for w in W_GRID:
        for lo in LO_GRID:
            for k in K_GRID:
                train_sum = h1_sum = h2_sum = None
                for p in pairs:
                    py, px = prices[p[0]], prices[p[1]]
                    sy, sx = gated_band_sim(
                        zs[(p, w)], py, px, lo,
                        paths[p]["cut"], paths[p]["adapting"], recent[p], k,
                    )
                    tr = account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_END)
                    h1 = account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_MID)
                    h2 = account_two_leg(sy, sx, py, px, TRAIN_MID + 1, TRAIN_END)
                    train_sum = tr if train_sum is None else train_sum + tr
                    h1_sum = h1 if h1_sum is None else h1_sum + h1
                    h2_sum = h2 if h2_sum is None else h2_sum + h2
                table.append(
                    {
                        "window": w,
                        "lo": lo,
                        "max_cuts": k,
                        "train": eval_score(train_sum.mean(), train_sum.std()),
                        "half1": eval_score(h1_sum.mean(), h1_sum.std()),
                        "half2": eval_score(h2_sum.mean(), h2_sum.std()),
                    }
                )
    table.sort(key=lambda r: -min(r["half1"], r["half2"]))
    for r in table[:10]:
        print(
            f"W {r['window']:>3} lo {r['lo']:.2f} maxcuts {r['max_cuts']:>5}: "
            f"train {r['train']:+8.1f} halves {r['half1']:+8.1f}/{r['half2']:+8.1f}"
        )
    chosen = table[0]
    print("chosen:", chosen)

    # Per-pair under the chosen config (test scored once for reporting).
    w, lo, k = chosen["window"], chosen["lo"], chosen["max_cuts"]
    print(f"\nper-pair at W={w} lo={lo} max_cuts={k}:")
    for p, nm in zip(pairs, ADAPTIVE_NAMES):
        py, px = prices[p[0]], prices[p[1]]
        sy, sx = gated_band_sim(
            zs[(p, w)], py, px, lo,
            paths[p]["cut"], paths[p]["adapting"], recent[p], k,
        )
        tr = account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_END)
        te = account_two_leg(sy, sx, py, px, TEST_START, TEST_END)
        days_in = int(((sy != 0) | (sx != 0)).sum())
        print(
            f"  {nm[0]}/{nm[1]:>4}: train {eval_score(tr.mean(), tr.std()):+7.2f} "
            f"test {eval_score(te.mean(), te.std()):+7.2f} days-in {days_in}"
        )
    json.dump(chosen, open("cusum_output/fifty_pairs/adaptive_gate.json", "w"), indent=2)


if __name__ == "__main__":
    main()

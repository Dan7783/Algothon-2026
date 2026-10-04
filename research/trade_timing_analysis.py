"""Entry/exit timing analysis for the fifty-pairs stable band books.

The user observes trades entering at low |z| ("much lower" than the later
peak) and missing the large-z excursions.  In band mode this has a precise
mechanical form worth quantifying:

  * entries happen at the 0.27 band edge, so a stretch to |z| = 2-3 is
    carried as adverse excursion from a weak entry level, and
  * the 4.0 break cap force-flattens the book exactly at the largest
    stretches, so the best fade levels are systematically exited, not
    entered.

This script measures both effects on the 16 stable pairs and then runs the
counterfactuals: later hysteresis entries (0.75-2.25 sigma) and wider / no
stop, on train 251-500 and test 501-750.
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

import numpy as np

from research.fifty_pairs_lab import (
    STABLE_NAMES,
    band_sim,
    kalman_beta_path,
    stable_z_path,
    ticker_indices,
)
from research.pair_book_lab import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    account_two_leg,
    eval_score,
    load_raw_prices,
)

LEG = 10_000.0
W = 145
LO = 0.27


def hysteresis_sim(z, py, px, entry, exit_, stop):
    T = len(py)
    sy = np.zeros(T)
    sx = np.zeros(T)
    side = 0
    for d in range(129, T):
        az = abs(z[d])
        if side == 0:
            if entry <= az < stop:
                side = -int(np.sign(z[d]))
        elif az <= exit_ or az >= stop:
            side = 0
        if side != 0:
            sy[d] = int(side * LEG / py[d])
            sx[d] = int(-side * LEG / px[d])
    return sy, sx


def episodes(side_path, z):
    out = []
    d = 0
    T = len(side_path)
    while d < T:
        if side_path[d] == 0:
            d += 1
            continue
        start = d
        while d < T and side_path[d] == side_path[start]:
            d += 1
        seg = slice(start, d)
        out.append(
            {
                "start": start,
                "end": d - 1,
                "entry_z": abs(z[start]),
                "max_z": float(np.abs(z[seg]).max()),
                "peak_day": start + int(np.abs(z[seg]).argmax()),
                "days": d - start,
            }
        )
    return out


def main() -> None:
    prices = load_raw_prices()
    idx = ticker_indices()
    pairs = [(idx[a], idx[b]) for a, b in STABLE_NAMES]
    log_prices = np.log(prices)

    all_eps = []
    stop_exit_days = 0
    high_z_flat_days = 0
    scored_days = 0
    for (i, j), nm in zip(pairs, STABLE_NAMES):
        py, px = prices[i], prices[j]
        beta = kalman_beta_path(log_prices[i], log_prices[j])
        z = stable_z_path(log_prices[i], log_prices[j], beta, W)
        sy, sx, side = band_sim(z, py, px, LO)
        pll = account_two_leg(sy, sx, py, px, TRAIN_START, TEST_END)
        for ep in episodes(side, z):
            lo_i = max(0, ep["start"] + 2 - TRAIN_START)
            hi_i = min(len(pll), ep["end"] + 3 - TRAIN_START)
            if hi_i <= lo_i:
                continue
            ep["pnl"] = float(pll[lo_i:hi_i].sum())
            peak_i = min(len(pll), ep["peak_day"] + 3 - TRAIN_START)
            ep["pnl_to_peak"] = float(pll[lo_i:peak_i].sum())
            ep["pair"] = f"{nm[0]}/{nm[1]}"
            all_eps.append(ep)
        scope = slice(TRAIN_START - 1, TEST_END)
        az = np.abs(z[scope])
        flat = side[scope] == 0
        stop_exit_days += int((az > 4.0).sum())
        high_z_flat_days += int(((az >= 2.0) & flat).sum())
        scored_days += az.size

    eps = [e for e in all_eps if "pnl" in e]
    print(f"{len(eps)} episodes across 16 stable pairs, days 251-750")
    entry_z = np.array([e["entry_z"] for e in eps])
    print(
        f"entry |z|: median {np.median(entry_z):.2f}, "
        f"90th pct {np.percentile(entry_z, 90):.2f} "
        f"(band edge {LO}) — entries cluster at the band edge"
    )
    print(
        f"days with |z|>4 (stop zone, forced flat): {stop_exit_days}; "
        f"days flat while |z|>=2: {high_z_flat_days} of {scored_days}"
    )
    for lo_b, hi_b in ((0.27, 0.75), (0.75, 1.5), (1.5, 2.5), (2.5, 9.9)):
        sel = [e for e in eps if lo_b <= e["entry_z"] < hi_b]
        if not sel:
            continue
        pnl = np.array([e["pnl"] for e in sel])
        days = np.array([e["days"] for e in sel])
        print(
            f"entry z in [{lo_b:4.2f},{hi_b:4.2f}): n {len(sel):4d}  "
            f"avg pnl {pnl.mean():+8.1f}  pnl/day-held "
            f"{pnl.sum() / max(days.sum(), 1):+6.2f}  win {np.mean(pnl > 0):.0%}"
        )
    big = [e for e in eps if e["max_z"] >= 2.0]
    if big:
        to_peak = np.array([e["pnl_to_peak"] for e in big])
        total = np.array([e["pnl"] for e in big])
        print(
            f"episodes reaching |z|>=2: n {len(big)}, "
            f"avg carry entry->peak {to_peak.mean():+8.1f} "
            f"(adverse), avg total {total.mean():+8.1f} — "
            f"the profit is peak->exit; early entry pays the carry"
        )

    # ---- counterfactuals ----
    print("\ncounterfactual sleeve totals (16 stable pairs summed):")
    variants = [
        ("band lo=0.27 hi=4.0 (shipped)", ("band", 0.27, 4.0)),
        ("band lo=0.27 hi=99 (no stop)", ("band", 0.27, 99.0)),
        ("band lo=0.27 hi=6.0", ("band", 0.27, 6.0)),
        ("hyst enter 0.75 exit 0.1 stop 99", ("hyst", 0.75, 99.0)),
        ("hyst enter 1.25 exit 0.1 stop 99", ("hyst", 1.25, 99.0)),
        ("hyst enter 1.75 exit 0.1 stop 99", ("hyst", 1.75, 99.0)),
        ("hyst enter 2.25 exit 0.1 stop 99", ("hyst", 2.25, 99.0)),
    ]
    for label, (mode, a, stop) in variants:
        tr_sum = te_sum = None
        for (i, j) in pairs:
            py, px = prices[i], prices[j]
            beta = kalman_beta_path(log_prices[i], log_prices[j])
            z = stable_z_path(log_prices[i], log_prices[j], beta, W)
            if mode == "band":
                import research.fifty_pairs_lab as fl

                old = fl.BAND_HI
                fl.BAND_HI = stop
                sy, sx, _ = band_sim(z, py, px, a)
                fl.BAND_HI = old
            else:
                sy, sx = hysteresis_sim(z, py, px, a, 0.1, stop)
            tr = account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_END)
            te = account_two_leg(sy, sx, py, px, TEST_START, TEST_END)
            tr_sum = tr if tr_sum is None else tr_sum + tr
            te_sum = te if te_sum is None else te_sum + te
        print(
            f"  {label:34s} train {eval_score(tr_sum.mean(), tr_sum.std()):+8.1f} "
            f"test {eval_score(te_sum.mean(), te_sum.std()):+8.1f}"
        )


if __name__ == "__main__":
    main()

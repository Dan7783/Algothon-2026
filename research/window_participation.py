"""Per-window participation audit: is the train window under-traded?

For every seeded pair, replays the exact PairAgent logic (same functions the
pair_seed_explorer notebook uses) and reports, separately for train
(251-500) and test (501-750):

    days in market, days blocked by the ADF gate, days blocked by the kappa
    gate, days flat purely because |z| < z_entry, mean |z|, and PnL.

That decomposition says WHY a window trades less: a gate is switching the
pair off, or the spread simply never stretches to the entry threshold.
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

import archive.old_tuners.daniel_filter as dfil
from research.pair_book_lab import load_raw_prices
from tune_daniel_pairs import (
    OUT_DIR,
    TEST_DECIDE_START,
    TEST_END,
    TEST_START,
    TRAIN_DECIDE_START,
    TRAIN_END,
    TRAIN_START,
    gate_and_z_paths,
    pnl_path,
    record_paths,
    trade_sim,
    window_stats,
)


def window_report(gates, sy, start, end, trade) -> dict:
    """Participation decomposition over an inclusive day range.

    Gates are counted only when the pair actually honours them: a pair with
    require_adf=False is not 'blocked' by an ADF flag it ignores.
    """
    lo, hi = start - 1, end  # day d is index d-1
    idx = slice(lo, hi)
    in_mkt = np.abs(sy[idx]) > 0
    adf = gates["adf_block"][idx] & bool(trade.require_adf)
    kap = gates["kappa_block"][idx] & bool(trade.require_kappa)
    z = np.abs(gates["z"][idx])
    n = hi - lo
    flat_low_z = (~in_mkt) & (~adf) & (~kap)
    return {
        "days": n,
        "in_market": int(in_mkt.sum()),
        "pct_in": 100.0 * in_mkt.sum() / n,
        "adf_blocked": int(adf.sum()),
        "kappa_blocked": int(kap.sum()),
        "flat_low_z": int(flat_low_z.sum()),
        "mean_abs_z": float(z[~adf].mean()) if (~adf).any() else 0.0,
    }


def main() -> None:
    prices = load_raw_prices()
    rows = []
    tot = {"train": np.zeros(4), "test": np.zeros(4)}

    print(f"{'pair':>11} | {'window':>5} | {'in mkt':>12} | {'adf':>5} "
          f"{'kap':>5} {'lowz':>5} | {'m|z|':>5} | {'score':>7} {'mean$':>8}")
    print("-" * 88)
    for spec in dfil.PAIR_CFGS:
        ly, lx = np.log(prices[spec.y]), np.log(prices[spec.x])
        py, px = prices[spec.y], prices[spec.x]
        path = record_paths(ly, lx, spec.filt)
        gates = gate_and_z_paths(ly, lx, path["beta"])
        sy_tr, sx_tr = trade_sim(gates, py, px, spec.trade, TRAIN_DECIDE_START)
        sy_te, sx_te = trade_sim(gates, py, px, spec.trade, TEST_DECIDE_START)
        tr_pnl = pnl_path(sy_tr, sx_tr, py, px)
        te_pnl = pnl_path(sy_te, sx_te, py, px)
        tr = window_report(gates, sy_tr, TRAIN_START, TRAIN_END, spec.trade)
        te = window_report(gates, sy_te, TEST_START, TEST_END, spec.trade)
        tr_s = window_stats(tr_pnl, TRAIN_START, TRAIN_END)
        te_s = window_stats(te_pnl, TEST_START, TEST_END)
        for tag, w, s in (("train", tr, tr_s), ("test", te, te_s)):
            print(
                f"{spec.name:>11} | {tag:>5} | {w['in_market']:4d}/{w['days']:3d} "
                f"{w['pct_in']:5.1f}% | {w['adf_blocked']:5d} {w['kappa_blocked']:5d} "
                f"{w['flat_low_z']:5d} | {w['mean_abs_z']:5.2f} | "
                f"{s['score']:7.2f} {s['mean']:8.2f}"
            )
            tot[tag] += np.array(
                [w["in_market"], w["adf_blocked"], w["kappa_blocked"], w["flat_low_z"]]
            )
        rows.append(
            {"name": spec.name, "pair": [spec.y, spec.x],
             "train": {**tr, **{"score": tr_s["score"], "mean": tr_s["mean"]}},
             "test": {**te, **{"score": te_s["score"], "mean": te_s["mean"]}}}
        )
        print("-" * 88)

    n_pairs = len(dfil.PAIR_CFGS)
    print(f"\nTOTALS across {n_pairs} pairs (pair-days):")
    for tag in ("train", "test"):
        a = tot[tag]
        denom = n_pairs * 250
        print(
            f"  {tag:>5}: in market {int(a[0]):5d} ({100 * a[0] / denom:5.1f}%)   "
            f"adf-blocked {int(a[1]):5d} ({100 * a[1] / denom:5.1f}%)   "
            f"kappa-blocked {int(a[2]):5d}   flat-low-z {int(a[3]):5d}"
        )
    d = tot["test"][0] - tot["train"][0]
    print(f"\n  test trades {int(d):+d} more pair-days than train "
          f"({100 * d / max(tot['train'][0], 1):+.1f}%)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "window_participation.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'window_participation.json'}")


if __name__ == "__main__":
    main()

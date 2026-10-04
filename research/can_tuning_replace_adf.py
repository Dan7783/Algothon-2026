"""Can per-pair threshold tuning replace the ADF gate?

The ADF gate filters on a property of the RELATIONSHIP (is the spread
stationary right now), while z_entry/z_exit/z_stop filter on a property of
the MOMENT (how stretched is it).  Those are different pieces of
information, but they might be empirically substitutable.

Test design — deliberately biased IN FAVOUR of the hypothesis:

  For each pair, with require_adf=False, sweep the full threshold grid and
  let the pair pick its best combo using ORACLE knowledge of both windows
  (i.e. choose the combo that maximises the worst window versus the gated
  baseline).  That is not achievable in practice; it is an upper bound.

  If even this oracle cannot match the gated baseline on both windows, the
  gate carries information tuning cannot replace -> definitive NO.
  If it can, the un-gated oracle configs are then portfolio-tested, because
  capacity crowding is a separate failure mode from trade selection.
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

import itertools
import json

import numpy as np
import pandas as pd

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
    portfolio_window,
    record_paths,
    trade_sim,
    window_stats,
)

# Wide threshold grid: selectivity is how tuning would have to replace the
# gate, so z_entry reaches well past the seeded values.
ENTRY = [0.12, 0.18, 0.27, 0.35, 0.45, 0.60, 0.80, 1.00, 1.25, 1.50]
EXIT = [0.02, 0.06, 0.12, 0.20]
STOP = [2.50, 3.25, 4.00, 4.75]

BASE = tuple(dfil.PAIR_CFGS)


def run(gates, py, px, trade) -> dict:
    sy_tr, sx_tr = trade_sim(gates, py, px, trade, TRAIN_DECIDE_START)
    sy_te, sx_te = trade_sim(gates, py, px, trade, TEST_DECIDE_START)
    tr = window_stats(pnl_path(sy_tr, sx_tr, py, px), TRAIN_START, TRAIN_END)
    te = window_stats(pnl_path(sy_te, sx_te, py, px), TEST_START, TEST_END)
    return {
        "train": tr["score"], "test": te["score"],
        "d_train": int((np.abs(sy_tr[TRAIN_START - 1:TRAIN_END]) > 0).sum()),
        "d_test": int((np.abs(sy_te[TEST_START - 1:TEST_END]) > 0).sum()),
    }


def main() -> None:
    prices = load_raw_prices()
    prices_df = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T

    print("Can un-gated thresholds match the GATED baseline?  (oracle choice)\n")
    print(f"{'pair':>11} | {'gated base':>16} | {'best un-gated':>16} | "
          f"{'z_ent':>5} | recovered?")
    print("-" * 76)

    oracle_cfgs = []
    recovered = failed = 0
    rows = []
    for spec in BASE:
        ly, lx = np.log(prices[spec.y]), np.log(prices[spec.x])
        py, px = prices[spec.y], prices[spec.x]
        path = record_paths(ly, lx, spec.filt)
        gates = gate_and_z_paths(ly, lx, path["beta"])

        base = run(gates, py, px, spec.trade)  # as seeded (gate as configured)

        # The pair's OWN seeded thresholds must be candidates, otherwise a
        # "cannot recover" verdict could just mean the grid missed the
        # baseline point.  With them included, any pair whose gate never
        # binds recovers exactly (margin 0) by construction.
        entry_axis = sorted(set(ENTRY) | {spec.trade.z_entry})
        exit_axis = sorted(set(EXIT) | {spec.trade.z_exit})
        stop_axis = sorted(set(STOP) | {spec.trade.z_stop})

        best = None
        for ze, zx, zs in itertools.product(entry_axis, exit_axis, stop_axis):
            trade = spec.trade._replace(
                require_adf=False,
                z_entry=ze,
                z_exit=min(zx, 0.85 * ze),
                z_stop=max(zs, 1.25 * ze),
            )
            r = run(gates, py, px, trade)
            # oracle objective: maximise the worse of the two window margins
            margin = min(r["train"] - base["train"], r["test"] - base["test"])
            if best is None or margin > best[0]:
                best = (margin, ze, zx, zs, r, trade)

        margin, ze, zx, zs, r, trade = best
        ok = margin >= -1e-9
        recovered += ok
        failed += not ok
        print(
            f"{spec.name:>11} | {base['train']:7.2f} {base['test']:7.2f} | "
            f"{r['train']:7.2f} {r['test']:7.2f} | {ze:5.2f} | "
            f"{'YES' if ok else f'NO  (short {margin:+.2f})'}"
        )
        rows.append({"name": spec.name, "gated": base, "ungated_best": r,
                     "z_entry": ze, "z_exit": zx, "z_stop": zs,
                     "margin": margin, "recovered": bool(ok)})
        oracle_cfgs.append(spec._replace(trade=trade))

    print(f"\n{recovered}/{len(BASE)} pairs recover the gated result with tuning "
          f"alone; {failed} cannot.")

    # Portfolio test of the oracle set (upper bound on what tuning could do).
    print("\nPORTFOLIO with oracle un-gated thresholds (upper bound):")
    b_tr = portfolio_window(prices, BASE, TRAIN_START, TRAIN_END)["score"]
    b_te = portfolio_window(prices, BASE, TEST_START, TEST_END)["score"]
    o_tr = portfolio_window(prices, tuple(oracle_cfgs), TRAIN_START, TRAIN_END)["score"]
    o_te = portfolio_window(prices, tuple(oracle_cfgs), TEST_START, TEST_END)["score"]
    print(f"  daniel core gated : train {b_tr:8.2f}  test {b_te:8.2f}")
    print(f"  daniel core oracle: train {o_tr:8.2f} ({o_tr - b_tr:+7.2f})  "
          f"test {o_te:8.2f} ({o_te - b_te:+7.2f})")

    from research.audit_robust_hybrid import configure, replay as hreplay

    dfil.apply_pair_cfgs(BASE)
    configure(discovery=True, extra=True, ridge=1.0, scale=1.0)
    hb_tr = hreplay(prices_df, TRAIN_START, TRAIN_END)["score"]
    hb_te = hreplay(prices_df, TEST_START, TEST_END)["score"]
    dfil.apply_pair_cfgs(tuple(oracle_cfgs))
    configure(discovery=True, extra=True, ridge=1.0, scale=1.0)
    ho_tr = hreplay(prices_df, TRAIN_START, TRAIN_END)["score"]
    ho_te = hreplay(prices_df, TEST_START, TEST_END)["score"]
    print(f"  hybrid      gated : train {hb_tr:8.2f}  test {hb_te:8.2f}")
    print(f"  hybrid      oracle: train {ho_tr:8.2f} ({ho_tr - hb_tr:+7.2f})  "
          f"test {ho_te:8.2f} ({ho_te - hb_te:+7.2f})")

    dfil.apply_pair_cfgs(BASE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tuning_vs_adf.json").write_text(
        json.dumps(
            {"pairs": rows, "recovered": recovered, "failed": failed,
             "core": {"gated": [b_tr, b_te], "oracle": [o_tr, o_te]},
             "hybrid": {"gated": [hb_tr, hb_te], "oracle": [ho_tr, ho_te]}},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_DIR / 'tuning_vs_adf.json'}  (configs restored)")


if __name__ == "__main__":
    main()

"""Full ablation: remove the ADF prior-gate from EVERY pair.

Shows the difference at three levels:
  1. per pair, standalone (score + days in market, both windows)
  2. aggregate participation
  3. portfolio — daniel_filter core and robust_hybrid, both windows

Non-invasive: configs are swapped in memory and restored at the end.
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

BASE = tuple(dfil.PAIR_CFGS)
NO_ADF = tuple(
    s._replace(trade=s.trade._replace(require_adf=False)) for s in BASE
)


def standalone(spec, prices, require_adf: bool) -> dict:
    ly, lx = np.log(prices[spec.y]), np.log(prices[spec.x])
    py, px = prices[spec.y], prices[spec.x]
    path = record_paths(ly, lx, spec.filt)
    gates = gate_and_z_paths(ly, lx, path["beta"])
    trade = spec.trade._replace(require_adf=require_adf)
    sy_tr, sx_tr = trade_sim(gates, py, px, trade, TRAIN_DECIDE_START)
    sy_te, sx_te = trade_sim(gates, py, px, trade, TEST_DECIDE_START)
    tr = window_stats(pnl_path(sy_tr, sx_tr, py, px), TRAIN_START, TRAIN_END)
    te = window_stats(pnl_path(sy_te, sx_te, py, px), TEST_START, TEST_END)
    return {
        "train": tr["score"], "test": te["score"],
        "d_train": int((np.abs(sy_tr[TRAIN_START - 1:TRAIN_END]) > 0).sum()),
        "d_test": int((np.abs(sy_te[TEST_START - 1:TEST_END]) > 0).sum()),
    }


def hybrid(prices_df, cfgs) -> tuple[float, float, float, float]:
    from research.audit_robust_hybrid import configure, replay

    dfil.apply_pair_cfgs(cfgs)
    configure(discovery=True, extra=True, ridge=1.0, scale=1.0)
    tr = replay(prices_df, TRAIN_START, TRAIN_END)
    te = replay(prices_df, TEST_START, TEST_END)
    return tr["score"], te["score"], tr["mean"], te["mean"]


def main() -> None:
    prices = load_raw_prices()
    prices_df = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T

    print("PER-PAIR STANDALONE  (ADF gate on -> off)\n")
    print(f"{'pair':>11} | {'train':>15} | {'test':>15} | {'days train':>13} | "
          f"{'days test':>13}")
    print("-" * 82)
    rows = []
    agg = {"on": np.zeros(2), "off": np.zeros(2)}
    for spec in BASE:
        on = standalone(spec, prices, True)
        off = standalone(spec, prices, False)
        rows.append({"name": spec.name, "on": on, "off": off})
        agg["on"] += [on["d_train"], on["d_test"]]
        agg["off"] += [off["d_train"], off["d_test"]]
        same = on["train"] == off["train"] and on["test"] == off["test"]
        tag = "  (gate never binds)" if same else ""
        print(
            f"{spec.name:>11} | {on['train']:6.2f} -> {off['train']:6.2f} | "
            f"{on['test']:6.2f} -> {off['test']:6.2f} | "
            f"{on['d_train']:5d} -> {off['d_train']:4d} | "
            f"{on['d_test']:5d} -> {off['d_test']:4d}{tag}"
        )

    n = len(BASE) * 250
    print("\nAGGREGATE PARTICIPATION (pair-days)")
    print(f"  gate ON : train {int(agg['on'][0]):5d} ({100*agg['on'][0]/n:5.1f}%)  "
          f"test {int(agg['on'][1]):5d} ({100*agg['on'][1]/n:5.1f}%)")
    print(f"  gate OFF: train {int(agg['off'][0]):5d} ({100*agg['off'][0]/n:5.1f}%)  "
          f"test {int(agg['off'][1]):5d} ({100*agg['off'][1]/n:5.1f}%)")
    print(f"  change  : train {int(agg['off'][0]-agg['on'][0]):+5d}  "
          f"test {int(agg['off'][1]-agg['on'][1]):+5d}")

    sum_on = (sum(r["on"]["train"] for r in rows), sum(r["on"]["test"] for r in rows))
    sum_off = (sum(r["off"]["train"] for r in rows), sum(r["off"]["test"] for r in rows))
    print("\nSUM OF STANDALONE PAIR SCORES (not additive in the book)")
    print(f"  gate ON : train {sum_on[0]:8.2f}  test {sum_on[1]:8.2f}")
    print(f"  gate OFF: train {sum_off[0]:8.2f}  test {sum_off[1]:8.2f}")
    print(f"  change  : train {sum_off[0]-sum_on[0]:+8.2f}  "
          f"test {sum_off[1]-sum_on[1]:+8.2f}")

    print("\nPORTFOLIO (the decisive level)")
    c_on_tr = portfolio_window(prices, BASE, TRAIN_START, TRAIN_END)
    c_on_te = portfolio_window(prices, BASE, TEST_START, TEST_END)
    c_off_tr = portfolio_window(prices, NO_ADF, TRAIN_START, TRAIN_END)
    c_off_te = portfolio_window(prices, NO_ADF, TEST_START, TEST_END)
    print(f"  daniel core   gate ON : train {c_on_tr['score']:8.2f}  "
          f"test {c_on_te['score']:8.2f}")
    print(f"  daniel core   gate OFF: train {c_off_tr['score']:8.2f} "
          f"({c_off_tr['score']-c_on_tr['score']:+7.2f})  "
          f"test {c_off_te['score']:8.2f} ({c_off_te['score']-c_on_te['score']:+7.2f})")

    h_on = hybrid(prices_df, BASE)
    h_off = hybrid(prices_df, NO_ADF)
    print(f"  robust_hybrid gate ON : train {h_on[0]:8.2f}  test {h_on[1]:8.2f}")
    print(f"  robust_hybrid gate OFF: train {h_off[0]:8.2f} ({h_off[0]-h_on[0]:+7.2f})  "
          f"test {h_off[1]:8.2f} ({h_off[1]-h_on[1]:+7.2f})")

    dfil.apply_pair_cfgs(BASE)
    payload = {
        "per_pair": rows,
        "participation": {"on": agg["on"].tolist(), "off": agg["off"].tolist()},
        "standalone_sum": {"on": sum_on, "off": sum_off},
        "core": {"on": [c_on_tr["score"], c_on_te["score"]],
                 "off": [c_off_tr["score"], c_off_te["score"]]},
        "hybrid": {"on": list(h_on[:2]), "off": list(h_off[:2])},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "adf_ablation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'adf_ablation.json'}  (configs restored)")


if __name__ == "__main__":
    main()

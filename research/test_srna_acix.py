"""SRNA/ACIX is the only persistently-cointegrated pair NOT in the atlas.

Panel-wide screen (in-window through-origin residual ADF): exactly 7 of 1225
pairs hold ADF < -3 in BOTH windows.  Six are already seeded; SRNA/ACIX
(-3.08 train / -3.07 test) is not.  This tests adding it.

Config is deliberately UNTUNED (library defaults) so the pair is judged on
its structure, not on a fit.  Three portfolios are compared: baseline,
baseline + SRNA/ACIX, and swapping out MDGI/HTRK (which the same screen
shows is not a designed pair: -3.34 train / -1.35 test, and 0 of 87,480
tuning candidates could improve its test window).
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
from archive.old_tuners.daniel_filter import FiltCfg, PairCfg, TradeCfg
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
Y, X = dfil.I["SRNA"], dfil.I["ACIX"]


def standalone(y, x, filt, trade, prices):
    ly, lx = np.log(prices[y]), np.log(prices[x])
    py, px = prices[y], prices[x]
    path = record_paths(ly, lx, filt)
    gates = gate_and_z_paths(ly, lx, path["beta"])
    sy_tr, sx_tr = trade_sim(gates, py, px, trade, TRAIN_DECIDE_START)
    sy_te, sx_te = trade_sim(gates, py, px, trade, TEST_DECIDE_START)
    tr = window_stats(pnl_path(sy_tr, sx_tr, py, px), TRAIN_START, TRAIN_END)
    te = window_stats(pnl_path(sy_te, sx_te, py, px), TEST_START, TEST_END)
    return tr, te, int(path["cut"].sum())


def hybrid(prices_df, cfgs):
    from research.audit_robust_hybrid import configure, replay

    dfil.apply_pair_cfgs(cfgs)
    configure(discovery=True, extra=True, ridge=1.0, scale=1.0)
    return (replay(prices_df, TRAIN_START, TRAIN_END)["score"],
            replay(prices_df, TEST_START, TEST_END)["score"])


def main() -> None:
    prices = load_raw_prices()
    prices_df = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T

    filt = FiltCfg()            # library defaults, nothing fitted
    trade = TradeCfg()          # z 0.27 / 0.10 / 4.0
    tr, te, cuts = standalone(Y, X, filt, trade, prices)
    print(f"SRNA/ACIX standalone (UNTUNED defaults, z "
          f"{trade.z_entry}/{trade.z_exit}/{trade.z_stop}):")
    print(f"  train {tr['score']:+7.2f} (mu {tr['mean']:+6.2f})   "
          f"test {te['score']:+7.2f} (mu {te['mean']:+6.2f})   cuts {cuts}\n")

    new_pair = PairCfg(
        idx=len(BASE), y=Y, x=X, name="SRNA/ACIX",
        priority=max(float(tr["score"]), 0.01), filt=filt, trade=trade,
    )
    added = BASE + (new_pair,)
    swapped = tuple(s for s in BASE if s.name != "MDGI/HTRK") + (new_pair,)
    swapped = tuple(s._replace(idx=i) for i, s in enumerate(swapped))

    print("PORTFOLIO (daniel core):")
    rows = {}
    for label, cfgs in (("baseline", BASE), ("+ SRNA/ACIX", added),
                        ("swap MDGI/HTRK -> SRNA/ACIX", swapped)):
        a = portfolio_window(prices, cfgs, TRAIN_START, TRAIN_END)["score"]
        b = portfolio_window(prices, cfgs, TEST_START, TEST_END)["score"]
        rows[label] = (a, b)
        base = rows["baseline"]
        print(f"  {label:30s} train {a:8.2f} ({a-base[0]:+7.2f})  "
              f"test {b:8.2f} ({b-base[1]:+7.2f})")

    print("\nPORTFOLIO (robust_hybrid):")
    hrows = {}
    for label, cfgs in (("baseline", BASE), ("+ SRNA/ACIX", added),
                        ("swap MDGI/HTRK -> SRNA/ACIX", swapped)):
        a, b = hybrid(prices_df, cfgs)
        hrows[label] = (a, b)
        base = hrows["baseline"]
        print(f"  {label:30s} train {a:8.2f} ({a-base[0]:+7.2f})  "
              f"test {b:8.2f} ({b-base[1]:+7.2f})")

    dfil.apply_pair_cfgs(BASE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "srna_acix_test.json").write_text(
        json.dumps({"standalone": {"train": tr, "test": te, "cuts": cuts},
                    "core": rows, "hybrid": hrows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_DIR / 'srna_acix_test.json'}  (configs restored)")


if __name__ == "__main__":
    main()

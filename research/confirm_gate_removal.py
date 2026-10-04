"""Portfolio confirmation for dropping require_adf on the unblocked pairs.

Standalone gains do not add: the EV-MWIS conflict layer means unblocking a
pair changes which OTHER pairs win their instruments.  This replays the
whole book (daniel core first, robust_hybrid for finalists) for the full
candidate set and, if that fails, for leave-one-out subsets.
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
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    portfolio_window,
)

CANDIDATES = [
    "NAYO/SRTX", "DIHO/RTTH", "MHRM/EAFC", "MTNS/ALUT",
    "MSDP/ULXY", "ACIX/ULXY", "EAFC/SRTX",
]

BASE = tuple(dfil.PAIR_CFGS)


def cfgs_with(names: set[str]):
    return tuple(
        s._replace(trade=s.trade._replace(require_adf=False))
        if s.name in names
        else s
        for s in BASE
    )


def core(prices, cfgs) -> tuple[float, float]:
    tr = portfolio_window(prices, cfgs, TRAIN_START, TRAIN_END)
    te = portfolio_window(prices, cfgs, TEST_START, TEST_END)
    return tr["score"], te["score"]


def hybrid(prices_df, cfgs) -> tuple[float, float]:
    from research.audit_robust_hybrid import configure, replay

    dfil.apply_pair_cfgs(cfgs)
    configure(discovery=True, extra=True, ridge=1.0, scale=1.0)
    tr = replay(prices_df, TRAIN_START, TRAIN_END)
    te = replay(prices_df, TEST_START, TEST_END)
    return tr["score"], te["score"]


def main() -> None:
    prices = load_raw_prices()
    prices_df = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T

    b_tr, b_te = core(prices, BASE)
    print(f"baseline core        train {b_tr:8.2f}  test {b_te:8.2f}")

    results = {}
    all_tr, all_te = core(prices, cfgs_with(set(CANDIDATES)))
    results["ALL 7"] = (all_tr, all_te)
    print(f"all 7 no_adf  core   train {all_tr:8.2f} ({all_tr - b_tr:+6.2f})  "
          f"test {all_te:8.2f} ({all_te - b_te:+6.2f})")

    print("\nleave-one-out (core):")
    loo = {}
    for name in CANDIDATES:
        subset = set(CANDIDATES) - {name}
        tr, te = core(prices, cfgs_with(subset))
        loo[name] = (tr, te)
        print(f"  drop {name:>11}: train {tr:8.2f} ({tr - b_tr:+6.2f})  "
              f"test {te:8.2f} ({te - b_te:+6.2f})")

    print("\nsingles (core):")
    singles = {}
    for name in CANDIDATES:
        tr, te = core(prices, cfgs_with({name}))
        singles[name] = (tr, te)
        print(f"  only {name:>11}: train {tr:8.2f} ({tr - b_tr:+6.2f})  "
              f"test {te:8.2f} ({te - b_te:+6.2f})")

    # Greedy forward build from the singles that help both windows.
    helpful = [n for n, (tr, te) in singles.items() if tr >= b_tr and te >= b_te]
    print(f"\nsingles improving/holding both windows: {helpful}")
    chosen: set[str] = set()
    cur_tr, cur_te = b_tr, b_te
    pool = list(helpful)
    while pool:
        best = None
        for name in pool:
            tr, te = core(prices, cfgs_with(chosen | {name}))
            if tr >= cur_tr - 1e-9 and te >= cur_te - 1e-9:
                gain = min(tr - cur_tr, te - cur_te)
                if best is None or gain > best[0]:
                    best = (gain, name, tr, te)
        if best is None:
            break
        _, name, cur_tr, cur_te = best
        chosen.add(name)
        pool.remove(name)
        print(f"  + {name:>11} -> train {cur_tr:8.2f} test {cur_te:8.2f}")

    print(f"\ngreedy core set: {sorted(chosen)}")
    print(f"  core train {cur_tr:8.2f} ({cur_tr - b_tr:+6.2f})  "
          f"test {cur_te:8.2f} ({cur_te - b_te:+6.2f})")

    # Hybrid confirmation for baseline, all-7 and the greedy set.
    print("\nrobust_hybrid confirmation:")
    hb = hybrid(prices_df, BASE)
    print(f"  baseline    train {hb[0]:8.2f}  test {hb[1]:8.2f}")
    ha = hybrid(prices_df, cfgs_with(set(CANDIDATES)))
    print(f"  all 7       train {ha[0]:8.2f} ({ha[0] - hb[0]:+6.2f})  "
          f"test {ha[1]:8.2f} ({ha[1] - hb[1]:+6.2f})")
    hg = hybrid(prices_df, cfgs_with(chosen)) if chosen else hb
    print(f"  greedy set  train {hg[0]:8.2f} ({hg[0] - hb[0]:+6.2f})  "
          f"test {hg[1]:8.2f} ({hg[1] - hb[1]:+6.2f})")

    dfil.apply_pair_cfgs(BASE)
    payload = {
        "baseline_core": [b_tr, b_te], "all7_core": [all_tr, all_te],
        "loo_core": loo, "singles_core": singles,
        "greedy_set": sorted(chosen), "greedy_core": [cur_tr, cur_te],
        "hybrid_baseline": list(hb), "hybrid_all7": list(ha),
        "hybrid_greedy": list(hg),
    }
    (OUT_DIR / "gate_removal_portfolio.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'gate_removal_portfolio.json'}")


if __name__ == "__main__":
    main()

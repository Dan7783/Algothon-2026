"""Portfolio confirmation for a tuned single-pair seed (step 4 of the proof).

Usage:  python research/confirm_pair_seed.py AENO NWIG

Reads the candidate written by tune_single_pair.py, applies it in memory,
and replays BOTH books on BOTH windows:

    daniel_filter  (pair atlas + extra lead-lag + ridge)
    robust_hybrid  (the above + trailing-OLS discovery)

A standalone pair gain is not evidence: the EV-MWIS conflict layer means a
changed pair alters which OTHER pairs win their instruments, so the only
decisive test is the portfolio replay.  Prints a clear APPLY / REJECT.
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
import robust_hybrid as rh
from archive.old_tuners.daniel_filter import FiltCfg, TradeCfg
from research.pair_book_lab import load_raw_prices
from tune_daniel_pairs import (
    OUT_DIR,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    portfolio_window,
)


def hybrid_windows(prices_df: np.ndarray) -> dict:
    """Self-contained robust_hybrid replayed on both windows."""
    from research.audit_robust_hybrid import configure, replay

    # Prefer the inlined live module when the audit harness still points at v2.
    import research.audit_robust_hybrid as audit

    audit.strategy = rh
    configure(discovery=True, extra=True, ridge=1.0, scale=1.0)
    return {
        "train": replay(prices_df, TRAIN_START, TRAIN_END),
        "test": replay(prices_df, TEST_START, TEST_END),
    }


def _apply_both(cfgs) -> None:
    """Keep daniel_filter and the inlined robust_hybrid atlas in sync."""
    dfil.apply_pair_cfgs(cfgs)
    rh.apply_pair_cfgs(cfgs)


def main() -> None:
    y_name, x_name = (_sys.argv[1].upper(), _sys.argv[2].upper()) if len(_sys.argv) >= 3 else ("AENO", "NWIG")
    tag = f"{y_name}_{x_name}".lower()
    cand_path = OUT_DIR / f"single_{tag}.json"
    if not cand_path.exists():
        raise SystemExit(f"no candidate at {cand_path} — run tune_single_pair first")
    cand = json.loads(cand_path.read_text(encoding="utf-8"))

    prices = load_raw_prices()
    prices_df = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    baseline_cfgs = tuple(dfil.PAIR_CFGS)
    y_i, x_i = cand["pair"]

    tuned_cfgs = tuple(
        s._replace(
            filt=FiltCfg(**cand["chosen"]["filt"]),
            trade=TradeCfg(**cand["chosen"]["trade"]),
        )
        if (s.y, s.x) == (y_i, x_i)
        else s
        for s in baseline_cfgs
    )

    print(f"portfolio confirmation for {cand['name']}\n")
    rows = {}
    for label, cfgs in (("baseline", baseline_cfgs), ("tuned", tuned_cfgs)):
        _apply_both(cfgs)
        core_tr = portfolio_window(prices, cfgs, TRAIN_START, TRAIN_END)
        core_te = portfolio_window(prices, cfgs, TEST_START, TEST_END)
        hyb = hybrid_windows(prices_df)
        rows[label] = {
            "core_train": core_tr["score"], "core_test": core_te["score"],
            "hyb_train": hyb["train"]["score"], "hyb_test": hyb["test"]["score"],
        }
        r = rows[label]
        print(
            f"  {label:8s} daniel_core train {r['core_train']:8.2f} test {r['core_test']:8.2f}"
            f"   robust_hybrid train {r['hyb_train']:8.2f} test {r['hyb_test']:8.2f}"
        )

    _apply_both(baseline_cfgs)  # leave both modules untouched

    b, t = rows["baseline"], rows["tuned"]
    deltas = {k: t[k] - b[k] for k in b}
    print("\n  deltas: " + "  ".join(f"{k} {v:+.2f}" for k, v in deltas.items()))
    ok = all(v >= -1e-9 for v in deltas.values()) and any(v > 0 for v in deltas.values())
    print("\n  VERDICT:", "APPLY — no book/window degrades" if ok else "REJECT — a book/window degrades")

    cand["portfolio"] = {"baseline": b, "tuned": t, "deltas": deltas, "apply": ok}
    cand_path.write_text(json.dumps(cand, indent=2), encoding="utf-8")
    print(f"  updated {cand_path}")


if __name__ == "__main__":
    main()

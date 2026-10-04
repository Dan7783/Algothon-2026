"""Does the ADF gate cost the train window trades that were worth taking?

The participation audit shows several pairs sitting out most of the TRAIN
window purely because the ADF prior-gate blocks them (DIHO/RTTH 219/250
blocked, MTNS/ALUT 223, EAFC/SRTX 187, MDGI/HTRK 124, ALUT/CCNS 122), and two
pairs (MSDP/ULXY, ACIX/ULXY) that are blocked 100% of BOTH windows and so
contribute exactly nothing.

For every pair this replays the seed with require_adf turned OFF (and,
separately, require_kappa OFF) and reports the standalone effect on each
window.  Nothing is applied here — candidates that improve both windows go
to the portfolio confirmation, which is the only decisive test.
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


def run(gates, py, px, trade) -> dict:
    sy_tr, sx_tr = trade_sim(gates, py, px, trade, TRAIN_DECIDE_START)
    sy_te, sx_te = trade_sim(gates, py, px, trade, TEST_DECIDE_START)
    tr = window_stats(pnl_path(sy_tr, sx_tr, py, px), TRAIN_START, TRAIN_END)
    te = window_stats(pnl_path(sy_te, sx_te, py, px), TEST_START, TEST_END)
    return {
        "train": tr["score"], "test": te["score"],
        "train_mean": tr["mean"], "test_mean": te["mean"],
        "days_train": int((np.abs(sy_tr[TRAIN_START - 1:TRAIN_END]) > 0).sum()),
        "days_test": int((np.abs(sy_te[TEST_START - 1:TEST_END]) > 0).sum()),
    }


def main() -> None:
    prices = load_raw_prices()
    out = []
    print(f"{'pair':>11} | {'variant':>10} | {'train':>7} {'test':>7} | "
          f"{'d.train':>7} {'d.test':>6} | verdict")
    print("-" * 78)
    for spec in dfil.PAIR_CFGS:
        ly, lx = np.log(prices[spec.y]), np.log(prices[spec.x])
        py, px = prices[spec.y], prices[spec.x]
        path = record_paths(ly, lx, spec.filt)
        gates = gate_and_z_paths(ly, lx, path["beta"])

        base = run(gates, py, px, spec.trade)
        variants = {"base": base}
        if spec.trade.require_adf:
            variants["no_adf"] = run(
                gates, py, px, spec.trade._replace(require_adf=False)
            )
        if spec.trade.require_kappa:
            variants["no_kappa"] = run(
                gates, py, px, spec.trade._replace(require_kappa=False)
            )
        if spec.trade.require_adf and spec.trade.require_kappa:
            variants["no_gates"] = run(
                gates, py, px,
                spec.trade._replace(require_adf=False, require_kappa=False),
            )

        for name, v in variants.items():
            if name == "base":
                verdict = ""
            elif v["train"] > base["train"] and v["test"] > base["test"]:
                verdict = "BOTH BETTER"
            elif v["train"] >= base["train"] and v["test"] >= base["test"]:
                verdict = "no worse"
            elif v["train"] > base["train"]:
                verdict = "train only"
            elif v["test"] > base["test"]:
                verdict = "test only"
            else:
                verdict = "worse"
            print(
                f"{spec.name:>11} | {name:>10} | {v['train']:7.2f} {v['test']:7.2f} | "
                f"{v['days_train']:7d} {v['days_test']:6d} | {verdict}"
            )
        out.append({"name": spec.name, "pair": [spec.y, spec.x],
                    "variants": variants})
        print("-" * 78)

    winners = [
        (r["name"], k, v)
        for r in out for k, v in r["variants"].items()
        if k != "base"
        and v["train"] > r["variants"]["base"]["train"]
        and v["test"] > r["variants"]["base"]["test"]
    ]
    print(f"\n{len(winners)} pair/variant combos improve BOTH windows standalone:")
    for name, k, v in winners:
        print(f"  {name:>11} {k:>9}: train {v['train']:7.2f} test {v['test']:7.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gate_toggle.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'gate_toggle.json'}")


if __name__ == "__main__":
    main()

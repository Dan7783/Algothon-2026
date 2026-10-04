#!/usr/bin/env python
"""Attribution flip/suppression for OU conflict modes (pinned env)."""
from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COMM_RATE = 0.0001
OUT = ROOT / "auto_tuning_results" / "_ou_conflict_attribution.json"

MERGED_KEYS = [
    "MERGED_CONSENSUS_MARGIN",
    "MERGED_PROTECTED_BOOST",
    "MERGED_RIDGE_WEIGHT",
    "MERGED_FLIP_TO_FLAT",
    "MERGED_MAG_WEIGHTED",
    "MERGED_EXCLUDE_COHORTS",
    "MERGED_EXCLUDE_PAIRS",
    "MERGED_CATALOG_JSON",
    "MERGED_OU_CONFLICT",
    "MERGED_OU_SLEEVE_FRAC",
    "MERGED_OU_SLEEVE_SPLIT",
]


def pin_env(extra: dict[str, str] | None = None) -> None:
    for k in MERGED_KEYS:
        os.environ.pop(k, None)
    if extra:
        os.environ.update(extra)


def load_prices():
    import latest_tuner

    prices, _nt, _ninst = latest_tuner.loadPrices("prices.txt")
    return np.asarray(prices, dtype=float)


def run_one(module_name: str, prices, start: int, end: int, oos_start: int):
    # Force re-exec so env-driven constants are picked up.
    if module_name in sys.modules:
        del sys.modules[module_name]
    mod = importlib.import_module(module_name)
    mod.reset()

    symbol_by_index = {i: s for s, i in mod.I.items()}
    names = [f"{symbol_by_index[s.y]}-{symbol_by_index[s.x]}" for s in mod.PAIR_CFGS]
    engines = [getattr(s, "engine", "zband") for s in mod.PAIR_CFGS]
    n = len(names)

    desired_shares = np.zeros((n, 2))
    pair_pnl = np.zeros(n)
    pair_pnl_oos = np.zeros(n)
    flips = np.zeros(n, dtype=int)
    suppressions = np.zeros(n, dtype=int)
    days = 0

    prev_pos = np.zeros(mod.N_INST)
    book_pnl = 0.0
    book_pnl_oos = 0.0

    for t in range(start, end - 1):
        prc = prices[:, : t + 1]
        if mod._STATE["atlas"] is None:
            mod._STATE["atlas"] = mod.AtlasBook()
        atlas = mod._STATE["atlas"]
        cands = atlas.candidates(prc)

        desired_dollars = np.zeros((n, 2))
        for _prio, agent, pos_y, pos_x in cands:
            desired_dollars[agent.spec.idx, 0] = pos_y
            desired_dollars[agent.spec.idx, 1] = pos_x

        px = prices[:, t]
        safe = np.maximum(px, 1e-12)
        for i, spec in enumerate(mod.PAIR_CFGS):
            desired_shares[i, 0] = desired_dollars[i, 0] / safe[spec.y]
            desired_shares[i, 1] = desired_dollars[i, 1] / safe[spec.x]

        shares = mod.getMyPosition(prc).astype(float)
        turnover = np.abs(shares - prev_pos)
        dpx = prices[:, t + 1] - prices[:, t]
        day_pnl = float(np.sum(shares * dpx) - COMM_RATE * np.sum(turnover * px))
        book_pnl += day_pnl
        if t >= oos_start:
            book_pnl_oos += day_pnl

        for i, spec in enumerate(mod.PAIR_CFGS):
            dy = desired_shares[i, 0] * dpx[spec.y]
            dx = desired_shares[i, 1] * dpx[spec.x]
            d_pnl = dy + dx
            pair_pnl[i] += d_pnl
            if t >= oos_start:
                pair_pnl_oos[i] += d_pnl

            for leg, idx in ((0, spec.y), (1, spec.x)):
                des = desired_dollars[i, leg]
                if abs(des) < 1e-9:
                    continue
                realized_dir = np.sign(shares[idx])
                des_dir = np.sign(des)
                if realized_dir == 0.0:
                    suppressions[i] += 1
                elif realized_dir != des_dir:
                    flips[i] += 1

        prev_pos = shares
        days += 1

    records = []
    for i in range(n):
        records.append(
            {
                "pair": names[i],
                "engine": engines[i],
                "pnl": float(pair_pnl[i]),
                "pnl_oos": float(pair_pnl_oos[i]),
                "flips": int(flips[i]),
                "suppressions": int(suppressions[i]),
            }
        )

    desired_sum = float(pair_pnl.sum())
    desired_sum_oos = float(pair_pnl_oos.sum())
    return {
        "module": module_name,
        "ou_conflict": getattr(mod, "OU_CONFLICT_MODE", "n/a"),
        "days": days,
        "book_pnl_full": book_pnl,
        "book_pnl_oos": book_pnl_oos,
        "desired_sum_full": desired_sum,
        "desired_sum_oos": desired_sum_oos,
        "flips_total": int(flips.sum()),
        "suppressions_total": int(suppressions.sum()),
        "flips_ou": int(sum(flips[i] for i in range(n) if engines[i] == "ou")),
        "flips_zband": int(sum(flips[i] for i in range(n) if engines[i] != "ou")),
        "records": records,
    }


def main() -> None:
    prices = load_prices()
    start, end, oos_start = 200, 1000, 801
    cases = [
        ("friend", "friend_book_latest", {}),
        ("shared", "merged_friend_ou", {"MERGED_OU_CONFLICT": "shared"}),
        ("sleeve", "merged_friend_ou", {"MERGED_OU_CONFLICT": "sleeve"}),
        (
            "sleeve_demand",
            "merged_friend_ou",
            {"MERGED_OU_CONFLICT": "sleeve", "MERGED_OU_SLEEVE_SPLIT": "demand"},
        ),
        ("ou_flat", "merged_friend_ou", {"MERGED_OU_CONFLICT": "ou_flat"}),
        ("twopass", "merged_friend_ou", {"MERGED_OU_CONFLICT": "twopass"}),
    ]
    out = {}
    for label, module, extra in cases:
        print(f"ATTR {label} ...", flush=True)
        pin_env(extra)
        # Drop cached module so constants rebind.
        if module in sys.modules:
            del sys.modules[module]
        r = run_one(module, prices, start, end, oos_start)
        out[label] = r
        print(
            f"  desired_oos={r['desired_sum_oos']:.0f} "
            f"book_oos={r['book_pnl_oos']:.0f} "
            f"flips={r['flips_total']} "
            f"(ou={r['flips_ou']} zb={r['flips_zband']}) "
            f"supp={r['suppressions_total']}",
            flush=True,
        )

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Attribution for friend_book_latest vs merged_friend_ou (pinned env).

Mirrors _pair_attribution.py but targets the new friend+OU modules.
"""
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
OUT = ROOT / "auto_tuning_results" / "_friend_ou_attribution.json"

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
    mod = importlib.import_module(module_name)
    importlib.reload(mod)
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
        # Desired pre-conflict from each agent
        if mod._STATE["atlas"] is None:
            mod._STATE["atlas"] = mod.AtlasBook()
        atlas = mod._STATE["atlas"]
        cands = atlas.candidates(prc)

        # Map agent idx -> desired dollars
        desired_dollars = np.zeros((n, 2))
        for _prio, agent, pos_y, pos_x in cands:
            desired_dollars[agent.spec.idx, 0] = pos_y
            desired_dollars[agent.spec.idx, 1] = pos_x

        px = prices[:, t]
        safe = np.maximum(px, 1e-12)
        for i, spec in enumerate(mod.PAIR_CFGS):
            desired_shares[i, 0] = desired_dollars[i, 0] / safe[spec.y]
            desired_shares[i, 1] = desired_dollars[i, 1] / safe[spec.x]

        # Realized book
        shares = mod.getMyPosition(prc).astype(float)
        # Commission on turnover
        turnover = np.abs(shares - prev_pos)
        dpx = prices[:, t + 1] - prices[:, t]
        day_pnl = float(np.sum(shares * dpx) - COMM_RATE * np.sum(turnover * px))
        book_pnl += day_pnl
        if t >= oos_start:
            book_pnl_oos += day_pnl

        # Per-pair desired PnL (no conflict)
        for i, spec in enumerate(mod.PAIR_CFGS):
            sy, sx = desired_shares[i]
            y_i, x_i = spec.y, spec.x
            # Approximate commission on desired share change omitted for desired;
            # mark-to-market only (matches prior attribution's desired path loosely)
            dp = sy * dpx[y_i] + sx * dpx[x_i]
            pair_pnl[i] += dp
            if t >= oos_start:
                pair_pnl_oos[i] += dp

            # Flip / suppression vs realized legs
            for leg_i, des in ((y_i, sy), (x_i, sx)):
                real = shares[leg_i]
                if abs(des) > 1e-9:
                    if abs(real) <= 1e-9:
                        suppressions[i] += 1
                    elif np.sign(real) != np.sign(des):
                        flips[i] += 1

        prev_pos = shares
        days += 1

    records = []
    for i, name in enumerate(names):
        records.append(
            {
                "pair": name,
                "engine": engines[i],
                "pnl": float(pair_pnl[i]),
                "pnl_oos": float(pair_pnl_oos[i]),
                "flips": int(flips[i]),
                "suppressions": int(suppressions[i]),
            }
        )

    desired_oos = float(np.sum(pair_pnl_oos))
    return {
        "module": module_name,
        "days": days,
        "book_pnl": book_pnl,
        "book_pnl_oos": book_pnl_oos,
        "desired_pnl": float(np.sum(pair_pnl)),
        "desired_pnl_oos": desired_oos,
        "realized_gap_oos": desired_oos - book_pnl_oos,
        "total_flips": int(flips.sum()),
        "total_suppressions": int(suppressions.sum()),
        "records": records,
    }


def main():
    prices = load_prices()
    variants = [
        ("friend", "friend_book_latest", {}),
        ("friend_ou", "merged_friend_ou", {}),
        (
            "friend_ou_m030",
            "merged_friend_ou",
            {"MERGED_CONSENSUS_MARGIN": "0.30"},
        ),
        (
            "friend_ou_flipflat",
            "merged_friend_ou",
            {"MERGED_FLIP_TO_FLAT": "1"},
        ),
        (
            "friend_ou_flipflat_m030",
            "merged_friend_ou",
            {"MERGED_FLIP_TO_FLAT": "1", "MERGED_CONSENSUS_MARGIN": "0.30"},
        ),
    ]
    out = {}
    for label, module, extra in variants:
        print(f"ATTR {label} ...", flush=True)
        pin_env(extra)
        # Force re-import with new env
        if module in sys.modules:
            del sys.modules[module]
        out[label] = run_one(module, prices, 200, 1000, 801)
        s = out[label]
        print(
            f"  desired_oos={s['desired_pnl_oos']:.0f} book_oos={s['book_pnl_oos']:.0f} "
            f"gap={s['realized_gap_oos']:.0f} flips={s['total_flips']} "
            f"supp={s['total_suppressions']}",
            flush=True,
        )

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

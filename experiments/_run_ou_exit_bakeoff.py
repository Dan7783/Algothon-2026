#!/usr/bin/env python
"""Pinned-env bake-off: optimal OU exit vs conventional mean-exit.

Windows (friend convention):
  primary OOS 750-1000; also 31-750, 501-750; optional 801-1000.

ALWAYS clears MERGED_* first to avoid PowerShell session leaks.
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
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "auto_tuning_results" / "_ou_exit_bakeoff.json"
ATTR_OUT = ROOT / "auto_tuning_results" / "_ou_exit_attribution.json"

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
    "MERGED_OU_SLEEVE_OPPOSE",
    "MERGED_OU_EXIT",
]

COMM_RATE = 0.0001


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for k in MERGED_KEYS:
        env.pop(k, None)
    if extra:
        env.update(extra)
    return env


def run_eval(module: str, start: str, end: str, env: dict[str, str]) -> dict:
    p = subprocess.run(
        [sys.executable, "eval.py", "--module", module, "--start", start, "--end", end],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    out = p.stdout + "\n" + p.stderr
    score = re.search(r"Score:\s*([-\d.]+)", out)
    sharpe = re.search(r"annSharpe\(PL\):\s*([-\d.]+)", out)
    return {
        "module": module,
        "start": int(start),
        "end": int(end),
        "score": float(score.group(1)) if score else None,
        "sharpe": float(sharpe.group(1)) if sharpe else None,
        "returncode": p.returncode,
        "tail": out[-600:],
    }


def pin_env(extra: dict[str, str] | None = None) -> None:
    for k in MERGED_KEYS:
        os.environ.pop(k, None)
    if extra:
        os.environ.update(extra)


def load_prices():
    import latest_tuner

    prices, _nt, _ninst = latest_tuner.loadPrices("prices.txt")
    return np.asarray(prices, dtype=float)


def run_attribution(module_name: str, prices, start: int, end: int, oos_start: int):
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
    side_changes = 0
    days_in_pos_ou = 0
    days_flat_ou = 0
    book_turnover = 0.0
    book_comm = 0.0

    prev_pos = np.zeros(mod.N_INST)
    prev_sides = np.zeros(n, dtype=int)
    book_pnl = 0.0
    book_pnl_oos = 0.0

    for t in range(start, end - 1):
        in_oos = t >= oos_start
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
        day_turn = float(np.sum(turnover * px))
        day_comm = COMM_RATE * day_turn
        dpx = prices[:, t + 1] - prices[:, t]
        day_pnl = float(np.sum(shares * dpx) - day_comm)
        book_pnl += day_pnl
        if in_oos:
            book_pnl_oos += day_pnl
            book_turnover += day_turn
            book_comm += day_comm

        for i, spec in enumerate(mod.PAIR_CFGS):
            sy, sx = desired_shares[i]
            y_i, x_i = spec.y, spec.x
            dp = sy * dpx[y_i] + sx * dpx[x_i]
            pair_pnl[i] += dp
            if in_oos:
                pair_pnl_oos[i] += dp
                for leg_i, des in ((y_i, sy), (x_i, sx)):
                    real = shares[leg_i]
                    if abs(des) > 1e-9:
                        if abs(real) <= 1e-9:
                            suppressions[i] += 1
                        elif np.sign(real) != np.sign(des):
                            flips[i] += 1

        for i, agent in enumerate(atlas.agents):
            side = int(agent.side)
            if in_oos:
                if side != prev_sides[i]:
                    side_changes += 1
                if engines[i] == "ou":
                    if side == 0:
                        days_flat_ou += 1
                    else:
                        days_in_pos_ou += 1
            prev_sides[i] = side

        prev_pos = shares

    desired_oos = float(np.sum(pair_pnl_oos))
    flips_ou = int(sum(flips[i] for i in range(n) if engines[i] == "ou"))
    flips_zb = int(sum(flips[i] for i in range(n) if engines[i] != "ou"))
    return {
        "module": module_name,
        "ou_exit": os.environ.get("MERGED_OU_EXIT", "optimal"),
        "oos_start": oos_start,
        "book_pnl": book_pnl,
        "book_pnl_oos": book_pnl_oos,
        "desired_pnl": float(np.sum(pair_pnl)),
        "desired_pnl_oos": desired_oos,
        "realized_gap_oos": desired_oos - book_pnl_oos,
        "total_flips": int(flips.sum()),
        "flips_ou": flips_ou,
        "flips_zband": flips_zb,
        "total_suppressions": int(suppressions.sum()),
        "side_changes": side_changes,
        "ou_days_in_pos": days_in_pos_ou,
        "ou_days_flat": days_flat_ou,
        "book_turnover_dollars": book_turnover,
        "book_commission": book_comm,
        "n_ou": sum(1 for e in engines if e == "ou"),
        "n_zband": sum(1 for e in engines if e != "ou"),
    }


def main() -> None:
    cases = [
        ("friend_baseline", "friend_book_latest", {}),
        (
            "optimal_m015",
            "merged_friend_ou",
            {"MERGED_CONSENSUS_MARGIN": "0.15", "MERGED_OU_EXIT": "optimal"},
        ),
        (
            "mean_m015",
            "merged_friend_ou",
            {"MERGED_CONSENSUS_MARGIN": "0.15", "MERGED_OU_EXIT": "mean"},
        ),
        (
            "mean_m015_ouflat",
            "merged_friend_ou",
            {
                "MERGED_CONSENSUS_MARGIN": "0.15",
                "MERGED_OU_EXIT": "mean",
                "MERGED_OU_CONFLICT": "ou_flat",
            },
        ),
        (
            "mean_m015_sleeve",
            "merged_friend_ou",
            {
                "MERGED_CONSENSUS_MARGIN": "0.15",
                "MERGED_OU_EXIT": "mean",
                "MERGED_OU_CONFLICT": "sleeve",
            },
        ),
    ]
    windows = (
        ("oos750", "750", "1000"),
        ("train31", "31", "750"),
        ("val501", "501", "750"),
        ("oos801", "801", "1000"),
    )

    jobs = []
    for label, module, extra in cases:
        for wname, start, end in windows:
            jobs.append((label, module, extra, wname, start, end))

    results = []
    print(f"Launching {len(jobs)} evals ...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {
            ex.submit(run_eval, module, start, end, clean_env(extra)): (
                label,
                wname,
                extra,
            )
            for label, module, extra, wname, start, end in jobs
        }
        for fut in as_completed(futs):
            label, wname, extra = futs[fut]
            r = fut.result()
            r["label"] = label
            r["window"] = wname
            r["env"] = extra
            results.append(r)
            print(
                f"  {label:22} {wname:8} score={r['score']} sharpe={r['sharpe']} rc={r['returncode']}",
                flush=True,
            )

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)

    # Attribution on primary OOS window (750-1000), start from 31 for warm state.
    print("Attribution optimal vs mean on 750-1000 ...", flush=True)
    prices = load_prices()
    attr = {}
    for label, extra in (
        ("optimal_m015", {"MERGED_CONSENSUS_MARGIN": "0.15", "MERGED_OU_EXIT": "optimal"}),
        ("mean_m015", {"MERGED_CONSENSUS_MARGIN": "0.15", "MERGED_OU_EXIT": "mean"}),
        ("friend", {}),
    ):
        pin_env(extra)
        module = "friend_book_latest" if label == "friend" else "merged_friend_ou"
        print(f"  ATTR {label} ...", flush=True)
        # Warm from day 31, attribute/judge from 750.
        attr[label] = run_attribution(module, prices, start=31, end=1000, oos_start=750)
        a = attr[label]
        print(
            f"    flips={a['total_flips']} (ou={a['flips_ou']}) "
            f"supp={a['total_suppressions']} side_chg={a['side_changes']} "
            f"ou_inpos={a['ou_days_in_pos']} turn={a['book_turnover_dollars']:.0f} "
            f"gap_oos={a['realized_gap_oos']:.0f} book_oos={a['book_pnl_oos']:.0f}",
            flush=True,
        )

    ATTR_OUT.write_text(json.dumps(attr, indent=2), encoding="utf-8")
    print(f"wrote {ATTR_OUT}", flush=True)


if __name__ == "__main__":
    main()

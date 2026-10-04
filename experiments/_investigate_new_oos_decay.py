#!/usr/bin/env python
"""Per-pair desired-PnL attribution for four_sigmas.py on the newly extended
prices.txt (now 1500 days, was 1000). Splits into:
  - "old" window: days used for tuning/selection (31-1000)
  - "new_oos" window: genuinely unseen days (1001-1500)

Desired PnL is conflict-free (each pair's own pos_y/pos_x from
sleeve_breakdown, marked to market with eval.py's share-rounding + 1bp
commission), so it isolates pair/engine decay from conflict-layer effects.

Usage: python _investigate_new_oos_decay.py
"""
from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import json

import numpy as np
import pandas as pd

import four_sigmas as fs

COMM_RATE = 0.0001
OLD_START, OLD_END = 31, 1000
NEW_START, NEW_END = 1001, 1500


def load_prices(fn="prices.txt"):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T.astype(float)


def run(prices, start, end):
    """Replay days [start, end]; PnL leg lands on day t using prcHist[:, :t]."""
    fs.reset()
    names = [fs._pair_name(spec) for spec in fs.PAIR_CFGS]
    n = len(names)
    desired_shares = np.zeros((n, 2))
    pair_pnl = np.zeros(n)
    active_days = np.zeros(n, dtype=int)
    pair_index = {name: i for i, name in enumerate(names)}

    algo_names = [fs._pair_name(spec) for spec in fs.ALGO_RESIDUAL_CFGS]
    n_algo = len(algo_names)
    algo_pnl = np.zeros(n_algo)
    algo_shares = np.zeros((n_algo, 1))
    algo_index = {name: i for i, name in enumerate(algo_names)}
    scale = fs.ALGO_RESIDUAL_SLOT / fs.LEG

    for t in range(start - 1, end):
        prc = prices[:, :t]
        px = prices[:, t - 1]
        nxt = prices[:, t]
        sleeves = fs.sleeve_breakdown(prc)
        if sleeves is None:
            continue

        for detail in sleeves["pair_contributor_details"]:
            name = detail["pair"]
            if name not in pair_index:
                continue
            i = pair_index[name]
            spec = fs.PAIR_CFGS[i]
            pos_y, pos_x = detail["pos_y"], detail["pos_x"]
            new_y = np.round(pos_y / px[spec.y])
            new_x = np.round(pos_x / px[spec.x])
            comm = COMM_RATE * (
                abs(new_y - desired_shares[i, 0]) * px[spec.y]
                + abs(new_x - desired_shares[i, 1]) * px[spec.x]
            )
            pnl_day = (
                new_y * (nxt[spec.y] - px[spec.y])
                + new_x * (nxt[spec.x] - px[spec.x])
                - comm
            )
            pair_pnl[i] += pnl_day
            desired_shares[i] = (new_y, new_x)
            if abs(pos_y) > 1e-9:
                active_days[i] += 1

        # ALGO residual basket (standalone desired, pre-hedge)
        for agent in fs._STATE["atlas"].algo_residual_agents:
            name = fs._pair_name(agent.spec)
            if name not in algo_index:
                continue
            j = algo_index[name]
            _, pos_x, _ = agent.dollar_positions()
            pos_x_scaled = scale * pos_x
            spec = agent.spec
            new_x = np.round(pos_x_scaled / px[spec.x])
            comm = COMM_RATE * abs(new_x - algo_shares[j, 0]) * px[spec.x]
            pnl_day = new_x * (nxt[spec.x] - px[spec.x]) - comm
            algo_pnl[j] += pnl_day
            algo_shares[j] = (new_x,)

    records = [
        {"pair": names[i], "pnl": round(float(pair_pnl[i]), 2), "active_days": int(active_days[i])}
        for i in range(n)
    ]
    algo_records = [
        {"pair": "ALGO-" + algo_names[i].split("-")[1], "pnl": round(float(algo_pnl[i]), 2)}
        for i in range(n_algo)
    ]
    return records, algo_records


def main():
    prices = load_prices()
    print(f"Loaded prices shape {prices.shape}")

    old_records, old_algo = run(prices, OLD_START, OLD_END)
    new_records, new_algo = run(prices, NEW_START, NEW_END)

    old_by_name = {r["pair"]: r for r in old_records}
    new_by_name = {r["pair"]: r for r in new_records}

    rows = []
    for name in old_by_name:
        o = old_by_name[name]
        n = new_by_name.get(name, {"pnl": 0.0, "active_days": 0})
        # normalize by number of days in each window for a fair per-day comparison
        o_per_day = o["pnl"] / (OLD_END - OLD_START + 1)
        n_per_day = n["pnl"] / (NEW_END - NEW_START + 1)
        rows.append({
            "pair": name,
            "old_pnl": o["pnl"],
            "old_per_day": o_per_day,
            "new_pnl": n["pnl"],
            "new_per_day": n_per_day,
            "delta_per_day": n_per_day - o_per_day,
            "old_active_days": o["active_days"],
            "new_active_days": n["active_days"],
        })

    df = pd.DataFrame(rows).sort_values("delta_per_day")

    print(f"\n=== Book totals (desired, conflict-free) ===")
    print(f"OLD  {OLD_START}-{OLD_END} ({OLD_END-OLD_START+1}d): sum pnl = {df['old_pnl'].sum():,.0f}  "
          f"per-day = {df['old_pnl'].sum()/(OLD_END-OLD_START+1):,.1f}")
    print(f"NEW  {NEW_START}-{NEW_END} ({NEW_END-NEW_START+1}d): sum pnl = {df['new_pnl'].sum():,.0f}  "
          f"per-day = {df['new_pnl'].sum()/(NEW_END-NEW_START+1):,.1f}")

    n_worse = (df["delta_per_day"] < 0).sum()
    n_better = (df["delta_per_day"] > 0).sum()
    n_flipped_sign = ((df["old_pnl"] > 0) & (df["new_pnl"] < 0)).sum()
    print(f"\nPairs worse per-day in new window: {n_worse}/{len(df)}")
    print(f"Pairs better per-day in new window: {n_better}/{len(df)}")
    print(f"Pairs flipped from net-positive (old) to net-negative (new): {n_flipped_sign}")

    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 100)
    print("\n=== WORST 20 pairs by per-day PnL delta (old -> new) ===")
    print(df.head(20).to_string(index=False))

    print("\n=== BEST 15 pairs by per-day PnL delta (old -> new) ===")
    print(df.tail(15).sort_values("delta_per_day", ascending=False).to_string(index=False))

    print("\n=== ALGO residual basket (standalone desired, pre-hedge) ===")
    old_algo_by = {r["pair"]: r["pnl"] for r in old_algo}
    for r in new_algo:
        name = r["pair"]
        o = old_algo_by.get(name, 0.0)
        o_pd = o / (OLD_END - OLD_START + 1)
        n_pd = r["pnl"] / (NEW_END - NEW_START + 1)
        print(f"{name:14} old/day={o_pd:8.2f}  new/day={n_pd:8.2f}  delta/day={n_pd-o_pd:8.2f}")

    out = {
        "old_window": [OLD_START, OLD_END],
        "new_window": [NEW_START, NEW_END],
        "records": rows,
        "algo_residual_old": old_algo,
        "algo_residual_new": new_algo,
    }
    with open("auto_tuning_results/_new_oos_decay.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved -> auto_tuning_results/_new_oos_decay.json")


if __name__ == "__main__":
    main()

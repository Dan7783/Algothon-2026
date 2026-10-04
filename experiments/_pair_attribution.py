#!/usr/bin/env python
"""Per-pair attribution for merged_strategy book variants.

For each catalog variant, replays the daily portfolio loop and records, per
pair: the pair's DESIRED dollar positions (pre-conflict) marked-to-market with
eval.py's accounting (shares, 1bp commissions), plus how often the final book
position on each leg was suppressed (zeroed) or flipped (opposite sign) by the
conflict/ridge voting layer.

Variants:
  tuned  - all 94 pairs on their tuned OU profiles (regenerated catalog)
  hybrid - current merged_strategy catalog (76 zband catalog + 18 ou tuned)

Usage: python _pair_attribution.py [--start 200] [--end 1000] [--oos_start 801]
"""

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import argparse
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout

import numpy as np

import latest_tuner
import merged_strategy as ms

COMM_RATE = 0.0001
N_INST = ms.N_INST
SYMBOL_BY_INDEX = {index: symbol for symbol, index in ms.I.items()}


def load_variant_rows(variant):
    if variant == "hybrid":
        return list(ms._MERGED_PAIR_PARAMS)
    # "tuned": regenerate the pre-hybrid catalog (no --hybrid) without
    # touching merged_strategy.py on disk.
    buf = io.StringIO()
    with redirect_stdout(buf):
        sys.argv = ["_build_merged_catalog.py"]
        import _build_merged_catalog
        _build_merged_catalog.main()
    literal = buf.getvalue()
    start = literal.index("_MERGED_PAIR_PARAMS = (")
    ns = {}
    exec(literal[start:], ns)
    return list(ns["_MERGED_PAIR_PARAMS"])


def install_variant(rows):
    ms._ACTIVE_PAIR_PARAMS = tuple(rows)
    ms.PAIR_CFGS = ms._build_selected_pair_cfgs()
    cohort = {}
    for row in rows:
        name = f"{row[1]}-{row[2]}"
        if name in ms.REJECT_COHORT_NAMES:
            cohort[name] = "reject_cohort"
        elif name in ms.VERIFIED_NEW_NAMES:
            cohort[name] = "verified_new"
        elif name in ms.TUNED_CURRENT_NAMES:
            cohort[name] = "tuned_current"
        elif name in ms.PROTECTED_PAIR_NAMES:
            cohort[name] = "protected"
        else:
            cohort[name] = "catalog"
    ms._COHORT_BY_NAME = cohort
    ms.reset()


def run_attribution(prices, start, end, oos_start):
    """Replay days [start, end); position at day t earns price[t] -> price[t+1]."""
    names = [f"{SYMBOL_BY_INDEX[s.y]}-{SYMBOL_BY_INDEX[s.x]}" for s in ms.PAIR_CFGS]
    engines = [s.engine for s in ms.PAIR_CFGS]
    cohorts = [ms._COHORT_BY_NAME.get(n, "?") for n in names]
    n = len(names)

    desired_shares = np.zeros((n, 2))  # per pair: shares in y leg, x leg
    pair_pnl = np.zeros(n)
    pair_pnl_oos = np.zeros(n)
    pair_comm = np.zeros(n)
    active_days = np.zeros(n, dtype=int)
    honored = np.zeros(n, dtype=int)
    suppressed = np.zeros(n, dtype=int)
    flipped = np.zeros(n, dtype=int)

    book_shares = np.zeros(N_INST)
    book_pnl = 0.0
    book_pnl_oos = 0.0

    pair_index = {name: i for i, name in enumerate(names)}

    # eval.py accounting: decision at iteration t uses history prices[:, :t]
    # (last close day t-1), trades at day t-1 prices, earns prices[t]-prices[t-1].
    # Iterations t in [start-1, end); PnL legs land on days [start, end].
    for t in range(start - 1, end):
        prc = prices[:, :t]
        px = prices[:, t - 1]
        nxt = prices[:, t]
        sleeves = ms.sleeve_breakdown(prc)
        if sleeves is None:
            continue
        oos = t >= oos_start - 1

        # --- book actual ---
        new_book = ms.getMyPosition(prc).astype(float)
        delta = new_book - book_shares
        book_comm = COMM_RATE * float(np.sum(np.abs(delta) * px))
        book_pnl_day = float(np.dot(new_book, nxt - px)) - book_comm
        book_pnl += book_pnl_day
        if oos:
            book_pnl_oos += book_pnl_day
        book_shares = new_book

        # --- per pair desired ---
        book_dollars = sleeves["dollars"]
        for detail in sleeves["pair_contributor_details"]:
            name = detail["pair"]
            if name not in pair_index:
                continue
            i = pair_index[name]
            spec = ms.PAIR_CFGS[i]
            pos_y, pos_x = detail["pos_y"], detail["pos_x"]
            new_y = np.round(pos_y / px[spec.y])
            new_x = np.round(pos_x / px[spec.x])
            comm = COMM_RATE * (
                abs(new_y - desired_shares[i, 0]) * px[spec.y]
                + abs(new_x - desired_shares[i, 1]) * px[spec.x]
            )
            pnl_day = (
                new_y * (prices[spec.y, t] - px[spec.y])
                + new_x * (prices[spec.x, t] - px[spec.x])
                - comm
            )
            pair_pnl[i] += pnl_day
            pair_comm[i] += comm
            if oos:
                pair_pnl_oos[i] += pnl_day
            desired_shares[i] = (new_y, new_x)
            active_days[i] += 1

            for leg_idx, pos in ((spec.y, pos_y), (spec.x, pos_x)):
                if abs(pos) < 1.0e-9:
                    continue
                book_sign = np.sign(book_dollars[leg_idx])
                want_sign = np.sign(pos)
                if book_sign == want_sign:
                    honored[i] += 1
                elif book_sign == 0.0:
                    suppressed[i] += 1
                else:
                    flipped[i] += 1

    records = [
        {
            "pair": names[i],
            "engine": engines[i],
            "cohort": cohorts[i],
            "pnl_full": round(float(pair_pnl[i]), 2),
            "pnl_oos": round(float(pair_pnl_oos[i]), 2),
            "comm_full": round(float(pair_comm[i]), 2),
            "active_days": int(active_days[i]),
            "honored": int(honored[i]),
            "suppressed": int(suppressed[i]),
            "flipped": int(flipped[i]),
        }
        for i in range(n)
    ]
    return {
        "book_pnl_full": round(book_pnl, 2),
        "book_pnl_oos": round(book_pnl_oos, 2),
        "desired_sum_full": round(float(pair_pnl.sum()), 2),
        "desired_sum_oos": round(float(pair_pnl_oos.sum()), 2),
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--oos_start", type=int, default=801)
    args = ap.parse_args()

    prices, nt, ninst = latest_tuner.loadPrices("prices.txt")
    prices = np.asarray(prices, dtype=float)

    out = {}
    for variant in ("tuned", "hybrid"):
        rows = load_variant_rows(variant)
        install_variant(rows)
        out[variant] = run_attribution(prices, args.start, args.end, args.oos_start)
        b = out[variant]
        print(f"[{variant}] book PnL full {b['book_pnl_full']:.0f} | OOS {b['book_pnl_oos']:.0f} "
              f"| desired-sum full {b['desired_sum_full']:.0f} | OOS {b['desired_sum_oos']:.0f}")

    path = "auto_tuning_results/_pair_attribution.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved -> {path}")

    # --- console digests ---
    tuned = {r["pair"]: r for r in out["tuned"]["records"]}
    hybrid = {r["pair"]: r for r in out["hybrid"]["records"]}

    print("\n=== WORST 15 desired OOS PnL (tuned OU variant) ===")
    print(f"{'pair':14} {'cohort':14} {'pnl_oos':>9} {'pnl_full':>9} {'act':>4} "
          f"{'hon%':>5} {'sup%':>5} {'flp%':>5} | hybrid_zband_oos")
    for r in sorted(out["tuned"]["records"], key=lambda r: r["pnl_oos"])[:15]:
        tot = r["honored"] + r["suppressed"] + r["flipped"] or 1
        hz = hybrid.get(r["pair"], {})
        print(f"{r['pair']:14} {r['cohort']:14} {r['pnl_oos']:9.0f} {r['pnl_full']:9.0f} "
              f"{r['active_days']:4} {100*r['honored']/tot:5.0f} {100*r['suppressed']/tot:5.0f} "
              f"{100*r['flipped']/tot:5.0f} | {hz.get('pnl_oos', float('nan')):9.0f}")

    print("\n=== BEST 15 desired OOS PnL (tuned OU variant) ===")
    for r in sorted(out["tuned"]["records"], key=lambda r: -r["pnl_oos"])[:15]:
        tot = r["honored"] + r["suppressed"] + r["flipped"] or 1
        hz = hybrid.get(r["pair"], {})
        print(f"{r['pair']:14} {r['cohort']:14} {r['pnl_oos']:9.0f} {r['pnl_full']:9.0f} "
              f"{r['active_days']:4} {100*r['honored']/tot:5.0f} {100*r['suppressed']/tot:5.0f} "
              f"{100*r['flipped']/tot:5.0f} | {hz.get('pnl_oos', float('nan')):9.0f}")

    print("\n=== cohort aggregates (tuned variant) ===")
    from collections import defaultdict
    agg = defaultdict(lambda: [0.0, 0.0, 0, 0, 0, 0])
    for r in out["tuned"]["records"]:
        a = agg[r["cohort"]]
        a[0] += r["pnl_full"]; a[1] += r["pnl_oos"]
        a[2] += r["honored"]; a[3] += r["suppressed"]; a[4] += r["flipped"]
        a[5] += 1
    print(f"{'cohort':14} {'n':>3} {'pnl_full':>10} {'pnl_oos':>10} {'hon%':>5} {'sup%':>5} {'flp%':>5}")
    for cohort, a in sorted(agg.items()):
        tot = a[2] + a[3] + a[4] or 1
        print(f"{cohort:14} {a[5]:3} {a[0]:10.0f} {a[1]:10.0f} "
              f"{100*a[2]/tot:5.0f} {100*a[3]/tot:5.0f} {100*a[4]/tot:5.0f}")

    print("\n=== engine delta on catalog pairs (desired OOS: tuned OU minus zband) ===")
    deltas = []
    for name, tr in tuned.items():
        hz = hybrid.get(name)
        if hz and tr["cohort"] == "catalog" and hz["engine"] == "zband":
            deltas.append((tr["pnl_oos"] - hz["pnl_oos"], name, tr["pnl_oos"], hz["pnl_oos"]))
    deltas.sort()
    for d, name, tp, hp in deltas[:12]:
        print(f"  OU worse: {name:14} ou {tp:8.0f} vs zband {hp:8.0f} (delta {d:8.0f})")
    for d, name, tp, hp in deltas[-8:]:
        print(f"  OU better: {name:14} ou {tp:8.0f} vs zband {hp:8.0f} (delta {d:+8.0f})")
    print(f"  catalog pairs: OU wins {sum(1 for d in deltas if d[0] > 0)}/{len(deltas)}, "
          f"sum delta {sum(d[0] for d in deltas):.0f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Reality check: are the pairs real, or factor exposure in disguise?

Three analyses:
  A. Placebo control: train->OOS decay of shortlist vs placebo pairs tuned
     through the identical pipeline (screen_tuning_results.json).
  B. Book-level factor loading: correlation of daily book PnL with ALGO
     returns, per book variant (tuned OU vs hybrid zband-catalog).
  C. Pair-level factor loading: per-pair correlation of desired daily PnL with
     ALGO returns + mean net dollar delta. Compares OU winners vs OU losers
     (cherrypick split) and OU vs zband engines on the same pairs.
"""

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import json
from collections import defaultdict

import numpy as np

import latest_tuner
import merged_strategy as ms
import _pair_attribution as pa

COMM_RATE = 0.0001
ALGO = ms.I["ALGO"]
SYMBOL_BY_INDEX = {index: symbol for symbol, index in ms.I.items()}


def replay_series(prices, start, end):
    """Per-pair desired daily PnL, book daily PnL, ALGO returns, net deltas."""
    names = [f"{SYMBOL_BY_INDEX[s.y]}-{SYMBOL_BY_INDEX[s.x]}" for s in ms.PAIR_CFGS]
    pair_index = {name: i for i, name in enumerate(names)}
    n = len(names)
    days = end - start + 1  # legs land on days [start, end]
    pair_pl = np.zeros((n, days))
    pair_delta = np.full((n, days), np.nan)  # net dollar delta (pos_y + pos_x)
    book_pl = np.zeros(days)
    algo_ret = np.zeros(days)
    desired_shares = np.zeros((n, 2))
    book_shares = np.zeros(ms.N_INST)
    logp = np.log(np.maximum(prices, 1.0e-12))

    for k, t in enumerate(range(start - 1, end)):
        prc = prices[:, :t]
        px = prices[:, t - 1]
        nxt = prices[:, t]
        sleeves = ms.sleeve_breakdown(prc)
        if sleeves is None:
            continue
        algo_ret[k] = logp[ALGO, t] - logp[ALGO, t - 1]

        new_book = ms.getMyPosition(prc).astype(float)
        comm = COMM_RATE * float(np.sum(np.abs(new_book - book_shares) * px))
        book_pl[k] = float(np.dot(new_book, nxt - px)) - comm
        book_shares = new_book

        for detail in sleeves["pair_contributor_details"]:
            name = detail["pair"]
            if name not in pair_index:
                continue
            i = pair_index[name]
            spec = ms.PAIR_CFGS[i]
            pos_y, pos_x = detail["pos_y"], detail["pos_x"]
            new_y = np.round(pos_y / px[spec.y])
            new_x = np.round(pos_x / px[spec.x])
            c = COMM_RATE * (
                abs(new_y - desired_shares[i, 0]) * px[spec.y]
                + abs(new_x - desired_shares[i, 1]) * px[spec.x]
            )
            pair_pl[i, k] = (
                new_y * (prices[spec.y, t] - px[spec.y])
                + new_x * (prices[spec.x, t] - px[spec.x]) - c
            )
            desired_shares[i] = (new_y, new_x)
            pair_delta[i, k] = pos_y + pos_x
    return names, pair_pl, pair_delta, book_pl, algo_ret


def corr_safe(a, b):
    if len(a) < 10 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    prices, nt, ninst = latest_tuner.loadPrices("prices.txt")
    prices = np.asarray(prices, dtype=float)

    print("=== A. placebo control (same tuner, real shortlist vs placebo pairs) ===")
    meta = json.load(open("auto_tuning_results/selection_meta.json"))
    placebo_keys = {frozenset(p["pair"]) for p in meta["placebos"]}
    tuning = json.load(open("auto_tuning_results/screen_tuning_results.json"))
    groups = {"shortlist": [], "placebo": []}
    for entry in tuning:
        key = frozenset((entry["y_symbol"], entry["x_symbol"]))
        g = "placebo" if key in placebo_keys else "shortlist"
        gm = entry["global"]
        tr = gm.get("train_metrics", {}).get("score")
        oo = gm.get("oos_metrics", {}).get("score")
        if tr is not None and oo is not None:
            groups[g].append((f"{entry['y_symbol']}-{entry['x_symbol']}", tr, oo))
    for g, rows in groups.items():
        if not rows:
            continue
        trs = np.array([r[1] for r in rows])
        oos = np.array([r[2] for r in rows])
        print(f"  {g:9} n={len(rows)}  train mean {trs.mean():7.2f}  "
              f"OOS mean {oos.mean():7.2f}  OOS>0: {(oos > 0).sum()}/{len(rows)}")
        for name, tr, oo in sorted(rows, key=lambda r: -r[2]):
            print(f"    {name:14} train {tr:8.2f} oos {oo:8.2f}")

    print("\n=== B/C. factor loading per variant (full 200-1000 + OOS 801-1000) ===")
    att = json.load(open("auto_tuning_results/_pair_attribution.json"))
    tuned_rec = {r["pair"]: r for r in att["tuned"]["records"]}
    zband_rec = {r["pair"]: r for r in att["hybrid"]["records"]}
    ou_winner = {
        name for name, tr in tuned_rec.items()
        if name in zband_rec and zband_rec[name]["engine"] == "zband"
        and tr["pnl_oos"] > zband_rec[name]["pnl_oos"]
    }

    oos_lo, oos_hi = 801, 1000
    for variant in ("tuned", "hybrid"):
        rows = pa.load_variant_rows(variant)
        pa.install_variant(rows)
        names, pair_pl, pair_delta, book_pl, algo_ret = replay_series(prices, 200, 1000)
        k0 = oos_lo - 200
        b_full = corr_safe(book_pl, algo_ret)
        b_oos = corr_safe(book_pl[k0:], algo_ret[k0:])
        print(f"\n[{variant}] book PnL x ALGO corr: full {b_full:+.3f} | OOS {b_oos:+.3f}")

        stats = defaultdict(list)
        deltas = defaultdict(list)
        engines = [s.engine for s in ms.PAIR_CFGS]
        cohorts = [ms._COHORT_BY_NAME.get(nm, "?") for nm in names]
        for i, nm in enumerate(names):
            c_full = corr_safe(pair_pl[i], algo_ret)
            c_oos = corr_safe(pair_pl[i][k0:], algo_ret[k0:])
            d = np.nanmean(np.abs(pair_delta[i]))
            grp = "ou_winner" if nm in ou_winner else (
                "ou_loser" if cohorts[i] == "catalog" else cohorts[i])
            stats[(engines[i], grp)].append(c_oos)
            deltas[(engines[i], grp)].append(d)
        print(f"  {'engine':6} {'group':14} {'n':>3} {'corr_oos mean':>14} "
              f"{'|corr| mean':>12} {'net|delta| mean':>16}")
        for (eng, grp), cs in sorted(stats.items()):
            cs = np.array([c for c in cs if not np.isnan(c)])
            ds = np.array(deltas[(eng, grp)])
            if len(cs) == 0:
                continue
            print(f"  {eng:6} {grp:14} {len(cs):3} {cs.mean():14.3f} "
                  f"{np.abs(cs).mean():12.3f} {ds.mean():16.0f}")


if __name__ == "__main__":
    main()

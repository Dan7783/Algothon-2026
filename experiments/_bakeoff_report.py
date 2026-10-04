#!/usr/bin/env python
"""Bake-off report: Sharpe-decay-primary comparison of tuner configs across pairs.

Sharpe retention (OOS/train) is the primary generalization metric because it is
moneyness-invariant; score retention is reported secondarily since score embeds
mean P&L (regime moneyness).

Reads:
  LT        : auto_tuning_results/_bakeoff_lt_rtth_backup.json + tuning_results.json
  W-current : auto_tuning_results/_bakeoff_wcurrent.json
  W-ratio   : auto_tuning_results/_bakeoff_wratio.json
  W-std0    : auto_tuning_results/_bakeoff_wstd0.json
  W-tiebreak: auto_tuning_results/_bakeoff_wtiebreak.json
"""

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import glob
import json
import os
import statistics as st

R = "auto_tuning_results"
SOURCES = {
    "LT": [os.path.join(R, "_bakeoff_lt_rtth_backup.json"),
           os.path.join(R, "_bakeoff_lt.json")],
    "W-cur": [os.path.join(R, "_bakeoff_wcurrent.json")],
    "W-ratio": [os.path.join(R, "_bakeoff_wratio.json")],
    "W-std0": [os.path.join(R, "_bakeoff_wstd0.json")],
    "W-tbrk": [os.path.join(R, "_bakeoff_wtiebreak.json")],
    "W-R+tb": [os.path.join(R, "_bakeoff_wratio_tbrk.json")],
}


def extract(cfg, entry):
    """Normalize the three result-file shapes to (train, oos) metric dicts."""
    if cfg == "LT":
        return entry["auto_metrics"], entry["test_metrics"]["auto"]
    if cfg in ("W-tbrk", "W-R+tb"):
        return entry["train_metrics"], entry["oos_metrics"]
    g = entry.get("global")
    return g["train_metrics"], g["oos_metrics"]


def main():
    table = {}  # table[pair][cfg] = dict
    for cfg, paths in SOURCES.items():
        for path in paths:
            if not os.path.exists(path):
                continue
            for e in json.load(open(path)):
                tr, oos = extract(cfg, e)
                table.setdefault(e["pair_name"], {})[cfg] = {
                    "tr_score": tr["score"], "oos_score": oos["score"],
                    "tr_s": tr["sharpe"], "oos_s": oos["sharpe"],
                }

    cfgs = [c for c in SOURCES if any(c in p for p in table.values())]
    pairs = sorted(table)

    print("=" * 100)
    print("SHARPE RETENTION (OOS/train) — primary, moneyness-invariant")
    print("=" * 100)
    hdr = f"{'pair':<14}" + "".join(f"{c:>9}" for c in cfgs)
    print(hdr + f"{'trainS':>9}{'oosS(LT)':>10}")
    print("-" * len(hdr))
    for p in pairs:
        row = f"{p:<14}"
        for c in cfgs:
            d = table[p].get(c)
            row += f"{(d['oos_s']/d['tr_s']):>9.0%}" if d and d["tr_s"] else f"{'-':>9}"
        lt = table[p].get("LT")
        row += f"{lt['tr_s']:>9.2f}{lt['oos_s']:>10.2f}" if lt else f"{'-':>9}{'-':>10}"
        print(row)

    print()
    print("=" * 100)
    print("SCORE RETENTION (OOS/train) — secondary, moneyness-laden")
    print("=" * 100)
    print(hdr)
    print("-" * len(hdr))
    for p in pairs:
        row = f"{p:<14}"
        for c in cfgs:
            d = table[p].get(c)
            row += f"{(d['oos_score']/d['tr_score']):>9.0%}" if d and d["tr_score"] else f"{'-':>9}"
        print(row)

    print()
    print("=" * 100)
    print("CONFIG SUMMARY")
    print("=" * 100)
    print(f"{'config':<9}{'medSret':>9}{'meanSret':>9}{'medScore':>9}{'meanScore':>10}"
          f"{'oosSharpe':>10}{'oosScore':>9}{'winsVsLT':>9}")
    lt_oos_s = {p: table[p]["LT"]["oos_s"] for p in pairs if "LT" in table[p]}
    for c in cfgs:
        sret = [table[p][c]["oos_s"] / table[p][c]["tr_s"] for p in pairs
                if c in table[p] and table[p][c]["tr_s"]]
        scret = [table[p][c]["oos_score"] / table[p][c]["tr_score"] for p in pairs
                 if c in table[p] and table[p][c]["tr_score"]]
        oos_s = [table[p][c]["oos_s"] for p in pairs if c in table[p]]
        oos_sc = [table[p][c]["oos_score"] for p in pairs if c in table[p]]
        wins = sum(1 for p in pairs if c in table[p] and p in lt_oos_s
                   and table[p][c]["oos_s"] > lt_oos_s[p])
        print(f"{c:<9}{st.median(sret):>9.0%}{st.mean(sret):>9.0%}"
              f"{st.median(scret):>9.0%}{st.mean(scret):>10.0%}"
              f"{st.mean(oos_s):>10.2f}{st.mean(oos_sc):>9.2f}{wins:>6}/{len(lt_oos_s)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build the merged pair catalog for merged_strategy.py.

Sources:
  1. friend_book._ACTIVE_PAIR_PARAMS  - the 88-pair old-engine catalog (z-band
     profiles). Filter constants carry over; the new engine's kalman_var_ewma_alpha
     is set to 0.03 (identical to the old hardcoded ewma_lambda=0.97) and the OU
     estimator gets the friend's hardcoded smoothing (window 100, alpha 0.1/0.1).
     Entry/exit is then the new engine's dynamic optimal-stopping a* instead of
     the fixed z bands (which have no meaning under the new engine).
  2. latest_strategy.RAW_PAIRS        - the user's 8 windowed-tuner profiles
     (new engine, ratio+tiebreak selected). These override by unordered pair key.
  3. screen_tuning_results.json       - the 10 tuned screen-cohort profiles
     (1 ADD, 2 REVIEW, 8 REJECT by the standalone bar). These override/append by
     unordered pair key. Placebo-cohort rows are never included.

Output: auto_tuning_results/merged_catalog.json + a Python literal on stdout.
"""

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import argparse
import json
import os

import friend_book
import latest_strategy

NEW_ENGINE_DEFAULTS = {
    "kalman_var_ewma_alpha": 0.03,
    "ou_window": 100,
    "kappa_ewma_alpha": 0.1,
    "sigma_ewma_alpha": 0.1,
}

PARAM_KEYS = (
    "q_beta", "kalman_var_ewma_alpha", "cusum_k", "h", "max_cov_inflate",
    "cov_decay", "ou_window", "kappa_ewma_alpha", "sigma_ewma_alpha",
)

PROTECTED_NAMES = friend_book.PROTECTED_PAIR_NAMES

VERIFIED_NEW = {frozenset(("MSDP", "ULXY")), frozenset(("EELT", "OTCS"))}


def ukey(y, x):
    return frozenset((y, x))


def old_row_to_params(row):
    (_, _, q_beta, cusum_k, h, max_cov_inflate, cov_decay,
     z_entry, z_exit, z_stop) = row
    return {
        "engine": "zband",
        "q_beta": q_beta,
        "cusum_k": cusum_k,
        "h": h,
        "max_cov_inflate": max_cov_inflate,
        "cov_decay": cov_decay,
        "z_entry": z_entry,
        "z_exit": z_exit,
        "z_stop": z_stop,
    }


def tuned_row_to_params(fp):
    return {"engine": "ou", **{k: fp[k] for k in PARAM_KEYS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hybrid", action="store_true",
        help="zband profiles for friend catalog pairs; ou only for tuned cohorts")
    parser.add_argument(
        "--cherrypick", action="store_true",
        help="per-pair engine split: catalog pairs keep their tuned ou profile "
             "only where attribution shows it beats zband OOS (else zband)")
    args = parser.parse_args()
    hybrid = args.hybrid or args.cherrypick

    ou_winners = set()
    if args.cherrypick:
        with open("auto_tuning_results/_pair_attribution.json") as f:
            att = json.load(f)
        tuned_rec = {r["pair"]: r for r in att["tuned"]["records"]}
        zband_rec = {r["pair"]: r for r in att["hybrid"]["records"]}
        for name, tr in tuned_rec.items():
            hz = zband_rec.get(name)
            if hz and hz["engine"] == "zband" and tr["pnl_oos"] > hz["pnl_oos"]:
                ou_winners.add(name)
        print(f"cherrypick: {len(ou_winners)} catalog pairs keep tuned ou profiles")

    catalog = {}  # ukey -> dict(y, x, params, cohort)
    order = []

    def upsert(y, x, params, cohort):
        k = ukey(y, x)
        if k not in catalog:
            order.append(k)
        catalog[k] = {"y": y, "x": x, "params": params, "cohort": cohort}

    # 1) friend's old catalog (already development-filtered); pairs that have
    #    been re-tuned on the new engine by the catalog88 windowed-tuner run
    #    get their tuned ou profile instead of the mapped zband guess — unless
    #    --hybrid, which keeps the proven zband profiles for catalog pairs
    #    (their tuned ou profiles decayed hard OOS) and ou only for tuned
    #    cohorts (protected/current/reject/verified).
    tuned88 = {}
    tuned88_path = "auto_tuning_results/catalog88_tuning_results.json"
    keep_tuned = not hybrid or args.cherrypick
    if os.path.exists(tuned88_path) and keep_tuned:
        with open(tuned88_path) as f:
            for entry in json.load(f):
                fp = entry.get("global", {}).get("final_params")
                if fp:
                    tuned88[ukey(entry["y_symbol"], entry["x_symbol"])] = fp
    for row in friend_book._ACTIVE_PAIR_PARAMS:
        y, x = row[0], row[1]
        name = f"{y}-{x}"
        cohort = "protected_old" if name in PROTECTED_NAMES else "catalog"
        fp = tuned88.get(ukey(y, x))
        if fp is not None and (not args.cherrypick or name in ou_winners):
            upsert(y, x, tuned_row_to_params(fp), cohort + "_tuned")
        else:
            upsert(y, x, old_row_to_params(row), cohort)

    # 2) user's tuned current-8 (override protected rows with tuned profiles)
    tuned_current = set()
    for row in latest_strategy.RAW_PAIRS:
        y, x = row["y"], row["x"]
        name = f"{y}-{x}"
        cohort = "protected" if name in PROTECTED_NAMES else "tuned_current"
        upsert(y, x, tuned_row_to_params(row), cohort)
        tuned_current.add(ukey(y, x))

    # 3) tuned screen cohort (10 new-pair rows; placebos are excluded upstream
    #    by selection_meta shortlist vs placebos)
    with open("auto_tuning_results/selection_meta.json") as f:
        selection = json.load(f)
    shortlist_keys = {ukey(*s["pair"]) for s in selection["shortlist"]}

    with open("auto_tuning_results/screen_tuning_results.json") as f:
        tuning = json.load(f)
    for entry in tuning:
        k = ukey(entry["y_symbol"], entry["x_symbol"])
        if k not in shortlist_keys:
            continue  # placebo arm
        fp = entry["global"]["final_params"]
        cohort = "verified_new" if k in VERIFIED_NEW else "reject_cohort"
        upsert(entry["y_symbol"], entry["x_symbol"], tuned_row_to_params(fp), cohort)

    rows = [catalog[k] for k in order]
    from collections import Counter
    counts = Counter(r["cohort"] for r in rows)
    print(f"merged catalog: {len(rows)} pairs | {dict(counts)}")

    with open("auto_tuning_results/merged_catalog.json", "w") as f:
        json.dump(rows, f, indent=1)

    # Emit the embeddable literal (engine-tagged rows; merged_strategy.py
    # dispatches on row[0]).
    print("_MERGED_PAIR_PARAMS = (")
    for r in rows:
        p = r["params"]
        if p["engine"] == "ou":
            vals = ", ".join(repr(p[k]) for k in PARAM_KEYS)
        else:
            vals = ", ".join(
                repr(p[k])
                for k in ("q_beta", "cusum_k", "h", "max_cov_inflate",
                          "cov_decay", "z_entry", "z_exit", "z_stop")
            )
        print(f'    ("{p["engine"]}", "{r["y"]}", "{r["x"]}", {vals}),  # {r["cohort"]}')
    print(")")


if __name__ == "__main__":
    main()

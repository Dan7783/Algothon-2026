#!/usr/bin/env python
"""Build/eval catalog keeping only OU pairs that beat friend z-band on TRAIN standalone.

Uses _standalone_exit_compare.json (train 31-750 primary). Global MERGED_OU_EXIT=mean
(majority of train winners prefer mean). Non-winners reverted to friend z-band.
OOS reported as verify only.
"""
from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "auto_tuning_results"
OUT_CAT = RESULTS / "_train_ou_winners_catalog.json"
OUT = RESULTS / "_train_ou_winners_bakeoff.json"

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
    "MERGED_ML_MODEL",
]


def clean_env(extra=None):
    env = os.environ.copy()
    for k in MERGED_KEYS:
        env.pop(k, None)
    if extra:
        env.update(extra)
    return env


def run_eval(module, start, end, extra):
    p = subprocess.run(
        [sys.executable, "eval.py", "--module", module, "--start", start, "--end", end],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=clean_env(extra),
    )
    m = re.search(r"Score:\s*([-\d.]+)", p.stdout + "\n" + p.stderr)
    return float(m.group(1)) if m else None


def friend_zband_lookup():
    for k in MERGED_KEYS:
        os.environ.pop(k, None)
    import friend_book_latest as fb

    return {(r[0], r[1]): ["zband", r[0], r[1], *r[2:]] for r in fb._ACTIVE_PAIR_PARAMS}


def main():
    cmp = json.loads((RESULTS / "_standalone_exit_compare.json").read_text(encoding="utf-8"))
    winners = {
        r["pair"]: r["train_31_750"]["best"]
        for r in cmp["pairs"]
        if r["train_31_750"]["best"] != "zband"
    }
    base = json.loads((RESULTS / "merged_friend_ou_catalog.json").read_text(encoding="utf-8"))
    friend_zb = friend_zband_lookup()
    new_rows = []
    kept_ou = []
    for row in base:
        name = f"{row[1]}-{row[2]}"
        if row[0] == "ou" and name in winners:
            new_rows.append(list(row))
            kept_ou.append({"pair": name, "best_exit": winners[name]})
        elif row[0] == "ou":
            zb = friend_zb.get((row[1], row[2])) or friend_zb.get((row[2], row[1]))
            if zb is None:
                # drop if no friend twin
                continue
            new_rows.append(list(zb))
        else:
            new_rows.append(list(row))

    # Cap 88
    new_rows = new_rows[:88]
    OUT_CAT.write_text(json.dumps(new_rows), encoding="utf-8")
    n_ou = sum(1 for r in new_rows if r[0] == "ou")
    print(f"catalog {len(new_rows)} rows, {n_ou} OU train-winners: {[k['pair'] for k in kept_ou]}", flush=True)

    cases = [
        ("friend", "friend_book_latest", {}),
        (
            "full_ou_mean_m015",
            "merged_friend_ou",
            {"MERGED_OU_EXIT": "mean", "MERGED_CONSENSUS_MARGIN": "0.15"},
        ),
        (
            "train_ou_winners_mean_m015",
            "merged_friend_ou",
            {
                "MERGED_OU_EXIT": "mean",
                "MERGED_CONSENSUS_MARGIN": "0.15",
                "MERGED_CATALOG_JSON": str(OUT_CAT),
            },
        ),
        (
            "train_ou_winners_optimal_m015",
            "merged_friend_ou",
            {
                "MERGED_OU_EXIT": "optimal",
                "MERGED_CONSENSUS_MARGIN": "0.15",
                "MERGED_CATALOG_JSON": str(OUT_CAT),
            },
        ),
        (
            "REJECTED_1057_oos_prune",
            "merged_friend_ou",
            {
                "MERGED_OU_EXIT": "mean",
                "MERGED_CONSENSUS_MARGIN": "0.20",
                "MERGED_CATALOG_JSON": str(RESULTS / "_mean_exit_best_catalog.json"),
            },
        ),
    ]
    scores = {}
    for label, module, extra in cases:
        sc = {
            "train_31_750": run_eval(module, "31", "750", extra),
            "val_501_750": run_eval(module, "501", "750", extra),
            "oos_750_1000_verify": run_eval(module, "750", "1000", extra),
        }
        scores[label] = sc
        print(f"  {label:32} {sc}", flush=True)

    report = {
        "kept_ou": kept_ou,
        "n_ou": n_ou,
        "catalog": str(OUT_CAT),
        "scores": scores,
        "friend_targets": {"train": 926.94, "val": 1042.37, "oos": 974.45},
        "note": "OU sleeve selected by train standalone only; OOS verify only",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

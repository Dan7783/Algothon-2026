#!/usr/bin/env python
"""Pinned-env bake-off: OU-specific conflict modes vs friend / shared voter.

ALWAYS clears MERGED_* env vars first to avoid PowerShell session leaks.
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
OUT = ROOT / "auto_tuning_results" / "_ou_conflict_bakeoff.json"

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
]


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for k in MERGED_KEYS:
        env.pop(k, None)
    if extra:
        env.update(extra)
    return env


def run_eval(module: str, start: str, end: str, env: dict[str, str]) -> dict:
    p = subprocess.run(
        [
            sys.executable,
            "eval.py",
            "--module",
            module,
            "--start",
            start,
            "--end",
            end,
        ],
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
        "tail": out[-800:],
    }


def main() -> None:
    cases = [
        ("friend_baseline", "friend_book_latest", {}),
        # Current default shared voter (margin 0.15 baked into module default)
        ("friend_ou_shared", "merged_friend_ou", {"MERGED_OU_CONFLICT": "shared"}),
        (
            "friend_ou_shared_m015",
            "merged_friend_ou",
            {"MERGED_OU_CONFLICT": "shared", "MERGED_CONSENSUS_MARGIN": "0.15"},
        ),
        ("friend_ou_sleeve", "merged_friend_ou", {"MERGED_OU_CONFLICT": "sleeve"}),
        (
            "friend_ou_sleeve_demand",
            "merged_friend_ou",
            {
                "MERGED_OU_CONFLICT": "sleeve",
                "MERGED_OU_SLEEVE_SPLIT": "demand",
            },
        ),
        ("friend_ou_ou_flat", "merged_friend_ou", {"MERGED_OU_CONFLICT": "ou_flat"}),
        ("friend_ou_twopass", "merged_friend_ou", {"MERGED_OU_CONFLICT": "twopass"}),
    ]

    windows = (("OOS", "801", "1000"), ("full", "200", "1000"))
    results = []
    for label, module, extra in cases:
        for wname, start, end in windows:
            print(f"RUN {label} {wname} ...", flush=True)
            r = run_eval(module, start, end, clean_env(extra))
            r["label"] = label
            r["window"] = wname
            r["env"] = extra
            results.append(r)
            print(
                f"  -> score={r['score']} sharpe={r['sharpe']} rc={r['returncode']}",
                flush=True,
            )

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

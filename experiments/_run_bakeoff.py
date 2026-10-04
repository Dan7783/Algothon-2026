#!/usr/bin/env python
"""Bake-off orchestrator: runs each tuner config as one process PER PAIR, each with its
own SQLite DB and output JSON under %TEMP% (no shared-writer lock contention, no
OneDrive fsync throttling), then merges per-pair JSONs (and studies, for windowed
configs) back into auto_tuning_results/_bakeoff_<phase>.{json,db}.

Phase groups run concurrently: (lt + wcurrent), then (wratio + wstd0).

Usage: python _run_bakeoff.py [--groups 0 1]   (default: both groups, sequentially)
"""

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "auto_tuning_results")
WORK = os.path.join(tempfile.gettempdir(), "_bakeoff")

PAIRS_ALL = json.load(open(os.path.join(ROOT, "experiments", "_bakeoff_pairs.json")))
PAIRS_LT = json.load(open(os.path.join(ROOT, "experiments", "_bakeoff_lt7.json")))

PHASES = {
    "lt": {"pairs": PAIRS_LT, "tuner": "latest_tuner.py", "extra": []},
    "wcurrent": {"pairs": PAIRS_ALL, "tuner": "windowed_tuner.py",
                 "extra": ["--norm_mode", "percentile", "--std_lambda", "1.0", "--embargo", "10"]},
    "wratio": {"pairs": PAIRS_ALL, "tuner": "windowed_tuner.py",
               "extra": ["--norm_mode", "ratio", "--std_lambda", "1.0", "--embargo", "10"]},
    "wstd0": {"pairs": PAIRS_ALL, "tuner": "windowed_tuner.py",
              "extra": ["--norm_mode", "percentile", "--std_lambda", "0", "--embargo", "0"]},
}
GROUPS = [["lt", "wcurrent"], ["wratio", "wstd0"]]


def spawn(phase, pair):
    pdir = os.path.join(WORK, phase)
    os.makedirs(pdir, exist_ok=True)
    name = f"{pair['y']}-{pair['x']}"
    pair_file = os.path.join(pdir, f"{name}_pair.json")
    with open(pair_file, "w") as f:
        json.dump([pair], f)
    cfg = PHASES[phase]
    cmd = [sys.executable, os.path.join(ROOT, cfg["tuner"]),
           "--pairs_file", pair_file,
           "--db_file", os.path.join(pdir, f"{name}.db"),
           "--out_file", os.path.join(pdir, f"{name}.json")]
    if cfg["tuner"] == "windowed_tuner.py":
        cmd += ["--workers", "1"] + cfg["extra"]
    log = open(os.path.join(pdir, f"{name}.log"), "w")
    return name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT), log


def merge_phase(phase):
    pdir = os.path.join(WORK, phase)
    merged = []
    for pair in PHASES[phase]["pairs"]:
        name = f"{pair['y']}-{pair['x']}"
        path = os.path.join(pdir, f"{name}.json")
        if os.path.exists(path):
            merged.extend(json.load(open(path)))
        else:
            print(f"  [{phase}] WARNING: no results for {name}")
    out_json = os.path.join(RESULTS, f"_bakeoff_{phase}.json")
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=4)
    print(f"  [{phase}] merged {len(merged)} pair results -> {out_json}")

    if PHASES[phase]["tuner"] != "windowed_tuner.py":
        return
    # Merge into a TEMP DB on the local disk first: per-trial inserts with fsync are
    # the dominant cost, and doing them on OneDrive makes the merge crawl. The finished
    # DB is then copied to RESULTS as one bulk write (fallback name if locked).
    tmp_db = os.path.join(pdir, f"_merged_{phase}.db")
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    to_storage = optuna.storages.RDBStorage(f"sqlite:///{tmp_db}")
    n = 0
    for pair in PHASES[phase]["pairs"]:
        name = f"{pair['y']}-{pair['x']}"
        src = os.path.join(pdir, f"{name}.db")
        if not os.path.exists(src):
            continue
        from_storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{src}", engine_kwargs={"connect_args": {"timeout": 120}})
        for s in optuna.get_all_study_summaries(storage=from_storage):
            optuna.copy_study(from_storage=from_storage, to_storage=to_storage,
                              from_study_name=s.study_name, to_study_name=s.study_name)
            n += 1
        print(f"  [{phase}] merged studies for {name} ({n} total)", flush=True)
    del to_storage
    out_db = os.path.join(RESULTS, f"_bakeoff_{phase}.db")
    for target in (out_db, out_db.replace(".db", "_merged.db")):
        try:
            if os.path.exists(target):
                os.remove(target)
            import shutil
            shutil.copyfile(tmp_db, target)
            print(f"  [{phase}] merged {n} studies -> {target}")
            return
        except PermissionError:
            time.sleep(5)
    print(f"  [{phase}] WARNING: could not write merged DB to RESULTS; left at {tmp_db}")


def run_phases(phases, only=None, max_concurrent=8, stagger=10):
    t0 = time.time()
    queue = [(phase, pair) for phase in phases for pair in PHASES[phase]["pairs"]
             if only is None or f"{pair['y']}-{pair['x']}" in only]
    running = {}
    failed = []
    while queue or running:
        # Staggered, capped spawns: every proc parses prices.txt (~300MB transient),
        # so a simultaneous spawn burst can OOM the machine.
        while queue and len(running) < max_concurrent:
            phase, pair = queue.pop(0)
            name, proc, log = spawn(phase, pair)
            running[(phase, name)] = (proc, log)
            print(f"[spawn] {phase}/{name} (pid {proc.pid})", flush=True)
            time.sleep(stagger)
        for key, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                status = "done" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                print(f"[{status}] {key[0]}/{key[1]} after {time.time()-t0:.0f}s", flush=True)
                if proc.returncode != 0:
                    failed.append(key)
                del running[key]
        if queue or running:
            time.sleep(5)
    for phase in phases:
        merge_phase(phase)
    print(f"[phases {phases}] finished in {time.time()-t0:.0f}s; failures: {failed or 'none'}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, nargs="*", default=None)
    ap.add_argument("--phases", nargs="*", default=None, help="phase names, e.g. wratio wstd0")
    ap.add_argument("--only", nargs="*", default=None, help="pair names to include")
    ap.add_argument("--max_concurrent", type=int, default=8)
    ap.add_argument("--stagger", type=int, default=10, help="seconds between spawns")
    args = ap.parse_args()
    if args.phases:
        run_phases(args.phases, args.only, args.max_concurrent, args.stagger)
    else:
        for gi in (args.groups if args.groups is not None else [0, 1]):
            run_phases(GROUPS[gi], args.only, args.max_concurrent, args.stagger)
    print("BAKEOFF COMPLETE")

"""Supervise the 80-pair OU-search tuner: relaunch on silent death until the
results JSON covers all candidates. Resume-aware optimize_study makes retries
cheap (only unflushed trials are re-run)."""

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import json
import os
import subprocess
import sys
import tempfile
import time

PAIRS_FILE = os.path.join("auto_tuning_results", "ou_search_pairs.json")
OUT_FILE = os.path.join("auto_tuning_results", "ou_search_tuning_results.json")
DB_FILE = os.path.join(tempfile.gettempdir(), "optuna_ou_search.db")
MAX_ATTEMPTS = 12


def results_complete():
    if not os.path.exists(OUT_FILE):
        return False
    try:
        done = {r["pair_name"] for r in json.load(open(OUT_FILE))}
        want = {f"{p['y']}-{p['x']}" for p in json.load(open(PAIRS_FILE))}
        return want <= done
    except Exception:
        return False


CMD = [
    sys.executable, "windowed_tuner.py",
    "--pairs_file", PAIRS_FILE,
    "--db_file", DB_FILE,
    "--out_file", OUT_FILE,
    "--workers", "24",
]

for attempt in range(1, MAX_ATTEMPTS + 1):
    if results_complete():
        break
    print(f"[supervisor] attempt {attempt}/{MAX_ATTEMPTS} starting {time.strftime('%H:%M:%S')}", flush=True)
    proc = subprocess.run(CMD)
    print(f"[supervisor] tuner exited code={proc.returncode} at {time.strftime('%H:%M:%S')}", flush=True)
    if proc.returncode == 0 and results_complete():
        break
    time.sleep(5)

if results_complete():
    print("[supervisor] Pipeline execution fully complete", flush=True)
else:
    print("[supervisor] FAILED: tuner did not complete all pairs after retries", flush=True)
    sys.exit(1)

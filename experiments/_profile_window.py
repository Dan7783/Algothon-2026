"""Profile a single windowed-tuner objective evaluation."""
# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import cProfile
import io
import os
import pstats
import time

os.environ["TUNER_Z_ENTRY_RULE"] = "new"

import latest_tuner
import windowed_tuner as wt

latest_tuner.GLOBAL_CONFIG["module"] = "testbedv3"

cfg = dict(wt.WINDOWED_CFG)
cfg["db_file"] = "auto_tuning_results/optuna_profile_probe.db"


def objective(trial):
    params = wt.suggest_params(trial, "RTTH", "LSST")
    return wt.eval_window_score(params, 201, 342)


t0 = time.perf_counter()
pr = cProfile.Profile()
pr.enable()
study = wt.optimize_study(
    "probe-RTTH-LSST-w0", objective, total_trials=60, sobol_trials=32,
    n_jobs=1, db_file=cfg["db_file"], show_progress=False,
)
pr.disable()
wall = time.perf_counter() - t0

s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(28)
print(s.getvalue())
print(f"WALL: {wall:.1f}s for 60 trials = {wall / 60 * 1000:.0f} ms/trial")

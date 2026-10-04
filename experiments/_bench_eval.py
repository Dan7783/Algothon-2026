"""Benchmark one official-engine evaluation of a testbedv3 pair parameter set."""
# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import time

import latest_tuner
from latest_tuner import loadPrices, run_official_eval, GLOBAL_CONFIG, pricesFile

GLOBAL_CONFIG["module"] = "testbedv3"
import os
os.environ["TUNER_Z_ENTRY_RULE"] = "new"

params = {
    "pair_y": "RTTH", "pair_x": "LSST",
    "q_beta": 5.43e-09, "h": 16.4, "cusum_k": 1.31, "max_cov_inflate": 273.0,
    "cov_decay": 0.00554, "kalman_var_ewma_alpha": 0.00575, "ou_window": 126,
    "kappa_ewma_alpha": 0.186, "sigma_ewma_alpha": 0.00551,
}

# price load cost (what every trial pays)
t0 = time.perf_counter()
for _ in range(3):
    loadPrices(pricesFile)
t_load = (time.perf_counter() - t0) / 3

# full trial eval, short window (Phase A shape)
t0 = time.perf_counter()
for _ in range(3):
    run_official_eval(params, start_day=201, end_day=342)
t_win = (time.perf_counter() - t0) / 3

# full trial eval, train span (Phase C global shape)
t0 = time.perf_counter()
run_official_eval(params, start_day=201, end_day=800)
t_glob = time.perf_counter() - t0

print(f"loadPrices alone:      {t_load * 1000:8.1f} ms")
print(f"trial eval 201-342:    {t_win * 1000:8.1f} ms  (backtest = {(t_win - t_load) * 1000:.1f} ms)")
print(f"trial eval 201-800:    {t_glob * 1000:8.1f} ms  (backtest = {(t_glob - t_load) * 1000:.1f} ms)")

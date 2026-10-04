"""Individual success evaluation for the 80-pair OU search.

Reads the completed Optuna studies from the temp WAL db and reports, per pair:
  - best walk-forward tuned value (in-sample tuning objective)
  - tuned OOS score via the tuner's own OOS evaluation, if available
  - pass/fail: OOS > 0 (individually profitable on the new engine)

Individual profitability on the OU engine = tuning success per the user's
success criterion. Portfolio integration is handled separately (greedy/conflict).
"""

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import json
import os
import sys
import tempfile

import optuna

DB = os.path.join(tempfile.gettempdir(), "optuna_ou_search.db")
PAIRS_FILE = "auto_tuning_results/ou_search_pairs.json"
OUT = "auto_tuning_results/ou_search_individual.json"

optuna.logging.set_verbosity(optuna.logging.WARNING)
storage = optuna.storages.RDBStorage(f"sqlite:///{DB}")

with open(PAIRS_FILE) as f:
    pairs = json.load(f)

results = []
for row in pairs:
    y, x = row["y"], row["x"]
    study_name = row.get("study_name") or f"{y}_{x}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        results.append({"y": y, "x": x, "error": str(e)})
        continue
    trials = [t for t in study.trials if t.value is not None]
    best = study.best_trial
    results.append({
        "y": y, "x": x,
        "n_trials": len(trials),
        "best_value": best.value,
        "best_params": best.params,
    })

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

done = [r for r in results if "best_value" in r]
pos = [r for r in done if r["best_value"] > 0]
print(f"pairs with completed studies: {len(done)}/{len(pairs)}")
print(f"individually positive (best tuning value > 0): {len(pos)}")
for r in sorted(done, key=lambda r: -r["best_value"])[:15]:
    print(f"  {r['y']}_{r['x']}: {r['best_value']:.2f}")

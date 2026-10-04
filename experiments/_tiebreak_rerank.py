#!/usr/bin/env python
"""Tiebreak selection test: re-rank the top-M trials of each pair's global study by
latest_tuner's exact purged walk-forward objective (mean - std, 4 folds, embargo 10),
then evaluate the winner on the full train range and OOS.

This answers: "same candidate pool, but selection by the purged-WF robustness criterion
instead of argmax of the normalized objective + perturbation gate — does it generalize
better?" It reuses an existing windowed global study, so no re-optimization is needed.

Example:
  python _tiebreak_rerank.py --db_file auto_tuning_results/_bakeoff_wcurrent.db \
      --pairs_file experiments/_bakeoff_pairs.json --out_file auto_tuning_results/_bakeoff_wtiebreak.json
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

import numpy as np
import optuna

import latest_tuner  # noqa: F401  (module import wires up the strategy/eval machinery)
from latest_tuner import evaluate_purged_walk_forward, run_official_eval
from windowed_tuner import get_storage, clean_trial_params

optuna.logging.set_verbosity(optuna.logging.WARNING)

CFG = {
    "start": 201, "end": 800, "oos_start": 801, "oos_end": 1000,
    "top_m": 20, "n_folds": 4, "fold_embargo": 10,  # identical to latest_tuner's objective
    "study_suffix": "globalnorm-percentile-std1",
    "db_file": os.path.join("auto_tuning_results", "_bakeoff_wcurrent.db"),
    "pairs_file": "experiments/_bakeoff_pairs.json",
    "out_file": os.path.join("auto_tuning_results", "_bakeoff_wtiebreak.json"),
}


def metrics(params, s, e):
    meanpl, ret, plstd, sharpe, dvol, days, pll, sc = run_official_eval(params, start_day=s, end_day=e)
    return {"meanpl": meanpl, "sharpe": sharpe, "score": sc,
            "ret": ret, "total_return": float(np.sum(pll))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db_file", default=CFG["db_file"])
    p.add_argument("--pairs_file", default=CFG["pairs_file"])
    p.add_argument("--out_file", default=CFG["out_file"])
    p.add_argument("--study_suffix", default=CFG["study_suffix"])
    p.add_argument("--top_m", type=int, default=CFG["top_m"])
    args = p.parse_args()

    os.environ.setdefault("TUNER_Z_ENTRY_RULE", "new")
    pairs = json.load(open(args.pairs_file))
    results = []
    for pair in pairs:
        y, x = pair["y"], pair["x"]
        name = f"{y}-{x}"
        study_name = f"{name}-{args.study_suffix}"
        # "{pair}" in db_file resolves to that pair's own DB (per-pair bake-off layout),
        # so no slow study-merge into one big DB is needed.
        db = args.db_file.replace("{pair}", name)
        study = optuna.load_study(study_name=study_name, storage=get_storage(db))
        done = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        top = sorted(done, key=lambda t: t.value, reverse=True)[:args.top_m]

        ranked = []
        for t in top:
            params = clean_trial_params(t.params, y, x)
            wf = evaluate_purged_walk_forward(params, CFG["start"], CFG["end"],
                                              n_folds=CFG["n_folds"], embargo=CFG["fold_embargo"])
            ranked.append({"trial": t.number, "objective": t.value, "wf_score": wf,
                           "params": params})
        ranked.sort(key=lambda r: r["wf_score"], reverse=True)
        winner = ranked[0]
        orig_rank = next(i for i, t in enumerate(top) if t.number == winner["trial"]) + 1

        train_m = metrics(winner["params"], CFG["start"], CFG["end"])
        oos_m = metrics(winner["params"], CFG["oos_start"], CFG["oos_end"])
        print(f"{name}: trial {winner['trial']} (obj rank {orig_rank}/{len(top)}, "
              f"wf {winner['wf_score']:.3f}) -> train {train_m['score']:.2f} "
              f"(S {train_m['sharpe']:.2f}) / OOS {oos_m['score']:.2f} (S {oos_m['sharpe']:.2f})",
              flush=True)

        results.append({
            "pair_name": name, "final_params": winner["params"],
            "tiebreak": {"top_m": len(top), "chosen_trial": winner["trial"],
                         "objective_rank": orig_rank, "wf_score": winner["wf_score"],
                         "ranked": [{"trial": r["trial"], "objective": r["objective"],
                                     "wf_score": r["wf_score"]} for r in ranked]},
            "train_metrics": train_m, "oos_metrics": oos_m,
        })

    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved tiebreak results to {args.out_file}")


if __name__ == "__main__":
    main()

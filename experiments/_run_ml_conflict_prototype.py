#!/usr/bin/env python
"""Train/eval non-bang-bang conflict prototypes (in-sample first).

1) Collect conflict features under shared voting on train days 31-500
2) Label = sign of next-day asset return (among {-1,0,1} with tiny deadzone)
3) Fit sklearn HistGradientBoostingClassifier on train only; validate 501-750
4) Eval book scores: friend vs shared vs flat_conflict vs stronger_vote vs ml
   on 31-750 / 501-750 / 750-1000 (OOS verify only — never tune on OOS)

Uses sklearn (xgboost not installed). Pins MERGED_* in subprocess evals.
"""
from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "auto_tuning_results"
MODEL_PATH = RESULTS / "_ml_conflict_model.joblib"
OUT = RESULTS / "_ml_conflict_prototype.json"

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

# Fit window (strict in-sample); val for model selection among fixed candidates only.
FIT_START, FIT_END = 31, 500
VAL_START, VAL_END = 501, 750
TRAIN_BOOK, TRAIN_BOOK_END = 31, 750
OOS_START, OOS_END = 750, 1000
DEADZONE = 1e-4  # |return| below → label 0


def clear_merged(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for k in MERGED_KEYS:
        env.pop(k, None)
    if extra:
        env.update(extra)
    return env


def pin(extra: dict[str, str] | None = None) -> None:
    for k in MERGED_KEYS:
        os.environ.pop(k, None)
    if extra:
        os.environ.update(extra)


def load_mod(extra: dict[str, str]):
    pin(extra)
    for name in list(sys.modules):
        if name == "merged_friend_ou" or name.startswith("merged_friend_ou."):
            del sys.modules[name]
    # Reset cached model handle if module reloaded
    return importlib.import_module("merged_friend_ou")


def load_prices():
    import latest_tuner

    prices, _nt, _ninst = latest_tuner.loadPrices("prices.txt")
    return np.asarray(prices, dtype=float)


def collect_conflict_dataset(mod, prices, start: int, end: int):
    """Rows where pair/ridge conflict; features + next-day return label."""
    mod.reset()
    X, y, meta = [], [], []
    for t in range(start, end - 1):
        prc = prices[:, : t + 1]
        sleeves = mod.sleeve_breakdown(prc)
        if sleeves is None:
            continue
        conflict = np.asarray(sleeves["conflict_mask"], dtype=bool)
        idx = np.flatnonzero(conflict)
        if idx.size == 0:
            continue
        ou_cands, zb_cands = mod._split_engine_candidates(
            mod._STATE["atlas"].candidates(prc)
        )
        ou_vote, _, _ = mod._pair_conflict_consensus(ou_cands)
        zb_vote, _, _ = mod._pair_conflict_consensus(zb_cands)
        feats = mod.ml_conflict_feature_matrix(
            np.asarray(sleeves["pair_vote"], dtype=float),
            np.asarray(sleeves["pair_active_mask"], dtype=bool),
            np.asarray(sleeves["pair_contributors"], dtype=float),
            np.sign(np.asarray(sleeves["ridge_raw"], dtype=float)),
            np.asarray(sleeves["ridge_strength"], dtype=float),
            np.asarray(sleeves["ridge_active_mask"], dtype=bool),
            np.asarray(sleeves["ridge_confirmation_sign"], dtype=float),
            np.asarray(sleeves["ridge_confirmation_active_mask"], dtype=bool),
            ou_vote,
            zb_vote,
            np.asarray(sleeves["combined_vote"], dtype=float),
            conflict,
        )
        rets = (prices[:, t + 1] - prices[:, t]) / np.maximum(prices[:, t], 1e-12)
        for i in idx:
            r = float(rets[i])
            if abs(r) < DEADZONE:
                lab = 0
            else:
                lab = int(np.sign(r))
            X.append(feats[i])
            y.append(lab)
            meta.append({"t": t, "asset": int(i), "ret": r})
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), meta


def run_eval(module: str, start: str, end: str, extra: dict[str, str]) -> float | None:
    p = subprocess.run(
        [sys.executable, "eval.py", "--module", module, "--start", start, "--end", end],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=clear_merged(extra),
    )
    m = re.search(r"Score:\s*([-\d.]+)", p.stdout + "\n" + p.stderr)
    return float(m.group(1)) if m else None


def three_windows(module: str, extra: dict[str, str]) -> dict:
    return {
        "train_31_750": run_eval(module, "31", "750", extra),
        "val_501_750": run_eval(module, "501", "750", extra),
        "oos_750_1000_verify": run_eval(module, "750", "1000", extra),
    }


def main() -> None:
    prices = load_prices()
    # Prefer mean-exit for conflict ease; margin 0.15 ship default.
    base_extra = {
        "MERGED_OU_EXIT": "mean",
        "MERGED_CONSENSUS_MARGIN": "0.15",
        "MERGED_OU_CONFLICT": "shared",
    }
    print("Collecting conflict training rows (shared/mean) ...", flush=True)
    mod = load_mod(base_extra)
    X_fit, y_fit, _ = collect_conflict_dataset(mod, prices, FIT_START, FIT_END)
    print(f"  fit rows={len(y_fit)} class_counts={np.bincount(y_fit + 1)}", flush=True)
    mod = load_mod(base_extra)
    X_val, y_val, _ = collect_conflict_dataset(mod, prices, VAL_START, VAL_END)
    print(f"  val rows={len(y_val)} class_counts={np.bincount(y_val + 1)}", flush=True)

    report: dict = {
        "fit_window": [FIT_START, FIT_END],
        "val_window": [VAL_START, VAL_END],
        "n_fit": int(len(y_fit)),
        "n_val": int(len(y_val)),
        "deps": {"sklearn": True, "xgboost": False, "note": "HistGradientBoostingClassifier"},
        "rejected_oos_overfit": "_mean_exit_best_catalog.json / 1057 path",
    }

    if len(y_fit) < 50:
        report["error"] = "too few conflict rows to train"
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report["error"], flush=True)
        return

    # Fixed hyperparams chosen a priori (no OOS tuning). Light grid on VAL only.
    candidates = [
        {"max_depth": 3, "learning_rate": 0.05, "max_iter": 100},
        {"max_depth": 4, "learning_rate": 0.08, "max_iter": 150},
        {"max_depth": 5, "learning_rate": 0.1, "max_iter": 200},
    ]
    best = None
    best_val = -1.0
    for hp in candidates:
        clf = HistGradientBoostingClassifier(
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            max_iter=hp["max_iter"],
            random_state=0,
        )
        clf.fit(X_fit, y_fit)
        val_acc = float(accuracy_score(y_val, clf.predict(X_val))) if len(y_val) else 0.0
        fit_acc = float(accuracy_score(y_fit, clf.predict(X_fit)))
        print(f"  hp={hp} fit_acc={fit_acc:.3f} val_acc={val_acc:.3f}", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best = (clf, hp, fit_acc, val_acc)

    clf, hp, fit_acc, val_acc = best
    joblib.dump(clf, MODEL_PATH)
    report["model_path"] = str(MODEL_PATH)
    report["chosen_hp"] = hp
    report["fit_acc"] = fit_acc
    report["val_acc"] = val_acc
    print(f"wrote model {MODEL_PATH}", flush=True)

    # Book bake-off — train primary
    cases = [
        ("friend_book_latest", "friend_book_latest", {}),
        (
            "merged_shared_mean_m015",
            "merged_friend_ou",
            {"MERGED_OU_EXIT": "mean", "MERGED_CONSENSUS_MARGIN": "0.15", "MERGED_OU_CONFLICT": "shared"},
        ),
        (
            "merged_optimal_m015",
            "merged_friend_ou",
            {"MERGED_OU_EXIT": "optimal", "MERGED_CONSENSUS_MARGIN": "0.15", "MERGED_OU_CONFLICT": "shared"},
        ),
        (
            "flat_conflict_mean_m015",
            "merged_friend_ou",
            {
                "MERGED_OU_EXIT": "mean",
                "MERGED_CONSENSUS_MARGIN": "0.15",
                "MERGED_OU_CONFLICT": "flat_conflict",
            },
        ),
        (
            "stronger_vote_mean_m015",
            "merged_friend_ou",
            {
                "MERGED_OU_EXIT": "mean",
                "MERGED_CONSENSUS_MARGIN": "0.15",
                "MERGED_OU_CONFLICT": "stronger_vote",
            },
        ),
        (
            "ml_conflict_mean_m015",
            "merged_friend_ou",
            {
                "MERGED_OU_EXIT": "mean",
                "MERGED_CONSENSUS_MARGIN": "0.15",
                "MERGED_OU_CONFLICT": "ml",
                "MERGED_ML_MODEL": str(MODEL_PATH),
            },
        ),
        (
            "REJECTED_oos_prune_1057",
            "merged_friend_ou",
            {
                "MERGED_OU_EXIT": "mean",
                "MERGED_CONSENSUS_MARGIN": "0.20",
                "MERGED_OU_CONFLICT": "shared",
                "MERGED_CATALOG_JSON": str(RESULTS / "_mean_exit_best_catalog.json"),
            },
        ),
    ]

    scores = {}
    print("=== book bake-off (train primary) ===", flush=True)
    for label, module, extra in cases:
        sc = three_windows(module, extra)
        scores[label] = sc
        print(
            f"  {label:28} train={sc['train_31_750']} val={sc['val_501_750']} "
            f"oos_verify={sc['oos_750_1000_verify']}",
            flush=True,
        )

    report["book_scores"] = scores
    friend_tr = scores["friend_book_latest"]["train_31_750"]
    beaters = {
        k: v
        for k, v in scores.items()
        if k != "friend_book_latest"
        and v["train_31_750"] is not None
        and friend_tr is not None
        and v["train_31_750"] > friend_tr
    }
    report["beats_friend_train"] = sorted(
        beaters.keys(),
        key=lambda k: scores[k]["train_31_750"],
        reverse=True,
    )
    report["friend_targets"] = {"train": 926.94, "val": 1042.37, "oos": 974.45}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    print(f"BEATS_FRIEND_TRAIN={report['beats_friend_train']}", flush=True)


if __name__ == "__main__":
    main()

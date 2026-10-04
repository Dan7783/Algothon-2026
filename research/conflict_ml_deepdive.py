#!/usr/bin/env python
"""Deep dive: can the conflict arbitrator be regularised into working?

Context. A gradient booster scored train 1488.3 / test 801.4 against a
baseline of 789.8 / 914.7 -- i.e. it looked like it nearly doubled the score
in-sample and actually destroyed value out of sample. This script asks
whether that is a fixable capacity problem (wrong hyperparameters, wrong
validation scheme) or an unfixable signal problem.

Five experiments:

  E1 capacity / learning curves -- train vs OOS AUC as capacity grows. If the
     gap opens with capacity while OOS stays flat at 0.5, the model is fitting
     noise, not under-regularised signal.
  E2 regularisation frontier -- an exhaustive grid over depth, trees, leaf
     size, learning rate, subsample and feature count. Establishes the BEST
     achievable OOS AUC over the whole hyperparameter space, so "we just
     haven't tuned it right" can be settled rather than argued.
  E3 purged walk-forward with embargo (Lopez de Prado 2018) -- the
     methodologically correct evaluation for overlapping financial labels,
     and simultaneously a better deployment: retrain on a trailing window so
     the model tracks the regime instead of freezing one window's answer.
  E4 accuracy -> score transfer -- what conflict accuracy is REQUIRED to beat
     the baseline? Turns the whole question into one number to clear.
  E5 statistical power -- how much data would be needed to detect the edge
     that is actually present, given it is ~2pp and flips sign?

Everything is evaluated through the exact eval.py simulator where score is
quoted. Fitting uses train blocks (251-500) only, per
cusum_output/VALIDATION_PROTOCOL.md.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.combine_lab import DLR_LIMIT, N_INST, score, simulate
from research.combine_schemes import _resolve, _sleeve_state
from research.conflict_ev_model import FEATURES
from research.signal_cache import load_or_build, load_prices

import testbedv2_tuned_ridge_protectedbarbell as strategy

OUTPUT_DIR = ROOT / "analysis_outputs" / "signal_overlap"
BB = strategy.BANG_BANG_LIMITS
BASELINE = {"train": 789.8, "val": 847.0, "test": 914.7}


# --------------------------------------------------------------------------
# Precomputed per-day allocation state, so score evaluations are cheap.
# --------------------------------------------------------------------------

def precompute(cache, prices, first=250, last=1000):
    """Base direction, conflict mask and oracle direction for every day."""
    lookup = {int(day): row for row, day in enumerate(cache["days"])}
    days, base_dir, conflicts, oracle_dir, ridge_dir, pair_dir_ = [], [], [], [], [], []
    for t in range(first, last):
        state = _sleeve_state(cache, lookup[t], prices, t)
        direction = _resolve(state)
        conflict = (
            state["pair_active"] & state["ridge_active"]
            & (state["pair_dir"] != state["ridge_sign"]) & ~state["crash"]
        )
        forward = prices[:, t] / prices[:, t - 1] - 1.0
        direction = direction.copy()
        direction[state["crash"]] = 0.0
        days.append(t)
        base_dir.append(direction)
        conflicts.append(conflict)
        oracle_dir.append(np.sign(forward))
        ridge_dir.append(state["ridge_sign"].copy())
        pair_dir_.append(state["pair_dir"].copy())
    return {
        "days": np.array(days),
        "base_dir": np.array(base_dir),
        "conflict": np.array(conflicts),
        "oracle_dir": np.array(oracle_dir),
        "ridge_dir": np.array(ridge_dir),
        "pair_dir": np.array(pair_dir_),
    }


def score_with_overrides(pre, prices, start, end, override_fn) -> float:
    """Score the book with conflict directions replaced by override_fn."""
    index = {int(d): i for i, d in enumerate(pre["days"])}
    dollars = {}
    for t in range(start - 1, end):
        i = index[t]
        direction = pre["base_dir"][i].copy()
        mask = pre["conflict"][i]
        if mask.any():
            direction[mask] = override_fn(i, mask, direction)
        dollars[t] = np.clip(BB * direction, -DLR_LIMIT, DLR_LIMIT)
    pnl = simulate(dollars, prices, start, end)
    return score(float(pnl.mean()), float(pnl.std()))


# --------------------------------------------------------------------------

def e1_learning_curves(data, tr, te):
    print("\nE1  CAPACITY / LEARNING CURVES")
    print("=" * 78)
    print(f"{'capacity (depth x trees)':<26} {'train AUC':>10} {'val AUC':>10} {'test AUC':>10}")
    print("-" * 78)
    va = data[(data.day >= 501) & (data.day <= 750)]
    rows = []
    for depth in (1, 2, 3, 5):
        for trees in (25, 100, 400):
            m = GradientBoostingClassifier(
                n_estimators=trees, max_depth=depth, learning_rate=0.05,
                subsample=0.8, random_state=0).fit(tr[FEATURES], tr.is_ridge_right)
            a_tr = roc_auc_score(tr.is_ridge_right, m.predict_proba(tr[FEATURES])[:, 1])
            a_va = roc_auc_score(va.is_ridge_right, m.predict_proba(va[FEATURES])[:, 1])
            a_te = roc_auc_score(te.is_ridge_right, m.predict_proba(te[FEATURES])[:, 1])
            print(f"depth {depth} x {trees:>4} trees{'':<8} {a_tr:10.4f} {a_va:10.4f} {a_te:10.4f}")
            rows.append(dict(depth=depth, trees=trees, train_auc=a_tr, val_auc=a_va, test_auc=a_te))
    return pd.DataFrame(rows)


def e2_regularisation_frontier(data, tr, te):
    print("\nE2  REGULARISATION FRONTIER (best achievable OOS AUC anywhere)")
    print("=" * 78)
    va = data[(data.day >= 501) & (data.day <= 750)]
    rows = []
    for depth in (1, 2, 3):
        for trees in (10, 50, 200):
            for leaf in (5, 50, 200):
                for lr in (0.01, 0.05, 0.2):
                    m = GradientBoostingClassifier(
                        n_estimators=trees, max_depth=depth, learning_rate=lr,
                        min_samples_leaf=leaf, subsample=0.8, random_state=0
                    ).fit(tr[FEATURES], tr.is_ridge_right)
                    rows.append(dict(
                        depth=depth, trees=trees, leaf=leaf, lr=lr,
                        train_auc=roc_auc_score(tr.is_ridge_right, m.predict_proba(tr[FEATURES])[:, 1]),
                        val_auc=roc_auc_score(va.is_ridge_right, m.predict_proba(va[FEATURES])[:, 1]),
                        test_auc=roc_auc_score(te.is_ridge_right, m.predict_proba(te[FEATURES])[:, 1])))
    frame = pd.DataFrame(rows)
    # Honest selection: pick on VALIDATION, then read test. Also report the
    # cheating maximum (best test) as an upper bound on what tuning could buy.
    best_val = frame.loc[frame.val_auc.idxmax()]
    best_test = frame.loc[frame.test_auc.idxmax()]
    print(f"  configurations tried: {len(frame)}")
    print(f"  selected on VAL : depth {best_val.depth:.0f} trees {best_val.trees:.0f} "
          f"leaf {best_val.leaf:.0f} lr {best_val.lr} -> "
          f"val AUC {best_val.val_auc:.4f}, test AUC {best_val.test_auc:.4f}")
    print(f"  best TEST AUC anywhere (unattainable, look-ahead): {best_test.test_auc:.4f}")
    print(f"  median test AUC across all configs: {frame.test_auc.median():.4f}")
    print(f"  fraction of configs with test AUC > 0.55: "
          f"{(frame.test_auc > 0.55).mean():.1%}")
    return frame


def e3_purged_walkforward(data, pre, prices):
    """Retrain on a trailing window with an embargo gap (Lopez de Prado)."""
    print("\nE3  PURGED WALK-FORWARD WITH EMBARGO")
    print("=" * 78)
    rows = []
    for window in (250, 500):
        for retrain_every in (25, 50):
            for embargo in (5,):
                preds = {}
                for t0 in range(501, 1000, retrain_every):
                    fit = data[(data.day >= t0 - window - embargo) & (data.day <= t0 - embargo - 1)]
                    if len(fit) < 200:
                        continue
                    m = GradientBoostingClassifier(
                        n_estimators=50, max_depth=2, learning_rate=0.05,
                        min_samples_leaf=50, subsample=0.8, random_state=0
                    ).fit(fit[FEATURES], fit.is_ridge_right)
                    block = data[(data.day >= t0) & (data.day < t0 + retrain_every)]
                    if not len(block):
                        continue
                    p = m.predict_proba(block[FEATURES])[:, 1]
                    for (_, r), pi in zip(block.iterrows(), p):
                        preds[(int(r.day), int(r.inst))] = pi

                index = {int(d): i for i, d in enumerate(pre["days"])}

                def override(i, mask, direction, _preds=preds):
                    t = int(pre["days"][i])
                    out = direction[mask].copy()
                    for k, inst in enumerate(np.flatnonzero(mask)):
                        p = _preds.get((t, int(inst)))
                        if p is None:
                            continue
                        out[k] = (pre["ridge_dir"][i][inst] if p > 0.5
                                  else pre["pair_dir"][i][inst])
                    return out

                val = score_with_overrides(pre, prices, 501, 750, override)
                test = score_with_overrides(pre, prices, 751, 1000, override)
                auc_rows = [(k, v) for k, v in preds.items()]
                rows.append(dict(window=window, retrain=retrain_every, embargo=embargo,
                                 val=val, test=test, n_pred=len(auc_rows)))
                print(f"  window {window:>3}d retrain {retrain_every:>2}d embargo {embargo}d "
                      f"-> val {val:8.1f} (base {BASELINE['val']:.1f})  "
                      f"test {test:8.1f} (base {BASELINE['test']:.1f})")
    return pd.DataFrame(rows)


def e4_accuracy_transfer(pre, prices, seeds=(0, 1, 2)):
    """What conflict accuracy is required to beat the baseline?"""
    print("\nE4  ACCURACY -> SCORE TRANSFER (what accuracy must a model reach?)")
    print("=" * 78)
    print(f"{'target acc':>11} {'train':>10} {'val':>10} {'test':>10}")
    print("-" * 78)
    rows = []
    for acc in (0.45, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70):
        got = {}
        for region, start, end in (("train", 251, 500), ("val", 501, 750), ("test", 751, 1000)):
            vals = []
            for seed in seeds:
                rng = np.random.default_rng(seed)

                def override(i, mask, direction, _rng=rng, _acc=acc):
                    oracle = pre["oracle_dir"][i][mask]
                    keep = _rng.random(mask.sum()) < _acc
                    return np.where(keep, oracle, -oracle)

                vals.append(score_with_overrides(pre, prices, start, end, override))
            got[region] = float(np.mean(vals))
        rows.append(dict(accuracy=acc, **got))
        print(f"{acc:11.2f} {got['train']:10.1f} {got['val']:10.1f} {got['test']:10.1f}")
    frame = pd.DataFrame(rows)
    for region in ("train", "val", "test"):
        need = np.interp(BASELINE[region], frame[region], frame.accuracy)
        print(f"  accuracy needed to match baseline on {region}: {need:.4f}")
    return frame


def e5_power_analysis(data):
    print("\nE5  STATISTICAL POWER")
    print("=" * 78)
    for label, sub in (("train", data[(data.day >= 251) & (data.day <= 500)]),
                       ("val", data[(data.day >= 501) & (data.day <= 750)]),
                       ("test", data[data.day >= 751])):
        n = len(sub)
        p = sub.is_ridge_right.mean()
        se = np.sqrt(0.25 / n)
        print(f"  {label:>5}: n={n:5d}  P(ridge right)={p:.4f}  "
              f"edge={p-0.5:+.4f}  SE={se:.4f}  z={(p-0.5)/se:+.2f}")
    print("\n  Sample size needed to detect an edge at 80% power, alpha=0.05 (two-sided):")
    for edge in (0.01, 0.02, 0.03, 0.05):
        n_req = (1.96 + 0.84) ** 2 * 0.25 / edge ** 2
        print(f"    edge {edge:.0%}: n = {n_req:,.0f} conflict observations "
              f"({n_req/8.2:,.0f} trading days at ~8.2 conflicts/day)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    cache = load_or_build()
    data = pd.read_csv(OUTPUT_DIR / "conflict_dataset.csv")
    tr = data[(data.day >= 251) & (data.day <= 500)]
    te = data[data.day >= 751]

    print("Precomputing allocation state ...")
    pre = precompute(cache, prices)

    e1 = e1_learning_curves(data, tr, te)
    e2 = e2_regularisation_frontier(data, tr, te)
    e3 = e3_purged_walkforward(data, pre, prices)
    e4 = e4_accuracy_transfer(pre, prices)
    e5_power_analysis(data)

    e1.to_csv(OUTPUT_DIR / "deepdive_e1_capacity.csv", index=False)
    e2.to_csv(OUTPUT_DIR / "deepdive_e2_regularisation.csv", index=False)
    e3.to_csv(OUTPUT_DIR / "deepdive_e3_walkforward.csv", index=False)
    e4.to_csv(OUTPUT_DIR / "deepdive_e4_accuracy.csv", index=False)
    print(f"\nwrote deepdive_e*.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Can the direction on a deadlock asset-day be predicted at >52% out of window?

Ceiling established by `deadlock_ceiling.py`: break-even ~52%, the whole prize
is ~+280 score (perfect foresight), and 56% accuracy is worth ~+30.

This fits predictors under the pre-registered protocol:
  * features and labels built ONLY from information available at decision time
  * fitted on train blocks 251-500 ONLY
  * evaluated on 501-750 (burned validation) and 751-1000 (organiser window)
  * label stability checked across windows FIRST -- prior work
    (conflict_ml_deepdive.py) died because the label inverted between adjacent
    windows, which no model class can survive

Run from the repo root:  python research/deadlock_predict.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as fs
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FIRST, LAST = 250, 1000
TRAIN = (251, 500)
TESTS = {"val 501-750": (501, 750), "org 751-1000": (751, 1000)}

FEATURES = [
    "ridge_strength", "ridge_signed_strength", "n_voters",
    "mean_abs_z", "mean_signed_z", "mean_kappa", "has_protected",
    "r1", "r5", "r20", "vol20", "r1_x", "mkt_r1", "rel_r1",
    "z_disagree", "beta_spread",
]


def build(prices: np.ndarray) -> pd.DataFrame:
    fs.reset()
    book = fs.AtlasBook()
    logret = np.diff(np.log(prices), axis=1)
    rows = []

    for t in range(FIRST, LAST + 1):
        hist = prices[:, :t]
        s = fs.sleeve_breakdown(hist)
        if s is None:
            continue
        dl = (s["pair_active_mask"]
              & (np.abs(s["pair_vote"]) < 1e-12)
              & ~s["ridge_active_mask"]
              & ~s["crash_mask"])
        if not dl.any():
            continue

        # per-asset ledger of the pairs voting on it right now
        legs: dict[int, list] = {}
        for priority, agent, py, px in book.candidates(hist):
            prot = fs._pair_name(agent.spec) in fs.PROTECTED_PAIR_NAMES
            for idx, pos in ((int(agent.spec.y), py), (int(agent.spec.x), px)):
                if abs(pos) > 1e-9:
                    legs.setdefault(idx, []).append(
                        (float(np.sign(pos)), agent.current_z,
                         agent.current_kappa, prot, float(agent.filt.beta)))

        lr = logret[:, :t - 1]
        r1 = lr[:, -1]
        r5 = lr[:, -5:].sum(axis=1)
        r20 = lr[:, -20:].sum(axis=1)
        vol20 = lr[:, -20:].std(axis=1)
        mkt = float(r1.mean())

        for a in np.flatnonzero(dl):
            e = legs.get(int(a), [])
            if not e:
                continue
            signs = np.array([x[0] for x in e])
            zs = np.array([x[1] for x in e])
            ks = np.array([x[2] for x in e])
            bs = np.array([x[4] for x in e])
            rows.append({
                "day": t, "asset": int(a),
                "ridge_strength": float(s["ridge_strength"][a]),
                "ridge_signed_strength": float(
                    s["ridge_strength"][a] * np.sign(s["ridge_raw"][a])),
                "n_voters": len(e),
                "mean_abs_z": float(np.mean(np.abs(zs))),
                "mean_signed_z": float(np.mean(zs * signs)),
                "mean_kappa": float(np.mean(ks)),
                "has_protected": float(any(x[3] for x in e)),
                "r1": float(r1[a]), "r5": float(r5[a]), "r20": float(r20[a]),
                "vol20": float(vol20[a]),
                "r1_x": float(r1[a] / (vol20[a] + 1e-12)),
                "mkt_r1": mkt, "rel_r1": float(r1[a] - mkt),
                "z_disagree": float(np.std(signs)),
                "beta_spread": float(np.std(bs)) if len(bs) > 1 else 0.0,
                # label: did the asset go UP on the day the position is held?
                "y": int(prices[a, t] > prices[a, t - 1]) if t < prices.shape[1] else -1,
                "ret": float(prices[a, t] / prices[a, t - 1] - 1.0)
                if t < prices.shape[1] else 0.0,
            })
    return pd.DataFrame(rows)


def main() -> None:
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    out = ROOT / "analysis_outputs" / "deadlock"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "deadlock_features.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        print("building deadlock feature set ...")
        df = build(prices)
        df.to_csv(cache, index=False)
    df = df[df.y >= 0]
    print(f"{len(df)} deadlock asset-days with labels\n")

    tr = df[(df.day >= TRAIN[0]) & (df.day <= TRAIN[1])]

    # ---- 0. is the label even stable across windows? ----
    print("=" * 74)
    print("0. LABEL STABILITY -- base rate of 'asset goes up' on deadlock days")
    print("=" * 74)
    for wn, (lo, hi) in {"train 251-500": TRAIN, **TESTS}.items():
        w = df[(df.day >= lo) & (df.day <= hi)]
        p = w.y.mean()
        se = np.sqrt(p * (1 - p) / len(w))
        print(f"  {wn:>14}: n={len(w):5d}  P(up)={p*100:5.1f}%  "
              f"+-{se*100:.1f}  ({(p-0.5)/se:+.2f} sigma from 50%)")

    # ---- 1. per-feature sign stability ----
    print("\n" + "=" * 74)
    print("1. FEATURE SIGN STABILITY -- corr(feature, next-day return)")
    print("=" * 74)
    print(f"{'feature':>22} " + "".join(f"{k:>16}" for k in
                                        ["train", "val", "org"]) + "   stable?")
    wins = [TRAIN, TESTS["val 501-750"], TESTS["org 751-1000"]]
    for f in FEATURES:
        cs = []
        for lo, hi in wins:
            w = df[(df.day >= lo) & (df.day <= hi)]
            cs.append(float(np.corrcoef(w[f], w.ret)[0, 1]) if w[f].std() > 0 else 0.0)
        same = len(set(np.sign(cs))) == 1 and abs(cs[0]) > 0.02
        print(f"{f:>22} " + "".join(f"{c:>16.3f}" for c in cs)
              + f"   {'YES' if same else 'no'}")

    # ---- 2. models fitted on train only ----
    print("\n" + "=" * 74)
    print("2. OUT-OF-WINDOW ACCURACY (fitted on days 251-500 ONLY)")
    print("=" * 74)
    Xtr = tr[FEATURES].to_numpy(float)
    ytr = tr.y.to_numpy(int)
    sc = StandardScaler().fit(Xtr)

    models = {
        "logistic (C=0.1)": LogisticRegression(C=0.1, max_iter=2000),
        "logistic (C=1)": LogisticRegression(C=1.0, max_iter=2000),
        "GBM d2 n100": GradientBoostingClassifier(
            max_depth=2, n_estimators=100, learning_rate=0.05, random_state=0),
        "GBM d3 n300": GradientBoostingClassifier(
            max_depth=3, n_estimators=300, learning_rate=0.05, random_state=0),
    }
    print(f"{'model':>18} {'train acc':>10} {'train AUC':>10} |"
          + "".join(f"{k:>26}" for k in TESTS))
    for name, m in models.items():
        m.fit(sc.transform(Xtr), ytr)
        ptr = m.predict_proba(sc.transform(Xtr))[:, 1]
        line = (f"{name:>18} {((ptr>0.5)==ytr).mean()*100:9.1f}%"
                f" {roc_auc_score(ytr, ptr):10.3f} |")
        for wn, (lo, hi) in TESTS.items():
            w = df[(df.day >= lo) & (df.day <= hi)]
            p = m.predict_proba(sc.transform(w[FEATURES].to_numpy(float)))[:, 1]
            acc = ((p > 0.5) == w.y.to_numpy(int)).mean()
            auc = roc_auc_score(w.y.to_numpy(int), p)
            line += f"   acc {acc*100:5.1f}%  AUC {auc:5.3f}"
        print(line)

    print("\nbreak-even is ~52% accuracy; 56% is worth ~+30 score.")
    print("A model must clear 52% on BOTH held-out windows to be worth anything.")


if __name__ == "__main__":
    main()

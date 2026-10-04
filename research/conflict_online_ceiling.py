#!/usr/bin/env python
"""Does continuously refeeding outcomes fix the arbitrator? Find its ceiling.

The hypothesis under test: the booster fails out of sample because it stops
learning at the train boundary, so an arbitrator that keeps ingesting
realised outcomes should keep up with the regime and survive.

That hypothesis is testable in a way that does not depend on picking a good
online learner, because every online method -- Hedge, SGD, incremental trees,
periodic retraining -- is ultimately trying to answer ONE question: *which
sleeve is better right now?* So the strict upper bound on all of them is an
oracle that already knows the answer, for free, with no estimation error and
no lag.

  REGIME ORACLE(W): partition the timeline into W-day blocks; within each
  block, follow whichever sleeve actually had positive mean EV in that block.
  Perfect regime knowledge at granularity W, zero learning cost.

If REGIME ORACLE(W) does not beat the shipped baseline, then no online
learner operating at that granularity can either -- it would have to exceed
a bound it is trying to estimate. Sweeping W turns "can online learning work"
into "how fine must regime detection be", which can then be compared against
the ~600 trading days that E5 showed are needed to detect a 2pp edge.

Also runs two genuine online learners for confirmation:
  * HEDGE / exponential weights over the two experts (Freund & Schapire 1997;
    Cesa-Bianchi & Lugosi 2006) -- the standard no-regret online combiner.
  * SGD logistic with partial_fit -- true incremental supervised learning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.linear_model import SGDClassifier

from research.combine_lab import DLR_LIMIT, score, simulate
from research.conflict_ev_model import FEATURES
from research.conflict_ml_deepdive import precompute, score_with_overrides
from research.signal_cache import load_or_build, load_prices

import testbedv2_tuned_ridge_protectedbarbell as strategy

OUTPUT_DIR = ROOT / "analysis_outputs" / "signal_overlap"
BB = strategy.BANG_BANG_LIMITS
BASELINE = {"train": 789.8, "val": 847.0, "test": 914.7}
REGIONS = (("train", 251, 500), ("val", 501, 750), ("test", 751, 1000))


def regime_oracle(pre, prices, data, window: int):
    """Follow the sleeve that actually wins each W-day block. Upper bound."""
    ev_by_day = data.groupby("day")["ev"].mean()
    block_sign = {}
    for day in ev_by_day.index:
        block = int(day) // window
        block_sign.setdefault(block, []).append(ev_by_day[day])
    sign_for_block = {b: (1.0 if np.mean(v) > 0 else -1.0) for b, v in block_sign.items()}

    def override(i, mask, direction):
        t = int(pre["days"][i])
        follow_ridge = sign_for_block.get(t // window, 1.0) > 0
        return (pre["ridge_dir"][i][mask] if follow_ridge else pre["pair_dir"][i][mask])

    return {name: score_with_overrides(pre, prices, a, b, override)
            for name, a, b in REGIONS}


def hedge_arbitrator(pre, prices, data, eta: float):
    """Exponential-weights over {follow pair, follow ridge}, updated every day."""
    ev_lookup = {(int(r.day), int(r.inst)): float(r.ev) for r in data.itertuples()}
    results = {}
    for name, start, end in REGIONS:
        weights = np.array([1.0, 1.0])  # [pair, ridge]
        index = {int(d): i for i, d in enumerate(pre["days"])}
        dollars = {}
        for t in range(250, end):
            i = index[t]
            direction = pre["base_dir"][i].copy()
            mask = pre["conflict"][i]
            if mask.any():
                follow_ridge = weights[1] >= weights[0]
                direction[mask] = (pre["ridge_dir"][i][mask] if follow_ridge
                                   else pre["pair_dir"][i][mask])
            if t >= start - 1:
                dollars[t] = np.clip(BB * direction, -DLR_LIMIT, DLR_LIMIT)
            # Update AFTER acting, using only now-observable outcomes.
            for inst in np.flatnonzero(mask):
                ev = ev_lookup.get((t, int(inst)))
                if ev is None:
                    continue
                weights[1] *= np.exp(eta * ev * 1000.0)
                weights[0] *= np.exp(-eta * ev * 1000.0)
            weights /= weights.sum()
            weights = np.clip(weights, 1e-6, None)
        pnl = simulate(dollars, prices, start, end)
        results[name] = score(float(pnl.mean()), float(pnl.std()))
    return results


def sgd_arbitrator(pre, prices, data, alpha: float):
    """True incremental supervised learning via partial_fit."""
    rows = {(int(r.day), int(r.inst)): r for r in data.itertuples()}
    means = data[FEATURES].mean().to_numpy()
    stds = data[FEATURES].std().to_numpy() + 1e-9
    results = {}
    for name, start, end in REGIONS:
        model = SGDClassifier(loss="log_loss", alpha=alpha, learning_rate="optimal",
                              random_state=0)
        started = False
        index = {int(d): i for i, d in enumerate(pre["days"])}
        dollars = {}
        for t in range(250, end):
            i = index[t]
            direction = pre["base_dir"][i].copy()
            mask = pre["conflict"][i]
            pending_x, pending_y = [], []
            for inst in np.flatnonzero(mask):
                r = rows.get((t, int(inst)))
                if r is None:
                    continue
                x = (np.array([getattr(r, f) for f in FEATURES]) - means) / stds
                if started:
                    p = float(model.predict_proba(x.reshape(1, -1))[0, 1])
                    direction[inst] = (pre["ridge_dir"][i][inst] if p > 0.5
                                       else pre["pair_dir"][i][inst])
                pending_x.append(x)
                pending_y.append(r.is_ridge_right)
            if t >= start - 1:
                dollars[t] = np.clip(BB * direction, -DLR_LIMIT, DLR_LIMIT)
            if pending_x:
                model.partial_fit(np.array(pending_x), np.array(pending_y), classes=[0.0, 1.0])
                started = True
        pnl = simulate(dollars, prices, start, end)
        results[name] = score(float(pnl.mean()), float(pnl.std()))
    return results


def main() -> None:
    prices = load_prices()
    cache = load_or_build()
    data = pd.read_csv(OUTPUT_DIR / "conflict_dataset.csv")
    pre = precompute(cache, prices)

    print("=" * 88)
    print("CEILING: perfect regime knowledge at granularity W (no estimation, no lag)")
    print("=" * 88)
    print(f"{'granularity':>14} {'train':>10} {'val':>10} {'test':>10}   {'beats baseline on test?':>24}")
    print("-" * 88)
    rows = []
    for window in (250, 125, 50, 25, 10, 5, 2, 1):
        got = regime_oracle(pre, prices, data, window)
        flag = "YES" if got["test"] > BASELINE["test"] else "no"
        print(f"{window:>11}d   {got['train']:10.1f} {got['val']:10.1f} {got['test']:10.1f}   {flag:>24}")
        rows.append(dict(window=window, **got))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "online_ceiling.csv", index=False)

    print(f"\n{'baseline (shipped)':>14} {BASELINE['train']:10.1f} "
          f"{BASELINE['val']:10.1f} {BASELINE['test']:10.1f}")

    print("\n" + "=" * 88)
    print("ACTUAL ONLINE LEARNERS (continuously refed, causal)")
    print("=" * 88)
    print(f"{'method':>34} {'train':>10} {'val':>10} {'test':>10}")
    print("-" * 88)
    for eta in (0.01, 0.05, 0.2, 1.0):
        got = hedge_arbitrator(pre, prices, data, eta)
        print(f"{'Hedge eta=' + str(eta):>34} {got['train']:10.1f} {got['val']:10.1f} {got['test']:10.1f}")
    for alpha in (1e-4, 1e-2, 1.0):
        got = sgd_arbitrator(pre, prices, data, alpha)
        print(f"{'SGD partial_fit alpha=' + str(alpha):>34} {got['train']:10.1f} {got['val']:10.1f} {got['test']:10.1f}")
    print(f"{'baseline (shipped)':>34} {BASELINE['train']:10.1f} {BASELINE['val']:10.1f} {BASELINE['test']:10.1f}")
    print(f"\nwrote {OUTPUT_DIR / 'online_ceiling.csv'}")


if __name__ == "__main__":
    main()

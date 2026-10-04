#!/usr/bin/env python
"""Cache the raw pair/ridge primitives once so combination schemes are cheap.

Both sleeves are pure functions of the price history:

* ``PairAgent.catch_up`` evolves its filter/OU state from prices alone and
  ``dollar_positions()`` only reads that state -- the allocator's output is
  never fed back into an agent.
* ``_ridge_forecast`` is a pure function of the return matrix.

So the expensive replay (~37s) can run ONCE, and every candidate way of
combining the two sleeves becomes a cheap array operation over the cache.
This is what makes it affordable to test many structurally different
combination schemes instead of tuning one scheme's knobs.

The cache stores, per day t (position decided from ``prices[:, :t]``):
  pair_pos_y/pair_pos_x  the pair's OWN requested leg dollars, including the
                         beta hedge ratio (pos_x = -beta * pos_y)
  pair_priority          kappa * |z|, the pair's own confidence
  ridge_raw              ridge's requested dollars (sign * RIDGE_LEG)
  ridge_strength         |prediction| / prediction_scale, ridge's confidence
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import testbedv2_tuned_ridge_protectedbarbell as strategy

CACHE_PATH = ROOT / "analysis_outputs" / "signal_overlap" / "signal_cache.npz"
# Day 250 is included because the PnL loop decides a position from
# prices[:, :start-1] before the first scored day.
FIRST_DAY = 250
LAST_DAY = 1000


def load_prices(path: Path | str = None) -> np.ndarray:
    path = Path(path) if path is not None else ROOT / "prices.txt"
    return pd.read_csv(path, sep=r"\s+", header=0).to_numpy(float).T


def build_cache(prices: np.ndarray, first: int = FIRST_DAY, last: int = LAST_DAY) -> dict:
    n_inst = strategy.N_INST
    n_pairs = len(strategy.PAIR_CFGS)
    days = np.arange(first, last + 1)
    n_days = len(days)

    pair_pos_y = np.zeros((n_days, n_pairs))
    pair_pos_x = np.zeros((n_days, n_pairs))
    pair_priority = np.zeros((n_days, n_pairs))
    ridge_raw = np.zeros((n_days, n_inst))
    ridge_strength = np.zeros((n_days, n_inst))

    book = strategy.AtlasBook()
    for row, t in enumerate(days):
        history = prices[:, :t]
        for agent in book.agents:
            agent.catch_up(history)
        for pair_idx, agent in enumerate(book.agents):
            pos_y, pos_x, abs_z = agent.dollar_positions()
            pair_pos_y[row, pair_idx] = pos_y
            pair_pos_x[row, pair_idx] = pos_x
            pair_priority[row, pair_idx] = agent.current_kappa * abs_z
        returns = np.diff(np.log(history), axis=1)
        raw, strength = strategy._ridge_forecast(returns)
        ridge_raw[row] = raw
        ridge_strength[row] = strength

    return {
        "days": days,
        "pair_pos_y": pair_pos_y,
        "pair_pos_x": pair_pos_x,
        "pair_priority": pair_priority,
        "ridge_raw": ridge_raw,
        "ridge_strength": ridge_strength,
        "pair_y_index": np.array([int(s.y) for s in strategy.PAIR_CFGS]),
        "pair_x_index": np.array([int(s.x) for s in strategy.PAIR_CFGS]),
        "pair_protected": np.array(
            [
                strategy._pair_name(s) in strategy.PROTECTED_PAIR_NAMES
                for s in strategy.PAIR_CFGS
            ]
        ),
    }


def load_or_build(rebuild: bool = False) -> dict:
    if CACHE_PATH.exists() and not rebuild:
        with np.load(CACHE_PATH) as data:
            return {key: data[key] for key in data.files}
    prices = load_prices()
    cache = build_cache(prices)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, **cache)
    return cache


if __name__ == "__main__":
    cache = load_or_build(rebuild="--rebuild" in sys.argv)
    print(f"cached days {cache['days'][0]}-{cache['days'][-1]}")
    print(f"pairs: {cache['pair_pos_y'].shape[1]}, instruments: {cache['ridge_raw'].shape[1]}")
    print(f"wrote {CACHE_PATH}")

#!/usr/bin/env python
"""Score the small "honest" 8-pair book (protected-7 union stable-8, since the
protected 7 are a subset of the stable-8 dynamic-OU set) in isolation, as a
z-band static-param variant and a dynamic-OU (a* optimal stopping) variant.
Compares against the full 88-pair four_sigmas.py book, on both the old
"OOS" window (750-1000, previously used for judgment) and the new true OOS
window (1001-1500), using eval.py's exact accounting (integer shares,
position limits, 1bp commission, score()).

The 8 pairs share no instruments with each other, so there is no conflict
layer to model -- summing each pair's desired dollar legs directly gives the
book's positions.
"""
from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import numpy as np
import pandas as pd

import four_sigmas as fs
import friend_stable8_ou as fs8

STABLE8_NAMES = set(fs8.STABLE8_OU_PARAMS.keys())
COMM_RATE = 0.0001
DEFAULT_COMM = 0.0001
INST0_COMM = 0.00002
DEFAULT_LIMIT = 10_000
INST0_LIMIT = 100_000
SCORE_PARAM = 1


def load_prices(fn="prices.txt"):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T.astype(float)


def score(mu, sigma, param=SCORE_PARAM):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


def build_book_fn(module, names):
    """Return a getMyPosition-style fn using only `names` pairs from `module`."""
    cfgs = [spec for spec in module.PAIR_CFGS if module._pair_name(spec) in names]
    assert len(cfgs) == len(names), (len(cfgs), names)

    def get_position(prices: np.ndarray) -> np.ndarray:
        nins, nt = prices.shape
        if nt < module.WARMUP + 10:
            return np.zeros(nins, dtype=int)
        dollars = np.zeros(nins)
        for spec in cfgs:
            agent = get_position.agents[module._pair_name(spec)]
            agent.catch_up(prices)
            pos_y, pos_x, _ = agent.dollar_positions()
            dollars[spec.y] += pos_y
            dollars[spec.x] += pos_x
        limit = np.full(nins, DEFAULT_LIMIT)
        limit[0] = INST0_LIMIT
        dollars = np.clip(dollars, -limit, limit)
        safe_prices = np.maximum(prices[:, -1], 1e-12)
        share_limits = np.floor(limit / safe_prices).astype(int)
        raw_shares = dollars / safe_prices
        return np.clip(np.round(raw_shares), -share_limits, share_limits).astype(int)

    get_position.agents = {
        module._pair_name(spec): module.PairAgent(spec) for spec in cfgs
    }
    return get_position


def calc_pl(get_position, prices, n_inst, start_day, end_day):
    cash = 0.0
    cur_pos = np.zeros(n_inst)
    tot_dvol = 0.0
    value = 0.0
    comm = 0.0
    pll = []
    comm_rate = np.full(n_inst, DEFAULT_COMM)
    comm_rate[0] = INST0_COMM

    for t in range(start_day - 1, end_day + 1):
        prc_hist = prices[:, :t]
        cur_prices = prc_hist[:, -1]
        if t < end_day:
            new_pos_orig = get_position(prc_hist)
            limit = np.full(n_inst, DEFAULT_LIMIT)
            limit[0] = INST0_LIMIT
            pos_limits = (limit / cur_prices).astype(int)
            new_pos = np.clip(new_pos_orig, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)
        delta = new_pos - cur_pos
        cash -= cur_prices.dot(delta) + comm
        dvolumes = cur_prices * np.abs(delta)
        tot_dvol += np.sum(dvolumes)
        comm = np.sum(dvolumes * comm_rate)
        cur_pos = np.array(new_pos)
        pos_value = cur_pos.dot(cur_prices)
        today_pl = cash + pos_value - value
        value = cash + pos_value
        if t >= start_day:
            pll.append(today_pl)
    pll = np.array(pll)
    mu, sigma = pll.mean(), pll.std()
    ann_sharpe = np.sqrt(250) * mu / sigma if sigma > 0 else 0.0
    return mu, sigma, ann_sharpe, tot_dvol, score(mu, sigma)


def main():
    prices = load_prices()
    n_inst = prices.shape[0]

    variants = {
        "8pair_zband (four_sigmas static params)": build_book_fn(fs, STABLE8_NAMES),
        "8pair_dynamicOU (friend_stable8_ou a*)": build_book_fn(fs8, STABLE8_NAMES),
    }

    windows = [("old_OOS 750-1000", 750, 1000), ("new_OOS 1001-1500", 1001, 1500)]

    print(f"Pairs in the honest book: {sorted(STABLE8_NAMES)}\n")
    header = f"{'variant':42} {'window':20} {'mean/day':>10} {'std':>10} {'sharpe':>7} {'score':>9}"
    print(header)
    print("-" * len(header))
    results = {}
    for label, fn in variants.items():
        for wlabel, s, e in windows:
            mu, sigma, sharpe, dvol, sc = calc_pl(fn, prices, n_inst, s, e)
            results[(label, wlabel)] = sc
            print(f"{label:42} {wlabel:20} {mu:10.1f} {sigma:10.1f} {sharpe:7.2f} {sc:9.2f}")

    print("\n=== Reference: full four_sigmas.py 88-pair book (from earlier eval.py runs) ===")
    print(f"{'four_sigmas (88 pairs)':42} {'old_OOS 750-1000':20} {'':>10} {'':>10} {8.78:7.2f} {978.61:9.2f}")
    print(f"{'four_sigmas (88 pairs)':42} {'new_OOS 1001-1500':20} {'':>10} {'':>10} {5.32:7.2f} {574.90:9.2f}")


if __name__ == "__main__":
    main()

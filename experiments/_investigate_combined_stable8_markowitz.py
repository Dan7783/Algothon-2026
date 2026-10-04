#!/usr/bin/env python
"""Combined sleeve book: stable-8 pairs get protected, exclusive control of
their 16 instruments; random_strat1's Markowitz signal fills the remaining
~35 instruments. No conflict layer needed -- the two sleeves never touch the
same instrument.

Tests both stable-8 engines (four_sigmas static z-band vs friend_stable8_ou
dynamic-OU a*) combined with Markowitz v14 (6-lambda ensemble), against:
  - old "OOS" window 750-1000 (previously used for judgment)
  - new true OOS window 1001-1500 (never touched by any selection/tuning)

Markowitz's getMyPosition() is called once per day (expensive: O(nins^3) x 6
lambdas) and its output is reused for both stable-8 engine variants, so the
combined-book cost is ~1x a solo Markowitz run, not 2x.
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
import random_strat1 as mkw

STABLE8_NAMES = set(fs8.STABLE8_OU_PARAMS.keys())
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


def stable_symbol_indices(module):
    idxs = set()
    for spec in module.PAIR_CFGS:
        if module._pair_name(spec) in STABLE8_NAMES:
            idxs.add(int(spec.y))
            idxs.add(int(spec.x))
    return idxs


def build_stable_dollars_fn(module):
    cfgs = [spec for spec in module.PAIR_CFGS if module._pair_name(spec) in STABLE8_NAMES]
    assert len(cfgs) == len(STABLE8_NAMES)
    agents = {module._pair_name(spec): module.PairAgent(spec) for spec in cfgs}

    def get_dollars(prices: np.ndarray, n_inst: int) -> np.ndarray:
        dollars = np.zeros(n_inst)
        for spec in cfgs:
            agent = agents[module._pair_name(spec)]
            agent.catch_up(prices)
            pos_y, pos_x, _ = agent.dollar_positions()
            dollars[spec.y] += pos_y
            dollars[spec.x] += pos_x
        return dollars

    return get_dollars


def to_shares(dollars, prices_last, n_inst):
    limit = np.full(n_inst, DEFAULT_LIMIT)
    limit[0] = INST0_LIMIT
    dollars = np.clip(dollars, -limit, limit)
    safe_prices = np.maximum(prices_last, 1e-12)
    share_limits = np.floor(limit / safe_prices).astype(int)
    raw = dollars / safe_prices
    return np.clip(np.round(raw), -share_limits, share_limits).astype(int)


def calc_pl_multi(prices, n_inst, start_day, end_day, book_fns, warmup):
    """Run several `book_fns[name](prc_hist) -> shares` books in one price pass."""
    state = {
        name: dict(cash=0.0, cur_pos=np.zeros(n_inst), value=0.0, comm=0.0, pll=[])
        for name in book_fns
    }
    comm_rate = np.full(n_inst, DEFAULT_COMM)
    comm_rate[0] = INST0_COMM

    for t in range(start_day - 1, end_day + 1):
        prc_hist = prices[:, :t]
        cur_prices = prc_hist[:, -1]
        for name, fn in book_fns.items():
            st = state[name]
            if t < end_day:
                if t < warmup:
                    new_pos = np.zeros(n_inst, dtype=int)
                else:
                    new_pos = fn(prc_hist)
            else:
                new_pos = np.array(st["cur_pos"])
            delta = new_pos - st["cur_pos"]
            st["cash"] -= cur_prices.dot(delta) + st["comm"]
            dvolumes = cur_prices * np.abs(delta)
            st["comm"] = np.sum(dvolumes * comm_rate)
            st["cur_pos"] = np.array(new_pos)
            pos_value = st["cur_pos"].dot(cur_prices)
            today_pl = st["cash"] + pos_value - st["value"]
            st["value"] = st["cash"] + pos_value
            if t >= start_day:
                st["pll"].append(today_pl)

    out = {}
    for name, st in state.items():
        pll = np.array(st["pll"])
        mu, sigma = pll.mean(), pll.std()
        sharpe = np.sqrt(250) * mu / sigma if sigma > 0 else 0.0
        out[name] = (mu, sigma, sharpe, score(mu, sigma))
    return out


def main():
    prices = load_prices()
    n_inst = prices.shape[0]

    zband_dollars_fn = build_stable_dollars_fn(fs)
    ou_dollars_fn = build_stable_dollars_fn(fs8)
    stable_idx_zband = stable_symbol_indices(fs)
    stable_idx_ou = stable_symbol_indices(fs8)
    assert stable_idx_zband == stable_idx_ou
    stable_idx = stable_idx_zband
    non_stable_mask = np.ones(n_inst, dtype=bool)
    for i in stable_idx:
        non_stable_mask[i] = False

    def combined_book(engine_dollars_fn):
        def fn(prc_hist):
            mkw_shares = mkw.getMyPosition(prc_hist)
            stable_dollars = engine_dollars_fn(prc_hist, n_inst)
            stable_shares = to_shares(stable_dollars, prc_hist[:, -1], n_inst)
            out = np.array(mkw_shares, dtype=int)
            out[list(stable_idx)] = stable_shares[list(stable_idx)]
            return out
        return fn

    def markowitz_only(prc_hist):
        return mkw.getMyPosition(prc_hist)

    def stable_only(engine_dollars_fn):
        def fn(prc_hist):
            stable_dollars = engine_dollars_fn(prc_hist, n_inst)
            return to_shares(stable_dollars, prc_hist[:, -1], n_inst)
        return fn

    book_fns = {
        "combined_zband+markowitz": combined_book(zband_dollars_fn),
        "combined_OU+markowitz": combined_book(ou_dollars_fn),
        "markowitz_only": markowitz_only,
    }

    windows = [("old_OOS 750-1000", 750, 1000), ("new_OOS 1001-1500", 1001, 1500)]
    # reset stateful modules before each window so hysteresis / agent state
    # doesn't leak across windows
    header = f"{'book':28} {'window':20} {'mean/day':>10} {'std':>10} {'sharpe':>7} {'score':>9}"
    print(f"Stable-8 pairs: {sorted(STABLE8_NAMES)}")
    print(f"Instruments they own: {sorted(fs._SYMBOL_BY_INDEX[i] for i in stable_idx)}\n")
    print(header)
    print("-" * len(header))
    for wlabel, s, e in windows:
        fs.reset()
        mkw.currentPos = np.zeros(n_inst, dtype=int)
        # rebuild stable-8 agents (fresh state) for this window
        zband_dollars_fn = build_stable_dollars_fn(fs)
        ou_dollars_fn = build_stable_dollars_fn(fs8)
        book_fns = {
            "combined_zband+markowitz": combined_book(zband_dollars_fn),
            "combined_OU+markowitz": combined_book(ou_dollars_fn),
            "markowitz_only": markowitz_only,
        }
        results = calc_pl_multi(prices, n_inst, s, e, book_fns, warmup=fs.WARMUP + 10)
        for name, (mu, sigma, sharpe, sc) in results.items():
            print(f"{name:28} {wlabel:20} {mu:10.1f} {sigma:10.1f} {sharpe:7.2f} {sc:9.2f}")

    print("\n=== Reference (from earlier eval.py runs) ===")
    print(f"{'four_sigmas (88 pairs)':28} {'old_OOS 750-1000':20} {'':>10} {'':>10} {8.78:7.2f} {978.61:9.2f}")
    print(f"{'four_sigmas (88 pairs)':28} {'new_OOS 1001-1500':20} {'':>10} {'':>10} {5.32:7.2f} {574.90:9.2f}")
    print(f"{'random_strat1 v14 solo':28} {'old_OOS 750-1000':20} {'':>10} {'':>10} {6.35:7.2f} {907.88:9.2f}")
    print(f"{'random_strat1 v14 solo':28} {'new_OOS 1001-1500':20} {'':>10} {'':>10} {5.24:7.2f} {723.78:9.2f}")


if __name__ == "__main__":
    main()

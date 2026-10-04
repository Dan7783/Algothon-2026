#!/usr/bin/env python
"""Combination-scheme lab for the pair and ridge sleeves.

Every scheme here consumes the SAME cached raw signals (see signal_cache.py)
and only differs in how the two sleeves are merged into dollar positions.
That isolates the combination layer as the single variable, which is exactly
the thing under study: both sleeves are known to work standalone, and the
question is what the merge is costing.

The simulator replicates eval.py's mechanics exactly (integer shares,
per-instrument limits and commissions, mark-only final day) so a scheme's
number here is directly comparable to the shipped evaluator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_cache import load_or_build, load_prices

import testbedv2_tuned_ridge_protectedbarbell as strategy

N_INST = strategy.N_INST
COMM_RATE = np.full(N_INST, 1.0e-4)
COMM_RATE[0] = 2.0e-5
DLR_LIMIT = np.full(N_INST, 10_000.0)
DLR_LIMIT[0] = 100_000.0

# Pre-registered regions (cusum_output/VALIDATION_PROTOCOL.md).
TRAIN_BLOCKS = (("train_a", 251, 375), ("train_b", 376, 500))
VAL_BLOCKS = (("val_a", 501, 625), ("val_b", 626, 750))
TEST_BLOCKS = (("test_a", 751, 875), ("test_b", 876, 1000))
TRAIN = ("train", 251, 500)
VALIDATION = ("validation", 501, 750)
TEST = ("test", 751, 1000)


def score(mean: float, std: float) -> float:
    if mean <= 0.0 or std < 1.0e-10:
        return float(mean)
    sharpe = np.sqrt(250.0) * mean / std
    return float(mean * sharpe * sharpe / (sharpe * sharpe + 1.0))


def simulate(dollars_by_day: dict[int, np.ndarray], prices: np.ndarray,
             start: int, end: int) -> np.ndarray:
    """Replicate eval.py's PnL loop for a precomputed dollar schedule."""
    cash = 0.0
    value = 0.0
    comm = 0.0
    cur_pos = np.zeros(N_INST)
    daily = []
    for t in range(start - 1, end + 1):
        cur_prices = prices[:, t - 1]
        if t < end:
            dollars = dollars_by_day[t]
            raw_shares = np.nan_to_num(dollars / np.maximum(cur_prices, 1e-12), nan=0.0)
            requested = np.round(raw_shares).astype(int)
            pos_limits = (DLR_LIMIT / cur_prices).astype(int)
            new_pos = np.clip(requested, -pos_limits, pos_limits).astype(float)
        else:
            new_pos = cur_pos.copy()
        delta = new_pos - cur_pos
        cash -= cur_prices.dot(delta) + comm
        comm = float(np.sum(cur_prices * np.abs(delta) * COMM_RATE))
        cur_pos = new_pos
        pos_value = cur_pos.dot(cur_prices)
        today = cash + pos_value - value
        value = cash + pos_value
        daily.append(today)
    return np.asarray(daily[1:])


def evaluate(scheme, cache: dict, prices: np.ndarray, start: int, end: int) -> np.ndarray:
    """Run a scheme over [start, end] and return its daily PnL."""
    days = cache["days"]
    lookup = {int(day): row for row, day in enumerate(days)}
    dollars_by_day = {}
    for t in range(start - 1, end):
        row = lookup[t]
        dollars_by_day[t] = scheme(cache, row, prices, t)
    return simulate(dollars_by_day, prices, start, end)


def region_report(pnl: np.ndarray, start: int, end: int,
                  blocks: tuple) -> dict:
    overall = score(float(pnl.mean()), float(pnl.std()))
    block_scores = []
    for _, b_start, b_end in blocks:
        seg = pnl[b_start - start:b_end - start + 1]
        block_scores.append(score(float(seg.mean()), float(seg.std())))
    return {
        "score": overall,
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "blocks": block_scores,
        "floor": float(min(block_scores)) if block_scores else float("nan"),
    }


# --------------------------------------------------------------------------
# Shared primitives used by the schemes
# --------------------------------------------------------------------------

def pair_consensus(cache: dict, row: int, boost: float, mode: str = "priority"):
    """The shipped asset-level pair vote: sign-only, priority/boost weighted."""
    numerator = np.zeros(N_INST)
    denominator = np.zeros(N_INST)
    y_idx = cache["pair_y_index"]
    x_idx = cache["pair_x_index"]
    protected = cache["pair_protected"]
    pos_y = cache["pair_pos_y"][row]
    pos_x = cache["pair_pos_x"][row]
    priority = cache["pair_priority"][row]
    for pair_idx in range(len(y_idx)):
        if abs(pos_y[pair_idx]) <= 1.0e-9:
            continue
        weight = (boost if protected[pair_idx] else 1.0)
        if mode == "priority":
            weight *= max(float(priority[pair_idx]), 1.0e-12)
        for index, position in ((int(y_idx[pair_idx]), pos_y[pair_idx]),
                                (int(x_idx[pair_idx]), pos_x[pair_idx])):
            if abs(position) <= 1.0e-9:
                continue
            numerator[index] += weight * np.sign(position)
            denominator[index] += weight
    consensus = np.divide(numerator, denominator,
                          out=np.zeros_like(numerator), where=denominator > 0.0)
    return consensus, denominator > 0.0


def pair_dollar_sum(cache: dict, row: int) -> np.ndarray:
    """Pairs' own requested dollars, summed, preserving each pair's hedge ratio."""
    dollars = np.zeros(N_INST)
    y_idx = cache["pair_y_index"]
    x_idx = cache["pair_x_index"]
    pos_y = cache["pair_pos_y"][row]
    pos_x = cache["pair_pos_x"][row]
    for pair_idx in range(len(y_idx)):
        if abs(pos_y[pair_idx]) <= 1.0e-9:
            continue
        dollars[int(y_idx[pair_idx])] += pos_y[pair_idx]
        dollars[int(x_idx[pair_idx])] += pos_x[pair_idx]
    return dollars


def crash_mask(prices: np.ndarray, t: int) -> np.ndarray:
    window = np.maximum(prices[:, t - 2:t], 1e-12)
    last_return = np.diff(np.log(window), axis=1)[:, 0]
    crash = np.abs(last_return) > 0.40
    crash[0] = False
    return crash


def finalize(dollars: np.ndarray, prices: np.ndarray, t: int) -> np.ndarray:
    """Apply the crash filter and the per-instrument dollar caps."""
    dollars = dollars.copy()
    dollars[crash_mask(prices, t)] = 0.0
    return np.clip(dollars, -DLR_LIMIT, DLR_LIMIT)

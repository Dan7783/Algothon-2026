"""Rejected research candidate: protected barbell plus an ALGO factor hedge.

ALGO is not treated as another tuned pair or as a directional alpha signal.
It is a low-cost, high-capacity proxy for the common market factor.  The
overlay estimates each live asset's beta to ALGO from the latest 60 returns
and uses ALGO to offset half of the resulting portfolio factor exposure.

All constants were frozen using days 1-750 only.  See
``analysis_outputs/algo_overlay/frozen_selection.json``.
The subsequent days-751-1000 test reduced score, so this module is retained
for reproducibility only and must not replace the production strategy.
"""

from __future__ import annotations

import numpy as np

import testbedv2_tuned_ridge_protectedbarbell as _base


ALGO_INDEX = 0
ALGO_BETA_WINDOW = 60
ALGO_HEDGE_RATIO = 0.5
ALGO_DOLLAR_CAP = 100_000.0
ALGO_BETA_SHRINK = 0.5
ALGO_BETA_MIN = -1.0
ALGO_BETA_MAX = 3.0


def reset() -> None:
    _base.reset()


def _algo_hedge_dollars(
    prices: np.ndarray,
    base_positions: np.ndarray,
) -> float:
    if prices.shape[1] < ALGO_BETA_WINDOW + 1:
        return 0.0

    recent = np.diff(
        np.log(prices[:, -(ALGO_BETA_WINDOW + 1) :]),
        axis=1,
    )
    algo = recent[ALGO_INDEX] - recent[ALGO_INDEX].mean()
    others = recent[1:] - recent[1:].mean(axis=1, keepdims=True)
    variance = float(algo @ algo)
    if variance <= 1.0e-14:
        return 0.0

    beta = (others @ algo) / variance
    beta = (
        (1.0 - ALGO_BETA_SHRINK) * beta
        + ALGO_BETA_SHRINK
    )
    beta = np.clip(beta, ALGO_BETA_MIN, ALGO_BETA_MAX)
    # Match the official evaluator before measuring exposure: the base
    # strategy rounds shares and the evaluator then clips to a floor-divided
    # $10k limit, which can differ by one share.
    share_limits = (
        _base.LEG / prices[1:, -1]
    ).astype(int)
    effective_positions = np.clip(
        base_positions[1:],
        -share_limits,
        share_limits,
    )
    current_dollars = effective_positions * prices[1:, -1]
    factor_loading = float(current_dollars @ beta)
    return float(
        np.clip(
            -ALGO_HEDGE_RATIO * factor_loading,
            -ALGO_DOLLAR_CAP,
            ALGO_DOLLAR_CAP,
        )
    )


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    prices = np.asarray(prcSoFar, dtype=float)
    base_positions = np.asarray(
        _base.getMyPosition(prices),
        dtype=int,
    ).copy()
    if (
        prices.ndim != 2
        or prices.shape[0] != _base.N_INST
        or prices.shape[1] == 0
        or not np.all(np.isfinite(prices[:, -1]))
        or np.any(prices[:, -1] <= 0.0)
    ):
        return base_positions

    hedge_dollars = _algo_hedge_dollars(prices, base_positions)
    base_positions[ALGO_INDEX] = int(
        np.round(hedge_dollars / prices[ALGO_INDEX, -1])
    )
    return base_positions

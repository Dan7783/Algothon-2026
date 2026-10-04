"""Experimental whole-vector expected-value combiner.

This module implements the locked model produced by
``research/whole_vector_ev_conflicts.py``. It is intentionally separate from
``four_sigmas.py`` because its one permitted held-out evaluation scored
845.95 versus the 886.39 baseline. The implementation is retained for
diagnosis and further forward-only observation; it is not the default
submission strategy.

Calibration:
* realised proposal PnL through day 500 only;
* hyperparameters selected on days 501-750;
* no recalibration or setting changes after the days 751-1000 evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import four_sigmas as core


_CALIBRATION_PATH = (
    Path(__file__).resolve().parent
    / "analysis_outputs"
    / "whole_vector_ev"
    / "frozen_calibration.json"
)
_CALIBRATION = json.loads(
    _CALIBRATION_PATH.read_text(encoding="utf-8")
)
_PAIR_CAL = _CALIBRATION["pair"]
_RIDGE_CAL = _CALIBRATION["ridge"]
_MIN_EDGE_Z = float(_CALIBRATION["selected"]["min_edge_z"])

_PAIR_OFFSETS = np.asarray(_PAIR_CAL["offsets"], dtype=float)
_PAIR_VARIANCES = np.asarray(_PAIR_CAL["variances"], dtype=float)
_RIDGE_OFFSETS = np.asarray(_RIDGE_CAL["offsets"], dtype=float)
_RIDGE_VARIANCES = np.asarray(_RIDGE_CAL["variances"], dtype=float)

_COMMISSION = np.full(core.N_INST, 1.0e-4)
_COMMISSION[core.I["ALGO"]] = 2.0e-5
_POSITION_LIMIT = np.full(core.N_INST, core.LEG)
_POSITION_LIMIT[core.I["ALGO"]] = 100_000.0

if len(_PAIR_OFFSETS) != len(core.PAIR_CFGS):
    raise RuntimeError("Frozen pair calibration does not match PAIR_CFGS")
if len(_RIDGE_OFFSETS) != core.N_INST:
    raise RuntimeError("Frozen ridge calibration does not match instruments")


def _predict(
    strength: float,
    identity: int,
    calibration: dict,
    offsets: np.ndarray,
) -> float:
    clipped = np.clip(
        strength,
        float(calibration["lower"]),
        float(calibration["upper"]),
    )
    return float(
        float(calibration["global_mean"])
        + float(calibration["slope"])
        * (clipped - float(calibration["center"]))
        + offsets[identity]
    )


class _WholeVectorEVBook:
    """Stateful complete-proposal EV allocator."""

    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self.agents = [core.PairAgent(spec) for spec in core.PAIR_CFGS]
        self.previous_pair_y = np.zeros(len(self.agents))
        self.previous_pair_x = np.zeros(len(self.agents))
        self.previous_ridge = np.zeros(core.N_INST)
        self.history_seen = 0
        self.current_dollars = np.zeros(core.N_INST)

    def _step(self, prices: np.ndarray) -> np.ndarray:
        history_size = prices.shape[1]
        current = prices[:, -1]
        aggregate = np.zeros(core.N_INST)

        for pair_index, agent in enumerate(self.agents):
            agent.catch_up(prices)
            pos_y = pos_x = abs_z = 0.0
            if history_size >= core.WARMUP + 10:
                pos_y, pos_x, abs_z = agent.dollar_positions()

            y_index = int(agent.spec.y)
            x_index = int(agent.spec.x)
            y_shares = pos_y / current[y_index]
            x_shares = pos_x / current[x_index]

            if abs(pos_y) > 1.0e-9:
                entry_cost = (
                    abs(y_shares - self.previous_pair_y[pair_index])
                    * current[y_index]
                    * _COMMISSION[y_index]
                    + abs(x_shares - self.previous_pair_x[pair_index])
                    * current[x_index]
                    * _COMMISSION[x_index]
                )
                exit_reserve = (
                    abs(pos_y) * _COMMISSION[y_index]
                    + abs(pos_x) * _COMMISSION[x_index]
                )
                kappa = max(float(agent.current_kappa), 0.0)
                sigma = max(float(agent.current_sigma), 0.0)
                strength = (
                    abs(pos_y)
                    * (1.0 - np.exp(-kappa))
                    * sigma
                    * abs_z
                )
                expected_gross = _predict(
                    strength,
                    pair_index,
                    _PAIR_CAL,
                    _PAIR_OFFSETS,
                )
                expected_net = expected_gross - entry_cost - exit_reserve
                variance = max(_PAIR_VARIANCES[pair_index], 1.0e-6)
                if expected_net > _MIN_EDGE_Z * np.sqrt(variance):
                    weight = expected_net / variance
                    # Both legs receive the same proposal weight. They enter
                    # the portfolio together before asset limits are applied.
                    aggregate[y_index] += weight * pos_y
                    aggregate[x_index] += weight * pos_x

            self.previous_pair_y[pair_index] = y_shares
            self.previous_pair_x[pair_index] = x_shares

        returns = np.diff(
            np.log(np.maximum(prices, core._EPS_DENOM)),
            axis=1,
        )
        ridge_position, ridge_strength = core._ridge_forecast(returns)
        ridge_shares = ridge_position / current
        ridge_cost = (
            np.abs(ridge_shares - self.previous_ridge)
            * current
            * _COMMISSION
            + np.abs(ridge_position) * _COMMISSION
        )
        for instrument in np.flatnonzero(
            np.abs(ridge_position) > 1.0e-9
        ):
            expected_gross = _predict(
                float(ridge_strength[instrument]),
                int(instrument),
                _RIDGE_CAL,
                _RIDGE_OFFSETS,
            )
            expected_net = expected_gross - ridge_cost[instrument]
            variance = max(_RIDGE_VARIANCES[instrument], 1.0e-6)
            if expected_net > _MIN_EDGE_Z * np.sqrt(variance):
                aggregate[instrument] += (
                    expected_net
                    / variance
                    * ridge_position[instrument]
                )
        self.previous_ridge = ridge_shares

        direction = np.sign(aggregate)
        if history_size >= 2:
            previous = np.maximum(prices[:, -2], core._EPS_DENOM)
            last_return = np.log(current) - np.log(previous)
            crash = np.abs(last_return) > 0.40
            crash[core.I["ALGO"]] = False
            direction[crash] = 0.0

        return core.BANG_BANG_LIMITS * direction

    def dollars(self, prices: np.ndarray) -> np.ndarray:
        history_size = prices.shape[1]
        if history_size < self.history_seen:
            self._reset_state()
        for size in range(self.history_seen + 1, history_size + 1):
            self.current_dollars = self._step(prices[:, :size])
        self.history_seen = history_size
        return self.current_dollars.copy()


_BOOK: _WholeVectorEVBook | None = None


def reset() -> None:
    global _BOOK
    _BOOK = None


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global _BOOK
    prices = np.asarray(prcSoFar, dtype=float)
    if (
        prices.ndim != 2
        or prices.shape[0] != core.N_INST
        or prices.shape[1] < core.WARMUP + 10
        or not np.all(np.isfinite(prices[:, -2:]))
        or np.any(prices[:, -2:] <= 0.0)
    ):
        return np.zeros(prices.shape[0], dtype=int)

    if _BOOK is None:
        _BOOK = _WholeVectorEVBook()
    dollars = _BOOK.dollars(prices)
    current = np.maximum(prices[:, -1], core._EPS_DENOM)
    limits = (_POSITION_LIMIT / current).astype(int)
    requested = np.round(dollars / current).astype(int)
    return np.clip(requested, -limits, limits).astype(int)

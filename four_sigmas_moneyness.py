"""Shared complete-vector dollar-moneyness engine.

The base class measures the full pair displacement to its exit. Subclasses
can convert that displacement to a common forecast horizon without changing
the ridge forecast, cost accounting, vector combination or final limits.
"""

from __future__ import annotations

import numpy as np

import four_sigmas as core


_COMMISSION = np.full(core.N_INST, 1.0e-4)
_COMMISSION[core.I["ALGO"]] = 2.0e-5
_POSITION_LIMIT = np.full(core.N_INST, core.LEG)
_POSITION_LIMIT[core.I["ALGO"]] = 100_000.0


def _ridge_raw_return_forecast(
    returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return current ridge positions and de-standardized return forecasts."""
    positions = np.zeros(core.N_INST)
    forecast_returns = np.zeros(core.N_INST)
    if returns.shape[1] + 1 < core.RIDGE_MIN_T:
        return positions, forecast_returns

    recent = returns[1:, -250:].T
    standardized = (
        recent - recent.mean(axis=0)
    ) / (recent.std(axis=0) + 1.0e-12)
    lagged_links = (
        standardized[:-1].T @ standardized[1:]
    ) / (len(recent) - 1)
    np.fill_diagonal(lagged_links, 0.0)
    significant_links = int(
        (
            np.abs(lagged_links)
            > 2.5 / np.sqrt(len(recent) - 1)
        ).sum()
    )
    if significant_links < core.RIDGE_GATE_N:
        return positions, forecast_returns

    history = returns[1:, :].T
    if core.RIDGE_LOOKBACK is not None:
        history = history[-core.RIDGE_LOOKBACK:]
    scale = history.std(axis=0) + 1.0e-12
    predictors = history[:-1] / scale
    targets = history[1:] / scale
    coefficients = np.linalg.solve(
        predictors.T @ predictors
        + core.RIDGE_LAM
        * len(predictors)
        * np.eye(core.N_INST - 1),
        predictors.T @ targets,
    )
    standardized_prediction = (
        history[-1] / scale
    ) @ coefficients
    prediction_scale = (
        predictors @ coefficients
    ).std(axis=0)
    strengths = np.abs(standardized_prediction) / np.maximum(
        prediction_scale,
        1.0e-12,
    )
    active = strengths > core.RIDGE_THETA
    active_indices = np.flatnonzero(active)
    if len(active_indices) > core.RIDGE_MAX_ACTIVE:
        keep = active_indices[
            np.argsort(strengths[active_indices])[
                -core.RIDGE_MAX_ACTIVE:
            ]
        ]
        active[:] = False
        active[keep] = True

    forecast_returns[1:] = standardized_prediction * scale
    positions[1:] = (
        core.RIDGE_LEG
        * np.sign(forecast_returns[1:])
        * active
    )
    return positions, forecast_returns


class _WholeVectorMoneynessBook:
    """Combine complete signals by predicted dollars available."""

    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self.agents = [
            core.PairAgent(spec) for spec in core.PAIR_CFGS
        ]
        self.previous_pair_y = np.zeros(len(self.agents))
        self.previous_pair_x = np.zeros(len(self.agents))
        self.previous_ridge = np.zeros(core.N_INST)
        self.history_seen = 0
        self.current_dollars = np.zeros(core.N_INST)
        self.last_pair_opportunity = 0.0
        self.last_ridge_opportunity = 0.0

    def _pair_gross_money(
        self,
        agent: core.PairAgent,
        pos_y: float,
        abs_z: float,
    ) -> float:
        """Dollar value of the full displacement remaining to z_exit."""
        spread_scale = (
            abs(float(agent.current_innov))
            / max(abs_z, core._EPS_DENOM)
        )
        remaining_displacement = spread_scale * max(
            abs_z - float(agent.spec.trade.z_exit),
            0.0,
        )
        return abs(pos_y) * remaining_displacement

    def _step(self, prices: np.ndarray) -> np.ndarray:
        history_size = prices.shape[1]
        current = prices[:, -1]
        opportunity_vector = np.zeros(core.N_INST)
        pair_opportunity = 0.0

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
                transition_cost = (
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
                gross_money = self._pair_gross_money(
                    agent,
                    pos_y,
                    abs_z,
                )
                net_money = (
                    gross_money - transition_cost - exit_reserve
                )
                gross_notional = abs(pos_y) + abs(pos_x)
                if net_money > 0.0 and gross_notional > 0.0:
                    opportunity_vector[y_index] += (
                        net_money * pos_y / gross_notional
                    )
                    opportunity_vector[x_index] += (
                        net_money * pos_x / gross_notional
                    )
                    pair_opportunity += net_money

            self.previous_pair_y[pair_index] = y_shares
            self.previous_pair_x[pair_index] = x_shares

        returns = np.diff(
            np.log(np.maximum(prices, core._EPS_DENOM)),
            axis=1,
        )
        ridge_position, forecast_return = (
            _ridge_raw_return_forecast(returns)
        )
        ridge_shares = ridge_position / current
        ridge_cost = (
            np.abs(ridge_shares - self.previous_ridge)
            * current
            * _COMMISSION
            + np.abs(ridge_position) * _COMMISSION
        )
        ridge_money = (
            np.abs(ridge_position * forecast_return)
            - ridge_cost
        )
        ridge_active = (
            (np.abs(ridge_position) > 1.0e-9)
            & (ridge_money > 0.0)
        )
        opportunity_vector[ridge_active] += (
            ridge_money[ridge_active]
            * np.sign(ridge_position[ridge_active])
        )
        self.previous_ridge = ridge_shares

        direction = np.sign(opportunity_vector)
        if history_size >= 2:
            previous = np.maximum(prices[:, -2], core._EPS_DENOM)
            last_return = np.log(current) - np.log(previous)
            crash = np.abs(last_return) > 0.40
            crash[core.I["ALGO"]] = False
            direction[crash] = 0.0

        self.last_pair_opportunity = pair_opportunity
        self.last_ridge_opportunity = float(
            np.sum(ridge_money[ridge_active])
        )
        return core.BANG_BANG_LIMITS * direction

    def dollars(self, prices: np.ndarray) -> np.ndarray:
        history_size = prices.shape[1]
        if history_size < self.history_seen:
            self._reset_state()
        for size in range(self.history_seen + 1, history_size + 1):
            self.current_dollars = self._step(prices[:, :size])
        self.history_seen = history_size
        return self.current_dollars.copy()


_BOOK: _WholeVectorMoneynessBook | None = None


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
        _BOOK = _WholeVectorMoneynessBook()
    dollars = _BOOK.dollars(prices)
    current = np.maximum(prices[:, -1], core._EPS_DENOM)
    limits = (_POSITION_LIMIT / current).astype(int)
    requested = np.round(dollars / current).astype(int)
    return np.clip(requested, -limits, limits).astype(int)

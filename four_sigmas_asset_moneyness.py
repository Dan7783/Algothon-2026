"""Per-asset next-move forecasts from pairs and ridge.

For every instrument this module forms:

1. one average next-day return forecast from all active pairs containing it;
2. the existing raw ridge next-return forecast, when active;
3. the sum of the two family forecasts in identical return units.

Pair forecasts use remaining spread displacement divided by dynamic OU
half-life. A minimum-norm decomposition maps the predicted spread move
``r_y - beta*r_x`` into individual y/x return forecasts. Multiple pair
forecasts are averaged so relationship count cannot mechanically overwhelm
ridge. There are no new tuned weights or probability estimates.
"""

from __future__ import annotations

from math import log

import numpy as np

import four_sigmas_moneyness as base


class _AssetMoneynessBook:
    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        core = base.core
        self.agents = [core.PairAgent(spec) for spec in core.PAIR_CFGS]
        self.previous_pair_y = np.zeros(len(self.agents))
        self.previous_pair_x = np.zeros(len(self.agents))
        self.previous_ridge = np.zeros(core.N_INST)
        self.history_seen = 0
        self.current_dollars = np.zeros(core.N_INST)
        self.last_pair_abs_forecast = 0.0
        self.last_ridge_abs_forecast = 0.0
        self.last_mean_pair_contributors = 0.0

    def _step(self, prices: np.ndarray) -> np.ndarray:
        core = base.core
        history_size = prices.shape[1]
        current = prices[:, -1]
        pair_forecast_sum = np.zeros(core.N_INST)
        pair_forecast_count = np.zeros(core.N_INST)

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
                spread_scale = (
                    abs(float(agent.current_innov))
                    / max(abs_z, core._EPS_DENOM)
                )
                remaining = spread_scale * max(
                    abs_z - float(agent.spec.trade.z_exit),
                    0.0,
                )
                kappa = max(
                    float(agent.current_kappa),
                    core._EPS_DENOM,
                )
                reversion_days = max(log(2.0) / kappa, 1.0)
                spread_forecast = (
                    -np.sign(float(agent.current_z))
                    * remaining
                    / reversion_days
                )

                beta = float(agent.filt.beta)
                denominator = 1.0 + beta * beta
                y_forecast = spread_forecast / denominator
                x_forecast = -beta * spread_forecast / denominator

                transition_cost = (
                    abs(y_shares - self.previous_pair_y[pair_index])
                    * current[y_index]
                    * base._COMMISSION[y_index]
                    + abs(x_shares - self.previous_pair_x[pair_index])
                    * current[x_index]
                    * base._COMMISSION[x_index]
                )
                exit_reserve = (
                    abs(pos_y) * base._COMMISSION[y_index]
                    + abs(pos_x) * base._COMMISSION[x_index]
                )
                predicted_gross = (
                    pos_y * y_forecast
                    + pos_x * x_forecast
                )
                if predicted_gross > 0.0:
                    net_fraction = max(
                        (
                            predicted_gross
                            - transition_cost
                            - exit_reserve
                        )
                        / predicted_gross,
                        0.0,
                    )
                    if net_fraction > 0.0:
                        pair_forecast_sum[y_index] += (
                            net_fraction * y_forecast
                        )
                        pair_forecast_sum[x_index] += (
                            net_fraction * x_forecast
                        )
                        pair_forecast_count[y_index] += 1.0
                        pair_forecast_count[x_index] += 1.0

            self.previous_pair_y[pair_index] = y_shares
            self.previous_pair_x[pair_index] = x_shares

        pair_forecast = np.divide(
            pair_forecast_sum,
            pair_forecast_count,
            out=np.zeros_like(pair_forecast_sum),
            where=pair_forecast_count > 0.0,
        )

        returns = np.diff(
            np.log(np.maximum(prices, core._EPS_DENOM)),
            axis=1,
        )
        ridge_position, ridge_forecast = (
            base._ridge_raw_return_forecast(returns)
        )
        ridge_shares = ridge_position / current
        ridge_cost = (
            np.abs(ridge_shares - self.previous_ridge)
            * current
            * base._COMMISSION
            + np.abs(ridge_position) * base._COMMISSION
        )
        ridge_gross = np.abs(ridge_position * ridge_forecast)
        ridge_net_fraction = np.divide(
            np.maximum(ridge_gross - ridge_cost, 0.0),
            ridge_gross,
            out=np.zeros_like(ridge_gross),
            where=ridge_gross > 0.0,
        )
        net_ridge_forecast = ridge_net_fraction * ridge_forecast
        self.previous_ridge = ridge_shares

        # These are two forecasts of the same next return. Summing versus
        # averaging the two families has the same sign; no family weight is
        # introduced. Pair-only and ridge-only assets retain their forecast.
        combined_forecast = pair_forecast + net_ridge_forecast
        direction = np.sign(combined_forecast)
        if history_size >= 2:
            previous = np.maximum(prices[:, -2], core._EPS_DENOM)
            last_return = np.log(current) - np.log(previous)
            crash = np.abs(last_return) > 0.40
            crash[core.I["ALGO"]] = False
            direction[crash] = 0.0

        pair_assets = pair_forecast_count > 0.0
        self.last_pair_abs_forecast = float(
            np.sum(np.abs(pair_forecast))
        )
        self.last_ridge_abs_forecast = float(
            np.sum(np.abs(net_ridge_forecast))
        )
        self.last_mean_pair_contributors = (
            float(np.mean(pair_forecast_count[pair_assets]))
            if np.any(pair_assets)
            else 0.0
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


_BOOK: _AssetMoneynessBook | None = None


def reset() -> None:
    global _BOOK
    _BOOK = None


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global _BOOK
    prices = np.asarray(prcSoFar, dtype=float)
    core = base.core
    if (
        prices.ndim != 2
        or prices.shape[0] != core.N_INST
        or prices.shape[1] < core.WARMUP + 10
        or not np.all(np.isfinite(prices[:, -2:]))
        or np.any(prices[:, -2:] <= 0.0)
    ):
        return np.zeros(prices.shape[0], dtype=int)

    if _BOOK is None:
        _BOOK = _AssetMoneynessBook()
    dollars = _BOOK.dollars(prices)
    current = np.maximum(prices[:, -1], core._EPS_DENOM)
    limits = (base._POSITION_LIMIT / current).astype(int)
    requested = np.round(dollars / current).astype(int)
    return np.clip(requested, -limits, limits).astype(int)

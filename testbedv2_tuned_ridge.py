"""The complete ``testbedv2_tuned`` pair book with ridge-first arbitration.

All 310 pair configurations, pair signals, conflict resolution and position
construction come directly from ``testbedv2_tuned`` without filtering.  The
ridge VAR(1) implementation and constants match the current ``four_sigmas``
ridge sleeve.

This is a barbell rather than an additive blend.  When ridge exceeds its
existing forecast threshold on an instrument, ridge receives that instrument
and any selected pair touching it is removed as a complete two-leg trade.
Below the ridge threshold the original pair allocation is retained.
"""

from __future__ import annotations

import numpy as np

import testbedv2_tuned as _base


# Re-export the complete tuned-pair strategy surface.
I = _base.I
N_INST = _base.N_INST
LEG = _base.LEG
WARMUP = _base.WARMUP
FiltCfg = _base.FiltCfg
TradeCfg = _base.TradeCfg
PairCfg = _base.PairCfg
PairSegFilter = _base.PairSegFilter
PairAgent = _base.PairAgent
PAIR_CFGS = _base.PAIR_CFGS


# Current four_sigmas ridge constants.
RIDGE_LAM = 1.0
RIDGE_THETA = 1.0
RIDGE_GATE_N = 35
RIDGE_LEG = 10_000.0
RIDGE_MIN_T = 252


class AtlasBook:
    """Original tuned-pair allocator with selected-pair attribution."""

    def __init__(self) -> None:
        self.agents = [PairAgent(spec) for spec in PAIR_CFGS]

    def dollars_with_pairs(
        self,
        prices: np.ndarray,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        nins, nt = prices.shape
        positions = np.zeros(nins)
        if nt < WARMUP + 10:
            return positions, ()

        for agent in self.agents:
            agent.catch_up(prices)

        candidates = []
        for agent in self.agents:
            pos_y, pos_x, abs_z = agent.dollar_positions()
            if abs(pos_y) < 1.0e-9:
                continue
            priority_score = agent.current_kappa * abs_z
            candidates.append((priority_score, agent, pos_y, pos_x))
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        taken: set[int] = set()
        selected: list[tuple[int, int]] = []
        for _, agent, pos_y, pos_x in candidates:
            y_index, x_index = agent.spec.y, agent.spec.x
            if y_index in taken or x_index in taken:
                continue
            taken.add(y_index)
            taken.add(x_index)
            positions[y_index] += pos_y
            positions[x_index] += pos_x
            selected.append((y_index, x_index))
        return positions, tuple(selected)

    def dollars(self, prices: np.ndarray) -> np.ndarray:
        positions, _ = self.dollars_with_pairs(prices)
        return positions


def _ridge_dollars(
    returns: np.ndarray,
    held_dollars: np.ndarray | None = None,
) -> np.ndarray:
    """Return ridge targets; ``held_dollars`` is accepted for API compatibility."""
    del held_dollars
    positions, _ = _ridge_forecast(returns)
    return positions


def _ridge_forecast(
    returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full-size ridge targets and normalized forecast strengths."""
    nins, n_returns = returns.shape
    positions = np.zeros(nins)
    strengths = np.zeros(nins)
    if n_returns + 1 < RIDGE_MIN_T:
        return positions, strengths

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
    if significant_links < RIDGE_GATE_N:
        return positions, strengths

    history = returns[1:, :].T
    scale = history.std(axis=0) + 1.0e-12
    predictors = history[:-1] / scale
    targets = history[1:] / scale
    coefficients = np.linalg.solve(
        predictors.T @ predictors
        + RIDGE_LAM * len(predictors) * np.eye(N_INST - 1),
        predictors.T @ targets,
    )
    prediction_scale = (predictors @ coefficients).std(axis=0)
    prediction = (history[-1] / scale) @ coefficients
    active = np.abs(prediction) > RIDGE_THETA * prediction_scale
    raw = (
        RIDGE_LEG
        * np.sign(prediction)
        * active
    )
    positions[1:] = raw
    strengths[1:] = np.abs(prediction) / np.maximum(
        prediction_scale,
        1.0e-12,
    )
    return positions, strengths


def _remove_ridge_conflicted_pairs(
    pair_dollars: np.ndarray,
    selected_pairs: tuple[tuple[int, int], ...],
    ridge_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove both legs of a pair if ridge claims either instrument."""
    retained = pair_dollars.copy()
    suppressed = np.zeros(len(pair_dollars), dtype=bool)
    for y_index, x_index in selected_pairs:
        if ridge_active[y_index] or ridge_active[x_index]:
            retained[y_index] = 0.0
            retained[x_index] = 0.0
            suppressed[y_index] = True
            suppressed[x_index] = True
    return retained, suppressed


_STATE: dict[str, AtlasBook | None] = {"atlas": None}


def reset() -> None:
    _STATE["atlas"] = None


def sleeve_breakdown(prcSoFar: np.ndarray) -> dict[str, np.ndarray] | None:
    """Return raw and barbell-arbitrated sleeves before final clipping."""
    prices = np.asarray(prcSoFar, dtype=float)
    nins, nt = prices.shape
    if nins != N_INST or nt < WARMUP + 10:
        return None
    if (
        not np.all(np.isfinite(prices[:, -2:]))
        or np.any(prices[:, -2:] <= 0.0)
    ):
        return None

    if _STATE["atlas"] is None:
        _STATE["atlas"] = AtlasBook()

    pair_raw, selected_pairs = _STATE["atlas"].dollars_with_pairs(prices)
    returns = np.diff(np.log(prices), axis=1)
    ridge_dollars, ridge_strength = _ridge_forecast(returns)
    ridge_active = np.abs(ridge_dollars) > 1.0
    pair_dollars, pair_suppressed = _remove_ridge_conflicted_pairs(
        pair_raw,
        selected_pairs,
        ridge_active,
    )
    return {
        "pairs_raw": pair_raw,
        "pairs": pair_dollars,
        "ridge": ridge_dollars,
        "ridge_strength": ridge_strength,
        "ridge_override_mask": ridge_active,
        "pair_suppressed_mask": pair_suppressed,
    }


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    prices = np.asarray(prcSoFar, dtype=float)
    nins, nt = prices.shape
    if nins != N_INST or nt < WARMUP + 10:
        return np.zeros(nins, dtype=int)

    sleeves = sleeve_breakdown(prices)
    if sleeves is None:
        return np.zeros(nins, dtype=int)

    dollars = sleeves["pairs"] + sleeves["ridge"]

    safe_prices = np.maximum(prices[:, -1], _base._EPS_DENOM)
    last_return = np.diff(
        np.log(np.maximum(prices[:, -2:], _base._EPS_DENOM)),
        axis=1,
    )[:, 0]
    crash = np.abs(last_return) > 0.40
    if len(crash) > 0:
        crash[0] = False
    dollars[crash] = 0.0

    if len(dollars) > 1:
        dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars[0] = 0.0

    raw_shares = dollars / safe_prices
    final_shares = np.round(raw_shares).astype(int)
    return np.nan_to_num(final_shares, nan=0).astype(int)

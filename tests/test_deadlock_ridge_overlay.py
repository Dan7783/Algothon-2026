"""Regression tests for the conservative ridge deadlock overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import four_sigmas as strategy


ROOT = Path(__file__).resolve().parents[1]
PRICES = np.loadtxt(ROOT / "prices.txt", skiprows=1).T


def sleeves_at(day: int) -> dict[str, object]:
    strategy.reset()
    sleeves = strategy.sleeve_breakdown(PRICES[:, :day])
    assert sleeves is not None
    return sleeves


def test_overlay_is_exactly_the_configured_fraction_of_instrument_limit():
    sleeves = sleeves_at(252)
    mask = np.asarray(sleeves["deadlock_ridge_overlay_mask"], dtype=bool)
    assert bool(sleeves["ridge_gate_pass"])
    assert mask.any()

    expected_mask = (
        np.asarray(sleeves["pair_active_mask"], dtype=bool)
        & (np.abs(np.asarray(sleeves["pair_vote"], dtype=float)) < 1.0e-12)
        & ~np.asarray(sleeves["ridge_active_mask"], dtype=bool)
        & (
            np.asarray(sleeves["ridge_strength"], dtype=float)
            > 0.0
        )
        & (
            np.asarray(sleeves["ridge_strength"], dtype=float)
            <= strategy.RIDGE_THETA
        )
        & (
            np.asarray(sleeves["ridge_latent_sign"], dtype=float)
            != 0.0
        )
        & ~np.asarray(sleeves["crash_mask"], dtype=bool)
    )
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_allclose(
        np.abs(np.asarray(sleeves["core_dollars"], dtype=float)[mask]),
        (
            strategy.DEADLOCK_RIDGE_OVERLAY_FRACTION
            * strategy.BANG_BANG_LIMITS[mask]
        ),
    )


def test_closed_global_ridge_gate_never_receives_deadlock_overlay():
    sleeves = sleeves_at(373)
    assert not bool(sleeves["ridge_gate_pass"])
    assert not np.asarray(
        sleeves["deadlock_ridge_overlay_mask"],
        dtype=bool,
    ).any()


def test_public_ridge_forecast_pair_remains_backward_compatible():
    returns = np.diff(np.log(PRICES[:, :252]), axis=1)
    positions, strengths = strategy._ridge_forecast(returns)
    detailed = strategy._ridge_forecast_details(returns)
    np.testing.assert_allclose(positions, detailed[0])
    np.testing.assert_allclose(strengths, detailed[1])

"""Regression tests for the zero-ALGO residual hedge basket."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import four_sigmas as strategy


ROOT = Path(__file__).resolve().parents[1]
PRICES = np.loadtxt(ROOT / "prices.txt", skiprows=1).T
ALGO = strategy.I["ALGO"]


def sleeves_at(
    day: int,
    enable_candidate: bool = True,
) -> dict[str, object]:
    original = strategy.USE_ALGO_RESIDUAL_BASKET
    try:
        strategy.USE_ALGO_RESIDUAL_BASKET = enable_candidate
        strategy.reset()
        sleeves = strategy.sleeve_breakdown(PRICES[:, :day])
        assert sleeves is not None
        return sleeves
    finally:
        strategy.reset()
        strategy.USE_ALGO_RESIDUAL_BASKET = original


def test_candidate_is_enabled_by_default():
    assert strategy.USE_ALGO_RESIDUAL_BASKET


def test_active_residual_overlay_has_zero_algo_and_zero_beta_dollars():
    sleeves = sleeves_at(600)
    alpha = np.asarray(sleeves["algo_residual_alpha"], dtype=float)
    hedge = np.asarray(sleeves["algo_residual_hedge"], dtype=float)
    overlay = np.asarray(sleeves["algo_residual_overlay"], dtype=float)
    dollars = np.asarray(sleeves["dollars"], dtype=float)

    assert bool(sleeves["algo_beta_ready"])
    assert np.any(np.abs(alpha[1:]) > 1.0e-9)
    assert np.any(np.abs(hedge[1:]) > 1.0e-9)
    assert alpha[ALGO] == 0.0
    assert hedge[ALGO] == 0.0
    assert overlay[ALGO] == 0.0
    assert dollars[ALGO] == 0.0
    assert abs(float(sleeves["algo_residual_beta_dollars"])) < 1.0e-6
    assert np.all(np.abs(dollars) <= strategy.BANG_BANG_LIMITS + 1.0e-8)


def test_residual_overlay_waits_for_causal_beta_warmup():
    sleeves = sleeves_at(50)
    assert not bool(sleeves["algo_beta_ready"])
    np.testing.assert_allclose(
        sleeves["algo_residual_overlay"],
        np.zeros(strategy.N_INST),
    )
    np.testing.assert_allclose(
        sleeves["dollars"],
        sleeves["core_dollars"],
    )


def test_public_position_never_holds_algo():
    strategy.reset()
    position = strategy.getMyPosition(PRICES[:, :600])
    assert position[ALGO] == 0
    limits = np.floor(
        strategy.BANG_BANG_LIMITS / PRICES[:, 599]
    ).astype(int)
    assert np.all(np.abs(position) <= limits)


def test_bounded_hedge_meets_feasible_beta_target():
    betas = np.linspace(0.4, 1.4, strategy.N_INST)
    lower = np.full(strategy.N_INST, -1_000.0)
    upper = np.full(strategy.N_INST, 1_000.0)
    eligible = np.ones(strategy.N_INST, dtype=bool)
    eligible[ALGO] = False
    target = -7_500.0

    hedge, delivery = strategy._bounded_minimum_norm_beta_hedge(
        target,
        betas,
        lower,
        upper,
        eligible,
    )

    assert delivery == 1.0
    assert hedge[ALGO] == 0.0
    assert abs(float(betas @ hedge) - target) < 1.0e-6
    assert np.all(hedge >= lower - 1.0e-9)
    assert np.all(hedge <= upper + 1.0e-9)

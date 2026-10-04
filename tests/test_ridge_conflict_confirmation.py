"""Regression tests for the dual-ridge conflict policy."""

from __future__ import annotations

import numpy as np

import four_sigmas as strategy


def resolve(
    *,
    pair_vote: float,
    pair_active: bool,
    ridge_sign: float,
    ridge_strength: float,
    ridge_active: bool,
    confirm_sign: float,
    confirm_active: bool,
) -> tuple[float, float, bool, bool]:
    result = strategy._resolve_pair_ridge_direction(
        np.asarray([pair_vote], dtype=float),
        np.asarray([pair_active], dtype=bool),
        np.asarray([ridge_sign], dtype=float),
        np.asarray([ridge_strength], dtype=float),
        np.asarray([ridge_active], dtype=bool),
        np.asarray([confirm_sign], dtype=float),
        np.asarray([confirm_active], dtype=bool),
    )
    return (
        float(result[0][0]),
        float(result[1][0]),
        bool(result[2][0]),
        bool(result[3][0]),
    )


def test_unconfirmed_ridge_cannot_overturn_an_opposing_pair_vote():
    combined, direction, conflict, confirmed = resolve(
        pair_vote=0.50,
        pair_active=True,
        ridge_sign=-1.0,
        ridge_strength=1.0,
        ridge_active=True,
        confirm_sign=1.0,
        confirm_active=True,
    )
    assert combined < 0.0
    assert direction == 1.0
    assert conflict
    assert not confirmed


def test_confirmed_ridge_can_overturn_an_opposing_pair_vote():
    combined, direction, conflict, confirmed = resolve(
        pair_vote=0.50,
        pair_active=True,
        ridge_sign=-1.0,
        ridge_strength=1.0,
        ridge_active=True,
        confirm_sign=-1.0,
        confirm_active=True,
    )
    assert combined < 0.0
    assert direction == -1.0
    assert conflict
    assert confirmed


def test_confirmation_does_not_turn_an_exact_conflict_tie_into_a_win():
    combined, direction, conflict, confirmed = resolve(
        pair_vote=0.75,
        pair_active=True,
        ridge_sign=-1.0,
        ridge_strength=1.0,
        ridge_active=True,
        confirm_sign=-1.0,
        confirm_active=True,
    )
    assert combined == 0.0
    assert direction == 1.0
    assert conflict
    assert not confirmed


def test_ridge_keeps_unused_and_deadlocked_asset_slots():
    for pair_active in (False, True):
        _, direction, conflict, confirmed = resolve(
            pair_vote=0.0,
            pair_active=pair_active,
            ridge_sign=-1.0,
            ridge_strength=1.0,
            ridge_active=True,
            confirm_sign=1.0,
            confirm_active=False,
        )
        assert direction == -1.0
        assert not conflict
        assert not confirmed

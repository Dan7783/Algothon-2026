"""Ridge-first barbell with six stable ``testbedv2_tuned`` pair agents.

This preserves the current Four Sigmas ridge VAR as a full standalone sleeve.
The pair side deliberately uses the same six disjoint relationships as the
earlier ``simple_pairs`` control, but retrieves their tuned filter/trading
parameters and PairAgent implementation from ``testbedv2_tuned``.

Allocation uses a confidence barbell:

* ridge enters normally above 1x its prediction scale;
* ordinary ridge signals fill idle or agreeing residual capacity;
* an exceptional forecast above 3x prediction scale receives a full
  +/-$10,000 target and suppresses any pair touching that instrument.

The separate entry and override thresholds preserve ridge's independent
signal without allowing marginal forecasts to evict a complete hedge.
"""

from __future__ import annotations

import numpy as np

import testbedv2_tuned as _pairs
import testbedv2_tuned_ridge as _ridge


I = _pairs.I
N_INST = _pairs.N_INST
LEG = _pairs.LEG
WARMUP = _pairs.WARMUP
FiltCfg = _pairs.FiltCfg
TradeCfg = _pairs.TradeCfg
PairCfg = _pairs.PairCfg
PairSegFilter = _pairs.PairSegFilter
PairAgent = _pairs.PairAgent

RIDGE_LAM = _ridge.RIDGE_LAM
RIDGE_THETA = _ridge.RIDGE_THETA
RIDGE_GATE_N = _ridge.RIDGE_GATE_N
RIDGE_LEG = _ridge.RIDGE_LEG
RIDGE_MIN_T = _ridge.RIDGE_MIN_T
RIDGE_OVERRIDE_THETA = 3.0


SELECTED_PAIR_NAMES = frozenset(
    {
        "AENO-NWIG",
        "CTGI-EELT",
        "HETT-ULXY",
        "HUXZ-ACAC",
        "RTTH-NAYO",
        "SMAH-ILVX",
    }
)

_SYMBOL_BY_INDEX = {index: symbol for symbol, index in I.items()}


def _pair_name(spec: PairCfg) -> str:
    return f"{_SYMBOL_BY_INDEX[spec.y]}-{_SYMBOL_BY_INDEX[spec.x]}"


PAIR_CFGS = tuple(
    spec._replace(idx=new_index)
    for new_index, spec in enumerate(
        spec
        for spec in _pairs.PAIR_CFGS
        if _pair_name(spec) in SELECTED_PAIR_NAMES
    )
)

if {_pair_name(spec) for spec in PAIR_CFGS} != SELECTED_PAIR_NAMES:
    missing = SELECTED_PAIR_NAMES - {_pair_name(spec) for spec in PAIR_CFGS}
    raise RuntimeError(f"Missing tuned pair configurations: {sorted(missing)}")


class AtlasBook:
    """Six disjoint tuned pairs with optional high-confidence suppression."""

    def __init__(self) -> None:
        self.agents = [PairAgent(spec) for spec in PAIR_CFGS]

    def dollars(
        self,
        prices: np.ndarray,
        ridge_active: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        nins, nt = prices.shape
        positions = np.zeros(nins)
        suppressed = np.zeros(nins, dtype=bool)
        if nt < WARMUP + 10:
            return positions, suppressed
        if ridge_active is None:
            ridge_active = np.zeros(nins, dtype=bool)

        for agent in self.agents:
            agent.catch_up(prices)

        for agent in self.agents:
            pos_y, pos_x, _ = agent.dollar_positions()
            if abs(pos_y) < 1.0e-9:
                continue
            y_index, x_index = agent.spec.y, agent.spec.x
            if ridge_active[y_index] or ridge_active[x_index]:
                suppressed[y_index] = True
                suppressed[x_index] = True
                continue
            positions[y_index] = pos_y
            positions[x_index] = pos_x
        return positions, suppressed


def _subordinate_ridge(
    ridge_raw: np.ndarray,
    pair_dollars: np.ndarray,
) -> np.ndarray:
    """Apply the original ridge residual-capacity allocation."""
    allocation = np.zeros(N_INST)
    held = pair_dollars[1:]
    raw = ridge_raw[1:]
    idle = np.abs(held) <= 1.0
    allocation[1:][idle] = raw[idle]

    agrees = (
        (~idle)
        & (np.abs(raw) > 1.0)
        & (np.sign(raw) == np.sign(held))
    )
    room = np.maximum(
        0.0,
        RIDGE_LEG - np.minimum(np.abs(held), RIDGE_LEG),
    )
    allocation[1:][agrees] = (
        np.sign(raw[agrees])
        * np.minimum(np.abs(raw[agrees]), room[agrees])
    )
    return allocation


_STATE: dict[str, AtlasBook | None] = {"atlas": None}


def reset() -> None:
    _STATE["atlas"] = None


def sleeve_breakdown(prcSoFar: np.ndarray) -> dict[str, np.ndarray] | None:
    """Return raw and allocated sleeves plus barbell diagnostics."""
    prices = np.asarray(prcSoFar, dtype=float)
    nins, nt = prices.shape
    if nins != N_INST or nt < WARMUP + 10:
        return None
    if (
        not np.all(np.isfinite(prices[:, -2:]))
        or np.any(prices[:, -2:] <= 0.0)
    ):
        return None

    returns = np.diff(np.log(prices), axis=1)
    ridge_raw, ridge_strength = _ridge._ridge_forecast(returns)
    ridge_active = np.abs(ridge_raw) > 1.0
    ridge_override = (
        ridge_active
        & (ridge_strength > RIDGE_OVERRIDE_THETA)
    )

    if _STATE["atlas"] is None:
        _STATE["atlas"] = AtlasBook()
    pair_raw, _ = _STATE["atlas"].dollars(
        prices,
        ridge_active=None,
    )
    pair_dollars, pair_suppressed = _STATE["atlas"].dollars(
        prices,
        ridge_active=ridge_override,
    )
    ridge_dollars = _subordinate_ridge(ridge_raw, pair_dollars)
    ridge_dollars[ridge_override] = ridge_raw[ridge_override]
    return {
        "pairs_raw": pair_raw,
        "pairs": pair_dollars,
        "ridge_raw": ridge_raw,
        "ridge": ridge_dollars,
        "ridge_strength": ridge_strength,
        "ridge_active_mask": ridge_active,
        "ridge_override_mask": ridge_override,
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

    dollars = sleeves["ridge"] + sleeves["pairs"]

    safe_prices = np.maximum(prices[:, -1], _pairs._EPS_DENOM)
    last_return = np.diff(
        np.log(np.maximum(prices[:, -2:], _pairs._EPS_DENOM)),
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

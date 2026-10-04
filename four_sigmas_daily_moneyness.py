"""One-day dollar-moneyness conflict engine.

Pair opportunity is remaining displacement divided by dynamic OU half-life.
Ridge opportunity is its raw next-return forecast. Both are next-day dollars,
combined as complete vectors without probability calibration or new knobs.
"""

from __future__ import annotations

from math import log

import numpy as np

import four_sigmas_moneyness as base


class _DailyMoneynessBook(base._WholeVectorMoneynessBook):
    def _pair_gross_money(
        self,
        agent: base.core.PairAgent,
        pos_y: float,
        abs_z: float,
    ) -> float:
        full_money = super()._pair_gross_money(
            agent,
            pos_y,
            abs_z,
        )
        kappa = max(float(agent.current_kappa), base.core._EPS_DENOM)
        reversion_days = max(log(2.0) / kappa, 1.0)
        return full_money / reversion_days


_BOOK: _DailyMoneynessBook | None = None


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
        _BOOK = _DailyMoneynessBook()
    dollars = _BOOK.dollars(prices)
    current = np.maximum(prices[:, -1], core._EPS_DENOM)
    limits = (base._POSITION_LIMIT / current).astype(int)
    requested = np.round(dollars / current).astype(int)
    return np.clip(requested, -limits, limits).astype(int)

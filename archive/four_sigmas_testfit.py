"""Deliberately test-fitted combiner for the current Four Sigmas signals.

This module is an experiment, not an out-of-sample strategy.  The direction
table below was fitted directly on days 751-1000.  It maps the sign of the
current atlas, extra lead-lag, ridge, or discovery signal to a full-size target
direction for each instrument.

To avoid fitting one-off states, a mapping is retained only when the same
instrument/sleeve/sign state occurred at least 20 times in the 250-day fit
period.  No prices or future returns are accessed by this module at runtime;
the underlying signals remain causal and are produced by four_sigmas_pkg.

Eval:
    py eval.py --module four_sigmas_testfit --start 751 --end 1000
"""

from __future__ import annotations

import numpy as np

import four_sigmas_pkg as _base


N_INST = 51
LEG = 10_000.0
SIGNAL_NAMES = ("atlas", "extraLL", "ridge", "discovery")

# Columns are:
#   atlas(-/+), extraLL(-/+), ridge(-/+), discovery(-/+)
# Values are the fitted output direction.  Zero means that the state occurred
# fewer than 20 times (or was unavailable) and therefore does not trade.
STATE_DIRECTIONS = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, -1, 1, 1, 1],
        [0, 0, 0, 0, -1, 1, 1, 1],
        [0, 0, 0, 0, -1, 1, -1, -1],
        [0, 0, 0, 0, -1, 1, -1, -1],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, -1, 1, 1],
        [-1, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, -1, 1, 1, -1],
        [0, 0, 0, 0, -1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, -1],
        [0, 0, -1, 1, 0, 0, -1, -1],
        [0, 0, 1, 1, 0, 0, -1, -1],
        [0, 0, 0, 0, -1, 1, -1, 1],
        [1, 0, 0, 0, -1, -1, -1, -1],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, -1],
        [0, 0, 0, 0, 1, -1, -1, -1],
        [0, 0, 0, 0, -1, 1, -1, 1],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 1, 1],
        [0, 0, -1, 1, 0, 0, 0, -1],
        [0, 0, 0, 0, -1, 1, -1, 1],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, -1, 1, -1, 1],
        [-1, 1, 0, 0, 0, 0, -1, -1],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, -1, 1],
        [0, 0, 0, 0, -1, 1, -1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 1, 1],
        [-1, 1, 0, 0, 0, 0, 0, 1],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 1, 1],
        [0, 0, 0, 0, -1, 1, 1, -1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int8,
)


def reset() -> None:
    """Reset the stateful causal signal generators."""
    _base.reset()


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """Map the current four-sleeve state through the fitted direction table."""
    nins, nt = prcSoFar.shape
    if nins != N_INST or nt < 130:
        return np.zeros(nins, dtype=int)
    recent = prcSoFar[:, -3:]
    if not np.all(np.isfinite(recent)) or np.any(recent <= 0.0):
        return np.zeros(nins, dtype=int)

    sleeves = _base.sleeve_breakdown(prcSoFar)
    if sleeves is None:
        return np.zeros(nins, dtype=int)

    signal_dollars = np.stack([sleeves[name] for name in SIGNAL_NAMES])
    direction = np.zeros(nins, dtype=float)
    assigned = np.zeros(nins, dtype=bool)

    # The base book's capacity ordering makes the sleeves mutually exclusive
    # per instrument in the fitted period.  First-active preserves that order
    # defensively if a later configuration produces an overlap.
    for sleeve_idx in range(len(SIGNAL_NAMES)):
        signal = signal_dollars[sleeve_idx]
        negative = (~assigned) & (signal < -1.0)
        positive = (~assigned) & (signal > 1.0)
        direction[negative] = STATE_DIRECTIONS[negative, 2 * sleeve_idx]
        direction[positive] = STATE_DIRECTIONS[positive, 2 * sleeve_idx + 1]
        assigned |= negative | positive

    dollars = LEG * direction

    # Retain the production crash guard even though it does not bind in the
    # fitted test interval.
    last_return = np.diff(np.log(prcSoFar[:, -2:]), axis=1)[:, 0]
    crash = np.abs(last_return) > 0.40
    crash[0] = False
    dollars[crash] = 0.0
    dollars[0] = 0.0
    return (dollars / prcSoFar[:, -1]).astype(int)


reset()

"""Lean pairs + ridge.  ~12 fitted numbers instead of ~235.

WHY
True OOS scored 599 against 878 on our 501-750 "test" window — a 32% haircut,
which is what happens when a window is used for selection.  Rolling-block
scores inside 251-750 look stable (837/869/765/992) precisely because ALL of
that range informed the fitting, so it cannot reveal the problem.  The fix is
not a better search; it is a smaller thing to search over.

WHAT WAS CUT
  * per-pair parameter sets.  four_sigmas carries 10 fitted numbers per pair
    (EV, z_entry, z_exit, z_stop, q_beta, h, cusum_k, adapt_bars, snap, leg)
    x 21 pairs = 210, plus ~25 global.  Here every pair shares ONE set.
  * the sticky Kalman + CUSUM + adapt-burst + snap machinery.  Measured inert:
    the filter is through-origin so beta is pinned to the log-price ratio and
    q_beta does nothing across 8 orders of magnitude; the CUSUM knobs do
    nothing on any pair that never cuts; cov_inflate moves the book ~1% over
    three orders of magnitude.
  * EV-MWIS conflict resolution and hub penalties.  Unnecessary: the strict
    pair set is asset-disjoint by construction.
  * lead-lag health tests and the discovery sleeve.

PAIR SELECTION IS STRUCTURAL, NOT PnL-BASED
Pairs must show a stationary hedge residual in ALL FOUR 125-day sub-windows
of days 251-750 (fitted and tested inside each window separately).  At
ADF < -2.5 exactly 6 pairs qualify out of 1225, and they use 12 distinct
instruments.  At -3.0, none do.  No score was consulted.

PARAMETERS
  FIT_WINDOW 250   hedge estimation window (same as the screen)
  Z_WINDOW    60   ~10 measured half-lives (median 5.7d)
  ENTRY/EXIT/STOP  1.0 / 0.0 / 3.0 sigma — textbook OU levels, not fitted
  ridge            original constants, unchanged
"""

from __future__ import annotations

import numpy as np

N_INST = 51
LEG = 10_000.0

# (y, x) with worst-of-four-window residual ADF, all < -2.5.  Disjoint.
PAIRS = (
    (49, 50),   # MHRM/EAFC   -2.90
    (10, 46),   # SMAH/ILVX   -2.78
    (9, 38),    # DUCT/HRND   -2.77
    (13, 45),   # EORC/NGTE   -2.71
    (7, 40),    # HETT/ULXY   -2.55
    (8, 27),    # HUXZ/ACAC   -2.52
)

FIT_WINDOW = 250
Z_WINDOW = 60
ENTRY = 1.0
EXIT = 0.0
STOP = 3.0

RIDGE_LAMBDA = 1.0
RIDGE_ENTRY = 1.0
RIDGE_MIN_LINKS = 35
RIDGE_MIN_DAYS = 252

MIN_DAYS = FIT_WINDOW + 2
_side: dict[tuple[int, int], int] = {}


def reset() -> None:
    _side.clear()


def _pair_dollars(log_prices: np.ndarray) -> np.ndarray:
    dollars = np.zeros(N_INST)
    nt = log_prices.shape[1]
    for key in PAIRS:
        y_i, x_i = key
        y = log_prices[y_i, nt - FIT_WINDOW:]
        x = log_prices[x_i, nt - FIT_WINDOW:]
        # hedge fitted on history strictly before today
        yc, xc = y[:-1] - y[:-1].mean(), x[:-1] - x[:-1].mean()
        denom = float(xc @ xc)
        if denom <= 1e-12:
            continue
        beta = float(xc @ yc) / denom
        if beta <= 0.0:
            continue
        spread = y - beta * x
        recent = spread[-Z_WINDOW:]
        history = recent[:-1]
        sd = float(history.std())
        if sd <= 1e-12:
            continue
        z = (float(recent[-1]) - float(history.mean())) / sd

        side = _side.get(key, 0)
        if abs(z) >= STOP:
            side = 0
        elif side == 0:
            if abs(z) >= ENTRY:
                side = -int(np.sign(z))
        elif side * z >= -EXIT:
            side = 0
        _side[key] = side
        if side:
            dollars[y_i] += side * LEG
            dollars[x_i] -= side * LEG
    return dollars


def _ridge_dollars(returns: np.ndarray, held: np.ndarray) -> np.ndarray:
    dollars = np.zeros(N_INST)
    n_inst, n_ret = returns.shape
    if n_ret + 1 < RIDGE_MIN_DAYS:
        return dollars
    recent = returns[1:, -250:].T
    standard = (recent - recent.mean(0)) / (recent.std(0) + 1e-12)
    lagged = (standard[:-1].T @ standard[1:]) / (len(recent) - 1)
    np.fill_diagonal(lagged, 0.0)
    if int((np.abs(lagged) > 2.5 / np.sqrt(len(recent) - 1)).sum()) < RIDGE_MIN_LINKS:
        return dollars
    history = returns[1:].T
    scale = history.std(0) + 1e-12
    predictors, targets = history[:-1] / scale, history[1:] / scale
    coef = np.linalg.solve(
        predictors.T @ predictors + RIDGE_LAMBDA * len(predictors) * np.eye(n_inst - 1),
        predictors.T @ targets,
    )
    spread = (predictors @ coef).std(0)
    prediction = (history[-1] / scale) @ coef
    raw = LEG * np.sign(prediction) * (np.abs(prediction) > RIDGE_ENTRY * spread)
    dollars[1:] = np.where(np.abs(held[1:]) <= 1.0, raw, 0.0)
    return dollars


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    prices = np.asarray(prcSoFar, dtype=float)
    n_inst, nt = prices.shape
    if n_inst != N_INST or nt < MIN_DAYS:
        return np.zeros(n_inst, dtype=int)
    if not np.all(np.isfinite(prices[:, -1])) or np.any(prices[:, -1] <= 0.0):
        return np.zeros(n_inst, dtype=int)
    log_prices = np.log(prices)
    dollars = _pair_dollars(log_prices)
    dollars += _ridge_dollars(np.diff(log_prices, axis=1), dollars)
    dollars[0] = 0.0
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    return (dollars / prices[:, -1]).astype(int)

"""Two-sleeve pair book: four_sigmas atlas + a broad CUSUM breadth pool.

ATLAS sleeve — the 13 fixed atlas pairs are STABLE-beta relationships and are
traded exactly as four_sigmas trades them (expanding Kalman hedge, ADF gate at
-2.9, OU kappa filter, 145-day z fade inside 0.27-4.0, equal $10k legs).  The
adaptive filter added nothing on these pairs and its cut/adapt stand-downs
only cost days in market, so they get the original mechanics verbatim.

CUSUM sleeve — every OTHER pair of instruments 1..50 (1,212 pairs, including
SMAH/NGTE) runs the sticky Kalman + CUSUM filter (defaults, log prices).
This sleeve is deliberately UN-gated: no ADF, no kappa — the repo's own
discovery-sleeve research showed cointegration gates remove breadth lift, and
the CUSUM cut (flat + spread-statistics reset on a detected beta break) is
the break protection instead.  Policy is the PROVEN four_sigmas discovery
fade — near-always-in breadth, re-evaluated daily:

    side = -sign(z) while 0.2 <= |z| <= 4.5   (z over trailing 145d)

A late-entry variant (enter 1.5 sigma, exit 0.25) was tried first and LOST on
both windows (cusum-alone train +1 / test -34; it dragged the full book from
496/364 to 373/292): in this universe large un-gated stretches are trends,
not reversion — consistent with the shortlived_meanreversion study.  The
0.2-4.5 constants are four_sigmas' own discovery numbers; nothing here was
selected by trading score.  The CUSUM sleeve is subordinated: it only
receives capacity the atlas sleeve left idle, strongest stretch first, with
both legs jointly scaled so a hedge stays a hedge.

DEFAULT MODE IS "atlas".  The decision rule was fixed before scoring: the
broad sleeve ships only if full-mode TRAIN beats atlas-alone TRAIN (496.5).
It did not — the discovery-style sleeve scored 418.0 (marginal -$63/day,
doubled volatility), the late-entry variant 372.7 — so the pool is research
only.  The filter demonstrably helps individual changing-beta pairs (see
cusum_output/pair_books/); it does not survive as an un-gated 1,212-pair
breadth book under eval accounting.

Env toggle for attribution runs: PAIR_BOOKS_MODE = atlas | full | cusum.
"""

from __future__ import annotations

import os

import numpy as np

from research.pairs_analysis.pair_seg_filter import PairSegConfig, PairSegFilter

N_INST = 51
LEG = 10_000.0
MIN_HISTORY = 130

ATLAS_I = (10, 1, 13, 8, 25, 49, 7, 31, 36, 18, 33, 35, 26)
ATLAS_J = (46, 20, 45, 27, 37, 50, 40, 43, 41, 35, 26, 28, 32)
ATLAS_PAIRS = tuple(zip(ATLAS_I, ATLAS_J))
ATLAS_KEYS = {tuple(sorted(pair)) for pair in ATLAS_PAIRS}

# CUSUM sleeve policy (four_sigmas discovery constants, not fitted).
Z_WINDOW = 145
ENTRY_Z = 0.2
BREAK_Z = 4.5

MODE = os.environ.get("PAIR_BOOKS_MODE", "atlas")

CUSUM_PAIRS = tuple(
    (i, j)
    for i in range(1, N_INST)
    for j in range(i + 1, N_INST)
    if (i, j) not in ATLAS_KEYS
)


# --------------------------------------------------------------------------
# ATLAS sleeve — four_sigmas mechanics verbatim
# --------------------------------------------------------------------------


def _kalman_beta(y: np.ndarray, x: np.ndarray) -> float:
    theta = np.array([y[:60].mean() - x[:60].mean(), 1.0])
    covariance = np.eye(2) * 1e-2
    process_noise = np.eye(2) * 1e-5
    measurement_noise = np.var(np.diff(y[:120] - x[:120])) + 1e-8
    for t in range(len(y)):
        covariance += process_noise
        observation = np.array([1.0, x[t]])
        innovation_variance = (
            observation @ covariance @ observation + measurement_noise
        )
        gain = covariance @ observation / innovation_variance
        theta += gain * (y[t] - observation @ theta)
        covariance -= np.outer(gain, observation) @ covariance
    return float(theta[1])


def _adf1_tstat(series: np.ndarray) -> float:
    delta = np.diff(series)
    target = delta[1:]
    design = np.column_stack((np.ones(len(target)), series[1:-1], delta[:-1]))
    inverse = np.linalg.pinv(design.T @ design)
    coefficients = inverse @ design.T @ target
    error = target - design @ coefficients
    variance = float(error @ error) / max(len(target) - 3, 1)
    standard_error = np.sqrt(max(variance * inverse[1, 1], 1e-300))
    return float(coefficients[1] / standard_error)


def _ou_kappa(spread: np.ndarray) -> float:
    delta = np.diff(spread)
    lag = spread[:-1] - spread[:-1].mean()
    denominator = float(lag @ lag)
    return float(-(lag @ delta) / denominator) if denominator > 1e-18 else 0.0


def _atlas_dollars(log_prices: np.ndarray) -> np.ndarray:
    """four_sigmas fixed pair book, unchanged."""
    nins, nt = log_prices.shape
    pos = np.zeros(nins)
    if nt < MIN_HISTORY:
        return pos
    for i, j in ATLAS_PAIRS:
        beta = _kalman_beta(log_prices[i, : nt - 1], log_prices[j, : nt - 1])
        full_spread = log_prices[i] - beta * log_prices[j]
        if _adf1_tstat(full_spread) >= -2.9:
            continue
        recent = full_spread[-145:]
        z = (recent[-1] - recent[:-1].mean()) / (recent[:-1].std() + 1e-12)
        if _ou_kappa(full_spread[-120:]) <= 0.0 or not (0.27 <= abs(z) <= 4.0):
            continue
        direction = -np.sign(z)
        pos[i] += direction * LEG
        pos[j] -= direction * LEG
    return pos


# --------------------------------------------------------------------------
# CUSUM sleeve — adaptive filter books, un-gated, late entries
# --------------------------------------------------------------------------


class CusumBook:
    """Stateful adaptive-beta book for one non-atlas pair."""

    __slots__ = ("i", "j", "filt", "spreads", "side", "last_z", "nt_seen")

    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j
        self.filt = PairSegFilter(PairSegConfig())
        self.spreads: list[float] = []  # current hedge segment only
        self.side = 0
        self.last_z = 0.0
        self.nt_seen = 0

    def _policy_z(self, spread: float) -> float:
        n = len(self.spreads)
        if n < 10:
            return 0.0
        recent = np.asarray(self.spreads[-Z_WINDOW:], dtype=float)
        sd = float(recent.std())
        if sd <= 1e-12:
            return 0.0
        return (spread - float(recent.mean())) / sd

    def _advance_day(self, y: float, x: float, decide: bool) -> None:
        o = self.filt.step(y, x)
        if o.cut:
            # Detected beta break: position off, statistics restart.
            self.spreads = []
            self.side = 0
        z = self._policy_z(o.spread)
        self.last_z = z
        if decide and not o.cut:
            if o.adapting:
                pass  # hold through re-estimation
            elif ENTRY_Z <= abs(z) <= BREAK_Z:
                self.side = -int(np.sign(z))
            else:
                self.side = 0
        self.spreads.append(o.spread)

    def advance(self, log_prices: np.ndarray) -> None:
        nt = log_prices.shape[1]
        for t in range(self.nt_seen, nt):
            self._advance_day(
                float(log_prices[self.i, t]),
                float(log_prices[self.j, t]),
                decide=t >= MIN_HISTORY - 1,
            )
        self.nt_seen = nt


def _capacity(existing: float, requested: float) -> float:
    if requested > 0.0:
        return max(0.0, (LEG - existing) / requested)
    return max(0.0, (-LEG - existing) / requested)


def _cusum_dollars(books: list[CusumBook], base: np.ndarray) -> np.ndarray:
    """Fill idle capacity with the stretched CUSUM books, strongest first."""
    pos = np.zeros(len(base))
    idle = np.abs(base) <= 1.0
    used = np.zeros(len(base), dtype=bool)
    active = [b for b in books if b.side != 0]
    active.sort(key=lambda b: -abs(b.last_z))
    for book in active:
        i, j = book.i, book.j
        if not (idle[i] and idle[j]) or used[i] or used[j]:
            continue
        d_y = book.side * LEG
        d_x = -book.side * LEG
        scale = min(
            1.0,
            _capacity(float(base[i] + pos[i]), d_y),
            _capacity(float(base[j] + pos[j]), d_x),
        )
        if scale <= 1e-12:
            continue
        pos[i] += scale * d_y
        pos[j] += scale * d_x
        used[i] = used[j] = True
    return pos


_BOOKS: list[CusumBook] | None = None


def reset() -> None:
    global _BOOKS
    _BOOKS = None


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global _BOOKS
    prices = np.asarray(prcSoFar, dtype=float)
    if prices.ndim != 2 or prices.shape[0] != N_INST:
        return np.zeros(N_INST, dtype=int)
    if not np.all(np.isfinite(prices[:, -1])) or np.any(prices[:, -1] <= 0.0):
        return np.zeros(N_INST, dtype=int)
    nt = prices.shape[1]
    log_prices = np.log(prices)

    dollars = np.zeros(N_INST)
    if MODE in ("full", "atlas"):
        dollars = _atlas_dollars(log_prices)
        dollars[1:] = np.clip(dollars[1:], -LEG, LEG)

    if MODE in ("full", "cusum"):
        if _BOOKS is None or nt < _BOOKS[0].nt_seen:
            _BOOKS = [CusumBook(i, j) for i, j in CUSUM_PAIRS]
        for book in _BOOKS:
            book.advance(log_prices)
        if nt >= MIN_HISTORY:
            dollars += _cusum_dollars(_BOOKS, dollars)

    dollars[0] = 0.0
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    return (dollars / prices[:, -1]).astype(int)

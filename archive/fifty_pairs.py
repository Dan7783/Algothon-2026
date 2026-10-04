"""Fifty labeled pairs: gateless-atlas stable sleeve + gated CUSUM adaptive sleeve.

User-classified universe (16 stable / 34 adaptive).  NO ADF gate and NO
OU-kappa decay filter anywhere, per instruction.

STABLE sleeve — four_sigmas atlas mechanics with the gates removed:
expanding intercept/slope Kalman hedge on log prices (beta always fitted on
days strictly before today), centered z over the trailing 145 days, fade
while 0.27 <= |z| <= 4.0, equal $10k legs, re-evaluated daily.  Sleeve
params (W=145, lo=0.27) chosen on train 251-500 by best worst-half score —
they coincide with four_sigmas' own constants.

ADAPTIVE sleeve — sticky CUSUM filter (defaults, log prices), centered z on
the CURRENT hedge segment (statistics reset on a cut), fade while
0.40 <= |z| <= 4.0, flat on a cut, hold through adapt bursts.  Break
protection is the CUSUM itself, not a stationarity statistic: a book may
hold or enter only while its filter has cut AT MOST once in the trailing
250 days — pairs that keep breaking have nothing stable to fade (this gate
rescued the sleeve from a ~0 train wash; W/lo/max_cuts selected on train
halves only).

Allocation follows the user's listed order, stable sleeve first; a book
whose instruments are already committed gets both legs jointly scaled so a
hedge can never become a directional position.

Attribution toggle: FIFTY_MODE = both | stable | adaptive.
"""

from __future__ import annotations

import os

import numpy as np

from research.pairs_analysis.pair_seg_filter import PairSegConfig, PairSegFilter

N_INST = 51
LEG = 10_000.0
MIN_HISTORY = 130

STABLE_PAIRS = (
    (1, 20), (46, 10), (13, 45), (27, 8), (7, 40), (49, 28), (50, 28),
    (32, 28), (12, 40), (36, 41), (33, 42), (15, 25), (31, 40), (17, 18),
    (49, 30), (8, 20),
)
ADAPTIVE_PAIRS = (
    (13, 10), (50, 15), (10, 45), (7, 28), (50, 37), (40, 4), (36, 10),
    (14, 40), (27, 25), (8, 25), (49, 46), (39, 2), (45, 46), (44, 18),
    (45, 25), (10, 25), (9, 36), (42, 41), (11, 35), (41, 25), (48, 41),
    (17, 47), (2, 42), (36, 48), (28, 22), (25, 22), (38, 24), (44, 12),
    (13, 46), (4, 34), (49, 24), (43, 46), (46, 7), (3, 34),
)

STABLE_W = 145
STABLE_LO = 0.27
ADAPTIVE_W = 145
ADAPTIVE_LO = 0.40
BAND_HI = 4.0
CUT_LOOKBACK = 250
MAX_RECENT_CUTS = 1

MODE = os.environ.get("FIFTY_MODE", "both")


class StableBook:
    """Gateless four_sigmas atlas mechanics, stepped incrementally."""

    __slots__ = ("i", "j", "theta", "cov", "noise", "assimilated", "side")

    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j
        self.theta: np.ndarray | None = None
        self.cov = np.eye(2) * 1e-2
        self.noise = 0.0
        self.assimilated = 0
        self.side = 0

    def update(self, ly: np.ndarray, lx: np.ndarray) -> None:
        nt = len(ly)
        if nt < MIN_HISTORY:
            return
        if self.theta is None:
            self.theta = np.array([ly[:60].mean() - lx[:60].mean(), 1.0])
            self.noise = float(np.var(np.diff(ly[:120] - lx[:120]))) + 1e-8
        # beta excludes today: assimilate days [assimilated, nt-1)
        for t in range(self.assimilated, nt - 1):
            self.cov += np.eye(2) * 1e-5
            observation = np.array([1.0, lx[t]])
            innovation_variance = (
                observation @ self.cov @ observation + self.noise
            )
            gain = self.cov @ observation / innovation_variance
            self.theta += gain * (ly[t] - observation @ self.theta)
            self.cov -= np.outer(gain, observation) @ self.cov
        self.assimilated = nt - 1

        beta = float(self.theta[1])
        recent = ly[-STABLE_W:] - beta * lx[-STABLE_W:]
        history = recent[:-1]
        sd = float(history.std())
        z = (recent[-1] - float(history.mean())) / sd if sd > 1e-12 else 0.0
        if STABLE_LO <= abs(z) <= BAND_HI:
            self.side = -int(np.sign(z))
        else:
            self.side = 0


class AdaptiveBook:
    """CUSUM filter book; break protection is the cut count, not ADF."""

    __slots__ = ("i", "j", "filt", "spreads", "cut_days", "side", "nt_seen")

    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j
        self.filt = PairSegFilter(PairSegConfig())
        self.spreads: list[float] = []  # current hedge segment only
        self.cut_days: list[int] = []
        self.side = 0
        self.nt_seen = 0

    def _advance_day(self, y: float, x: float, decide: bool) -> None:
        t = self.nt_seen
        o = self.filt.step(y, x)
        if o.cut:
            self.spreads = []
            self.cut_days.append(t)
            self.side = 0
        n = len(self.spreads)
        z = 0.0
        if n >= 10:
            recent = np.asarray(self.spreads[-ADAPTIVE_W:], dtype=float)
            sd = float(recent.std())
            if sd > 1e-12:
                z = (o.spread - float(recent.mean())) / sd
        recent_cuts = sum(1 for d in self.cut_days if d > t - CUT_LOOKBACK)
        if decide:
            if o.cut or recent_cuts > MAX_RECENT_CUTS:
                self.side = 0
            elif o.adapting:
                pass
            elif ADAPTIVE_LO <= abs(z) <= BAND_HI:
                self.side = -int(np.sign(z))
            else:
                self.side = 0
        self.spreads.append(o.spread)
        self.nt_seen = t + 1

    def advance(self, log_prices: np.ndarray) -> None:
        nt = log_prices.shape[1]
        for t in range(self.nt_seen, nt):
            self._advance_day(
                float(log_prices[self.i, t]),
                float(log_prices[self.j, t]),
                decide=t >= MIN_HISTORY - 1,
            )


def _capacity(existing: float, requested: float) -> float:
    """Room for a same-direction addition; zero if it would net a hedge.

    An opposite-sign request must NOT consume an earlier book's leg — that
    would silently turn the earlier pair into a directional position.
    """
    if requested > 0.0:
        if existing < -1.0:
            return 0.0
        return max(0.0, (LEG - existing) / requested)
    if existing > 1.0:
        return 0.0
    return max(0.0, (-LEG - existing) / requested)


_STABLE: list[StableBook] | None = None
_ADAPTIVE: list[AdaptiveBook] | None = None


def reset() -> None:
    global _STABLE, _ADAPTIVE
    _STABLE = None
    _ADAPTIVE = None


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global _STABLE, _ADAPTIVE
    prices = np.asarray(prcSoFar, dtype=float)
    if prices.ndim != 2 or prices.shape[0] != N_INST:
        return np.zeros(N_INST, dtype=int)
    if not np.all(np.isfinite(prices[:, -1])) or np.any(prices[:, -1] <= 0.0):
        return np.zeros(N_INST, dtype=int)
    nt = prices.shape[1]
    log_prices = np.log(prices)

    if _STABLE is None or (_ADAPTIVE and nt < _ADAPTIVE[0].nt_seen):
        _STABLE = [StableBook(i, j) for i, j in STABLE_PAIRS]
        _ADAPTIVE = [AdaptiveBook(i, j) for i, j in ADAPTIVE_PAIRS]

    books: list = []
    if MODE in ("both", "stable"):
        for book in _STABLE:
            book.update(log_prices[book.i], log_prices[book.j])
            books.append(book)
    if MODE in ("both", "adaptive"):
        for book in _ADAPTIVE:
            book.advance(log_prices)
            books.append(book)
    if nt < MIN_HISTORY:
        return np.zeros(N_INST, dtype=int)

    dollars = np.zeros(N_INST)
    for book in books:
        if book.side == 0:
            continue
        d_y = book.side * LEG
        d_x = -book.side * LEG
        scale = min(
            1.0,
            _capacity(dollars[book.i], d_y),
            _capacity(dollars[book.j], d_x),
        )
        if scale <= 1e-12:
            continue
        dollars[book.i] += scale * d_y
        dollars[book.j] += scale * d_x

    dollars[0] = 0.0
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    return (dollars / prices[:, -1]).astype(int)

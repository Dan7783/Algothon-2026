"""Slim rewrite of the four_sigmas book.  Same signals, far less surface.

WHAT WAS ACTUALLY IN THERE
FiltCfg had 20 fields and TradeCfg 9, per pair, for 21 pairs.  Measured
across the live seeds:

  * 18 of those 29 fields hold the SAME value for every pair — they are
    module constants wearing a config costume.
  * of the 11 that vary, sharing 8 of them (q_beta, h, cusum_k, adapt_bars,
    snap_overshoot, z_entry, z_exit, z_stop) costs 2 points of block floor.
  * the variance-CUSUM branch is DEAD: vol_cut is False everywhere, so
    cv_up/cv_dn/run_v_*/vol_alarm accumulate, reset, and change nothing.
  * only EV (conflict priority) and the enabled/require_adf flags genuinely
    need to be per pair — flattening EV costs 80 points of floor.

SO THIS MODULE KEEPS
  per pair : y, x, EV, plus two booleans
  global   : 9 filter constants + 3 thresholds
and deletes the variance CUSUM, adapt_extend, cut_confirm_bars,
adapt_p_floor, snap_max_rel, vol_cut, core, beta_mode, trade_while_adapting,
require_kappa and leg_scale entirely.

Magic numbers that remain are named (DISAGREE_ABS/DISAGREE_REL/SEG_EWMA/
DRIFT_MIN) rather than buried inline, so they can be seen and argued with.

Sleeves are unchanged: adaptive-beta pairs -> extra lead-lag -> ridge.
"""

from __future__ import annotations

from collections import defaultdict
from math import sqrt

import numpy as np

N_INST = 51
LEG = 10_000.0
Z_W = 145
ADF_WINDOW = 375
PRIOR_GATE_T = -2.4
MIN_DAYS = 130

# ---- the whole filter parameter set (was 20 fields x 21 pairs) ----
Q_BETA = 8.284e-06
Q_BETA_ADAPT = 5e-3
EWMA_LAMBDA = 0.97
WARMUP = 30
CUSUM_K = 0.6
H0 = 5.0
H_GAIN = 0.05
TARGET_SEG = 50
MIN_SEG_LEN = 40
COV_INFLATE = 25.0
SNAP_WINDOW = 12
ADAPT_BARS = 9
SNAP_OVERSHOOT = 1.15
INIT_WINDOW = 20

# ---- the whole trade parameter set (was 9 fields x 21 pairs) ----
Z_ENTRY = 0.45
Z_EXIT = 0.05
Z_STOP = 4.00

# ---- named constants that used to be inline magic numbers ----
DISAGREE_ABS = 0.025     # floor on "filter beta disagrees with short OLS"
DISAGREE_REL = 0.03      # ... as a fraction of beta
DRIFT_MIN = 8            # min consecutive disagreement days to force a cut
SEG_EWMA = 0.1           # weight on the newest segment length
H_FLOOR = 2.0

# ---- pairs: y, x, EV(conflict priority), use_adf_gate ----
# EV stays per-pair: it decides which pair wins an instrument conflict and
# flattening it costs 80 points of block floor.  Pairs dropped by the
# marginal-contribution audit are simply absent.
PAIRS = (
    (18, 35, 107.1, True),   # RTTH/NAYO
    (25, 37, 94.5, True),    # CTGI/EELT
    (7, 40, 63.9, True),     # HETT/ULXY
    (15, 25, 46.0, True),    # HRET/CTGI
    (8, 27, 45.1, True),     # HUXZ/ACAC
    (35, 28, 44.7, True),    # NAYO/SRTX
    (1, 20, 44.7, True),     # AENO/NWIG
    (17, 18, 41.2, True),    # DIHO/RTTH
    (49, 50, 39.5, True),    # MHRM/EAFC
    (13, 45, 34.9, True),    # EORC/NGTE
    (26, 32, 32.5, True),    # ALUT/CCNS
    (10, 46, 30.7, True),    # SMAH/ILVX
    (33, 42, 23.1, True),    # MTNS/BENI
    (36, 41, 19.1, True),    # FWWG/BLBT
    (31, 43, 16.0, True),    # ACIX/ITPA
)

LEADLAG_EXTRA = (
    (25, 1), (37, 1), (9, 38), (2, 27), (1, 20), (3, 14), (49, 40), (40, 7),
    (41, 15), (23, 14), (36, 5), (3, 29), (20, 1), (40, 29), (3, 40), (37, 41),
    (25, 40), (34, 17), (47, 40), (40, 38), (37, 28), (42, 29), (39, 48), (38, 26),
)
LOOK_OLD, LOOK_NEW = 160, 90
Z_FULL, Z_NEW = 1.5, 1.0

DISCOVERY_FIT = 250      # trailing hedge window for the broad reversion sleeve
DISCOVERY_ENTRY = 0.2
DISCOVERY_BREAK = 4.5

RIDGE_LAMBDA = 1.0
RIDGE_ENTRY = 1.0
RIDGE_MIN_LINKS = 35
RIDGE_MIN_DAYS = 252


def _adf_t(spread: np.ndarray) -> float:
    if len(spread) < 40:
        return 0.0
    d = np.diff(spread)
    y = d[1:]
    X = np.column_stack((np.ones(len(y)), spread[1:-1], d[:-1]))
    inv = np.linalg.pinv(X.T @ X)
    c = inv @ X.T @ y
    e = y - X @ c
    v = float(e @ e) / max(len(y) - 3, 1)
    return float(c[1] / sqrt(max(v * inv[1, 1], 1e-300)))


def _ou_kappa(spread: np.ndarray) -> float:
    d = np.diff(spread)
    lag = spread[:-1] - spread[:-1].mean()
    v = float(lag @ lag)
    return float(-(lag @ d) / v) if v > 1e-18 else 0.0


class Filter:
    """Sticky through-origin beta with a mean CUSUM.  Live paths only."""

    __slots__ = ("beta", "p22", "innov_var", "cm_up", "cm_dn", "run_up",
                 "run_dn", "seg_start", "h_live", "seg_ewma", "adapt_left",
                 "disagree_run", "t", "_y", "_x")

    def __init__(self) -> None:
        self.beta = None
        self.p22 = 1.0
        self.innov_var = None
        self.cm_up = self.cm_dn = 0.0
        self.run_up = self.run_dn = None
        self.seg_start = None
        self.h_live = H0
        self.seg_ewma = None
        self.adapt_left = 0
        self.disagree_run = 0
        self.t = -1
        self._y: list[float] = []
        self._x: list[float] = []

    def _ols(self, window: int) -> float:
        n = min(window, len(self._x))
        xs = np.asarray(self._x[-n:])
        ys = np.asarray(self._y[-n:])
        d = float(xs @ xs)
        return float(xs @ ys) / d if d > 1e-12 else (self.beta or 1.0)

    def step(self, y: float, x: float) -> tuple[float, bool]:
        self.t += 1
        self._y.append(y)
        self._x.append(x)
        if self.beta is None:
            self.beta = self._ols(INIT_WINDOW)
            self.seg_start = self.t

        q = Q_BETA_ADAPT if self.adapt_left > 0 else Q_BETA
        pp = self.p22 + q
        innov = y - self.beta * x
        s = max(pp * x * x + 1.0, 1e-12)
        g = (pp * x) / s
        self.beta += g * innov
        a = 1.0 - g * x
        self.p22 = max(a * a * pp + g * g, 1e-18)

        if self.adapt_left > 0:                     # hard-track short OLS
            self.beta = self._ols(SNAP_WINDOW)
            self.p22 = max(self.p22, 2.0)
            self.adapt_left -= 1

        self.innov_var = (innov * innov if self.innov_var is None
                          else EWMA_LAMBDA * self.innov_var
                          + (1.0 - EWMA_LAMBDA) * innov * innov)
        z = innov / sqrt(self.innov_var) if self.innov_var > 0 else 0.0

        cut = False
        if self.t >= WARMUP:
            pu, pd = self.cm_up, self.cm_dn
            self.cm_up = max(0.0, self.cm_up + z - CUSUM_K)
            self.cm_dn = max(0.0, self.cm_dn - z - CUSUM_K)
            if self.cm_up > 0 and pu == 0:
                self.run_up = self.t
            elif self.cm_up == 0:
                self.run_up = None
            if self.cm_dn > 0 and pd == 0:
                self.run_dn = self.t
            elif self.cm_dn == 0:
                self.run_dn = None

            alarm = self.cm_up > self.h_live or self.cm_dn > self.h_live
            ols = self._ols(MIN_SEG_LEN)
            thr = max(DISAGREE_ABS, DISAGREE_REL * abs(self.beta))
            disagree = abs(ols - self.beta) >= thr
            self.disagree_run = self.disagree_run + 1 if (
                disagree and self.adapt_left == 0) else 0

            drift = self.disagree_run >= max(DRIFT_MIN, TARGET_SEG // 4)
            if (alarm and disagree and self.adapt_left == 0) or drift:
                runs = [r for r in (self.run_up, self.run_dn) if r is not None]
                bar = max(min(runs) if runs else self.t, self.seg_start + 1)
                closed = bar - self.seg_start
                self.cm_up = self.cm_dn = 0.0
                self.run_up = self.run_dn = None
                if closed >= MIN_SEG_LEN:
                    cut = True
                    self.disagree_run = 0
                    target = self._ols(SNAP_WINDOW)
                    self.beta += SNAP_OVERSHOOT * (target - self.beta)
                    self.p22 = max(self.p22 * COV_INFLATE, 2.0)
                    self.adapt_left = ADAPT_BARS
                    self.seg_ewma = (float(min(closed, 3 * TARGET_SEG))
                                     if self.seg_ewma is None
                                     else SEG_EWMA * closed
                                     + (1 - SEG_EWMA) * self.seg_ewma)
                    err = max(-0.5, min(0.5,
                              (TARGET_SEG - self.seg_ewma) / TARGET_SEG))
                    self.h_live = max(H_FLOOR, self.h_live + H_GAIN * err * self.h_live)
                    self.seg_start = bar
            elif alarm and not disagree and self.adapt_left == 0:
                self.cm_up = self.cm_dn = 0.0
                self.run_up = self.run_dn = None
        return self.beta, cut


class Agent:
    __slots__ = ("y", "x", "ev", "use_adf", "filt", "beta_lag", "z", "side", "seen")

    def __init__(self, y, x, ev, use_adf):
        self.y, self.x, self.ev, self.use_adf = y, x, ev, use_adf
        self.filt = Filter()
        self.beta_lag = 1.0
        self.z = 0.0
        self.side = 0
        self.seen = 0

    def advance(self, log_prices: np.ndarray) -> None:
        nt = log_prices.shape[1]
        if self.seen > nt:
            self.__init__(self.y, self.x, self.ev, self.use_adf)
        for t in range(self.seen, nt):
            self.beta_lag = self.filt.beta if self.filt.beta is not None else 1.0
            self.filt.step(float(log_prices[self.y, t]), float(log_prices[self.x, t]))
            self.seen = t + 1

    def decide(self, LP: np.ndarray) -> int:
        nt = LP.shape[1]
        b = self.beta_lag
        if self.use_adf:
            lo = max(0, nt - 1 - ADF_WINDOW)
            if _adf_t(LP[self.y, lo:nt - 1] - b * LP[self.x, lo:nt - 1]) >= PRIOR_GATE_T:
                self.side = 0
                return 0
        s = LP[self.y, nt - Z_W:nt] - b * LP[self.x, nt - Z_W:nt]
        if _ou_kappa(LP[self.y, nt - 120:nt] - b * LP[self.x, nt - 120:nt]) <= 0.0:
            self.side = 0
            return 0
        self.z = float((s[-1] - s[:-1].mean()) / (s[:-1].std() + 1e-12))
        a = abs(self.z)
        if self.side == 0:
            if Z_ENTRY <= a < Z_STOP:
                self.side = -int(np.sign(self.z))
        elif a <= Z_EXIT or a >= Z_STOP:
            self.side = 0
        elif a >= Z_ENTRY and -int(np.sign(self.z)) != self.side:
            self.side = -int(np.sign(self.z))
        return self.side


_AGENTS: list[Agent] | None = None
_HUB: dict[int, float] = {}


def _hub_penalty() -> dict[int, float]:
    deg: dict[int, int] = defaultdict(int)
    for y, x, _, _ in PAIRS:
        deg[y] += 1
        deg[x] += 1
    return {i: sqrt(max(deg[y], 1) * max(deg[x], 1))
            for i, (y, x, _, _) in enumerate(PAIRS)}


def reset() -> None:
    global _AGENTS, _HUB
    _AGENTS = [Agent(*p) for p in PAIRS]
    _HUB = _hub_penalty()


def _pair_dollars(prices: np.ndarray, LP: np.ndarray) -> np.ndarray:
    pos = np.zeros(N_INST)
    for ag in _AGENTS:
        ag.advance(LP)
    live = []
    for i, ag in enumerate(_AGENTS):
        side = ag.decide(LP)
        if side:
            live.append((ag.ev * abs(ag.z) / max(_HUB.get(i, 1.0), 1.0), i, side))
    live.sort(reverse=True)
    taken: set[int] = set()
    for _, i, side in live:
        ag = _AGENTS[i]
        if ag.y in taken or ag.x in taken:
            continue
        taken.add(ag.y)
        taken.add(ag.x)
        pos[ag.y] += side * LEG
        pos[ag.x] -= side * LEG
    return pos


def _corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _leadlag_dollars(r: np.ndarray, held: np.ndarray) -> np.ndarray:
    pos = np.zeros(N_INST)
    m = r.shape[1]
    look = LOOK_OLD + LOOK_NEW
    if m < look + 2:
        return pos
    for i, j in LEADLAG_EXTRA:
        if abs(held[j]) > 1.0:
            continue
        x, y = r[i, m - look - 1:m - 1], r[j, m - look:m]
        c_old = _corr(x[:LOOK_OLD], y[:LOOK_OLD])
        if c_old == 0.0:
            continue
        c_new = _corr(x[LOOK_OLD:], y[LOOK_OLD:])
        sign = float(np.sign(c_old))
        ok = (abs(_corr(x, y)) * sqrt(len(x)) >= Z_FULL
              and abs(c_new) * sqrt(LOOK_NEW) >= Z_NEW
              and np.sign(c_new) == sign
              and float(np.sum(sign * np.sign(x[LOOK_OLD:]) * y[LOOK_OLD:])) > 0.0)
        if ok:
            pos[j] += LEG * sign * np.sign(r[i, -1])
    return pos


def _ridge_dollars(r: np.ndarray, held: np.ndarray) -> np.ndarray:
    pos = np.zeros(N_INST)
    if r.shape[1] + 1 < RIDGE_MIN_DAYS:
        return pos
    recent = r[1:, -250:].T
    std = (recent - recent.mean(0)) / (recent.std(0) + 1e-12)
    lag = (std[:-1].T @ std[1:]) / (len(recent) - 1)
    np.fill_diagonal(lag, 0.0)
    if int((np.abs(lag) > 2.5 / sqrt(len(recent) - 1)).sum()) < RIDGE_MIN_LINKS:
        return pos
    hist = r[1:].T
    sc = hist.std(0) + 1e-12
    X, Y = hist[:-1] / sc, hist[1:] / sc
    B = np.linalg.solve(X.T @ X + RIDGE_LAMBDA * len(X) * np.eye(N_INST - 1), X.T @ Y)
    pred = (hist[-1] / sc) @ B
    raw = LEG * np.sign(pred) * (np.abs(pred) > RIDGE_ENTRY * (X @ B).std(0))
    idle = np.abs(held[1:]) <= 1.0
    room = np.maximum(0.0, LEG - np.minimum(np.abs(held[1:]), LEG))
    same = (~idle) & (np.abs(raw) > 1.0) & (np.sign(raw) == np.sign(held[1:]))
    out = np.where(idle, raw, 0.0)
    out[same] = np.sign(raw[same]) * np.minimum(np.abs(raw[same]), room[same])
    pos[1:] = out
    return pos


_ATLAS_KEYS = {tuple(sorted((y, x))) for y, x, _, _ in PAIRS}
_ti, _tj = np.triu_indices(N_INST - 1, k=1)
_DI, _DJ = _ti + 1, _tj + 1
_keep = np.fromiter(
    (tuple(sorted((int(i), int(j)))) not in _ATLAS_KEYS for i, j in zip(_DI, _DJ)),
    dtype=bool, count=len(_DI))
_DI, _DJ = _DI[_keep], _DJ[_keep]


def _discovery_dollars(LP: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Broad trailing-OLS reversion on instruments the other sleeves left idle."""
    pos = np.zeros(N_INST)
    if LP.shape[1] < DISCOVERY_FIT + 1:
        return pos
    w = LP[:, -(DISCOVERY_FIT + 1):]
    train = w[:, :-1]
    y, x = train[_DI], train[_DJ]
    yc = y - y.mean(axis=1, keepdims=True)
    xc = x - x.mean(axis=1, keepdims=True)
    beta = (yc * xc).sum(axis=1) / np.maximum((xc * xc).sum(axis=1), 1e-18)
    alpha = y.mean(axis=1) - beta * x.mean(axis=1)
    resid = y - alpha[:, None] - beta[:, None] * x
    cur = LP[_DI, -1] - alpha - beta * LP[_DJ, -1]
    z = cur / (resid.std(axis=1) + 1e-12)
    idle = np.abs(held) <= 1.0
    ok = ((beta > 0.0) & np.isfinite(z) & (np.abs(z) >= DISCOVERY_ENTRY)
          & (np.abs(z) < DISCOVERY_BREAK) & idle[_DI] & idle[_DJ])
    used = np.zeros(N_INST, dtype=bool)
    for k in np.argsort(-np.abs(z)):
        if not ok[k]:
            continue
        i, j = int(_DI[k]), int(_DJ[k])
        if used[i] or used[j]:
            continue
        d = -np.sign(z[k])
        leg = LEG / max(1.0, abs(float(beta[k])))
        pos[i] += d * leg
        pos[j] -= d * float(beta[k]) * leg
        used[i] = used[j] = True
    return pos


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    prices = np.asarray(prcSoFar, dtype=float)
    n, nt = prices.shape
    if n != N_INST or nt < MIN_DAYS:
        return np.zeros(n, dtype=int)
    if not np.all(np.isfinite(prices[:, -3:])) or np.any(prices[:, -3:] <= 0.0):
        return np.zeros(n, dtype=int)
    if _AGENTS is None:
        reset()

    LP = np.log(prices)
    r = np.diff(LP, axis=1)
    dollars = _pair_dollars(prices, LP)
    dollars += _leadlag_dollars(r, dollars)
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars += _ridge_dollars(r, dollars)
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars += _discovery_dollars(LP, dollars)

    crash = np.abs(r[:, -1]) > 0.40
    crash[0] = False
    dollars[crash] = 0.0
    dollars[0] = 0.0
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    return (dollars / prices[:, -1]).astype(int)


reset()

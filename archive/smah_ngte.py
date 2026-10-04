"""SMAH / NGTE single-pair mean-reversion (self-contained).

Inlined sticky through-origin Kalman + CUSUM beta filter.
Entry/exit z = Kalman innovation z = innov / sqrt(innov_var).
Thresholds = mult * live CUSUM h (filter-tied).
Hedge: beta-scaled dollar legs for |β|≥1 and |β|<1 inside $10k/leg.

NamedTuples only (eval.py exec loader breaks dataclasses).
"""

from __future__ import annotations

from math import sqrt
from typing import NamedTuple

import numpy as np


N_INST = 51
LEG = 10_000.0
MIN_HISTORY = 80

Y_IDX = 10  # SMAH
X_IDX = 46  # NGTE

_INIT_WINDOW = 20


# ---------------------------------------------------------------------------
# Adaptive beta filter (inlined)
# ---------------------------------------------------------------------------


class FiltCfg(NamedTuple):
    q_beta: float = 3e-8
    q_beta_adapt: float = 5e-3
    ewma_lambda: float = 0.97
    warmup: int = 30
    cusum_k: float = 0.5
    auto_h: bool = True
    target_seg: int = 40
    h: float = 10.0
    h_gain: float = 0.05
    min_seg_len: int = 40
    vol_cut: bool = False
    cov_inflate: float = 25.0
    snap_window: int = 12
    adapt_chase: float = 1.0
    adapt_bars: int = 8
    adapt_extend: bool = False
    snap_overshoot: float = 1.25


class StepOut(NamedTuple):
    beta: float
    spread: float
    z: float
    h: float
    cut: bool
    adapting: bool
    vol_shift: bool
    cm_up: float
    cm_dn: float
    cv: float


class PairSegFilter:
    """Sticky Kalman beta + CUSUM snaps. z = innov / sqrt(innov_var)."""

    def __init__(self, cfg: FiltCfg | None = None) -> None:
        self.cfg = cfg if cfg is not None else FiltCfg()
        self.beta: float | None = None
        self.p22: float = 1.0
        self.innov_var: float | None = None
        self.resid_var: float | None = None
        self.cm_up: float = 0.0
        self.cm_dn: float = 0.0
        self.run_m_up: int | None = None
        self.run_m_dn: int | None = None
        self.cv_up: float = 0.0
        self.cv_dn: float = 0.0
        self.run_v_up: int | None = None
        self.run_v_dn: int | None = None
        self.seg_start: int | None = None
        self.h_live: float | None = None
        self.seg_len_ewma: float | None = None
        self.adapt_left: int = 0
        self.disagree_run: int = 0
        self.snap_dir: float = 0.0
        self.t: int = -1
        self._y_hist: list[float] = []
        self._x_hist: list[float] = []

    def _ols_beta(self, window: int) -> float:
        n = len(self._x_hist)
        w = min(window, n)
        xs = np.asarray(self._x_hist[-w:], dtype=float)
        ys = np.asarray(self._y_hist[-w:], dtype=float)
        denom = float(xs @ xs)
        if denom < 1e-12:
            return float(self.beta) if self.beta is not None else 1.0
        return float(xs @ ys) / denom

    def _accel_target(self) -> float:
        ols_s = self._ols_beta(self.cfg.snap_window)
        b = float(self.beta) if self.beta is not None else ols_s
        if self.snap_dir > 0.0:
            return max(ols_s, b)
        if self.snap_dir < 0.0:
            return min(ols_s, b)
        return ols_s

    def step(self, y: float, x: float) -> StepOut:
        cfg = self.cfg
        self.t += 1
        t = self.t
        cut = False
        vol_shift = False
        yf, xf = float(y), float(x)
        self._y_hist.append(yf)
        self._x_hist.append(xf)

        if self.beta is None:
            self.beta = self._ols_beta(_INIT_WINDOW)
            self.seg_start = t
            self.h_live = cfg.h

        q_b = cfg.q_beta_adapt if self.adapt_left > 0 else cfg.q_beta
        adapting = self.adapt_left > 0

        b = float(self.beta)
        pp22 = self.p22 + q_b
        innov = yf - b * xf
        s_pred = pp22 * xf * xf + 1.0
        if s_pred < 1e-12:
            s_pred = 1e-12
        g = (pp22 * xf) / s_pred
        self.beta = b + g * innov
        a = 1.0 - g * xf
        self.p22 = max(a * a * pp22 + g * g, 1e-18)

        if self.adapt_left > 0 and cfg.adapt_chase > 0.0:
            target = self._accel_target()
            ch = min(1.0, max(0.0, cfg.adapt_chase))
            self.beta = (1.0 - ch) * float(self.beta) + ch * target
            self.p22 = max(float(self.p22), 2.0)

        if self.innov_var is None:
            self.innov_var = innov * innov
        else:
            lam = cfg.ewma_lambda
            self.innov_var = lam * self.innov_var + (1.0 - lam) * innov * innov
        z = innov / sqrt(self.innov_var) if self.innov_var > 0.0 else 0.0

        resid = yf - float(self.beta) * xf
        if self.resid_var is None:
            self.resid_var = resid * resid
        else:
            lam = cfg.ewma_lambda
            self.resid_var = lam * self.resid_var + (1.0 - lam) * resid * resid

        h_work = (
            (self.h_live if self.h_live is not None else cfg.h)
            if cfg.auto_h
            else cfg.h
        )

        if t >= cfg.warmup:
            prev_up, prev_dn = self.cm_up, self.cm_dn
            self.cm_up = max(0.0, self.cm_up + z - cfg.cusum_k)
            self.cm_dn = max(0.0, self.cm_dn - z - cfg.cusum_k)

            if self.cm_up > 0.0 and prev_up == 0.0:
                self.run_m_up = t
            elif self.cm_up == 0.0:
                self.run_m_up = None

            if self.cm_dn > 0.0 and prev_dn == 0.0:
                self.run_m_dn = t
            elif self.cm_dn == 0.0:
                self.run_m_dn = None

            d_v = (z * z - 1.0) / sqrt(2.0)
            prev_v_up, prev_v_dn = self.cv_up, self.cv_dn
            self.cv_up = max(0.0, self.cv_up + d_v - cfg.cusum_k)
            self.cv_dn = max(0.0, self.cv_dn - d_v - cfg.cusum_k)

            if self.cv_up > 0.0 and prev_v_up == 0.0:
                self.run_v_up = t
            elif self.cv_up == 0.0:
                self.run_v_up = None

            if self.cv_dn > 0.0 and prev_v_dn == 0.0:
                self.run_v_dn = t
            elif self.cv_dn == 0.0:
                self.run_v_dn = None

            mean_alarm = self.cm_up > h_work or self.cm_dn > h_work
            vol_alarm = self.cv_up > h_work or self.cv_dn > h_work
            if vol_alarm:
                vol_shift = True
                if not cfg.vol_cut:
                    self.cv_up = self.cv_dn = 0.0
                    self.run_v_up = self.run_v_dn = None

            ols = self._ols_beta(cfg.min_seg_len)
            thr = max(0.025, 0.03 * abs(float(self.beta)))
            disagree = abs(ols - float(self.beta)) >= thr
            if disagree and self.adapt_left == 0:
                self.disagree_run += 1
            else:
                self.disagree_run = 0

            cusum_cut = (
                (mean_alarm or (cfg.vol_cut and vol_alarm))
                and disagree
                and self.adapt_left == 0
            )
            drift_need = max(8, cfg.target_seg // 4)
            drift_cut = self.disagree_run >= drift_need and self.adapt_left == 0
            want_cut = cusum_cut or drift_cut

            if mean_alarm and not disagree and self.adapt_left == 0:
                self.cm_up = self.cm_dn = 0.0
                self.cv_up = self.cv_dn = 0.0
                self.run_m_up = self.run_m_dn = None
                self.run_v_up = self.run_v_dn = None

            if want_cut:
                best = -1.0
                cut_bar: int | None = None
                if self.cm_up > h_work and self.cm_up > best:
                    best = self.cm_up
                    cut_bar = self.run_m_up
                if self.cm_dn > h_work and self.cm_dn > best:
                    best = self.cm_dn
                    cut_bar = self.run_m_dn
                if cfg.vol_cut and self.cv_up > h_work and self.cv_up > best:
                    best = self.cv_up
                    cut_bar = self.run_v_up
                if cfg.vol_cut and self.cv_dn > h_work and self.cv_dn > best:
                    best = self.cv_dn
                    cut_bar = self.run_v_dn
                if cut_bar is None:
                    cut_bar = t

                assert self.seg_start is not None
                cut_bar = max(int(cut_bar), self.seg_start + 1)
                closed_len = cut_bar - self.seg_start

                self.cm_up = self.cm_dn = 0.0
                self.cv_up = self.cv_dn = 0.0
                self.run_m_up = self.run_m_dn = None
                self.run_v_up = self.run_v_dn = None

                if closed_len >= cfg.min_seg_len:
                    cut = True
                    self.disagree_run = 0
                    target = self._ols_beta(cfg.snap_window)
                    pre = float(self.beta)
                    delta = target - pre
                    self.snap_dir = 1.0 if delta >= 0.0 else -1.0
                    self.beta = pre + cfg.snap_overshoot * delta
                    self.p22 = max(self.p22 * cfg.cov_inflate, 2.0)
                    self.adapt_left = cfg.adapt_bars
                    resid = yf - float(self.beta) * xf

                    if cfg.auto_h:
                        if self.seg_len_ewma is None:
                            self.seg_len_ewma = float(
                                min(closed_len, 3 * cfg.target_seg)
                            )
                        else:
                            self.seg_len_ewma = (
                                0.1 * closed_len + 0.9 * self.seg_len_ewma
                            )
                        err = (cfg.target_seg - self.seg_len_ewma) / cfg.target_seg
                        err = max(-0.5, min(0.5, err))
                        h0 = self.h_live if self.h_live is not None else cfg.h
                        self.h_live = max(2.0, h0 + cfg.h_gain * err * h0)

                    self.seg_start = cut_bar

        if self.adapt_left > 0:
            self.adapt_left -= 1
            adapting = True
            if self.adapt_left == 0:
                if cfg.adapt_extend:
                    ols_now = self._ols_beta(cfg.snap_window)
                    n = len(self._x_hist)
                    if n > cfg.snap_window + 5:
                        lag = min(cfg.snap_window // 2, 6)
                        xs = np.asarray(
                            self._x_hist[-(cfg.snap_window + lag) : -lag], dtype=float
                        )
                        ys = np.asarray(
                            self._y_hist[-(cfg.snap_window + lag) : -lag], dtype=float
                        )
                        denom = float(xs @ xs)
                        ols_lag = (
                            float(xs @ ys) / denom if denom > 1e-12 else ols_now
                        )
                    else:
                        ols_lag = ols_now
                    still_trending = abs(ols_now - ols_lag) >= 0.02
                    still_off = abs(ols_now - float(self.beta)) >= 0.03
                    if still_trending or still_off:
                        self.adapt_left = max(10, cfg.adapt_bars // 4)
                        adapting = True
                    else:
                        self.snap_dir = 0.0
                        self.p22 = min(float(self.p22), 1e-4)
                else:
                    self.snap_dir = 0.0
                    self.p22 = min(float(self.p22), 1e-4)

        h_out = (
            (self.h_live if self.h_live is not None else cfg.h)
            if cfg.auto_h
            else cfg.h
        )
        return StepOut(
            beta=float(self.beta),
            spread=float(resid),
            z=float(z),
            h=float(h_out),
            cut=cut,
            adapting=adapting,
            vol_shift=vol_shift,
            cm_up=float(self.cm_up),
            cm_dn=float(self.cm_dn),
            cv=max(self.cv_up, self.cv_dn),
        )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

# Tuned for SMAH/NGTE regime break (~day 340).
FILT_CFG = FiltCfg(
    q_beta=1e-6,
    h=6.0,
    cusum_k=0.45,
    min_seg_len=30,
    target_seg=50,
    adapt_bars=14,
    snap_overshoot=1.15,
    snap_window=8,
    cov_inflate=50.0,
)

ENTRY_H_MULT = 0.30
EXIT_H_MULT = 0.13
STOP_H_MULT = 0.45


def _beta_dollar_legs(side: int, beta: float) -> tuple[float, float]:
    """Dollar notionals with beta hedge, both legs ≤ LEG."""
    if side == 0 or abs(beta) < 1e-12:
        return 0.0, 0.0
    ab = abs(beta)
    if ab >= 1.0:
        d_x = -side * np.sign(beta) * LEG
        d_y = side * (LEG / ab)
    else:
        d_y = side * LEG
        d_x = -side * beta * LEG
    return float(d_y), float(d_x)


class SmahNgteBook:
    def __init__(self) -> None:
        self.filt = PairSegFilter(FILT_CFG)
        self.nt_seen = 0
        self.side = 0
        self.hedge_beta = 0.0
        self.z = 0.0
        self.live_beta = 1.0
        self.spread = 0.0
        self.adapting = False
        self.cut = False
        self.h_live = float(FILT_CFG.h)

    def _thresholds(self) -> tuple[float, float, float]:
        h = max(2.0, self.h_live)
        return ENTRY_H_MULT * h, EXIT_H_MULT * h, STOP_H_MULT * h

    def _step(self, y: float, x: float) -> None:
        o = self.filt.step(y, x)
        self.live_beta = float(o.beta)
        self.spread = float(o.spread)
        self.adapting = bool(o.adapting)
        self.cut = bool(o.cut)
        self.h_live = float(o.h)
        self.z = float(o.z)  # innov / sqrt(innov_var)

    def _desired(self) -> tuple[int, float]:
        entry_z, exit_z, stop_z = self._thresholds()
        if self.cut and self.side != 0:
            return 0, 0.0
        if self.adapting:
            if self.side != 0:
                return self.side, self.hedge_beta
            return 0, 0.0
        if self.side == 0:
            if self.z >= entry_z:
                return -1, self.live_beta
            if self.z <= -entry_z:
                return 1, self.live_beta
            return 0, 0.0
        if abs(self.z) <= exit_z or abs(self.z) >= stop_z:
            return 0, 0.0
        return self.side, self.hedge_beta

    def update(self, prices: np.ndarray) -> np.ndarray:
        nins, nt = prices.shape
        if nins != N_INST or nt < MIN_HISTORY:
            return np.zeros(N_INST, dtype=int)
        start = self.nt_seen
        if start > nt:
            self.__init__()
            start = 0
        for t in range(start, nt):
            self._step(float(prices[Y_IDX, t]), float(prices[X_IDX, t]))
            self.nt_seen = t + 1

        side, beta = self._desired()
        self.side = side
        self.hedge_beta = beta if side != 0 else 0.0

        shares = np.zeros(N_INST, dtype=int)
        if side == 0:
            return shares
        py = float(prices[Y_IDX, -1])
        px = float(prices[X_IDX, -1])
        if py <= 0.0 or px <= 0.0:
            return shares
        d_y, d_x = _beta_dollar_legs(side, beta)
        shares[Y_IDX] = int(d_y / py)
        shares[X_IDX] = int(d_x / px)
        return shares


_BOOK: SmahNgteBook | None = None


def reset() -> None:
    global _BOOK
    _BOOK = None


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global _BOOK
    prices = np.asarray(prcSoFar, dtype=float)
    if prices.ndim != 2 or prices.shape[0] != N_INST:
        return np.zeros(N_INST, dtype=int)
    if not np.all(np.isfinite(prices[:, -1])) or np.any(prices[:, -1] <= 0.0):
        return np.zeros(N_INST, dtype=int)
    if _BOOK is None:
        _BOOK = SmahNgteBook()
    return _BOOK.update(prices)
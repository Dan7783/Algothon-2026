"""Sticky through-origin beta + Pine CUSUM segmentation.

Lean pairs hedge filter. Runtime state matches the Pine ``var`` block;
inputs are a small panel. No chase/drift/trend-exit/OLS-persist gates.

    y_t = beta_t * x_t + eps_t   (alpha frozen at 0)

PIPELINE
  1. Sticky Kalman on beta (tiny q_beta); R = 1
  2. EWMA innov variance → z
  3. Two-sided mean CUSUM; cuts backdated to run start
  4. Optional variance CUSUM
  5. Reject cuts shorter than min_seg_len
  6. On cut: OLS snap, cov inflate, brief elevated q (adapting)
  7. Adaptive h from target segment length
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import NamedTuple

import numpy as np


# ---- Pine input panel (knobs only; everything else is state / derived) ----
DEFAULT_Q_BETA: float = 3e-8  # sticky publish noise
DEFAULT_Q_BETA_ADAPT: float = 5e-3  # brief burst of process noise on cut
DEFAULT_EWMA_LAMBDA: float = 0.97
DEFAULT_WARMUP: int = 30
DEFAULT_CUSUM_K: float = 0.5
DEFAULT_AUTO_H: bool = True
DEFAULT_TARGET_SEG: int = 40
DEFAULT_H: float = 10.0
DEFAULT_H_GAIN: float = 0.05
DEFAULT_MIN_SEG_LEN: int = 40
DEFAULT_VOL_CUT: bool = False
DEFAULT_COV_INFLATE: float = 25.0  # hard reopen of P on cut
DEFAULT_SNAP_WINDOW: int = 12  # short OLS target for snap / chase
DEFAULT_ADAPT_CHASE: float = 1.0  # hard-track short OLS during brief adapt
DEFAULT_ADAPT_BARS: int = 8  # short burst — then lock sticky
DEFAULT_ADAPT_EXTEND: bool = False  # do not linger in adapt
DEFAULT_SNAP_OVERSHOOT: float = 1.25  # large jump on cut (compensates short adapt)
DEFAULT_INIT_WINDOW: int = 20


@dataclass(frozen=True)
class PairSegConfig:
    """Pine inputs — not runtime state."""

    q_beta: float = DEFAULT_Q_BETA
    q_beta_adapt: float = DEFAULT_Q_BETA_ADAPT
    ewma_lambda: float = DEFAULT_EWMA_LAMBDA
    warmup: int = DEFAULT_WARMUP
    cusum_k: float = DEFAULT_CUSUM_K
    auto_h: bool = DEFAULT_AUTO_H
    target_seg: int = DEFAULT_TARGET_SEG
    h: float = DEFAULT_H
    h_gain: float = DEFAULT_H_GAIN
    min_seg_len: int = DEFAULT_MIN_SEG_LEN
    vol_cut: bool = DEFAULT_VOL_CUT
    cov_inflate: float = DEFAULT_COV_INFLATE
    snap_window: int = DEFAULT_SNAP_WINDOW
    adapt_chase: float = DEFAULT_ADAPT_CHASE
    adapt_bars: int = DEFAULT_ADAPT_BARS
    adapt_extend: bool = DEFAULT_ADAPT_EXTEND
    snap_overshoot: float = DEFAULT_SNAP_OVERSHOOT


# Prior long-adapt acceleration (for comparison plots only)
BASELINE_CFG = PairSegConfig(
    q_beta_adapt=2e-3,
    cov_inflate=25.0,
    snap_window=12,
    adapt_chase=1.0,
    adapt_bars=80,
    adapt_extend=True,
    snap_overshoot=1.15,
)


class StepOut(NamedTuple):
    alpha: float
    beta: float
    spread: float  # y - beta * x
    z: float
    h: float
    cut: bool
    adapting: bool
    vol_shift: bool
    cm_up: float
    cm_dn: float
    cv: float


class PairSegFilter:
    """Online state — mirrors the Pine ``var`` block (+ brief adapt countdown).

    Pine name     →  attribute
    ---------------------------
    kLevel        →  alpha  (always 0; through-origin)
    kSlope        →  beta
    p11,p12,p22   →  p11, p12, p22
    innovVar      →  innov_var
    residVar      →  resid_var
    cmUp/cmDn     →  cm_up / cm_dn
    cvUp/cvDn     →  cv_up / cv_dn
    runMUp/...    →  run_m_up / ...
    segStartBar   →  seg_start
    hLive         →  h_live
    segLenEwma    →  seg_len_ewma
    """

    def __init__(self, cfg: PairSegConfig | None = None) -> None:
        self.cfg = cfg or PairSegConfig()
        self.alpha: float = 0.0
        self.beta: float | None = None
        self.p11: float = 1e-18
        self.p12: float = 0.0
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
        self.snap_dir: float = 0.0  # +1 up / -1 down / 0 none
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
        """Short OLS, ratcheted in the snap direction (never reverse mid-adapt)."""
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
            self.beta = self._ols_beta(DEFAULT_INIT_WINDOW)
            self.seg_start = t
            self.h_live = cfg.h

        q_b = cfg.q_beta_adapt if self.adapt_left > 0 else cfg.q_beta
        adapting = self.adapt_left > 0

        # ---- through-origin Kalman (H=[x], state=beta, R=1) ----
        b = float(self.beta)
        pp22 = self.p22 + q_b
        innov = yf - b * xf
        s_pred = pp22 * xf * xf + 1.0
        if s_pred < 1e-12:
            s_pred = 1e-12
        g = (pp22 * xf) / s_pred
        self.beta = b + g * innov
        # Joseph scalar: P = (1 - g x)^2 P_pred + g^2 R
        a = 1.0 - g * xf
        self.p22 = max(a * a * pp22 + g * g, 1e-18)
        self.p11 = 1e-18
        self.p12 = 0.0
        self.alpha = 0.0

        # While adapting: hard-track short OLS (acceleration). Kalman step above
        # is overwritten so publish beta follows the snap target immediately.
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

            # Cut on CUSUM+disagreement, or sustained OLS drift
            cusum_cut = (
                (mean_alarm or (cfg.vol_cut and vol_alarm))
                and disagree
                and self.adapt_left == 0
            )
            drift_need = max(8, cfg.target_seg // 4)
            drift_cut = self.disagree_run >= drift_need and self.adapt_left == 0
            want_cut = cusum_cut or drift_cut
            if mean_alarm and not disagree and self.adapt_left == 0:
                # CUSUM noise without a real level shift
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
                    # Hard snap with optional overshoot past short OLS
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
                            self.seg_len_ewma = float(min(closed_len, 3 * cfg.target_seg))
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
                        self.p22 = min(float(self.p22), 1e-4)  # lock sticky
                else:
                    # Brief burst done — lock covariance so beta stabilises
                    self.snap_dir = 0.0
                    self.p22 = min(float(self.p22), 1e-4)

        h_out = (
            (self.h_live if self.h_live is not None else cfg.h)
            if cfg.auto_h
            else cfg.h
        )
        return StepOut(
            alpha=0.0,
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


@dataclass
class PairSegResult:
    alpha: np.ndarray
    beta: np.ndarray
    spread: np.ndarray
    z: np.ndarray
    h: np.ndarray
    cut: np.ndarray
    adapting: np.ndarray
    vol_shift: np.ndarray
    cm_up: np.ndarray
    cm_dn: np.ndarray
    cv: np.ndarray
    config: PairSegConfig


def run_pair_seg(
    y: np.ndarray,
    x: np.ndarray,
    cfg: PairSegConfig | None = None,
) -> PairSegResult:
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if y.shape != x.shape:
        raise ValueError("y and x length mismatch")
    cfg = cfg or PairSegConfig()
    filt = PairSegFilter(cfg)
    T = y.size
    alpha = np.empty(T)
    beta = np.empty(T)
    spread = np.empty(T)
    z = np.empty(T)
    h = np.empty(T)
    cut = np.zeros(T, dtype=bool)
    adapting = np.zeros(T, dtype=bool)
    vol = np.zeros(T, dtype=bool)
    cm_up = np.empty(T)
    cm_dn = np.empty(T)
    cv = np.empty(T)
    for t in range(T):
        o = filt.step(float(y[t]), float(x[t]))
        alpha[t] = o.alpha
        beta[t] = o.beta
        spread[t] = o.spread
        z[t] = o.z
        h[t] = o.h
        cut[t] = o.cut
        adapting[t] = o.adapting
        vol[t] = o.vol_shift
        cm_up[t] = o.cm_up
        cm_dn[t] = o.cm_dn
        cv[t] = o.cv
    return PairSegResult(
        alpha=alpha,
        beta=beta,
        spread=spread,
        z=z,
        h=h,
        cut=cut,
        adapting=adapting,
        vol_shift=vol,
        cm_up=cm_up,
        cm_dn=cm_dn,
        cv=cv,
        config=cfg,
    )


def hedge_step(filt: PairSegFilter, y: float, x: float) -> dict:
    o = filt.step(y, x)
    return {
        "alpha": o.alpha,
        "beta": o.beta,
        "spread": o.spread,
        "z": o.z,
        "h": o.h,
        "cut": o.cut,
        "adapting": o.adapting,
        "vol_shift": o.vol_shift,
    }


def _rolling_through_origin(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    T = len(y)
    out = np.full(T, np.nan)
    for i in range(window, T):
        xi, yi = x[i - window : i], y[i - window : i]
        d = float(xi @ xi)
        if d > 1e-12:
            out[i] = float(xi @ yi) / d
    return out


def plot_smoothed_vs_raw(
    y: np.ndarray,
    x: np.ndarray,
    result: PairSegResult,
    *,
    y_ticker: str = "Y",
    x_ticker: str = "X",
    path: str | Path,
    roll_window: int = 100,
    break_window: tuple[int, int] = (280, 560),
) -> None:
    """Match ``cusum_kalman_smoothed_vs_raw_beta.png`` layout."""
    import matplotlib.pyplot as plt

    T = len(result.beta)
    t = np.arange(T)
    raw = _rolling_through_origin(y, x, roll_window)
    cuts = np.flatnonzero(result.cut)
    adapt = result.adapting.astype(float)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax = axes[0]
    ax.plot(t, y / y[0], color="#1f4e79", lw=1.1, label=y_ticker)
    ax.plot(t, x / x[0], color="#e67e22", lw=1.1, label=x_ticker)
    ax.axvspan(
        break_window[0],
        break_window[1],
        color="gray",
        alpha=0.12,
        label="break window",
    )
    ax.set_ylabel("price / day1")
    ax.set_title(f"{y_ticker}/{x_ticker}: sticky beta + brief burst adapt (lean)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(t, 0.0, adapt, step="pre", color="#f4a261", alpha=0.45, label="adapting")
    ax.plot(t, raw, color="black", lw=2.0, alpha=0.85, label=f"raw rolling {roll_window}d")
    ax.plot(t, result.beta, color="#1f4e79", lw=1.3, label="Kalman beta")
    for idx in cuts:
        ax.axvline(
            idx,
            color="#c44e52",
            ls="--",
            lw=0.9,
            alpha=0.75,
            label="CUSUM snap" if idx == cuts[0] else None,
        )
    ax.set_ylabel("beta")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, result.spread, color="#2ca02c", lw=1.0, label="spread = y - beta·x")
    ax.axhline(0.0, color="gray", lw=0.7)
    for idx in cuts:
        ax.axvline(idx, color="#c44e52", ls="--", lw=0.9, alpha=0.75)
    ax.set_ylabel("spread")
    ax.set_xlabel("day index")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_snap_speed_comparison(
    y: np.ndarray,
    x: np.ndarray,
    slow: PairSegResult,
    fast: PairSegResult,
    *,
    y_ticker: str = "Y",
    x_ticker: str = "X",
    path: str | Path,
    roll_window: int = 100,
    break_window: tuple[int, int] = (280, 560),
) -> None:
    """Overlay baseline vs fast-snap beta / spread (+ break zoom)."""
    import matplotlib.pyplot as plt

    T = len(fast.beta)
    t = np.arange(T)
    raw = _rolling_through_origin(y, x, roll_window)
    short = _rolling_through_origin(y, x, fast.config.snap_window)
    cuts_f = np.flatnonzero(fast.cut)

    def _tti(beta: np.ndarray, thr: float, start: int = 280) -> int | None:
        for i in range(start, len(beta)):
            if beta[i] >= thr:
                return int(i - start)
        return None

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=False)

    ax = axes[0]
    ax.plot(t, y / y[0], color="#1f4e79", lw=1.1, label=y_ticker)
    ax.plot(t, x / x[0], color="#e67e22", lw=1.1, label=x_ticker)
    ax.axvspan(
        break_window[0],
        break_window[1],
        color="gray",
        alpha=0.12,
        label="break window",
    )
    ax.set_ylabel("price / day1")
    ax.set_title(
        f"{y_ticker}/{x_ticker}: adapt burst — long(80d) vs short(8d) lock"
    )
    ax.set_xlim(0, T - 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(
        t,
        0.0,
        fast.adapting.astype(float),
        step="pre",
        color="#f4a261",
        alpha=0.35,
        label="fast adapting",
    )
    ax.plot(t, raw, color="black", lw=2.0, alpha=0.75, label=f"raw rolling {roll_window}d")
    ax.plot(
        t,
        short,
        color="#8e44ad",
        lw=1.0,
        alpha=0.7,
        label=f"short OLS {fast.config.snap_window}d",
    )
    ax.plot(
        t,
        slow.beta,
        color="#7f8c8d",
        lw=1.5,
        ls="--",
        label=(
            f"long adapt ({slow.config.adapt_bars}d, extend={slow.config.adapt_extend})"
        ),
    )
    ax.plot(
        t,
        fast.beta,
        color="#1f4e79",
        lw=1.6,
        label=(
            f"burst adapt ({fast.config.adapt_bars}d, overshoot={fast.config.snap_overshoot})"
        ),
    )
    for i, idx in enumerate(cuts_f):
        ax.axvline(
            idx,
            color="#c44e52",
            ls="--",
            lw=0.9,
            alpha=0.8,
            label="fast CUSUM snap" if i == 0 else None,
        )
    ax.set_ylabel("beta")
    ax.set_xlim(0, T - 1)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Zoom on break: where acceleration matters
    ax = axes[2]
    z0, z1 = break_window[0] - 20, min(T - 1, break_window[1] + 40)
    tz = t[z0 : z1 + 1]
    ax.plot(tz, raw[z0 : z1 + 1], color="black", lw=2.0, alpha=0.75, label="raw 100d")
    ax.plot(tz, short[z0 : z1 + 1], color="#8e44ad", lw=1.0, alpha=0.75, label="short OLS")
    ax.plot(
        tz,
        slow.beta[z0 : z1 + 1],
        color="#7f8c8d",
        lw=1.6,
        ls="--",
        label="long-adapt β",
    )
    ax.plot(tz, fast.beta[z0 : z1 + 1], color="#1f4e79", lw=1.8, label="burst β")
    ax.fill_between(
        tz,
        0.0,
        fast.adapting[z0 : z1 + 1].astype(float),
        step="pre",
        color="#f4a261",
        alpha=0.3,
    )
    for thr, col in ((1.2, "#e74c3c"), (1.5, "#c0392b")):
        ax.axhline(thr, color=col, ls=":", lw=0.8, alpha=0.7)
        db, df = _tti(slow.beta, thr), _tti(fast.beta, thr)
        if df is not None:
            ax.annotate(
                f"burst→{thr} in {df}d (long {db}d)",
                xy=(280 + df, thr),
                xytext=(280 + df + 15, thr + 0.12),
                fontsize=7,
                color=col,
                arrowprops=dict(arrowstyle="->", color=col, lw=0.7),
            )
    for idx in cuts_f:
        if z0 <= idx <= z1:
            ax.axvline(idx, color="#c44e52", ls="--", lw=0.9, alpha=0.8)
    ax.set_ylabel("beta (break zoom)")
    ax.set_xlim(z0, z1)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(t, slow.spread, color="#7f8c8d", lw=1.0, ls="--", label="long-adapt spread")
    ax.plot(t, fast.spread, color="#2ca02c", lw=1.1, label="burst spread")
    ax.axhline(0.0, color="gray", lw=0.7)
    for idx in cuts_f:
        ax.axvline(idx, color="#c44e52", ls="--", lw=0.9, alpha=0.75)
    ax.set_ylabel("spread")
    ax.set_xlabel("day index")
    ax.set_xlim(0, T - 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _mad_diff(beta: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = np.abs(np.diff(beta))
    if mask is not None:
        d = d[mask[1:]]
    return float(np.mean(d)) if d.size else float("nan")


def run_smah_ngte_example(
    prices_path: str = "prices.txt",
    out_dir: str = "cusum_output/smah_ngte",
    cfg: PairSegConfig | None = None,
) -> PairSegResult:
    from calibrate import extract_pair, load_prices

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or PairSegConfig()
    y, x = extract_pair(load_prices(prices_path), "SMAH", "NGTE")
    result = run_pair_seg(y, x, cfg)
    baseline = run_pair_seg(y, x, BASELINE_CFG)

    plot_path = out / "cusum_kalman_smoothed_vs_raw_beta.png"
    plot_smoothed_vs_raw(
        y, x, result, y_ticker="SMAH", x_ticker="NGTE", path=plot_path
    )
    cmp_path = out / "snap_speed_comparison.png"
    plot_snap_speed_comparison(
        y, x, baseline, result, y_ticker="SMAH", x_ticker="NGTE", path=cmp_path
    )

    adapt_dir = Path("cusum_output/smah_ngte_adaptive")
    adapt_dir.mkdir(parents=True, exist_ok=True)
    plot_smoothed_vs_raw(
        y,
        x,
        result,
        y_ticker="SMAH",
        x_ticker="NGTE",
        path=adapt_dir / "smah_ngte_adaptive_beta.png",
    )
    plot_snap_speed_comparison(
        y,
        x,
        baseline,
        result,
        y_ticker="SMAH",
        x_ticker="NGTE",
        path=adapt_dir / "snap_speed_comparison.png",
    )

    cuts = np.flatnonzero(result.cut)
    cuts_b = np.flatnonzero(baseline.cut)
    break_m = np.zeros(len(y), dtype=bool)
    break_m[280:561] = True

    def _tti(beta: np.ndarray, thr: float, start: int = 280) -> int | None:
        for i in range(start, len(beta)):
            if beta[i] >= thr:
                return int(i - start)
        return None

    tti_lines = []
    for thr in (1.2, 1.5):
        db, df = _tti(baseline.beta, thr), _tti(result.beta, thr)
        saved = "" if db is None or df is None else f" (saved {db - df}d)"
        tti_lines.append(
            f"- Days from break onset(280) to β≥{thr}: "
            f"long-adapt `{db}` / burst `{df}`{saved}"
        )

    report = "\n".join(
        [
            "# SMAH/NGTE sticky beta + Pine CUSUM (brief burst adapt)",
            "",
            f"- Burst cuts: `{int(result.cut.sum())}` at `{cuts.tolist()}`",
            f"- Long-adapt cuts: `{int(baseline.cut.sum())}` at `{cuts_b.tolist()}`",
            f"- Burst adapting fraction: `{float(result.adapting.mean()):.3f}` "
            f"(long `{float(baseline.adapting.mean()):.3f}`)",
            *tti_lines,
            f"- mean (burst − long) beta in break: "
            f"`{float((result.beta[280:561] - baseline.beta[280:561]).mean()):.4f}`",
            f"- Final h: `{float(result.h[-1]):.3f}` (auto={cfg.auto_h})",
            f"- adapt_bars: `{cfg.adapt_bars}`, extend: `{cfg.adapt_extend}`, "
            f"overshoot: `{cfg.snap_overshoot}`",
            f"- snap_window: `{cfg.snap_window}`, adapt_chase: `{cfg.adapt_chase}`",
            "",
            "Plots:",
            "- `cusum_kalman_smoothed_vs_raw_beta.png` (burst defaults)",
            "- `snap_speed_comparison.png` (long adapt vs burst lock)",
            "",
        ]
    )
    (out / "cusum_kalman_smoothed_vs_raw_beta_report.md").write_text(
        report, encoding="utf-8"
    )
    (adapt_dir / "smah_ngte_adaptive_beta_report.md").write_text(report, encoding="utf-8")
    return result


if __name__ == "__main__":
    run_smah_ngte_example()

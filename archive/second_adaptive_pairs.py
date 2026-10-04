"""Isolated pairs strategy with beta speed and CUSUM calibrated in sequence.

The model explicitly separates the cointegrating hedge ratio from the spread::

    y[t] = alpha[t] + beta[t] * x[t] + spread[t] + measurement_error[t]
    spread[t] = phi * spread[t-1] + spread_shock[t]
    beta[t] = beta[t-1] + beta_shock[t]

That distinction matters.  A levels Kalman regression with no spread state can
make its residual look stationary simply by moving beta.  A return regression
avoids that failure, but estimates short-run co-movement rather than the
cointegrating vector.  Here levels identify beta while an explicit stationary
state captures ordinary residual excursions.

Calibration is deliberately sequential:

1. CUSUM is forced off while ``calibrate_kalman_speed`` chooses ordinary beta
   process noise from the first 250 observations.  Selection uses prequential
   Gaussian loss and the one-standard-error rule, never trading PnL.
2. The selected speed is frozen.  Only then may a likelihood-ratio CUSUM reset
   beta uncertainty after evidence of a discrete relationship change.

The module trades only the fixed atlas relationships.  Ridge and generic pair
discovery are absent so they cannot mask the filter's behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import NamedTuple

import numpy as np


N_INST = 51
LEG = 10_000.0

PAIR_I = np.array([10, 1, 13, 8, 25, 49, 7, 31, 36, 18, 33, 35, 26], dtype=int)
PAIR_J = np.array([46, 20, 45, 27, 37, 50, 40, 43, 41, 35, 26, 28, 32], dtype=int)
PAIRS = tuple(zip(PAIR_I.tolist(), PAIR_J.tolist()))

CALIBRATION_OBSERVATIONS = 250
INITIAL_OBSERVATIONS = int(np.ceil(CALIBRATION_OBSERVATIONS ** 0.75))
ENABLE_CUSUM = False
FAMILY_FALSE_ALARM_RATE = 0.01
ADF_FIVE_PERCENT_CRITICAL = -2.86
ROUND_TRIP_LOG_COST = 4.0e-4
EPSILON = 1.0e-12

# Exact calibration prefixes are immutable.  Evaluation calls the strategy
# once per day, so repeating the same grid search would be pure computation.
# Bytes are used as the key to avoid ever reusing a speed for different data.
_SPEED_CACHE: dict[tuple[bytes, bytes], tuple[float, np.ndarray, np.ndarray]] = {}


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


class StepOut(NamedTuple):
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
    """Online state — mirrors the Pine ``var`` block (+ brief adapt countdown)."""

    def __init__(self, cfg: PairSegConfig | None = None) -> None:
        self.cfg = cfg or PairSegConfig()
        self.beta: float | None = None
        self.p22: float = 1.0  # beta variance
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
        "beta": o.beta,
        "spread": o.spread,
        "z": o.z,
        "h": o.h,
        "cut": o.cut,
        "adapting": o.adapting,
        "vol_shift": o.vol_shift,
    }


# --- Trade-signal policy for the PairSegFilter book -------------------------
SEG_ENTRY_Z = 1.5        # enter when the current spread is this many sigma out
SEG_BREAK_Z = 4.0        # beyond this looks like a break, not a fade
SEG_Z_WINDOW = 60        # window used to standardise the current spread
SEG_ADF_MAX_T = ADF_FIVE_PERCENT_CRITICAL   # require a stationary recent spread


def _seg_spread_zscore(spread: np.ndarray, window: int) -> float:
    """Current adaptive-beta residual, in local sigma."""
    recent = spread[-window:]
    if recent.size < 10:
        return 0.0
    history = recent[:-1]
    scale = float(history.std())
    if scale <= np.sqrt(EPSILON):
        return 0.0
    return float((recent[-1] - history.mean()) / scale)


def _seg_adf_t(spread: np.ndarray, window: int) -> float:
    """ADF(1) t-stat on the recent adaptive-beta residual."""
    s = np.asarray(spread[-window:], dtype=float)
    if s.size < 20 or float(s.std()) <= EPSILON:
        return 0.0
    delta = np.diff(s)
    target = delta[1:]
    design = np.column_stack((np.ones(target.size), s[1:-1], delta[:-1]))
    inverse = np.linalg.pinv(design.T @ design)
    coef = inverse @ design.T @ target
    error = target - design @ coef
    var = float(error @ error) / max(target.size - design.shape[1], 1)
    se = np.sqrt(max(var * inverse[1, 1], EPSILON))
    return float(coef[1] / se)


def _pair_candidates(
    log_prices: np.ndarray,
) -> list[tuple[float, int, int, float, float]]:
    """Run the online PairSegFilter on each atlas pair; keep stretched, stable,
    stationary spreads.  Returns ``(abs(z), i, j, beta, z)`` tuples.

    The filter supplies the adaptive hedge ratio and its residual; a pair is
    skipped while the filter is re-segmenting (a detected beta break), so a
    hedge-ratio jump is never mistaken for a tradeable spread excursion.
    """
    candidates = []
    for i, j in PAIRS:
        result = run_pair_seg(log_prices[i], log_prices[j])
        beta = float(result.beta[-1])
        if not np.isfinite(beta) or beta <= 0.0:
            continue
        if bool(result.adapting[-1]) or bool(result.cut[-1]):
            continue
        z = _seg_spread_zscore(result.spread, SEG_Z_WINDOW)
        if not (SEG_ENTRY_Z <= abs(z) < SEG_BREAK_Z):
            continue
        if _seg_adf_t(result.spread, 2 * SEG_Z_WINDOW) >= SEG_ADF_MAX_T:
            continue
        candidates.append((abs(z), i, j, beta, z))
    return candidates


def _allocate_pairs(
    candidates: list[tuple[float, int, int, float, float]],
) -> np.ndarray:
    """Allocate disjoint beta-neutral pairs, most-stretched first."""
    dollars = np.zeros(N_INST)
    used = np.zeros(N_INST, dtype=bool)
    for _, i, j, beta, z in sorted(candidates, key=lambda item: -item[0]):
        if used[i] or used[j] or z == 0.0:
            continue
        direction = -np.sign(z)                    # fade the deviation
        leg = LEG / max(1.0, abs(beta))
        dollars[i] = direction * leg
        dollars[j] = -direction * beta * leg
        used[i] = used[j] = True
    return dollars


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """Return target shares for the isolated adaptive-beta pair book."""
    n_instruments, n_days = prcSoFar.shape
    if n_instruments != N_INST or n_days <= CALIBRATION_OBSERVATIONS:
        return np.zeros(n_instruments, dtype=int)
    if not np.all(np.isfinite(prcSoFar[:, -1])) or np.any(prcSoFar[:, -1] <= 0.0):
        return np.zeros(n_instruments, dtype=int)
    candidates = _pair_candidates(np.log(prcSoFar))
    dollars = _allocate_pairs(candidates)
    dollars[0] = 0.0
    return (dollars / prcSoFar[:, -1]).astype(int)

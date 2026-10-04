"""CUSUM regime-adaptive Kalman for pairs beta (RESEARCH / EXPERIMENTAL).

Trading path: use ``adaptive_beta.PairSegFilter`` — Pine-faithful state only
(kLevel/kSlope/P/innovVar/CUSUMs/hLive/…).  This module accreted sticky-snap-
chase / OLS-drift / trend-exit knobs and should not be used as the production
filter.

Stable regimes: sticky through-origin beta (tiny ``q_beta``) so day-to-day
spread mean-reversion stays in the residual.

Regime change: on CUSUM + real OLS disagreement, snap beta to the short-window
through-origin OLS, inflate P, and briefly run elevated ``q_beta_adapt`` so
the hedge accelerates onto the new level — then lock sticky again.

Legacy ``adapt_mode="q_switch"`` (sustained Q×mult) is kept for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from research.cusum_legacy.calibrate import CalibrationResult, calibrate_pair, extract_pair, load_prices
from research.cusum_legacy.cusum import (
    DEFAULT_EXIT_BAND,
    DEFAULT_H,
    DEFAULT_KAPPA,
    CusumState,
    cusum_step,
    within_exit_band,
)
from research.cusum_legacy.kalman_pair import KalmanPairConfig, ols_init, run_kalman_pair


# ---- named tunable defaults ----
# Publish beta must be sticky: day-to-day spread MR belongs in the residual,
# not in beta.  MLE Q is far too large for that.  Default is through-origin
# with a tiny absolute Q_beta (matches ~100d rolling smoothness).
DEFAULT_ADAPT_MODE: Literal["p_inflate", "q_switch"] = "p_inflate"
DEFAULT_THROUGH_ORIGIN: bool = True
DEFAULT_Q_BETA: float = 3e-8  # slightly less glacial between snaps
DEFAULT_Q_BETA_ADAPT: float = 1.5e-4  # climb hard while adapting
DEFAULT_Q_SMOOTH_SCALE: float = 0.001  # used only if through_origin=False
DEFAULT_Q_ALPHA_FREEZE: float = 1e-4
DEFAULT_Q_MULTIPLIER: float = 75.0  # adapt_mode="q_switch" only
DEFAULT_Q_ADAPT_MULT: float = 1.0
DEFAULT_ADAPT_BARS: int = 8  # minimum elevate bars (FP can abort after this)
DEFAULT_MAX_ADAPT_BARS: int = 60  # cover ~40d tradable beta windows
DEFAULT_ADAPT_EXIT_MOVE: float = 0.03  # |OLS_short - beta| caught-up band
DEFAULT_ADAPT_CHASE: float = 0.35  # each adapt bar: pull beta toward short OLS
DEFAULT_TREND_EXIT_BARS: int = 10  # require short OLS flat this long before exit
DEFAULT_TREND_EXIT_MOVE: float = 0.025  # |OLS_now - OLS_lag| = flat
DEFAULT_MIN_CUT_GAP: int = 25  # allow multiple snaps inside a break
DEFAULT_RELEVEL_MIN_MOVE: float = 0.04  # absolute OLS disagreement floor
DEFAULT_RELEVEL_MOVE_FRAC: float = 0.04  # also: frac * |beta| (scale-aware)
DEFAULT_RELEVEL_ON_CUT: bool = True
DEFAULT_DETECT_PERSIST: int = 2  # bars of OLS disagree + CUSUM
DEFAULT_DRIFT_PERSIST: int = 5  # OLS-only drift trigger (no CUSUM required)
DEFAULT_EWMA_LAMBDA: float = 0.97
DEFAULT_EWMA_H: float = 10.0  # more CUSUM detections
DEFAULT_HYSTERESIS_N: int = 10
DEFAULT_RESEED_WINDOW: int = 15  # short OLS: detect + aggressive snap target
DEFAULT_SNAP_WINDOW: int = 15  # snap to short OLS (big jump)
DEFAULT_P_INFLATE: float = 5.0
DEFAULT_P_BETA_FLOOR: float = 0.5
DEFAULT_INIT_WINDOW: int = 20
DEFAULT_P_GATE_MULT: float = 3.0
DEFAULT_P_REF_EMA: float = 0.05

LEGACY_Z_THRESHOLD: float = 4.5
LEGACY_Z_CONSECUTIVE: int = 3


class Mode(IntEnum):
    STABLE = 0
    TURBULENT = 1  # cooldown after a cut


@dataclass(frozen=True)
class RegimeSwitchConfig:
    """All overridable regime-switch parameters."""

    kappa: float = DEFAULT_KAPPA
    h: float = DEFAULT_H
    exit_band: float = DEFAULT_EXIT_BAND
    hysteresis_n: int = DEFAULT_HYSTERESIS_N
    q_multiplier: float = DEFAULT_Q_MULTIPLIER
    through_origin: bool = DEFAULT_THROUGH_ORIGIN
    q_beta: float = DEFAULT_Q_BETA
    q_beta_adapt: float = DEFAULT_Q_BETA_ADAPT
    q_smooth_scale: float = DEFAULT_Q_SMOOTH_SCALE
    q_alpha_freeze: float = DEFAULT_Q_ALPHA_FREEZE
    q_adapt_mult: float = DEFAULT_Q_ADAPT_MULT
    adapt_bars: int = DEFAULT_ADAPT_BARS
    max_adapt_bars: int = DEFAULT_MAX_ADAPT_BARS
    adapt_exit_move: float = DEFAULT_ADAPT_EXIT_MOVE
    adapt_chase: float = DEFAULT_ADAPT_CHASE
    trend_exit_bars: int = DEFAULT_TREND_EXIT_BARS
    trend_exit_move: float = DEFAULT_TREND_EXIT_MOVE
    min_cut_gap: int = DEFAULT_MIN_CUT_GAP
    relevel_min_move: float = DEFAULT_RELEVEL_MIN_MOVE
    relevel_move_frac: float = DEFAULT_RELEVEL_MOVE_FRAC
    relevel_on_cut: bool = DEFAULT_RELEVEL_ON_CUT
    detect_persist: int = DEFAULT_DETECT_PERSIST
    drift_persist: int = DEFAULT_DRIFT_PERSIST
    ewma_lambda: float = DEFAULT_EWMA_LAMBDA
    ewma_h: float = DEFAULT_EWMA_H
    reseed_window: int = DEFAULT_RESEED_WINDOW
    snap_window: int = DEFAULT_SNAP_WINDOW
    p_inflate: float = DEFAULT_P_INFLATE
    p_beta_floor: float = DEFAULT_P_BETA_FLOOR
    p_gate_mult: float = DEFAULT_P_GATE_MULT
    adapt_mode: str = DEFAULT_ADAPT_MODE

@dataclass
class RegimeSwitchResult:
    """Auditable full-path history of the regime-switching filter."""

    alpha: np.ndarray
    beta: np.ndarray
    z: np.ndarray
    innov_var: np.ndarray
    s_pos: np.ndarray
    s_neg: np.ndarray
    mode: np.ndarray  # Mode enum values
    alarm_events: np.ndarray  # bool: CUSUM entry this step
    reseed_events: np.ndarray  # bool: hysteresis exit / reseed this step
    dwell: np.ndarray  # hysteresis counter path
    p_alpha: np.ndarray  # P[0,0] path (for alpha/x uncertainty diagnostics)
    p_beta: np.ndarray  # P[1,1] path
    config: RegimeSwitchConfig
    stable_config: KalmanPairConfig


def _one_step(
    y_t: float,
    x_t: float,
    alpha: float,
    beta: float,
    P: np.ndarray,
    Q: np.ndarray,
    R: float,
    *,
    min_innov_var: float = 1e-12,
) -> tuple[float, float, np.ndarray, float, float, float]:
    """Single predict/update step mirroring ``kalman_pair.run_kalman_pair``.

    Returns (alpha, beta, P_new, innovation, S, z).
    """
    # Predict
    P_pred = P + Q
    P_pred = 0.5 * (P_pred + P_pred.T)

    h0, h1 = 1.0, float(x_t)
    yhat = alpha + beta * h1
    innov = float(y_t) - yhat
    S = float(h0 * (h0 * P_pred[0, 0] + h1 * P_pred[1, 0])
              + h1 * (h0 * P_pred[0, 1] + h1 * P_pred[1, 1])
              + R)
    if S < min_innov_var:
        S = min_innov_var

    k0 = (P_pred[0, 0] * h0 + P_pred[0, 1] * h1) / S
    k1 = (P_pred[1, 0] * h0 + P_pred[1, 1] * h1) / S
    alpha = alpha + k0 * innov
    beta = beta + k1 * innov

    # Joseph form (same structure as kalman_pair._joseph_update)
    a00 = 1.0 - k0 * h0
    a01 = -k0 * h1
    a10 = -k1 * h0
    a11 = 1.0 - k1 * h1
    M = np.array([[a00, a01], [a10, a11]], dtype=float) @ P_pred
    P_new = M @ np.array([[a00, a10], [a01, a11]], dtype=float) + np.outer(
        [k0, k1], [k0, k1]
    ) * R
    P_new = 0.5 * (P_new + P_new.T)

    z = innov / np.sqrt(S)
    return alpha, beta, P_new, innov, S, z


def _ols_window_state(
    y: np.ndarray,
    x: np.ndarray,
    t: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """OLS [alpha, beta] and P from y[t-window+1 : t+1] (inclusive of t)."""
    start = max(0, t - window + 1)
    y_w = y[start : t + 1]
    x_w = x[start : t + 1]
    if len(y_w) < 2:
        return np.array([0.0, 1.0]), np.eye(2)
    x0, P0, _ = ols_init(y_w, x_w, init_window=len(y_w))
    return x0, P0


def _p_converged(
    P: np.ndarray,
    p_ref_alpha: float,
    p_ref_beta: float,
    *,
    mult: float = DEFAULT_P_GATE_MULT,
) -> bool:
    """True when filtered P diagonals are within ``mult`` of stable-regime refs."""
    return bool(
        P[0, 0] <= mult * p_ref_alpha and P[1, 1] <= mult * p_ref_beta
    )


def _through_origin_beta(
    y: np.ndarray,
    x: np.ndarray,
    t: int,
    window: int,
) -> float:
    """Rolling through-origin OLS beta on y[t-window+1 : t+1]."""
    start = max(0, t - window + 1)
    xw = x[start : t + 1]
    yw = y[start : t + 1]
    denom = float(xw @ xw)
    if denom < 1e-12:
        return 1.0
    return float(xw @ yw) / denom


def run_regime_switch(
    y: np.ndarray,
    x: np.ndarray,
    stable_config: KalmanPairConfig,
    *,
    kappa: float = DEFAULT_KAPPA,
    h: float = DEFAULT_H,
    exit_band: float = DEFAULT_EXIT_BAND,
    hysteresis_n: int = DEFAULT_HYSTERESIS_N,
    q_multiplier: float = DEFAULT_Q_MULTIPLIER,
    through_origin: bool = DEFAULT_THROUGH_ORIGIN,
    q_beta: float = DEFAULT_Q_BETA,
    q_beta_adapt: float = DEFAULT_Q_BETA_ADAPT,
    q_smooth_scale: float = DEFAULT_Q_SMOOTH_SCALE,
    q_alpha_freeze: float = DEFAULT_Q_ALPHA_FREEZE,
    q_adapt_mult: float = DEFAULT_Q_ADAPT_MULT,
    adapt_bars: int = DEFAULT_ADAPT_BARS,
    max_adapt_bars: int = DEFAULT_MAX_ADAPT_BARS,
    adapt_exit_move: float = DEFAULT_ADAPT_EXIT_MOVE,
    adapt_chase: float = DEFAULT_ADAPT_CHASE,
    trend_exit_bars: int = DEFAULT_TREND_EXIT_BARS,
    trend_exit_move: float = DEFAULT_TREND_EXIT_MOVE,
    min_cut_gap: int = DEFAULT_MIN_CUT_GAP,
    relevel_min_move: float = DEFAULT_RELEVEL_MIN_MOVE,
    relevel_move_frac: float = DEFAULT_RELEVEL_MOVE_FRAC,
    relevel_on_cut: bool = DEFAULT_RELEVEL_ON_CUT,
    detect_persist: int = DEFAULT_DETECT_PERSIST,
    drift_persist: int = DEFAULT_DRIFT_PERSIST,
    ewma_lambda: float = DEFAULT_EWMA_LAMBDA,
    ewma_h: float = DEFAULT_EWMA_H,
    reseed_window: int = DEFAULT_RESEED_WINDOW,
    snap_window: int = DEFAULT_SNAP_WINDOW,
    p_inflate: float = DEFAULT_P_INFLATE,
    p_beta_floor: float = DEFAULT_P_BETA_FLOOR,
    p_gate_mult: float = DEFAULT_P_GATE_MULT,
    p_ref_ema: float = DEFAULT_P_REF_EMA,
    adapt_mode: Literal["p_inflate", "q_switch"] = DEFAULT_ADAPT_MODE,
) -> RegimeSwitchResult:
    """Run CUSUM-gated adaptive Kalman over aligned pair series.

    ``adapt_mode="p_inflate"`` (default)
        Sticky through-origin publish filter (``q_beta``) so spread MR stays
        in the residual during stable regimes.  CUSUM runs on
        EWMA-standardized innovations with threshold ``ewma_h``.  A cut
        requires CUSUM alarm + ``detect_persist`` consecutive OLS disagreements
        of size ``>= relevel_min_move``.  On cut: snap, floor P, elevate Q for
        only ``adapt_bars`` (short — false positives can re-correct quickly).
        Adapt aborts early if the short OLS no longer disagrees with the
        pre-snap beta.

    ``adapt_mode="q_switch"`` (legacy)
        Sustained high-Q turbulent window (noisy — kept for comparison).
    """
    if adapt_mode not in ("p_inflate", "q_switch"):
        raise ValueError(f"adapt_mode must be 'p_inflate' or 'q_switch', got {adapt_mode!r}")
    if q_beta <= 0.0:
        raise ValueError(f"q_beta must be > 0, got {q_beta}")
    if q_beta_adapt < q_beta:
        raise ValueError("q_beta_adapt must be >= q_beta")
    if min_cut_gap < 1:
        raise ValueError(f"min_cut_gap must be >= 1, got {min_cut_gap}")

    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if y.shape != x.shape:
        raise ValueError("y and x length mismatch")

    cfg = RegimeSwitchConfig(
        kappa=kappa,
        h=h,
        exit_band=exit_band,
        hysteresis_n=hysteresis_n,
        q_multiplier=q_multiplier,
        through_origin=through_origin,
        q_beta=q_beta,
        q_beta_adapt=q_beta_adapt,
        q_smooth_scale=q_smooth_scale,
        q_alpha_freeze=q_alpha_freeze,
        q_adapt_mult=q_adapt_mult,
        adapt_bars=adapt_bars,
        max_adapt_bars=max_adapt_bars,
        adapt_exit_move=adapt_exit_move,
        adapt_chase=adapt_chase,
        trend_exit_bars=trend_exit_bars,
        trend_exit_move=trend_exit_move,
        min_cut_gap=min_cut_gap,
        relevel_min_move=relevel_min_move,
        relevel_move_frac=relevel_move_frac,
        relevel_on_cut=relevel_on_cut,
        detect_persist=detect_persist,
        drift_persist=drift_persist,
        ewma_lambda=ewma_lambda,
        ewma_h=ewma_h,
        reseed_window=reseed_window,
        snap_window=snap_window,
        p_inflate=p_inflate,
        p_beta_floor=p_beta_floor,
        p_gate_mult=p_gate_mult,
        adapt_mode=adapt_mode,
    )

    T = y.size
    Q_cal = np.asarray(stable_config.Q, dtype=float).copy()
    R = stable_config.R_scalar

    if through_origin:
        b0 = _through_origin_beta(y, x, min(DEFAULT_INIT_WINDOW - 1, T - 1), DEFAULT_INIT_WINDOW)
        alpha = 0.0
        beta = b0
        Q_pub = np.diag([1e-18, float(q_beta)])
        Q_adapt = np.diag([1e-18, float(q_beta_adapt)])
        P = np.diag([1e-18, 1.0])
    else:
        alpha = float(stable_config.x0[0])
        beta = float(stable_config.x0[1])
        Q_pub = Q_cal * q_smooth_scale
        Q_pub[0, 0] *= q_alpha_freeze
        Q_pub[0, 1] = 0.0
        Q_pub[1, 0] = 0.0
        Q_pub = 0.5 * (Q_pub + Q_pub.T)
        Q_adapt = Q_pub.copy()
        Q_adapt[1, 1] = max(float(Q_adapt[1, 1]), float(q_beta_adapt))
        P = np.asarray(stable_config.P0, dtype=float).copy()

    Q_turb = q_multiplier * Q_cal
    Q = Q_pub.copy()
    mode = Mode.STABLE
    cusum = CusumState()
    dwell = 0
    cooldown = 0
    last_cut = -min_cut_gap
    settling = False
    innov_var_ewma: float | None = None
    disagree_run = 0
    beta_pre_snap = float(beta)

    p_ref_alpha = float(P[0, 0])
    p_ref_beta = float(P[1, 1])

    alpha_out = np.empty(T)
    beta_out = np.empty(T)
    z_out = np.empty(T)
    S_out = np.empty(T)
    s_pos = np.empty(T)
    s_neg = np.empty(T)
    mode_out = np.empty(T, dtype=int)
    alarm_events = np.zeros(T, dtype=bool)
    reseed_events = np.zeros(T, dtype=bool)
    dwell_out = np.zeros(T, dtype=int)
    p_alpha_out = np.empty(T)
    p_beta_out = np.empty(T)

    h_cusum = float(ewma_h) if adapt_mode == "p_inflate" else float(h)

    for t in range(T):
        alpha, beta, P, innov, S, z_kalman = _one_step(
            y[t], x[t], alpha, beta, P, Q, R
        )
        if through_origin:
            alpha = 0.0  # hard constraint every step

        # CUSUM scale: EWMA of innov^2 (Pine / RiskMetrics) — NOT Kalman S.
        if innov_var_ewma is None:
            innov_var_ewma = innov * innov
        else:
            innov_var_ewma = (
                ewma_lambda * innov_var_ewma
                + (1.0 - ewma_lambda) * innov * innov
            )
        z = innov / np.sqrt(innov_var_ewma) if innov_var_ewma > 0.0 else 0.0
        cusum, alarmed = cusum_step(z, cusum, kappa=kappa, h=h_cusum)

        if adapt_mode == "p_inflate":
            if through_origin:
                detect_b = _through_origin_beta(y, x, t, reseed_window)
            else:
                x0_d, _ = _ols_window_state(y, x, t, reseed_window)
                detect_b = float(x0_d[1])
            move_now = abs(detect_b - beta)
            # Scale-aware gate: pairs with small |β| need a smaller absolute thr.
            thr = max(relevel_min_move, relevel_move_frac * abs(beta))
            if move_now >= thr:
                disagree_run += 1
            else:
                disagree_run = 0

            if cooldown > 0 or mode == Mode.TURBULENT:
                # Chase the short OLS while the regime is still trending.
                # Exit only when OLS has flattened AND beta has caught up
                # (or max bars / FP abort).
                mode = Mode.TURBULENT
                dwell += 1

                if relevel_on_cut and adapt_chase > 0.0:
                    beta = (1.0 - adapt_chase) * beta + adapt_chase * float(detect_b)
                    if through_origin:
                        alpha = 0.0
                    P[1, 1] = max(float(P[1, 1]), 0.25 * p_beta_floor)
                    P = 0.5 * (P + P.T)
                    move_now = abs(detect_b - beta)

                # Flat short-OLS trend?
                lag_t = t - trend_exit_bars
                if lag_t >= 0:
                    ols_lag = _through_origin_beta(y, x, lag_t, reseed_window)
                    ols_flat = abs(detect_b - ols_lag) < trend_exit_move
                else:
                    ols_flat = False

                fp_gone = abs(detect_b - beta_pre_snap) < 0.5 * thr
                caught_up = (
                    move_now < adapt_exit_move
                    and ols_flat
                    and dwell >= adapt_bars
                )
                timed_out = dwell >= max_adapt_bars

                if fp_gone or caught_up or timed_out:
                    Q = Q_pub.copy()
                    mode = Mode.STABLE
                    cooldown = 0
                    reseed_events[t] = True
                    dwell = 0
                    if fp_gone and relevel_on_cut:
                        beta = 0.7 * beta_pre_snap + 0.3 * beta
                        if through_origin:
                            alpha = 0.0
                        P[1, 1] = min(float(P[1, 1]), 0.05)
                        P = 0.5 * (P + P.T)
                        last_cut = t - min_cut_gap
                        disagree_run = 0
                else:
                    cooldown = 1  # remain adapting next bar
            elif (t - last_cut) >= min_cut_gap and (
                (alarmed and disagree_run >= detect_persist)
                or (disagree_run >= drift_persist)
            ):
                # CUSUM+OLS cut, OR sustained OLS drift without CUSUM
                # (pairs where sticky Q makes EWMA-z quiet but hedge still wanders).
                if through_origin:
                    target = float(detect_b)
                else:
                    x0, P0 = _ols_window_state(y, x, t, snap_window)
                    target_a, target_b = float(x0[0]), float(x0[1])
                    P0_snap = P0
                    target = target_b

                beta_pre_snap = float(beta)
                if relevel_on_cut:
                    if through_origin:
                        beta = float(target)
                        alpha = 0.0
                        P[0, 0] = 1e-18
                        P[1, 1] = max(float(P[1, 1]) * p_inflate, p_beta_floor)
                    else:
                        alpha, beta = target_a, target_b
                        P = p_inflate * P0_snap
                        P[1, 1] = max(float(P[1, 1]), p_beta_floor)
                    P = 0.5 * (P + P.T)

                Q = Q_adapt.copy()
                cusum = CusumState()
                last_cut = t
                disagree_run = 0
                cooldown = 1
                mode = Mode.TURBULENT
                dwell = 0
                alarm_events[t] = True
            else:
                if alarmed:
                    # CUSUM alone without sustained OLS disagreement → ignore.
                    cusum = CusumState()
                mode = Mode.STABLE
                p_ref_alpha = (1.0 - p_ref_ema) * p_ref_alpha + p_ref_ema * float(P[0, 0])
                p_ref_beta = (1.0 - p_ref_ema) * p_ref_beta + p_ref_ema * float(P[1, 1])

        else:
            if mode == Mode.STABLE and alarmed:
                mode = Mode.TURBULENT
                Q = Q_turb.copy()
                x0, P0 = _ols_window_state(y, x, t, reseed_window)
                alpha, beta = float(x0[0]), float(x0[1])
                P = p_inflate * P0
                P = 0.5 * (P + P.T)
                dwell = 0
                settling = False
                alarm_events[t] = True

            if mode == Mode.TURBULENT:
                if within_exit_band(cusum, exit_band=exit_band):
                    dwell += 1
                else:
                    dwell = 0

                if (not settling) and dwell >= hysteresis_n:
                    settling = True
                    Q = Q_cal.copy()

                if settling:
                    cusum = CusumState()
                    if _p_converged(P, p_ref_alpha, p_ref_beta, mult=p_gate_mult):
                        mode = Mode.STABLE
                        P = np.array(
                            [[p_ref_alpha, 0.0], [0.0, p_ref_beta]], dtype=float
                        )
                        Q = Q_cal.copy()
                        cusum = CusumState()
                        dwell = 0
                        settling = False
                        reseed_events[t] = True

            elif mode == Mode.STABLE:
                p_ref_alpha = (1.0 - p_ref_ema) * p_ref_alpha + p_ref_ema * float(P[0, 0])
                p_ref_beta = (1.0 - p_ref_ema) * p_ref_beta + p_ref_ema * float(P[1, 1])

        alpha_out[t] = alpha
        beta_out[t] = beta
        z_out[t] = z
        S_out[t] = S
        s_pos[t] = cusum.s_pos
        s_neg[t] = cusum.s_neg
        mode_out[t] = int(mode)
        dwell_out[t] = dwell
        p_alpha_out[t] = P[0, 0]
        p_beta_out[t] = P[1, 1]

    return RegimeSwitchResult(
        alpha=alpha_out,
        beta=beta_out,
        z=z_out,
        innov_var=S_out,
        s_pos=s_pos,
        s_neg=s_neg,
        mode=mode_out,
        alarm_events=alarm_events,
        reseed_events=reseed_events,
        dwell=dwell_out,
        p_alpha=p_alpha_out,
        p_beta=p_beta_out,
        config=cfg,
        stable_config=stable_config,
    )


def legacy_z_gate_events(
    z: np.ndarray,
    *,
    threshold: float = LEGACY_Z_THRESHOLD,
    consecutive: int = LEGACY_Z_CONSECUTIVE,
) -> np.ndarray:
    """Boolean mask: True on the step where ``consecutive`` |z|>=threshold ends."""
    z = np.asarray(z, dtype=float).ravel()
    events = np.zeros(z.size, dtype=bool)
    run = 0
    for t in range(z.size):
        if np.isfinite(z[t]) and abs(z[t]) >= threshold:
            run += 1
            if run >= consecutive:
                events[t] = True
                run = 0  # fire once per streak
        else:
            run = 0
    return events


def run_smah_ngte_worked_example(
    prices_path: str | Path = "prices.txt",
    *,
    calm_start: int = 0,
    calm_end: int = 170,
    kappa: float = DEFAULT_KAPPA,
    h: float | None = None,
    arl0: float | None = None,
    target_break_onset: int = 170,
    output_dir: str | Path = "cusum_output/smah_ngte",
    q_multiplier: float = DEFAULT_Q_MULTIPLIER,
    hysteresis_n: int = DEFAULT_HYSTERESIS_N,
) -> tuple[RegimeSwitchResult, CalibrationResult, dict]:
    """Full-series SMAH/NGTE pipeline with comparison to the legacy z-gate."""
    prices = load_prices(prices_path)
    y, x = extract_pair(prices, "SMAH", "NGTE")

    if h is None:
        from research.cusum_legacy.arl_calibration import run_arl_calibration

        arl = run_arl_calibration(prices_path, output_dir="cusum_output/arl_calibration")
        h = arl.chosen_h
        arl0 = arl.chosen_arl0
        kappa = arl.kappa
    elif arl0 is None:
        arl0 = float("nan")

    cal = calibrate_pair(
        y,
        x,
        y_ticker="SMAH",
        x_ticker="NGTE",
        calm_start=calm_start,
        calm_end=calm_end,
    )
    result = run_regime_switch(
        y,
        x,
        cal.config,
        kappa=kappa,
        h=h,
        q_multiplier=q_multiplier,
        hysteresis_n=hysteresis_n,
    )

    # Fast reference: always-turbulent / high-Q filter for the "climb" chart
    fast_cfg = KalmanPairConfig(
        Q=q_multiplier * cal.Q,
        R=cal.R,
        x0=cal.x0,
        P0=cal.P0 * DEFAULT_P_INFLATE,
    )
    fast = run_kalman_pair(y, x, fast_cfg)

    # Legacy gate on the stable-only filter innovations
    stable_only = run_kalman_pair(y, x, cal.config)
    legacy_events = legacy_z_gate_events(stable_only.z)

    alarm_idx = np.flatnonzero(result.alarm_events)
    reseed_idx = np.flatnonzero(result.reseed_events)
    legacy_idx = np.flatnonzero(legacy_events)

    first_alarm = int(alarm_idx[0]) if alarm_idx.size else None
    first_reseed = int(reseed_idx[0]) if reseed_idx.size else None
    first_legacy = int(legacy_idx[0]) if legacy_idx.size else None

    meta = {
        "h": h,
        "kappa": kappa,
        "arl0": arl0,
        "first_alarm": first_alarm,
        "first_reseed": first_reseed,
        "first_legacy": first_legacy,
        "target_break_onset": target_break_onset,
        "alarm_indices": alarm_idx,
        "reseed_indices": reseed_idx,
        "legacy_indices": legacy_idx,
        "fast_beta": fast.beta,
        "stable_only_beta": stable_only.beta,
        "stable_only_z": stable_only.z,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _plot_worked_example(out, result, meta, y_ticker="SMAH", x_ticker="NGTE")
    _write_worked_report(out, result, cal, meta)

    return result, cal, meta


def _plot_worked_example(
    out: Path,
    result: RegimeSwitchResult,
    meta: dict,
    *,
    y_ticker: str,
    x_ticker: str,
) -> None:
    T = len(result.beta)
    t = np.arange(T)
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    ax = axes[0]
    adapt = getattr(result.config, "adapt_mode", "p_inflate")
    ax.plot(t, meta["fast_beta"], color="#c44e52", lw=1.0, alpha=0.85, label="fast (always Q×mult)")
    ax.plot(t, result.beta, color="#1f4e79", lw=1.2, label=f"operational ({adapt})")
    ax.plot(t, meta["stable_only_beta"], color="#999999", lw=0.9, alpha=0.7, label="stable-only (no switch)")
    onset = meta["target_break_onset"]
    ax.axvline(onset, color="orange", ls=":", lw=1.2, label=f"known onset ~{onset}")
    for idx in meta["alarm_indices"]:
        ax.axvline(idx, color="#c44e52", ls="--", lw=0.7, alpha=0.6)
    for idx in meta["reseed_indices"]:
        ax.axvline(idx, color="#2ca02c", ls="--", lw=0.7, alpha=0.6)
    if meta["first_legacy"] is not None:
        ax.axvline(
            meta["first_legacy"],
            color="purple",
            ls="-.",
            lw=1.2,
            label=f"legacy |z|>={LEGACY_Z_THRESHOLD} x{LEGACY_Z_CONSECUTIVE} @ {meta['first_legacy']}",
        )
    ax.set_ylabel("beta")
    ax.set_title(f"{y_ticker}/{x_ticker}: CUSUM regime switch ({adapt})")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, result.s_pos, color="#1f4e79", lw=1.0, label="S+")
    ax.plot(t, result.s_neg, color="#c44e52", lw=1.0, label="S-")
    ax.axhline(result.config.h, color="gray", ls="--", lw=0.8)
    ax.axhline(-result.config.h, color="gray", ls="--", lw=0.8)
    ax.axhline(result.config.exit_band, color="green", ls=":", lw=0.7)
    ax.axhline(-result.config.exit_band, color="green", ls=":", lw=0.7)
    ax.set_ylabel("CUSUM")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.fill_between(t, 0, result.mode, step="pre", color="#f4a261", alpha=0.5, label="adapting")
    ax.set_ylabel("mode")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["stable", "adapting"])
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(t, result.z, color="#1f4e79", lw=0.8, label="operational z")
    ax.axhline(LEGACY_Z_THRESHOLD, color="purple", ls=":", lw=0.8)
    ax.axhline(-LEGACY_Z_THRESHOLD, color="purple", ls=":", lw=0.8)
    ax.set_ylabel("z_k")
    ax.set_xlabel("day index")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "smah_ngte_regime_switch.png", dpi=140)
    plt.close(fig)


def _write_worked_report(
    out: Path,
    result: RegimeSwitchResult,
    cal: CalibrationResult,
    meta: dict,
) -> None:
    fa = meta["first_alarm"]
    fr = meta["first_reseed"]
    fl = meta["first_legacy"]
    onset = meta["target_break_onset"]

    lines = [
        "# SMAH/NGTE CUSUM regime-switch worked example",
        "",
        "## Calibration",
        f"- Calm window: `[{cal.calm_start}, {cal.calm_end})`",
        f"- Q: `{np.diag(cal.Q).tolist()}` (diagonal)",
        f"- R: `{float(cal.R[0, 0]):.6g}`",
        f"- Optimizer success: `{cal.success}`",
        "",
        "## CUSUM / ARL",
        f"- kappa: `{meta['kappa']}`",
        f"- h (from pooled ARL): `{meta['h']}`",
        f"- empirical ARL0 at chosen h: `{meta['arl0']}`",
        f"- adapt_mode: `{getattr(result.config, 'adapt_mode', 'p_inflate')}`",
        f"- through_origin / q_beta: "
        f"`{getattr(result.config, 'through_origin', True)}` / "
        f"`{getattr(result.config, 'q_beta', 'n/a')}`",
        f"- min_cut_gap / reseed_window: "
        f"`{getattr(result.config, 'min_cut_gap', 'n/a')}` / "
        f"`{result.config.reseed_window}`",
        f"- P inflate: `{result.config.p_inflate}`",
        f"- Q multiplier (q_switch only): `{result.config.q_multiplier}`",
        f"- adapting fraction: `{float(result.mode.mean()):.3f}`",
        f"- mean |Δβ|: `{float(np.mean(np.abs(np.diff(result.beta)))):.5f}`",
        "",
        "## Detection timing vs known break (~day 170-200)",
        f"- Known onset reference: day `{onset}`",
        f"- First CUSUM alarm: day `{fa}`"
        + (
            f" (delay vs onset: {fa - onset} obs)"
            if fa is not None
            else " (none)"
        ),
        f"- First reseed: day `{fr}`",
        f"- Legacy gate (3 consecutive |z|>=4.5 on stable-only filter): day `{fl}`",
        "",
    ]

    if fa is not None and fl is not None:
        if fa < fl:
            lines.append(
                f"**CUSUM fired {fl - fa} observations earlier than the legacy "
                f"|z|>=4.5 x3 gate** (alarm {fa} vs legacy {fl})."
            )
        elif fa > fl:
            lines.append(
                f"**Discrepancy:** CUSUM fired later than the legacy gate "
                f"(alarm {fa} vs legacy {fl}). Reported rather than omitted."
            )
        else:
            lines.append("CUSUM and legacy gate fired on the same day.")
    elif fl is None and fa is not None:
        lines.append(
            f"**Legacy gate never fired** on the stable-only innovation stream "
            f"(no 3-in-a-row |z|>=4.5). CUSUM first alarm at day {fa}. "
            f"Relative to the known onset ~{onset}: delay = "
            f"{fa - onset if fa is not None else 'n/a'} observations. "
            "Near onset the calibrated-Q filter absorbs much of the beta "
            "transition (max |z| in days 150-220 is only ~2.5), so CUSUM "
            "accumulates from the residual drift rather than a single spike."
        )
    elif fa is None:
        lines.append(
            "**Discrepancy:** CUSUM never fired on the full series — "
            "reported rather than omitted."
        )

    lines.extend(
        [
            "",
            f"- Total alarm events: `{int(result.alarm_events.sum())}`",
            f"- Total reseed events: `{int(result.reseed_events.sum())}`",
            f"- Adapting fraction: `{float(result.mode.mean()):.3f}`",
            "",
            "Plot: `smah_ngte_regime_switch.png`",
            "",
        ]
    )
    (out / "smah_ngte_regime_switch_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

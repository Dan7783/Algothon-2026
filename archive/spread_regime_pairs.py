"""Atlas-pair research strategy with an explicit spread-regime state machine.

This module leaves ``adaptive_beta_pairs.py`` unchanged as the control.  It
implements the lifecycle

    STABLE -> RESPONSIVE -> VALIDATING -> STABLE or INACTIVE

causally after a fixed 250-close calibration prefix.  CUSUM monitors the
standardised OU forecast error ``spread[t] - phi * spread[t-1]``.  An alarm
only raises Kalman process noise; it never deactivates a pair.  Deactivation is
possible only after a fixed-alpha/fixed-beta post-adaptation spread has enough
observations and fails both stationarity and positive-mean-reversion checks.

No transition parameter is selected using trading PnL.  Recovery and
validation sample requirements are derived from the calibration spread's
half-life and coefficient-estimation uncertainty.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import numpy as np

import adaptive_beta_pairs as control


N_INST = control.N_INST
LEG = control.LEG
PAIRS = control.PAIRS
CALIBRATION_OBSERVATIONS = 250
INITIAL_OBSERVATIONS = control.INITIAL_OBSERVATIONS
FAMILY_FALSE_ALARM_RATE = control.FAMILY_FALSE_ALARM_RATE
ADF_FIVE_PERCENT_CRITICAL = control.ADF_FIVE_PERCENT_CRITICAL
ROUND_TRIP_LOG_COST = control.ROUND_TRIP_LOG_COST
EPSILON = control.EPSILON
NORMAL_CRITICAL = 1.959963984540054
ENABLE_REGIME_MACHINE = True


class Lifecycle(IntEnum):
    STABLE = 0
    RESPONSIVE = 1
    VALIDATING = 2
    INACTIVE = 3


class RegimeResult(NamedTuple):
    alpha: float
    beta: float
    spread_deviation: float
    spread_scale: float
    z: float
    kappa: float
    convergence_t: float
    adf_t: float
    expected_edge: float
    lifecycle: Lifecycle
    lifecycle_path: np.ndarray
    alarm_path: np.ndarray
    beta_path: np.ndarray
    spread_path: np.ndarray
    ou_innovation_path: np.ndarray
    cusum_path: np.ndarray
    q_multiplier_path: np.ndarray
    beta_response_path: np.ndarray
    transition_days: tuple[int, ...]
    recovery_observations: int
    validation_observations: int
    validation_adf_t: float
    validation_kappa: float
    validation_start: int
    validation_alpha: float
    validation_beta: float


class Calibration(NamedTuple):
    yn: np.ndarray
    xn: np.ndarray
    theta: np.ndarray
    covariance: np.ndarray
    observation_variance: float
    slow_process_covariance: np.ndarray
    responsive_process_covariance: np.ndarray
    regime_covariance: np.ndarray
    variance_lambda: float
    y_centre: float
    x_centre: float
    y_scale: float
    x_scale: float
    phi: float
    ou_variance: float
    beta_score_mean: float
    beta_score_variance: float
    recovery_observations: int
    validation_observations: int


def _kalman_update(
    theta: np.ndarray,
    covariance: np.ndarray,
    observation_variance: float,
    process_covariance: np.ndarray,
    x_value: float,
    y_value: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Random-walk alpha/beta Kalman update in Joseph covariance form."""
    prior_covariance = covariance + process_covariance
    observation = np.array((1.0, x_value))
    innovation = float(y_value - observation @ theta)
    innovation_variance = max(
        float(observation @ prior_covariance @ observation)
        + observation_variance,
        EPSILON,
    )
    gain = prior_covariance @ observation / innovation_variance
    theta = theta + gain * innovation
    identity_minus_gain = np.eye(2) - np.outer(gain, observation)
    covariance = (
        identity_minus_gain
        @ prior_covariance
        @ identity_minus_gain.T
        + np.outer(gain, gain) * observation_variance
    )
    covariance = 0.5 * (covariance + covariance.T)
    return theta, covariance, innovation, innovation_variance


def _ar_parameters(spread: np.ndarray) -> tuple[float, float, float]:
    """Return stationary AR coefficient, OU-shock variance and half-life."""
    spread = np.asarray(spread, dtype=float)
    centred = spread - float(np.mean(spread))
    denominator = max(float(centred[:-1] @ centred[:-1]), EPSILON)
    phi = float(centred[:-1] @ centred[1:] / denominator)
    phi = float(np.clip(phi, -0.98, 0.98))
    shock = centred[1:] - phi * centred[:-1]
    variance = control._robust_variance(shock)
    if 0.0 < phi < 1.0:
        half_life = float(np.log(0.5) / np.log(phi))
    else:
        half_life = float(np.sqrt(len(spread)))
    return phi, variance, max(half_life, 1.0)


def _long_run_variance(values: np.ndarray) -> float:
    """Newey-West variance with sample-size-derived cube-root bandwidth."""
    values = np.asarray(values, dtype=float)
    centred = values - float(np.mean(values))
    n = len(centred)
    variance = float(centred @ centred) / max(n, 1)
    bandwidth = max(1, int(np.ceil(n ** (1.0 / 3.0))))
    for lag in range(1, min(bandwidth, n - 1) + 1):
        covariance = float(centred[lag:] @ centred[:-lag]) / n
        variance += 2.0 * (1.0 - lag / (bandwidth + 1.0)) * covariance
    return max(variance, EPSILON)


def _calibrate(y: np.ndarray, x: np.ndarray) -> Calibration:
    """Freeze all ordinary-filter and lifecycle scales on closes 1..250."""
    prefix_y = y[:CALIBRATION_OBSERVATIONS]
    prefix_x = x[:CALIBRATION_OBSERVATIONS]
    (
        _,
        _,
        theta,
        covariance,
        observation_variance,
        variance_lambda,
        y_centre,
        x_centre,
        minimum_segment,
        slow_process_covariance,
        regime_covariance,
        y_scale,
        x_scale,
    ) = control._initial_filter_model(prefix_y, prefix_x)

    yn = (y - y_centre) / y_scale
    xn = (x - x_centre) / x_scale
    theta = theta.copy()
    covariance = covariance.copy()
    for t in range(INITIAL_OBSERVATIONS, CALIBRATION_OBSERVATIONS):
        theta, covariance, _, _ = _kalman_update(
            theta,
            covariance,
            observation_variance,
            slow_process_covariance,
            float(xn[t]),
            float(yn[t]),
        )

    calibration_spread = (
        yn[:CALIBRATION_OBSERVATIONS]
        - theta[0]
        - theta[1] * xn[:CALIBRATION_OBSERVATIONS]
    )
    phi, ou_variance, half_life = _ar_parameters(calibration_spread)
    calibration_dx = np.diff(xn[:CALIBRATION_OBSERVATIONS])
    calibration_dy = np.diff(yn[:CALIBRATION_OBSERVATIONS])
    calibration_beta_score = (
        calibration_dy - theta[1] * calibration_dx
    ) * calibration_dx
    beta_score_mean = float(np.mean(calibration_beta_score))
    beta_score_variance = _long_run_variance(calibration_beta_score)
    recovery_observations = int(
        np.ceil(max(np.sqrt(CALIBRATION_OBSERVATIONS), 2.0 * half_life))
    )
    validation_observations = int(
        np.ceil(max(minimum_segment, 3.0 * half_life))
    )

    # Across a complete responsive episode the added process variance equals
    # the empirically inferred coefficient-jump covariance.  This makes fast Q
    # pair-specific without an arbitrary covariance multiplier or bar count.
    responsive_process_covariance = (
        slow_process_covariance
        + regime_covariance / max(recovery_observations, 1)
    )
    return Calibration(
        yn,
        xn,
        theta,
        covariance,
        observation_variance,
        slow_process_covariance,
        responsive_process_covariance,
        regime_covariance,
        variance_lambda,
        y_centre,
        x_centre,
        y_scale,
        x_scale,
        phi,
        ou_variance,
        beta_score_mean,
        beta_score_variance,
        recovery_observations,
        validation_observations,
    )


def _cusum_threshold(test_number: int, family_size: int) -> float:
    """Two-sided family Page-CUSUM likelihood boundary.

    ``test_number`` is accepted so callers expose chronology explicitly, but a
    Page CUSUM uses a fixed boundary: its reset statistic already represents
    the maximum likelihood over possible change starts.  Confirmation is the
    separate fixed-spread validation state, not a growing alarm threshold.
    """
    del test_number
    return float(
        np.log(
            2.0 * max(family_size, 1) / FAMILY_FALSE_ALARM_RATE
        )
    )


def _family_normal_critical(family_size: int) -> float:
    """Conservative simultaneous normal-score boundary for beta escalation."""
    return float(
        np.sqrt(
            2.0
            * np.log(
                2.0 * max(family_size, 1) / FAMILY_FALSE_ALARM_RATE
            )
        )
    )


def _innovations_normalised(values: list[float], required: int) -> bool:
    """Test recent standardised OU errors for zero mean and unit second moment."""
    if len(values) < required:
        return False
    recent = np.asarray(values[-required:], dtype=float)
    mean = float(np.mean(recent))
    second_moment = float(np.mean(recent * recent))
    mean_limit = NORMAL_CRITICAL / np.sqrt(required)
    variance_limit = NORMAL_CRITICAL * np.sqrt(2.0 / required)
    return abs(mean) <= mean_limit and abs(second_moment - 1.0) <= variance_limit


def _fixed_spread_decision(
    spread: np.ndarray,
) -> tuple[Lifecycle | None, float, float]:
    """Three-way fixed-spread decision.

    Failure to reject a unit root is not evidence for one.  A positive kappa
    with weak ADF power therefore remains inconclusive and gathers more data.
    """
    adf_t = control._adf1_tstat(spread)
    centred = spread - float(np.mean(spread))
    lag = centred[:-1]
    delta = np.diff(centred)
    denominator = float(lag @ lag)
    kappa = (
        float(-(lag @ delta) / denominator)
        if denominator > EPSILON
        else 0.0
    )
    if adf_t < ADF_FIVE_PERCENT_CRITICAL and kappa > 0.0:
        decision: Lifecycle | None = Lifecycle.STABLE
    elif kappa <= 0.0:
        decision = Lifecycle.INACTIVE
    else:
        decision = None
    return decision, adf_t, kappa


def filter_pair(
    y: np.ndarray,
    x: np.ndarray,
    family_size: int = len(PAIRS),
) -> RegimeResult | None:
    """Replay the complete causal lifecycle for one fixed pair."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    if len(x) != n or n <= CALIBRATION_OBSERVATIONS:
        return None
    model = _calibrate(y, x)
    theta = model.theta.copy()
    covariance = model.covariance.copy()
    phi = model.phi
    ou_variance = model.ou_variance
    lifecycle = Lifecycle.STABLE
    positive_cusum = negative_cusum = 0.0
    previous_spread = float(
        model.yn[CALIBRATION_OBSERVATIONS - 1]
        - theta[0]
        - theta[1] * model.xn[CALIBRATION_OBSERVATIONS - 1]
    )
    response_z: list[float] = []
    frozen_theta: np.ndarray | None = None
    validation_spread: list[float] = []
    validation_start = -1
    validation_adf_t = np.nan
    validation_kappa = np.nan
    transition_days: list[int] = []

    lifecycle_path = np.full(n, int(Lifecycle.STABLE), dtype=np.int8)
    alarm_path = np.zeros(n, dtype=bool)
    beta_path = np.full(n, np.nan)
    spread_path = np.full(n, np.nan)
    ou_innovation_path = np.full(n, np.nan)
    cusum_path = np.zeros(n)
    q_multiplier_path = np.ones(n)
    beta_response_path = np.zeros(n, dtype=bool)
    beta_path[:CALIBRATION_OBSERVATIONS] = (
        theta[1] * model.y_scale / model.x_scale
    )
    active_responsive_covariance = model.responsive_process_covariance.copy()
    beta_fast_active = False
    beta_score_sum = 0.0
    beta_score_count = 0

    for t in range(CALIBRATION_OBSERVATIONS, n):
        process_covariance = (
            active_responsive_covariance
            if lifecycle == Lifecycle.RESPONSIVE
            else model.slow_process_covariance
        )
        q_multiplier_path[t] = float(
            np.trace(process_covariance)
            / max(np.trace(model.slow_process_covariance), EPSILON)
        )

        # The forecast residual is the spread.  Removing its expected AR(1)
        # continuation leaves the innovation CUSUM should monitor.
        observation = np.array((1.0, float(model.xn[t])))
        forecast_spread = float(model.yn[t] - observation @ theta)
        ou_innovation = forecast_spread - phi * previous_spread
        z_ou = ou_innovation / np.sqrt(max(ou_variance, EPSILON))
        spread_path[t] = forecast_spread * model.y_scale
        ou_innovation_path[t] = ou_innovation * model.y_scale

        if lifecycle == Lifecycle.STABLE:
            # Gaussian log-likelihood CUSUM for a one-standard-deviation shift.
            positive_cusum = max(0.0, positive_cusum + z_ou - 0.5)
            negative_cusum = max(0.0, negative_cusum - z_ou - 0.5)
            test_number = t - CALIBRATION_OBSERVATIONS + 1
            threshold = _cusum_threshold(test_number, family_size)
            if (
                ENABLE_REGIME_MACHINE
                and max(positive_cusum, negative_cusum) >= threshold
            ):
                lifecycle = Lifecycle.RESPONSIVE
                alarm_path[t] = True
                transition_days.append(t + 1)
                response_z = [z_ou]
                # A spread alarm supports faster intercept motion.  Beta starts
                # slow and requires subsequent differenced-regression evidence;
                # the alarm observation itself may be a one-off level jump.
                jump_per_observation = (
                    model.regime_covariance
                    / max(model.recovery_observations, 1)
                )
                jump_per_observation = jump_per_observation.copy()
                jump_per_observation[1, :] = 0.0
                jump_per_observation[:, 1] = 0.0
                beta_fast_active = False
                beta_score_sum = 0.0
                beta_score_count = 0
                active_responsive_covariance = (
                    model.slow_process_covariance + jump_per_observation
                )
                positive_cusum = negative_cusum = 0.0
                process_covariance = active_responsive_covariance
                q_multiplier_path[t] = float(
                    np.trace(process_covariance)
                    / max(np.trace(model.slow_process_covariance), EPSILON)
                )
            cusum_path[t] = max(positive_cusum, negative_cusum) / threshold
        elif lifecycle == Lifecycle.RESPONSIVE:
            response_z.append(z_ou)
            if not beta_fast_active:
                x_change = float(model.xn[t] - model.xn[t - 1])
                y_change = float(model.yn[t] - model.yn[t - 1])
                beta_score_sum += (
                    (y_change - theta[1] * x_change) * x_change
                    - model.beta_score_mean
                )
                beta_score_count += 1
                beta_score_z = beta_score_sum / np.sqrt(
                    max(
                        beta_score_count * model.beta_score_variance,
                        EPSILON,
                    )
                )
                if abs(beta_score_z) >= _family_normal_critical(family_size):
                    beta_fast_active = True
                    beta_response_path[t] = True
                    active_responsive_covariance = (
                        model.responsive_process_covariance.copy()
                    )
                    process_covariance = active_responsive_covariance

        theta, covariance, _, _ = _kalman_update(
            theta,
            covariance,
            model.observation_variance,
            process_covariance,
            float(model.xn[t]),
            float(model.yn[t]),
        )
        beta_path[t] = theta[1] * model.y_scale / model.x_scale
        previous_spread = forecast_spread

        if lifecycle == Lifecycle.RESPONSIVE and _innovations_normalised(
            response_z,
            model.recovery_observations,
        ):
            lifecycle = Lifecycle.VALIDATING
            frozen_theta = theta.copy()
            validation_spread = []
            validation_start = t + 1
            transition_days.append(t + 1)
        elif lifecycle == Lifecycle.VALIDATING and frozen_theta is not None:
            fixed_spread = float(model.yn[t] - observation @ frozen_theta)
            validation_spread.append(fixed_spread)
            if len(validation_spread) >= model.validation_observations:
                decision, validation_adf_t, validation_kappa = _fixed_spread_decision(
                    np.asarray(validation_spread)
                )
                if decision == Lifecycle.STABLE:
                    transition_days.append(t + 1)
                    lifecycle = Lifecycle.STABLE
                    phi, ou_variance, _ = _ar_parameters(
                        np.asarray(validation_spread)
                    )
                    previous_spread = validation_spread[-1]
                    positive_cusum = negative_cusum = 0.0
                    response_z = []
                    active_responsive_covariance = (
                        model.responsive_process_covariance.copy()
                    )
                    beta_fast_active = False
                    beta_score_sum = 0.0
                    beta_score_count = 0
                    frozen_theta = None
                    validation_spread = []
                elif decision == Lifecycle.INACTIVE:
                    transition_days.append(t + 1)
                    lifecycle = Lifecycle.INACTIVE

        # Ordinary robust variance adaptation is allowed only in a stable
        # regime; otherwise it would normalize away the very shift being tested.
        if lifecycle == Lifecycle.STABLE:
            family_z2 = 2.0 * np.log(
                2.0 * max(family_size, 1) / FAMILY_FALSE_ALARM_RATE
            )
            capped = min(
                ou_innovation * ou_innovation,
                family_z2 * ou_variance,
            )
            ou_variance = (
                model.variance_lambda * ou_variance
                + (1.0 - model.variance_lambda) * capped
            )
        lifecycle_path[t] = int(lifecycle)

    beta = float(theta[1] * model.y_scale / model.x_scale)
    alpha = float(
        model.y_centre
        + model.y_scale * theta[0]
        - beta * model.x_centre
    )
    (
        deviation,
        scale,
        z,
        kappa,
        convergence_t,
        adf_t,
    ) = control._fixed_beta_spread_health(
        y,
        x,
        alpha,
        beta,
        INITIAL_OBSERVATIONS,
    )
    expected_edge = max(0.0, kappa) * abs(deviation)
    if frozen_theta is None:
        validation_alpha = np.nan
        validation_beta = np.nan
    else:
        validation_beta = float(
            frozen_theta[1] * model.y_scale / model.x_scale
        )
        validation_alpha = float(
            model.y_centre
            + model.y_scale * frozen_theta[0]
            - validation_beta * model.x_centre
        )
    return RegimeResult(
        alpha,
        beta,
        deviation,
        scale,
        z,
        kappa,
        convergence_t,
        adf_t,
        expected_edge,
        lifecycle,
        lifecycle_path,
        alarm_path,
        beta_path,
        spread_path,
        ou_innovation_path,
        cusum_path,
        q_multiplier_path,
        beta_response_path,
        tuple(transition_days),
        model.recovery_observations,
        model.validation_observations,
        float(validation_adf_t),
        float(validation_kappa),
        validation_start,
        validation_alpha,
        validation_beta,
    )


def _pair_candidates(
    log_prices: np.ndarray,
) -> list[tuple[float, int, int, RegimeResult]]:
    """Keep tradable relationships; an alarm alone never removes a pair."""
    candidates = []
    for i, j in PAIRS:
        result = filter_pair(log_prices[i], log_prices[j])
        if result is None or result.lifecycle == Lifecycle.INACTIVE:
            continue
        stationary = result.adf_t < ADF_FIVE_PERCENT_CRITICAL
        converging = result.kappa > 0.0 and result.convergence_t < 0.0
        economical = result.expected_edge > ROUND_TRIP_LOG_COST
        finite = np.isfinite(result.z) and result.beta > 0.0
        if stationary and converging and economical and finite:
            candidates.append(
                (result.expected_edge - ROUND_TRIP_LOG_COST, i, j, result)
            )
    return candidates


def _allocate_pairs(
    candidates: list[tuple[float, int, int, RegimeResult]],
) -> np.ndarray:
    """Allocate disjoint beta-neutral pairs, strongest expected edge first."""
    dollars = np.zeros(N_INST)
    used = np.zeros(N_INST, dtype=bool)
    for _, i, j, result in sorted(candidates, key=lambda item: -item[0]):
        if used[i] or used[j] or result.spread_deviation == 0.0:
            continue
        direction = -np.sign(result.spread_deviation)
        leg = LEG / max(1.0, abs(result.beta))
        dollars[i] = direction * leg
        dollars[j] = -direction * result.beta * leg
        used[i] = used[j] = True
    return dollars


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """Return positions for the isolated spread-regime state-machine book."""
    n_instruments, n_days = prcSoFar.shape
    if n_instruments != N_INST or n_days <= CALIBRATION_OBSERVATIONS:
        return np.zeros(n_instruments, dtype=int)
    if not np.all(np.isfinite(prcSoFar[:, -1])) or np.any(prcSoFar[:, -1] <= 0.0):
        return np.zeros(n_instruments, dtype=int)
    dollars = _allocate_pairs(_pair_candidates(np.log(prcSoFar)))
    dollars[0] = 0.0
    return (dollars / prcSoFar[:, -1]).astype(int)

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


class PairFilterResult(NamedTuple):
    """Final trading state plus complete diagnostic paths."""

    alpha: float
    beta: float
    innovation: float
    innovation_scale: float
    z: float
    frozen_kappa: float
    convergence_t: float
    adf_t: float
    expected_edge: float
    regime_change: bool
    segment_age: int
    cusum_threshold: float
    minimum_segment: int
    q_beta_over_spread: float
    beta_path: np.ndarray
    innovation_path: np.ndarray
    z_path: np.ndarray
    regime_path: np.ndarray


class StateModel(NamedTuple):
    """Normalized beta-plus-AR-spread state-space specification."""

    y: np.ndarray
    x: np.ndarray
    y_centre: float
    x_centre: float
    y_scale: float
    x_scale: float
    initial_state: np.ndarray
    initial_covariance: np.ndarray
    spread_phi: float
    spread_process_variance: float
    measurement_variance: float


def _robust_variance(values: np.ndarray) -> float:
    """Normal-consistent MAD variance with a finite variance fallback."""
    values = np.asarray(values, dtype=float)
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre))) / 0.6744897501960817
    variance = mad * mad
    if not np.isfinite(variance) or variance <= EPSILON:
        variance = float(np.var(values))
    return max(variance, EPSILON)


def _adf1_tstat(series: np.ndarray) -> float:
    """ADF(1) level-coefficient t-statistic."""
    series = np.asarray(series, dtype=float)
    if len(series) < 10 or float(np.std(series)) <= EPSILON:
        return np.inf
    delta = np.diff(series)
    target = delta[1:]
    design = np.column_stack(
        (np.ones(len(target)), series[1:-1], delta[:-1])
    )
    inverse = np.linalg.pinv(design.T @ design)
    coefficients = inverse @ design.T @ target
    error = target - design @ coefficients
    variance = float(error @ error) / max(len(target) - design.shape[1], 1)
    standard_error = np.sqrt(max(variance * inverse[1, 1], EPSILON))
    return float(coefficients[1] / standard_error)


def _initial_state_model(y: np.ndarray, x: np.ndarray) -> StateModel:
    """Identify scale, cointegrating state and AR spread from the prefix."""
    initial = min(INITIAL_OBSERVATIONS, len(y))
    initial_y = y[:initial]
    initial_x = x[:initial]
    y_centre = float(np.mean(initial_y))
    x_centre = float(np.mean(initial_x))
    y_scale = max(float(np.std(initial_y)), np.sqrt(EPSILON))
    x_scale = max(float(np.std(initial_x)), np.sqrt(EPSILON))
    yn = (y - y_centre) / y_scale
    xn = (x - x_centre) / x_scale

    design = np.column_stack((np.ones(initial), xn[:initial]))
    normal_inverse = np.linalg.pinv(design.T @ design)
    coefficients = normal_inverse @ design.T @ yn[:initial]
    spread = yn[:initial] - design @ coefficients
    lag = spread[:-1]
    denominator = max(float(lag @ lag), EPSILON)
    spread_phi = float(lag @ spread[1:] / denominator)
    # A stationary spread is part of the model definition.  Values at the
    # numerical boundary otherwise imply infinite unconditional variance.
    spread_phi = float(np.clip(spread_phi, -0.98, 0.98))
    spread_shock = spread[1:] - spread_phi * lag
    spread_process_variance = _robust_variance(spread_shock)

    # With one close per day, spread-process and quote noise are only weakly
    # identifiable.  Treat the estimated daily spread shock as process noise;
    # the small regularizer is its variance divided by available information,
    # not a separately tuned measurement-noise parameter.
    measurement_variance = max(
        spread_process_variance / initial,
        EPSILON,
    )
    coefficient_covariance = _robust_variance(spread) * normal_inverse
    initial_covariance = np.zeros((3, 3))
    initial_covariance[:2, :2] = coefficient_covariance
    initial_covariance[2, 2] = _robust_variance(spread)
    initial_state = np.array(
        [coefficients[0], coefficients[1], spread[-1]],
        dtype=float,
    )
    return StateModel(
        yn,
        xn,
        y_centre,
        x_centre,
        y_scale,
        x_scale,
        initial_state,
        initial_covariance,
        spread_phi,
        spread_process_variance,
        measurement_variance,
    )


def _kalman_step(
    state: np.ndarray,
    covariance: np.ndarray,
    model: StateModel,
    q_beta_over_spread: float,
    t: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """One Joseph-form update of the explicit-spread state model."""
    transition = np.diag((1.0, 1.0, model.spread_phi))
    process_covariance = np.diag(
        (
            0.0,
            q_beta_over_spread * model.spread_process_variance,
            model.spread_process_variance,
        )
    )
    prior_state = transition @ state
    prior_covariance = (
        transition @ covariance @ transition.T + process_covariance
    )
    observation = np.array((1.0, model.x[t], 1.0))
    innovation = float(model.y[t] - observation @ prior_state)
    innovation_variance = max(
        float(observation @ prior_covariance @ observation)
        + model.measurement_variance,
        EPSILON,
    )
    gain = prior_covariance @ observation / innovation_variance
    state = prior_state + gain * innovation
    identity_minus_gain = np.eye(3) - np.outer(gain, observation)
    covariance = (
        identity_minus_gain @ prior_covariance @ identity_minus_gain.T
        + np.outer(gain, gain) * model.measurement_variance
    )
    covariance = 0.5 * (covariance + covariance.T)
    return state, covariance, innovation, innovation_variance


def _speed_loss(
    model: StateModel,
    q_beta_over_spread: float,
    stop: int,
) -> np.ndarray:
    """Prequential Gaussian loss with CUSUM structurally absent."""
    state = model.initial_state.copy()
    covariance = model.initial_covariance.copy()
    losses = []
    for t in range(INITIAL_OBSERVATIONS, stop):
        state, covariance, innovation, variance = _kalman_step(
            state,
            covariance,
            model,
            q_beta_over_spread,
            t,
        )
        losses.append(0.5 * (np.log(variance) + innovation * innovation / variance))
    return np.asarray(losses)


def _speed_grid(calibration_observations: int) -> np.ndarray:
    """Broad sample-size-derived beta-noise/spread-noise grid."""
    points = 2 * int(np.ceil(np.log2(calibration_observations))) + 1
    positive = np.geomspace(
        calibration_observations ** -4.0,
        calibration_observations ** -0.5,
        points,
    )
    return np.concatenate(([0.0], positive))


def calibrate_kalman_speed(
    y: np.ndarray,
    x: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Choose ordinary beta process noise without CUSUM or trading score.

    The one-standard-error rule selects the smoothest speed whose prequential
    loss is statistically indistinguishable from the empirical minimum.  It
    allows fast continuous beta only when its likelihood gain is material.
    """
    calibration_observations = min(
        CALIBRATION_OBSERVATIONS,
        len(y),
    )
    calibration_y = np.ascontiguousarray(
        y[:calibration_observations], dtype=float
    )
    calibration_x = np.ascontiguousarray(
        x[:calibration_observations], dtype=float
    )
    cache_key = (calibration_y.tobytes(), calibration_x.tobytes())
    cached = _SPEED_CACHE.get(cache_key)
    if cached is not None:
        return cached
    model = _initial_state_model(calibration_y, calibration_x)
    if calibration_observations <= INITIAL_OBSERVATIONS:
        result = (0.0, np.array([0.0]), np.array([np.nan]))
        _SPEED_CACHE[cache_key] = result
        return result
    grid = _speed_grid(calibration_observations - 1)
    paths = [
        _speed_loss(model, speed, calibration_observations)
        for speed in grid
    ]
    mean_loss = np.asarray([float(np.mean(path)) for path in paths])
    optimum = int(np.argmin(mean_loss))
    standard_error = float(np.std(paths[optimum], ddof=1)) / np.sqrt(
        len(paths[optimum])
    )
    eligible = np.flatnonzero(mean_loss <= mean_loss[optimum] + standard_error)
    selected = int(eligible[0])
    result = (float(grid[selected]), grid, mean_loss)
    _SPEED_CACHE[cache_key] = result
    return result


def _cusum_model(
    model: StateModel,
    calibration_stop: int,
) -> tuple[float, float]:
    """Infer a detectable beta jump independently of alarm frequency."""
    block_length = max(3, int(np.ceil(calibration_stop ** (2.0 / 3.0))))
    local_beta = []
    local_variance = []
    for start in range(0, calibration_stop - block_length + 1, block_length):
        stop = start + block_length
        block_x = model.x[start:stop]
        block_y = model.y[start:stop]
        design = np.column_stack((np.ones(block_length), block_x))
        normal_inverse = np.linalg.pinv(design.T @ design)
        coefficients = normal_inverse @ design.T @ block_y
        residual = block_y - design @ coefficients
        # Inflate the iid slope variance by the AR(1) long-run factor.
        ar_factor = (1.0 + model.spread_phi) / max(
            1.0 - model.spread_phi,
            EPSILON,
        )
        variance = (
            _robust_variance(residual)
            * normal_inverse[1, 1]
            * ar_factor
        )
        local_beta.append(float(coefficients[1]))
        local_variance.append(float(variance))

    if local_variance:
        detectable_shift = float(np.sqrt(np.median(local_variance)))
    else:
        detectable_shift = float(
            np.sqrt(max(model.initial_covariance[1, 1], EPSILON))
        )
    if len(local_beta) >= 2:
        observed_change = np.diff(np.asarray(local_beta)) ** 2
        estimation_change = (
            np.asarray(local_variance[:-1]) + np.asarray(local_variance[1:])
        )
        excess_change = np.maximum(observed_change - estimation_change, 0.0)
        jump_variance = max(
            float(np.median(excess_change)),
            detectable_shift * detectable_shift,
        )
    else:
        jump_variance = detectable_shift * detectable_shift
    return detectable_shift, jump_variance


def _cusum_threshold(monitoring_test: int, family_size: int) -> float:
    """Anytime-valid two-sided family threshold for repeated monitoring.

    Allocate the family false-alarm probability over monitoring times with
    ``6 / (pi^2 k^2)``.  Those weights sum to one, so repeatedly checking the
    CUSUM does not silently turn a 1% policy into near-certain eventual alarms.
    """
    test = max(int(monitoring_test), 1)
    return float(
        np.log(
            2.0
            * max(family_size, 1)
            * np.pi
            * np.pi
            * test
            * test
            / (6.0 * FAMILY_FALSE_ALARM_RATE)
        )
    )


def _run_state_filter(
    model: StateModel,
    q_beta_over_spread: float,
    family_size: int,
    enable_cusum: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Run frozen-speed Kalman updates, then optional beta-break CUSUM."""
    n = len(model.y)
    state = model.initial_state.copy()
    covariance = model.initial_covariance.copy()
    beta_path = np.full(n, np.nan)
    spread_path = np.full(n, np.nan)
    forecast_z_path = np.full(n, np.nan)
    regime_path = np.zeros(n, dtype=bool)
    state_path = np.full((n, 3), np.nan)
    state_path[:INITIAL_OBSERVATIONS, 0] = state[0]
    state_path[:INITIAL_OBSERVATIONS, 1] = state[1]
    beta_path[:INITIAL_OBSERVATIONS] = (
        state[1] * model.y_scale / model.x_scale
    )
    spread_path[:INITIAL_OBSERVATIONS] = (
        model.y[:INITIAL_OBSERVATIONS]
        - state[0]
        - state[1] * model.x[:INITIAL_OBSERVATIONS]
    ) * model.y_scale

    detectable_shift, jump_variance = _cusum_model(
        model,
        CALIBRATION_OBSERVATIONS - 1,
    )
    threshold = _cusum_threshold(1, family_size)
    positive = negative = 0.0
    positive_start = negative_start = CALIBRATION_OBSERVATIONS
    segment_start = INITIAL_OBSERVATIONS
    # Efficient slope scores must be orthogonal to the nuisance intercept.
    # Centre x on information strictly preceding the monitored observation.
    x_sum = float(np.sum(model.x[:CALIBRATION_OBSERVATIONS]))
    x_count = CALIBRATION_OBSERVATIONS

    for t in range(INITIAL_OBSERVATIONS, n):
        # Compute the same prior and forecast used by _kalman_step so the CUSUM
        # increment is an exact Gaussian likelihood ratio for a beta jump.
        transition = np.diag((1.0, 1.0, model.spread_phi))
        process_covariance = np.diag(
            (
                0.0,
                q_beta_over_spread * model.spread_process_variance,
                model.spread_process_variance,
            )
        )
        prior_state = transition @ state
        prior_covariance = (
            transition @ covariance @ transition.T + process_covariance
        )
        observation = np.array((1.0, model.x[t], 1.0))
        innovation = float(model.y[t] - observation @ prior_state)
        innovation_variance = max(
            float(observation @ prior_covariance @ observation)
            + model.measurement_variance,
            EPSILON,
        )
        forecast_z_path[t] = innovation / np.sqrt(innovation_variance)

        if t >= CALIBRATION_OBSERVATIONS:
            monitoring_test = t - CALIBRATION_OBSERVATIONS + 1
            threshold = _cusum_threshold(monitoring_test, family_size)
            beta_loading = model.x[t] - x_sum / max(x_count, 1)
            information = beta_loading * beta_loading / innovation_variance
            score = beta_loading * innovation / innovation_variance
            penalty = 0.5 * detectable_shift * detectable_shift * information
            previous_positive = positive
            previous_negative = negative
            positive = max(
                0.0,
                positive + detectable_shift * score - penalty,
            )
            negative = max(
                0.0,
                negative - detectable_shift * score - penalty,
            )
            if previous_positive == 0.0 and positive > 0.0:
                positive_start = t
            if previous_negative == 0.0 and negative > 0.0:
                negative_start = t
            if enable_cusum and max(positive, negative) >= threshold:
                change_start = (
                    positive_start if positive >= negative else negative_start
                )
                # A beta jump at tau must be accompanied by
                # delta_alpha=-x[tau]*delta_beta to avoid inventing a price or
                # spread discontinuity.  Restore uncertainty only along that
                # joint direction; the spread state and ordinary q stay fixed.
                jump_direction = np.array(
                    (-model.x[change_start], 1.0, 0.0)
                )
                covariance += jump_variance * np.outer(
                    jump_direction,
                    jump_direction,
                )
                regime_path[t] = True
                segment_start = change_start
                positive = negative = 0.0
                positive_start = negative_start = t + 1
            x_sum += float(model.x[t])
            x_count += 1

        # Reuse the forecast already computed for CUSUM; recomputing the same
        # 3x3 prediction here approximately doubled live runtime.
        gain = prior_covariance @ observation / innovation_variance
        state = prior_state + gain * innovation
        identity_minus_gain = np.eye(3) - np.outer(gain, observation)
        covariance = (
            identity_minus_gain
            @ prior_covariance
            @ identity_minus_gain.T
            + np.outer(gain, gain) * model.measurement_variance
        )
        covariance = 0.5 * (covariance + covariance.T)
        state_path[t] = state
        beta_path[t] = state[1] * model.y_scale / model.x_scale
        spread_path[t] = state[2] * model.y_scale

    return (
        state_path,
        beta_path,
        spread_path,
        forecast_z_path,
        regime_path,
        segment_start,
        threshold,
    )


def _fixed_beta_spread_health(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float,
    beta: float,
    start: int,
) -> tuple[float, float, float, float, float, float]:
    """Test realized convergence with the current alpha and beta frozen."""
    spread = y[start:] - alpha - beta * x[start:]
    if len(spread) < 10:
        return 0.0, EPSILON, 0.0, 0.0, 0.0, np.inf
    history = spread[:-1]
    deviation = float(spread[-1] - np.mean(history))
    scale = max(float(np.std(history)), np.sqrt(EPSILON))
    z = deviation / scale
    centred = spread - float(np.mean(history))
    lag = centred[:-1]
    delta = np.diff(centred)
    denominator = float(lag @ lag)
    if denominator <= EPSILON:
        return deviation, scale, z, 0.0, 0.0, _adf1_tstat(spread)
    coefficient = float(lag @ delta / denominator)
    error = delta - coefficient * lag
    variance = float(error @ error) / max(len(error) - 1, 1)
    standard_error = np.sqrt(max(variance / denominator, EPSILON))
    return (
        deviation,
        scale,
        z,
        -coefficient,
        coefficient / standard_error,
        _adf1_tstat(spread),
    )


def filter_pair(
    y: np.ndarray,
    x: np.ndarray,
    family_size: int = len(PAIRS),
    enable_cusum: bool | None = None,
) -> PairFilterResult | None:
    """Estimate one pair; CUSUM cannot affect its calibrated Kalman speed."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    if len(x) != n or n <= CALIBRATION_OBSERVATIONS:
        return None

    q_beta_over_spread, _, _ = calibrate_kalman_speed(
        y[:CALIBRATION_OBSERVATIONS],
        x[:CALIBRATION_OBSERVATIONS],
    )
    model = _initial_state_model(y, x)
    use_cusum = ENABLE_CUSUM if enable_cusum is None else enable_cusum
    (
        state_path,
        beta_path,
        spread_path,
        forecast_z_path,
        regime_path,
        segment_start,
        cusum_threshold,
    ) = _run_state_filter(
        model,
        q_beta_over_spread,
        family_size,
        use_cusum,
    )
    current_state = state_path[-1]
    normalized_beta = float(current_state[1])
    beta = normalized_beta * model.y_scale / model.x_scale
    alpha = (
        model.y_centre
        + model.y_scale * float(current_state[0])
        - beta * model.x_centre
    )
    valid_start = max(INITIAL_OBSERVATIONS, segment_start)
    (
        frozen_deviation,
        frozen_scale,
        frozen_z,
        frozen_kappa,
        convergence_t,
        adf_t,
    ) = _fixed_beta_spread_health(
        y,
        x,
        alpha,
        beta,
        valid_start,
    )

    # The state spread is the residual after the Kalman update; fixed-beta
    # health above prevents beta movement from being credited as convergence.
    current_spread = float(spread_path[-1])
    stationary_scale = (
        np.sqrt(
            model.spread_process_variance
            / max(1.0 - model.spread_phi * model.spread_phi, EPSILON)
        )
        * model.y_scale
    )
    spread_z = current_spread / max(stationary_scale, np.sqrt(EPSILON))
    expected_edge = max(0.0, frozen_kappa) * abs(current_spread)
    return PairFilterResult(
        alpha=alpha,
        beta=beta,
        innovation=current_spread,
        innovation_scale=stationary_scale,
        z=spread_z,
        frozen_kappa=frozen_kappa,
        convergence_t=convergence_t,
        adf_t=adf_t,
        expected_edge=expected_edge,
        regime_change=bool(regime_path[-1]),
        segment_age=int(n - 1 - valid_start),
        cusum_threshold=cusum_threshold,
        minimum_segment=0,
        q_beta_over_spread=q_beta_over_spread,
        beta_path=beta_path,
        innovation_path=spread_path,
        z_path=forecast_z_path,
        regime_path=regime_path,
    )


def _family_z_limit(family_size: int) -> float:
    """Simultaneous Gaussian tail limit for an obvious spread break."""
    return float(
        np.sqrt(
            2.0
            * np.log(
                2.0 * max(family_size, 1) / FAMILY_FALSE_ALARM_RATE
            )
        )
    )


def _pair_candidates(
    log_prices: np.ndarray,
) -> list[tuple[float, int, int, PairFilterResult]]:
    """Validate fixed relationships independently of portfolio PnL."""
    candidates = []
    z_limit = _family_z_limit(len(PAIRS))
    for i, j in PAIRS:
        result = filter_pair(log_prices[i], log_prices[j])
        if result is None:
            continue
        stationary = result.adf_t < ADF_FIVE_PERCENT_CRITICAL
        converging = result.frozen_kappa > 0.0 and result.convergence_t < 0.0
        economical = result.expected_edge > ROUND_TRIP_LOG_COST
        intact = (
            result.beta > 0.0
            and np.isfinite(result.z)
            and abs(result.z) <= z_limit
            and not result.regime_change
        )
        if stationary and converging and economical and intact:
            candidates.append(
                (result.expected_edge - ROUND_TRIP_LOG_COST, i, j, result)
            )
    return candidates


def _allocate_pairs(
    candidates: list[tuple[float, int, int, PairFilterResult]],
) -> np.ndarray:
    """Allocate disjoint beta-neutral pairs by expected convergence."""
    dollars = np.zeros(N_INST)
    used = np.zeros(N_INST, dtype=bool)
    for _, i, j, result in sorted(candidates, key=lambda item: -item[0]):
        if used[i] or used[j] or result.innovation == 0.0:
            continue
        direction = -np.sign(result.innovation)
        leg = LEG / max(1.0, abs(result.beta))
        dollars[i] = direction * leg
        dollars[j] = -direction * result.beta * leg
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

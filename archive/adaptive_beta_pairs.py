"""Pairs-only strategy with smooth beta tracking and CUSUM regime adaptation.

This is a research-isolated port of the useful idea in the supplied Pine
script.  A pair is modelled as

    log(P_i,t) = alpha_t + beta_t log(P_j,t) + innovation_t.

The intercept and hedge ratio follow a deliberately slow random walk.  An
optional two-sided CUSUM monitors the beta score (hedged return error times the
explanatory return), while a variance CUSUM monitors innovation dispersion.
The beta-specific detector is implemented for research but defaults off: on
the supplied data its regime updates worsen both frozen-beta convergence and
out-of-sample innovation diagnostics.  This is intentionally different from
blindly applying the Pine script's price-innovation CUSUM to a spread.

Pair selection, filtering, spread validation, signal construction and
portfolio allocation are separate functions.  There is no ridge forecast or
generic pair discovery here, so the beta model can be assessed without another
signal masking its behaviour.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


N_INST = 51
LEG = 10_000.0

# The fixed relationships currently used by the atlas sleeve.  Keeping the
# universe fixed is intentional: beta estimation must not be evaluated while
# pair discovery is changing the sample at the same time.
PAIR_I = np.array([10, 1, 13, 8, 25, 49, 7, 31, 36, 18, 33, 35, 26], dtype=int)
PAIR_J = np.array([46, 20, 45, 27, 37, 50, 40, 43, 41, 35, 26, 28, 32], dtype=int)
PAIRS = tuple(zip(PAIR_I.tolist(), PAIR_J.tolist()))

# Statistical policy rather than pair-specific tuning.  Sixty observations is
# used only to identify a two-state regression and its noise; regime duration,
# variance memory, process noise and the CUSUM threshold are data-derived.
INITIAL_OBSERVATIONS = 60
FAMILY_FALSE_ALARM_RATE = 0.01
TARGET_BETA_SCORE_SHIFT = 1.0
ENABLE_CUSUM = False
ADF_FIVE_PERCENT_CRITICAL = -2.86
ROUND_TRIP_LOG_COST = 4.0e-4
EPSILON = 1.0e-12


class PairFilterResult(NamedTuple):
    """Final signal state plus paths used by diagnostics."""

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
    q_beta_over_r: float
    beta_path: np.ndarray
    innovation_path: np.ndarray
    z_path: np.ndarray
    regime_path: np.ndarray


def _robust_variance(values: np.ndarray) -> float:
    """Normal-consistent MAD variance with a conventional variance fallback."""
    values = np.asarray(values, dtype=float)
    centre = np.median(values)
    mad_scale = np.median(np.abs(values - centre)) / 0.6744897501960817
    variance = float(mad_scale * mad_scale)
    if not np.isfinite(variance) or variance <= EPSILON:
        variance = float(np.var(values))
    return max(variance, EPSILON)


def _ols_state(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS coefficient state, covariance and robust observation variance."""
    design = np.column_stack((np.ones(len(x)), x))
    normal_inverse = np.linalg.pinv(design.T @ design)
    theta = normal_inverse @ design.T @ y
    residual = y - design @ theta
    observation_variance = _robust_variance(residual)
    covariance = observation_variance * normal_inverse
    covariance += np.eye(2) * EPSILON
    return theta, covariance, observation_variance


def _adf1_tstat(series: np.ndarray) -> float:
    """ADF(1) level-coefficient t-statistic, used as a health diagnostic."""
    series = np.asarray(series, dtype=float)
    if len(series) < 10 or np.std(series) <= EPSILON:
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


def _initial_filter_model(
    y: np.ndarray,
    x: np.ndarray,
) -> tuple:
    """Scale a pair and infer pair-specific state/observation noise.

    Ordinary process noise is the local coefficient-estimation uncertainty
    amortized over all available prior observations.  Differences between
    local regressions estimate the size of a *regime jump*, not daily beta
    noise.  This separation is what lets beta remain smooth normally and move
    quickly only after CUSUM supplies evidence of a break.
    """
    initial_y = y[:INITIAL_OBSERVATIONS]
    initial_x = x[:INITIAL_OBSERVATIONS]
    y_centre = float(initial_y.mean())
    x_centre = float(initial_x.mean())
    y_scale = max(float(initial_y.std()), np.sqrt(EPSILON))
    x_scale = max(float(initial_x.std()), np.sqrt(EPSILON))
    yn = (y - y_centre) / y_scale
    xn = (x - x_centre) / x_scale

    theta, covariance, initial_observation_variance = _ols_state(
        yn[:INITIAL_OBSERVATIONS], xn[:INITIAL_OBSERVATIONS]
    )

    # Infer ordinary coefficient drift from multiple local regressions over all
    # observations already closed before the current decision.  The median of
    # uncertainty-adjusted changes is robust to an isolated regime break; the
    # old two-half estimate treated any one break as permanent daily noise.
    calibration_end = max(INITIAL_OBSERVATIONS, len(y) - 1)
    local_length = max(
        2 * (len(theta) + 1),
        int(np.ceil(calibration_end ** (2.0 / 3.0))),
    )
    local_theta = []
    local_covariance = []
    local_variance = []
    for start in range(0, calibration_end - local_length + 1, local_length):
        stop = start + local_length
        local = _ols_state(yn[start:stop], xn[start:stop])
        local_theta.append(local[0])
        local_covariance.append(local[1])
        local_variance.append(local[2])
    if len(local_theta) < 2:
        local_theta = [theta]
        local_covariance = [covariance]
        local_variance = [initial_observation_variance]

    local_theta_array = np.asarray(local_theta)
    covariance_diagonal = np.asarray(
        [np.diag(item) for item in local_covariance]
    )
    observation_variance = float(np.median(local_variance))
    if len(local_theta_array) >= 2:
        observed_shift = np.diff(local_theta_array, axis=0) ** 2
        estimation_noise = covariance_diagonal[:-1] + covariance_diagonal[1:]
        regime_diagonal = np.median(
            np.maximum(observed_shift, estimation_noise), axis=0
        )
    else:
        regime_diagonal = covariance_diagonal[0]

    calibration_design = np.column_stack(
        (np.ones(calibration_end), xn[:calibration_end])
    )
    calibration_covariance = observation_variance * np.linalg.pinv(
        calibration_design.T @ calibration_design
    )
    covariance_floor = np.diag(calibration_covariance) / calibration_end
    # Do not reinterpret local-regression disagreement as continuous beta
    # motion.  That made beta chase the spread.  The slow covariance floor is
    # pair-specific; CUSUM uses ``regime_diagonal`` for fast event-driven moves.
    process_diagonal = covariance_floor
    process_covariance = np.diag(process_diagonal)

    # A detected break restores empirically observed coefficient uncertainty;
    # there is no universal covariance-inflation multiplier.
    regime_covariance = np.diag(regime_diagonal)

    initial_residual = (
        yn[:INITIAL_OBSERVATIONS]
        - theta[0]
        - theta[1] * xn[:INITIAL_OBSERVATIONS]
    )
    lag = initial_residual[:-1] - initial_residual[:-1].mean()
    lead = initial_residual[1:] - initial_residual[1:].mean()
    phi = float(lag @ lead / max(lag @ lag, EPSILON))
    if 0.0 < phi < 1.0:
        half_life = float(np.log(0.5) / np.log(phi))
    else:
        half_life = float(np.sqrt(INITIAL_OBSERVATIONS))
    memory_half_life = max(half_life, np.sqrt(INITIAL_OBSERVATIONS))
    variance_lambda = float(np.exp(np.log(0.5) / memory_half_life))
    minimum_segment = int(
        np.ceil(
            min(
                calibration_end / 2.0,
                max(local_length, 2.0 * half_life),
            )
        )
    )

    return (
        yn,
        xn,
        theta,
        covariance,
        observation_variance,
        variance_lambda,
        y_centre,
        x_centre,
        minimum_segment,
        process_covariance,
        regime_covariance,
        y_scale,
        x_scale,
    )


def _fixed_beta_spread_health(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float,
    beta: float,
    start: int,
) -> tuple[float, float, float, float, float, float]:
    """Validate the current regime with alpha and beta held fixed.

    This is deliberately separate from Kalman innovations.  A state update
    cannot improve these statistics: only realised movement of the two assets
    can make the fixed-beta spread converge.
    """
    spread = y[start:] - alpha - beta * x[start:]
    if len(spread) < 10:
        return 0.0, EPSILON, 0.0, 0.0, 0.0, np.inf
    history = spread[:-1]
    deviation = float(spread[-1] - history.mean())
    scale = max(float(history.std()), np.sqrt(EPSILON))
    z = deviation / scale

    lag = spread[:-1] - spread[:-1].mean()
    delta = np.diff(spread) - np.diff(spread).mean()
    denominator = float(lag @ lag)
    if denominator <= EPSILON:
        return deviation, scale, z, 0.0, 0.0, _adf1_tstat(spread)
    coefficient = float(lag @ delta / denominator)
    error = delta - coefficient * lag
    variance = float(error @ error) / max(len(error) - 2, 1)
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
) -> PairFilterResult | None:
    """Run the causal beta filter and return the current pair state."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    if n <= INITIAL_OBSERVATIONS or len(x) != n:
        return None

    (
        yn,
        xn,
        theta,
        covariance,
        observation_variance,
        variance_lambda,
        y_centre,
        x_centre,
        minimum_segment,
        process_covariance,
        regime_covariance,
        y_scale,
        x_scale,
    ) = _initial_filter_model(y, x)

    # Four simultaneously monitored streams: positive/negative innovation
    # mean and positive/negative innovation variance.  This Gaussian tail
    # bound makes the threshold depend on family size and the desired false
    # alarm rate instead of a fitted target segment length.
    base_cusum_threshold = float(
        np.sqrt(
            2.0
            * np.log(
                4.0 * max(family_size, 1) / FAMILY_FALSE_ALARM_RATE
            )
        )
    )
    allowance = TARGET_BETA_SCORE_SHIFT / 2.0

    beta_path = np.full(n, np.nan)
    alpha_path = np.full(n, np.nan)
    innovation_path = np.full(n, np.nan)
    innovation_scale_path = np.full(n, np.nan)
    z_path = np.full(n, np.nan)
    regime_path = np.zeros(n, dtype=bool)

    beta_up = beta_down = variance_up = variance_down = 0.0
    run_beta_up = run_beta_down = run_variance_up = run_variance_down = -1
    segment_start = INITIAL_OBSERVATIONS
    innovation_variance = observation_variance
    initial_dx = np.diff(xn[:INITIAL_OBSERVATIONS])
    initial_dy = np.diff(yn[:INITIAL_OBSERVATIONS])
    initial_beta_score = (
        initial_dy - theta[1] * initial_dx
    ) * initial_dx
    beta_score_variance = _robust_variance(initial_beta_score)
    identity = np.eye(2)
    cusum_threshold = base_cusum_threshold

    for t in range(INITIAL_OBSERVATIONS, n):
        # Sequential monitoring creates repeated opportunities for a false
        # alarm.  Account for both pair-family size and elapsed segment time.
        segment_tests = max(t - segment_start + 1, 1)
        cusum_threshold = float(
            np.sqrt(
                2.0
                * np.log(
                    4.0
                    * max(family_size, 1)
                    * segment_tests
                    / FAMILY_FALSE_ALARM_RATE
                )
            )
        )
        prior_covariance = covariance + process_covariance
        observation = np.array([1.0, xn[t]])
        prediction = float(observation @ theta)
        innovation = float(yn[t] - prediction)
        kalman_innovation_variance = max(
            float(observation @ prior_covariance @ observation)
            + observation_variance,
            EPSILON,
        )
        scale2 = max(kalman_innovation_variance, innovation_variance, EPSILON)
        z = innovation / np.sqrt(scale2)

        beta = float(theta[1] * y_scale / x_scale)
        alpha = float(y_centre + y_scale * theta[0] - beta * x_centre)
        beta_path[t] = beta
        alpha_path[t] = alpha
        innovation_path[t] = innovation * y_scale
        innovation_scale_path[t] = np.sqrt(scale2) * y_scale
        z_path[t] = z

        # A pair beta change is identified by the regression score: hedged
        # return errors become systematically aligned with x returns.  A plain
        # innovation-mean CUSUM would mistake the stationary spread excursion
        # we want to trade for a coefficient break.
        x_change = float(xn[t] - xn[t - 1])
        y_change = float(yn[t] - yn[t - 1])
        beta_score = (y_change - theta[1] * x_change) * x_change
        z_beta_score = beta_score / np.sqrt(max(beta_score_variance, EPSILON))

        previous = beta_up
        beta_up = max(0.0, beta_up + z_beta_score - allowance)
        if beta_up > 0.0 and previous == 0.0:
            run_beta_up = t
        elif beta_up == 0.0:
            run_beta_up = -1

        previous = beta_down
        beta_down = max(0.0, beta_down - z_beta_score - allowance)
        if beta_down > 0.0 and previous == 0.0:
            run_beta_down = t
        elif beta_down == 0.0:
            run_beta_down = -1

        variance_score = (z * z - 1.0) / np.sqrt(2.0)
        previous = variance_up
        variance_up = max(0.0, variance_up + variance_score - allowance)
        if variance_up > 0.0 and previous == 0.0:
            run_variance_up = t
        elif variance_up == 0.0:
            run_variance_up = -1

        previous = variance_down
        variance_down = max(0.0, variance_down - variance_score - allowance)
        if variance_down > 0.0 and previous == 0.0:
            run_variance_down = t
        elif variance_down == 0.0:
            run_variance_down = -1

        beta_statistics = np.array([beta_up, beta_down])
        beta_run_starts = np.array([run_beta_up, run_beta_down])
        winner = int(np.argmax(beta_statistics))
        alarm = bool(
            ENABLE_CUSUM and beta_statistics[winner] > cusum_threshold
        )
        accepted_change = False
        reinitialised = False
        if alarm:
            proposed_start = int(beta_run_starts[winner])
            if proposed_start < 0:
                proposed_start = t
            proposed_start = max(proposed_start, segment_start + 1)
            if proposed_start - segment_start >= minimum_segment:
                accepted_change = True
                segment_start = proposed_start
                severity = float(beta_statistics[winner] / cusum_threshold)
                run_x = xn[proposed_start : t + 1]
                run_y = yn[proposed_start : t + 1]
                regime_prior_variance = (
                    np.diag(covariance) + severity * np.diag(regime_covariance)
                )

                # The beta-score alarm is a change in return co-movement, so
                # estimate the new slope from that same identifiable quantity.
                # Differencing removes intercept ambiguity; the Bayesian prior
                # prevents a quiet run from producing an extreme beta.
                run_dx = np.diff(run_x)
                run_dy = np.diff(run_y)
                return_variance = 2.0 * observation_variance
                beta_precision = (
                    1.0 / max(regime_prior_variance[1], EPSILON)
                    + float(run_dx @ run_dx) / return_variance
                )
                beta_numerator = (
                    theta[1] / max(regime_prior_variance[1], EPSILON)
                    + float(run_dx @ run_dy) / return_variance
                )
                theta[1] = beta_numerator / beta_precision
                beta_variance = 1.0 / beta_precision

                # Once beta is identified from changes, update the regime
                # intercept from the mean level error with the same Bayesian
                # uncertainty accounting.
                alpha_observation = float(np.mean(run_y - theta[1] * run_x))
                alpha_observation_variance = observation_variance / len(run_x)
                alpha_precision = (
                    1.0 / max(regime_prior_variance[0], EPSILON)
                    + 1.0 / alpha_observation_variance
                )
                theta[0] = (
                    theta[0] / max(regime_prior_variance[0], EPSILON)
                    + alpha_observation / alpha_observation_variance
                ) / alpha_precision
                alpha_variance = 1.0 / alpha_precision
                # The run has already been assimilated in one batch.  Do not
                # leave a large posterior covariance that would make beta keep
                # chasing ordinary residuals for dozens of subsequent days.
                coefficient_variance_cap = np.diag(process_covariance) * max(t, 1)
                covariance = np.diag(
                    np.minimum(
                        [alpha_variance, beta_variance],
                        coefficient_variance_cap,
                    )
                )
                reinitialised = True
                regime_path[t] = True

            beta_up = beta_down = variance_up = variance_down = 0.0
            run_beta_up = run_beta_down = -1
            run_variance_up = run_variance_down = -1

        # As in the supplied Pine script's default, volatility CUSUMs remain
        # diagnostics.  A variance change alone is not evidence that beta has
        # changed, so it cannot reset the coefficient state.
        if max(variance_up, variance_down) > cusum_threshold:
            variance_up = variance_down = 0.0
            run_variance_up = run_variance_down = -1

        if not reinitialised:
            gain = prior_covariance @ observation / kalman_innovation_variance
            theta = theta + gain * innovation
            gain_observation = np.outer(gain, observation)
            covariance = (
                (identity - gain_observation)
                @ prior_covariance
                @ (identity - gain_observation).T
                + np.outer(gain, gain) * observation_variance
            )
            covariance = 0.5 * (covariance + covariance.T)

        # Robust covariance matching.  Regime-sized outliers are winsorized so
        # they trigger coefficient adaptation instead of permanently inflating
        # measurement noise and suppressing subsequent detection.
        capped_square = min(
            innovation * innovation,
            cusum_threshold * cusum_threshold * innovation_variance,
        )
        innovation_variance = (
            variance_lambda * innovation_variance
            + (1.0 - variance_lambda) * capped_square
        )
        capped_beta_score_square = min(
            beta_score * beta_score,
            cusum_threshold * cusum_threshold * beta_score_variance,
        )
        if not accepted_change:
            beta_score_variance = (
                variance_lambda * beta_score_variance
                + (1.0 - variance_lambda) * capped_beta_score_square
            )

    valid_start = max(segment_start, INITIAL_OBSERVATIONS)
    current_beta = float(beta_path[-1])
    current_alpha = float(alpha_path[-1])
    (
        spread_deviation,
        spread_scale,
        spread_z,
        frozen_kappa,
        convergence_t,
        _fixed_beta_adf_t,
    ) = _fixed_beta_spread_health(
        y,
        x,
        current_alpha,
        current_beta,
        valid_start,
    )
    # Stationarity belongs to the complete causal dynamic spread path.  The
    # separate frozen-beta convergence statistic above prevents a moving state
    # from being credited for next-day reversion.
    dynamic_spread = innovation_path[INITIAL_OBSERVATIONS:]
    dynamic_spread = dynamic_spread[np.isfinite(dynamic_spread)]
    adf_t = _adf1_tstat(dynamic_spread)
    expected_edge = max(0.0, frozen_kappa) * abs(spread_deviation)
    return PairFilterResult(
        alpha=current_alpha,
        beta=current_beta,
        innovation=spread_deviation,
        innovation_scale=spread_scale,
        z=spread_z,
        frozen_kappa=float(frozen_kappa),
        convergence_t=float(convergence_t),
        adf_t=float(adf_t),
        expected_edge=float(expected_edge),
        regime_change=bool(regime_path[-1]),
        segment_age=int(n - 1 - valid_start),
        cusum_threshold=cusum_threshold,
        minimum_segment=minimum_segment,
        q_beta_over_r=float(
            process_covariance[1, 1] / max(observation_variance, EPSILON)
        ),
        beta_path=beta_path,
        innovation_path=innovation_path,
        z_path=z_path,
        regime_path=regime_path,
    )


def _pair_candidates(log_prices: np.ndarray) -> list[tuple[float, int, int, PairFilterResult]]:
    """Estimate and validate every fixed relationship independently."""
    candidates = []
    for i, j in PAIRS:
        result = filter_pair(log_prices[i], log_prices[j])
        if result is None:
            continue
        stationary = result.adf_t < ADF_FIVE_PERCENT_CRITICAL
        converging = result.frozen_kappa > 0.0 and result.convergence_t < 0.0
        economical = result.expected_edge > ROUND_TRIP_LOG_COST
        intact = (
            np.isfinite(result.z)
            and abs(result.z) <= result.cusum_threshold
            and result.beta > 0.0
            and not result.regime_change
        )
        if stationary and converging and economical and intact:
            net_edge = result.expected_edge - ROUND_TRIP_LOG_COST
            candidates.append((net_edge, i, j, result))
    return candidates


def _allocate_pairs(
    candidates: list[tuple[float, int, int, PairFilterResult]],
) -> np.ndarray:
    """Allocate disjoint beta-neutral pairs, strongest expected edge first."""
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
    """Return integer target shares for the isolated adaptive-beta pair book."""
    n_instruments, n_days = prcSoFar.shape
    if n_instruments != N_INST or n_days <= INITIAL_OBSERVATIONS:
        return np.zeros(n_instruments, dtype=int)
    if not np.all(np.isfinite(prcSoFar[:, -1])) or np.any(prcSoFar[:, -1] <= 0.0):
        return np.zeros(n_instruments, dtype=int)

    log_prices = np.log(prcSoFar)
    candidates = _pair_candidates(log_prices)
    dollars = _allocate_pairs(candidates)
    dollars[0] = 0.0
    return (dollars / prcSoFar[:, -1]).astype(int)

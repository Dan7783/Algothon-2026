"""Algothon 2026: persistent atlas + ridge + validated residual pairs.

The generic pair sleeve uses a causal formation/validation split.  A candidate
must earn positive forward validation P&L, retain a stable positive hedge ratio,
remain stationary, and have a plausible half-life before it can trade.  Only the
two strongest disjoint pairs may use capacity left idle by atlas and ridge.
"""

from __future__ import annotations

import numpy as np


N_INST = 51
LEG = 10_000.0

ATLAS_I = np.array([10, 1, 13, 8, 25, 49, 7, 31, 36, 18, 33, 35, 26], dtype=int)
ATLAS_J = np.array([46, 20, 45, 27, 37, 50, 40, 43, 41, 35, 26, 28, 32], dtype=int)
ATLAS_PAIRS = tuple(zip(ATLAS_I.tolist(), ATLAS_J.tolist()))
ATLAS_KEYS = {tuple(sorted(pair)) for pair in ATLAS_PAIRS}

RIDGE_ENTRY = 0.75

# Walk-forward generic-pair policy selected on chronological holdouts.
FORMATION = 250
VALIDATION = 80
VALIDATION_Z = 1.0
MIN_VALIDATION_TRADES = 6
MIN_VALIDATION_T = 0.25
BETA_RATIO_MAX = 1.7
HALF_LIFE_MIN = 1.5
HALF_LIFE_MAX = 35.0
PAIR_ENTRY_Z = 1.75
PAIR_BREAK_Z = 3.75
PAIR_ADF_T = -2.2
MAX_GENERIC_PAIRS = 2

_ti, _tj = np.triu_indices(N_INST - 1, k=1)
_pi, _pj = _ti + 1, _tj + 1
_generic = np.fromiter(
    (tuple(sorted((int(i), int(j)))) not in ATLAS_KEYS for i, j in zip(_pi, _pj)),
    dtype=bool,
    count=len(_pi),
)
PAIR_I, PAIR_J = _pi[_generic], _pj[_generic]


def _ols(y: np.ndarray, x: np.ndarray):
    yc = y - y.mean(axis=1, keepdims=True)
    xc = x - x.mean(axis=1, keepdims=True)
    beta = (yc * xc).sum(axis=1) / np.maximum((xc * xc).sum(axis=1), 1e-18)
    alpha = y.mean(axis=1) - beta * x.mean(axis=1)
    residual = y - alpha[:, None] - beta[:, None] * x
    return alpha, beta, residual


def _kalman_beta(y: np.ndarray, x: np.ndarray) -> float:
    theta = np.array([y[:60].mean() - x[:60].mean(), 1.0])
    covariance = np.eye(2) * 1e-2
    process_noise = np.eye(2) * 1e-5
    measurement_noise = np.var(np.diff(y[:120] - x[:120])) + 1e-8
    for t in range(len(y)):
        covariance += process_noise
        observation = np.array([1.0, x[t]])
        innovation_variance = observation @ covariance @ observation + measurement_noise
        gain = covariance @ observation / innovation_variance
        theta += gain * (y[t] - observation @ theta)
        covariance -= np.outer(gain, observation) @ covariance
    return float(theta[1])


def _adf1_tstats(series: np.ndarray) -> np.ndarray:
    """Vectorised ADF(1) level-coefficient t-statistic for series columns."""
    delta = np.diff(series, axis=0)
    target = delta[1:]
    level = series[1:-1]
    lag_delta = delta[:-1]
    n, m = target.shape

    s1, s2 = level.sum(0), lag_delta.sum(0)
    s11 = (level * level).sum(0)
    s12 = (level * lag_delta).sum(0)
    s22 = (lag_delta * lag_delta).sum(0)
    sy = target.sum(0)
    s1y = (level * target).sum(0)
    s2y = (lag_delta * target).sum(0)
    syy = (target * target).sum(0)

    normal = np.empty((m, 3, 3))
    rhs = np.empty((m, 3))
    normal[:, 0, 0] = n
    normal[:, 0, 1] = normal[:, 1, 0] = s1
    normal[:, 0, 2] = normal[:, 2, 0] = s2
    normal[:, 1, 1] = s11
    normal[:, 1, 2] = normal[:, 2, 1] = s12
    normal[:, 2, 2] = s22
    rhs[:, 0], rhs[:, 1], rhs[:, 2] = sy, s1y, s2y

    inverse = np.linalg.inv(normal)
    coefficients = np.einsum("mij,mj->mi", inverse, rhs)
    variance = (syy - np.einsum("mi,mi->m", coefficients, rhs)) / max(n - 3, 1)
    standard_error = np.sqrt(np.maximum(variance * inverse[:, 1, 1], 1e-300))
    return coefficients[:, 1] / standard_error


def _ou_kappa(spread: np.ndarray) -> float:
    delta = np.diff(spread)
    lag = spread[:-1] - spread[:-1].mean()
    denominator = (lag * lag).sum()
    return float(-(lag @ delta) / denominator) if denominator > 1e-18 else 0.0


def _atlas_dollars(log_prices: np.ndarray) -> np.ndarray:
    """Long-lived relationships retain first claim on their two legs."""
    pos = np.zeros(log_prices.shape[0])
    nt = log_prices.shape[1]
    if nt < 130:
        return pos
    for i, j in ATLAS_PAIRS:
        beta = _kalman_beta(log_prices[i, : nt - 1], log_prices[j, : nt - 1])
        spread = log_prices[i] - beta * log_prices[j]
        if _adf1_tstats(spread.reshape(-1, 1))[0] >= -2.9:
            continue
        recent = spread[-145:]
        z = (recent[-1] - recent[:-1].mean()) / (recent[:-1].std() + 1e-12)
        if _ou_kappa(spread[-120:]) <= 0.0 or not (0.27 <= abs(z) <= 4.0):
            continue
        direction = -np.sign(z)
        pos[i] += direction * LEG
        pos[j] -= direction * LEG
    return pos


def _ridge_dollars(returns: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Regularised next-day cross-asset forecast, subordinate to atlas."""
    pos = np.zeros(returns.shape[0])
    if returns.shape[1] + 1 < 252:
        return pos
    recent = returns[1:, -250:].T
    standard = (recent - recent.mean(0)) / (recent.std(0) + 1e-12)
    lag_corr = (standard[:-1].T @ standard[1:]) / (len(recent) - 1)
    np.fill_diagonal(lag_corr, 0.0)
    breadth = int((np.abs(lag_corr) > 2.5 / np.sqrt(len(recent) - 1)).sum())
    if breadth < 35:
        return pos

    history = returns[1:].T
    scale = history.std(0) + 1e-12
    predictors, targets = history[:-1] / scale, history[1:] / scale
    coefficients = np.linalg.solve(
        predictors.T @ predictors + len(predictors) * np.eye(50),
        predictors.T @ targets,
    )
    prediction_scale = (predictors @ coefficients).std(0)
    prediction = (history[-1] / scale) @ coefficients
    raw = LEG * np.sign(prediction) * (
        np.abs(prediction) > RIDGE_ENTRY * prediction_scale
    )

    existing = held[1:]
    allocation = np.zeros_like(raw)
    idle = np.abs(existing) <= 1.0
    allocation[idle] = raw[idle]
    agrees = ~idle & (np.abs(raw) > 1.0) & (np.sign(raw) == np.sign(existing))
    room = np.maximum(0.0, LEG - np.minimum(np.abs(existing), LEG))
    allocation[agrees] = np.sign(raw[agrees]) * np.minimum(
        np.abs(raw[agrees]), room[agrees]
    )
    pos[1:] = allocation
    return pos


def _capacity(existing: float, requested: float) -> float:
    if requested > 0.0:
        return max(0.0, (LEG - existing) / requested)
    return max(0.0, (-LEG - existing) / requested)


def _pair_statistics(log_prices: np.ndarray):
    """Causally validate all non-atlas pairs before calculating today's signal."""
    stop = log_prices.shape[1] - 1
    start = max(0, stop - FORMATION)
    sample = log_prices[:, start:stop]
    if sample.shape[1] < VALIDATION + 100:
        return None
    split = sample.shape[1] - VALIDATION

    old_y, old_x = sample[PAIR_I, :split], sample[PAIR_J, :split]
    alpha_old, beta_old, residual_old = _ols(old_y, old_x)
    val_y, val_x = sample[PAIR_I, split:], sample[PAIR_J, split:]
    residual_val = val_y - alpha_old[:, None] - beta_old[:, None] * val_x
    z_val = (
        residual_val - residual_old.mean(1)[:, None]
    ) / (residual_old.std(1)[:, None] + 1e-12)

    direction = -np.sign(z_val[:, :-1])
    active = (
        (np.abs(z_val[:, :-1]) >= VALIDATION_Z)
        & (np.abs(z_val[:, :-1]) < PAIR_BREAK_Z)
    )
    validation_leg = LEG / np.maximum(1.0, np.abs(beta_old))
    target_i = direction * active * validation_leg[:, None]
    target_j = -direction * active * beta_old[:, None] * validation_leg[:, None]
    ret_i = np.exp(np.diff(val_y, axis=1)) - 1.0
    ret_j = np.exp(np.diff(val_x, axis=1)) - 1.0
    gross = target_i * ret_i + target_j * ret_j
    previous_i = np.column_stack([np.zeros(len(PAIR_I)), target_i[:, :-1]])
    previous_j = np.column_stack([np.zeros(len(PAIR_I)), target_j[:, :-1]])
    fees = 0.0001 * (
        np.abs(target_i - previous_i) + np.abs(target_j - previous_j)
    )
    pnl = gross - fees
    mean = pnl.mean(1)
    std = pnl.std(1) + 1e-12
    tstat = mean / std * np.sqrt(pnl.shape[1])
    half = pnl.shape[1] // 2
    first_pnl = pnl[:, :half].sum(1)
    second_pnl = pnl[:, half:].sum(1)
    recent_pnl = pnl[:, -min(30, pnl.shape[1]) :].sum(1)

    y, x = sample[PAIR_I], sample[PAIR_J]
    alpha, beta, residual = _ols(y, x)
    current = log_prices[PAIR_I, -1] - alpha - beta * log_prices[PAIR_J, -1]
    z = current / (residual.std(1) + 1e-12)
    adf = _adf1_tstats(residual.T)

    lag = residual[:, :-1] - residual[:, :-1].mean(1, keepdims=True)
    lead = residual[:, 1:] - residual[:, 1:].mean(1, keepdims=True)
    phi = (lag * lead).sum(1) / np.maximum((lag * lag).sum(1), 1e-18)
    half_life = np.full(len(PAIR_I), np.inf)
    ar_ok = (phi > 0.0) & (phi < 1.0)
    half_life[ar_ok] = -np.log(2.0) / np.log(phi[ar_ok])

    _, beta_recent, _ = _ols(y[:, split:], x[:, split:])
    beta_high = np.maximum(np.abs(beta_old), np.abs(beta_recent))
    beta_low = np.maximum(np.minimum(np.abs(beta_old), np.abs(beta_recent)), 1e-6)
    return {
        "beta": beta,
        "beta_old": beta_old,
        "beta_recent": beta_recent,
        "z": z,
        "adf": adf,
        "half_life": half_life,
        "beta_ratio": beta_high / beta_low,
        "trades": active.sum(1),
        "tstat": tstat,
        "first_pnl": first_pnl,
        "second_pnl": second_pnl,
        "recent_pnl": recent_pnl,
    }


def _validated_pair_dollars(
    log_prices: np.ndarray, base_dollars: np.ndarray
) -> np.ndarray:
    pos = np.zeros(N_INST)
    stats = _pair_statistics(log_prices)
    if stats is None:
        return pos

    first_floor = -0.20 * np.maximum(np.abs(stats["second_pnl"]), LEG * 0.01)
    idle = np.abs(base_dollars) <= 1.0
    eligible = (
        (stats["beta"] > 0.0)
        & (stats["beta_old"] > 0.0)
        & (stats["beta_recent"] > 0.0)
        & np.isfinite(stats["z"])
        & (stats["adf"] < PAIR_ADF_T)
        & (stats["half_life"] >= HALF_LIFE_MIN)
        & (stats["half_life"] <= HALF_LIFE_MAX)
        & (stats["beta_ratio"] <= BETA_RATIO_MAX)
        & (stats["trades"] >= MIN_VALIDATION_TRADES)
        & (stats["tstat"] >= MIN_VALIDATION_T)
        & (stats["first_pnl"] >= first_floor)
        & (stats["second_pnl"] > 0.0)
        & (stats["recent_pnl"] >= 0.0)
        & (np.abs(stats["z"]) >= PAIR_ENTRY_Z)
        & (np.abs(stats["z"]) < PAIR_BREAK_Z)
        & idle[PAIR_I]
        & idle[PAIR_J]
    )

    convergence = np.abs(stats["z"]) * (
        1.0 - np.exp(-np.log(2.0) / stats["half_life"])
    )
    rank = convergence * (np.clip(stats["tstat"], 0.0, 3.0) + 0.25)
    used = np.zeros(N_INST, dtype=bool)
    accepted = 0
    for k in np.argsort(-rank):
        if not eligible[k]:
            continue
        i, j = int(PAIR_I[k]), int(PAIR_J[k])
        if used[i] or used[j]:
            continue
        direction = -np.sign(stats["z"][k])
        beta = float(stats["beta"][k])
        leg = LEG / max(1.0, abs(beta))
        raw_i, raw_j = direction * leg, -direction * beta * leg
        scale = min(
            _capacity(float(base_dollars[i] + pos[i]), raw_i),
            _capacity(float(base_dollars[j] + pos[j]), raw_j),
        )
        if scale <= 1e-12:
            continue
        pos[i] += scale * raw_i
        pos[j] += scale * raw_j
        used[i] = used[j] = True
        accepted += 1
        if accepted >= MAX_GENERIC_PAIRS:
            break
    return pos


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """Return 51 integer target positions for the official evaluator."""
    prices = np.asarray(prcSoFar, dtype=float)
    if prices.ndim != 2:
        return np.zeros(N_INST, dtype=int)
    nins, nt = prices.shape
    if nins != N_INST or nt < 180:
        return np.zeros(nins, dtype=int)
    if not np.all(np.isfinite(prices[:, -3:])) or np.any(prices[:, -3:] <= 0.0):
        return np.zeros(nins, dtype=int)

    log_prices = np.log(prices)
    returns = np.diff(log_prices, axis=1)
    atlas = _atlas_dollars(log_prices)
    ridge = _ridge_dollars(returns, atlas)
    dollars = atlas + ridge
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars += _validated_pair_dollars(log_prices, dollars)

    crash = np.abs(returns[:, -1]) > 0.40
    crash[0] = False
    dollars[crash] = 0.0
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars[0] = 0.0
    return (dollars / prices[:, -1]).astype(int)

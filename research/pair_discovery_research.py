"""Configurable, causal pair-discovery ablation around ``teamName``.

This is a research harness, not the submission module.  It keeps the atlas and
ridge implementations unchanged while varying the generic pair estimator,
relationship gate, ranking rule, and allocation priority.
"""

from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import numpy as np

import four_sigmas as base


N_INST = base.N_INST
LEG = base.LEG

# Values are deliberately overrideable by benchmark_teamname.py.
DISCOVERY_LOOKBACK = 250       # 0 means all available history
DISCOVERY_HEALTH = "full"     # none, adf, full
DISCOVERY_RANK = "z"          # z, adf, speed
DISCOVERY_FIRST = True
DISCOVERY_ENTRY_Z = 1.5
DISCOVERY_MAX_Z = 4.0
DISCOVERY_SCALE = 1.0
DISCOVERY_ONLY = False


def _ols(y: np.ndarray, x: np.ndarray):
    yc = y - y.mean(axis=1, keepdims=True)
    xc = x - x.mean(axis=1, keepdims=True)
    beta = (yc * xc).sum(1) / np.maximum((xc * xc).sum(1), 1e-18)
    alpha = y.mean(1) - beta * x.mean(1)
    residual = y - alpha[:, None] - beta[:, None] * x
    return alpha, beta, residual


def _half_life(residual: np.ndarray) -> np.ndarray:
    lag = residual[:, :-1]
    lead = residual[:, 1:]
    lag = lag - lag.mean(1, keepdims=True)
    lead = lead - lead.mean(1, keepdims=True)
    phi = (lag * lead).sum(1) / np.maximum((lag * lag).sum(1), 1e-18)
    valid = (phi > 0.0) & (phi < 1.0)
    result = np.full(len(phi), np.inf)
    result[valid] = -np.log(2.0) / np.log(phi[valid])
    return result


def _beta_ratio(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    middle = y.shape[1] // 2
    _, first, _ = _ols(y[:, :middle], x[:, :middle])
    _, second, _ = _ols(y[:, middle:], x[:, middle:])
    high = np.maximum(np.abs(first), np.abs(second))
    low = np.maximum(np.minimum(np.abs(first), np.abs(second)), 1e-6)
    return high / low


def _discovery_dollars(log_prices: np.ndarray, base_dollars: np.ndarray) -> np.ndarray:
    pos = np.zeros(log_prices.shape[0])
    if DISCOVERY_SCALE <= 0.0 or log_prices.shape[1] < 121:
        return pos

    stop = log_prices.shape[1] - 1
    start = 0 if DISCOVERY_LOOKBACK <= 0 else max(0, stop - DISCOVERY_LOOKBACK)
    train = log_prices[:, start:stop]
    if train.shape[1] < 120:
        return pos

    pi, pj = base.PAIR_I, base.PAIR_J
    y, x = train[pi], train[pj]
    alpha, beta, residual = _ols(y, x)
    current = log_prices[pi, -1] - alpha - beta * log_prices[pj, -1]
    z = current / (residual.std(1) + 1e-12)

    healthy = (beta > 0.0) & np.isfinite(z)
    pair_t = np.zeros(len(pi))
    half_life = np.ones(len(pi))
    if DISCOVERY_HEALTH in {"adf", "full"}:
        pair_t = base._adf1_tstats(residual.T)
        healthy &= pair_t < -2.0
    if DISCOVERY_HEALTH == "full":
        _, reverse_beta, reverse_residual = _ols(x, y)
        reverse_t = base._adf1_tstats(reverse_residual.T)
        pair_t = np.maximum(pair_t, reverse_t)
        half_life = _half_life(residual)
        healthy &= (
            (pair_t < -2.0)
            & (_beta_ratio(y, x) <= 2.0)
            & (half_life >= 1.5)
            & (half_life <= 40.0)
            & (reverse_beta > 0.0)
        )
    elif DISCOVERY_RANK == "speed":
        half_life = _half_life(residual)

    idle = np.abs(base_dollars) <= 1.0
    eligible = (
        healthy
        & (np.abs(z) >= DISCOVERY_ENTRY_Z)
        & (np.abs(z) < DISCOVERY_MAX_Z)
        & idle[pi]
        & idle[pj]
    )
    if DISCOVERY_RANK == "adf":
        rank = -pair_t
    elif DISCOVERY_RANK == "speed":
        # Approximate next-day OU convergence in spread-standard-deviation units.
        rank = np.abs(z) * (1.0 - np.exp(-np.log(2.0) / half_life))
    else:
        rank = np.abs(z)

    used = np.zeros(log_prices.shape[0], dtype=bool)
    for k in np.argsort(-rank):
        if not eligible[k]:
            continue
        i, j = int(pi[k]), int(pj[k])
        if used[i] or used[j]:
            continue
        direction = -np.sign(z[k])
        leg = LEG / max(1.0, abs(float(beta[k])))
        raw_i = direction * leg
        raw_j = -direction * float(beta[k]) * leg
        scale = min(
            DISCOVERY_SCALE,
            base._capacity(float(base_dollars[i] + pos[i]), raw_i),
            base._capacity(float(base_dollars[j] + pos[j]), raw_j),
        )
        if scale <= 1e-12:
            continue
        pos[i] += scale * raw_i
        pos[j] += scale * raw_j
        used[i] = used[j] = True
    return pos


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    nins, nt = prcSoFar.shape
    if nins != N_INST or nt < 130:
        return np.zeros(nins, dtype=int)
    recent = prcSoFar[:, -3:]
    if not np.all(np.isfinite(recent)) or np.any(recent <= 0.0):
        return np.zeros(nins, dtype=int)

    log_prices = np.log(prcSoFar)
    returns = np.diff(log_prices, axis=1)
    if DISCOVERY_ONLY:
        dollars = _discovery_dollars(log_prices, np.zeros(nins))
        crash = np.abs(returns[:, -1]) > 0.40
        crash[0] = False
        dollars[crash] = 0.0
        dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
        dollars[0] = 0.0
        return (dollars / prcSoFar[:, -1]).astype(int)

    atlas = base._atlas_dollars(log_prices)

    if DISCOVERY_FIRST:
        discovery = _discovery_dollars(log_prices, atlas)
        held = atlas + discovery
        ridge = base._ridge_dollars(returns, held)
        dollars = held + ridge
    else:
        ridge = base._ridge_dollars(returns, atlas)
        held = atlas + ridge
        held[1:] = np.clip(held[1:], -LEG, LEG)
        dollars = held + _discovery_dollars(log_prices, held)

    crash = np.abs(returns[:, -1]) > 0.40
    crash[0] = False
    dollars[crash] = 0.0
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars[0] = 0.0
    return (dollars / prcSoFar[:, -1]).astype(int)

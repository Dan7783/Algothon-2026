"""Test strategy: ALGO sleeve from combined rolling asset returns.

Signal (causal):
  1. For every non-ALGO asset, compute a LOOKBACK-day rolling log-return,
     lagged by one day (known at t-1).
  2. On history available at the decision time, estimate each asset's IC
     versus ALGO's next HORIZON-day forward return, using only fully
     completed label windows (ended at least one day before now).
  3. Form either:
       - top_k_absic: equal-weight the TOP_K names by |IC|, trade ALGO with
         -sign(basket) when median IC is negative (contrarian default), or
       - ic_weighted: score = sum_j IC_j * feature_j, trade +sign(score).
  4. Hold only ALGO, sized to DOLLAR_CAP (bang-bang by sign).

This module is intentionally standalone for CPCV / eval.py testing. It is not
wired into the live pair book.
"""

from __future__ import annotations

import numpy as np

N_INST = 51
ALGO = 0
LOOKBACK = 20
HORIZON = 30
TOP_K = 10
MIN_ABS_IC = 0.05
MIN_IC_SAMPLES = 80
MIN_HISTORY_DAYS = 120  # price columns before first trade
DOLLAR_CAP = 50_000.0
MODE = "ic_weighted"  # "ic_weighted" | "top_k_absic"
RETRAIN_EVERY = 5

COMMISSION = np.full(N_INST, 1.0e-4)
COMMISSION[ALGO] = 2.0e-5
LIMIT = np.full(N_INST, 10_000.0)
LIMIT[ALGO] = 100_000.0

_STATE: dict = {}


def reset() -> None:
    _STATE.clear()


def _log_returns(prices: np.ndarray) -> np.ndarray:
    """prices: (n_inst, n_days) -> (n_inst, n_days-1) log returns."""
    return np.diff(np.log(prices), axis=1)


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling sum ending at each index; NaN until window-1."""
    n = len(x)
    out = np.full(n, np.nan)
    if window <= 0 or n < window:
        return out
    c = np.concatenate([[0.0], np.cumsum(x)])
    out[window - 1 :] = c[window:] - c[: n - window + 1]
    return out


def _feature_matrix(returns: np.ndarray, lookback: int) -> np.ndarray:
    """Lagged rolling lookback returns for assets 1..N-1.

    features[t, j] uses asset j+1 returns [t-lookback, t) i.e. ending at t-1
    when indexed against a target that starts at return index t.
    Shape: (n_rets, n_other).
    """
    n_rets = returns.shape[1]
    n_other = returns.shape[0] - 1
    feats = np.full((n_rets, n_other), np.nan)
    for j in range(n_other):
        roll = _rolling_sum(returns[j + 1], lookback)
        # lag by 1: feature at t = roll ending at t-1
        feats[1:, j] = roll[:-1]
    return feats


def _forward_algo(returns: np.ndarray, horizon: int) -> np.ndarray:
    """fwd[t] = sum(ALGO returns[t : t+horizon])."""
    algo = returns[ALGO]
    n = len(algo)
    fwd = np.full(n, np.nan)
    if horizon <= 0:
        return fwd
    c = np.concatenate([[0.0], np.cumsum(algo)])
    # sum algo[t:t+h] = c[t+h] - c[t]
    last = n - horizon
    if last <= 0:
        return fwd
    idx = np.arange(last)
    fwd[idx] = c[idx + horizon] - c[idx]
    return fwd


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    da = float(a @ a)
    db = float(b @ b)
    if da < 1e-18 or db < 1e-18:
        return 0.0
    return float((a @ b) / np.sqrt(da * db))


def estimate_ics(
    features: np.ndarray,
    forward: np.ndarray,
    train_idx: np.ndarray,
) -> np.ndarray:
    """IC of each feature column vs forward ALGO return on train_idx."""
    n_feat = features.shape[1]
    ics = np.zeros(n_feat)
    for j in range(n_feat):
        x = features[train_idx, j]
        y = forward[train_idx]
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < MIN_IC_SAMPLES:
            ics[j] = 0.0
            continue
        ics[j] = _corr(x[m], y[m])
    return ics


def build_score(
    features_row: np.ndarray,
    ics: np.ndarray,
    mode: str = MODE,
) -> float:
    """Scalar signal from one day's feature row and fitted ICs."""
    if mode == "ic_weighted":
        mask = np.isfinite(features_row) & (np.abs(ics) >= MIN_ABS_IC)
        if int(mask.sum()) < 3:
            mask = np.isfinite(features_row) & (np.abs(ics) > 0.0)
        if int(mask.sum()) < 1:
            return 0.0
        w = ics.copy()
        w[~mask] = 0.0
        denom = float(np.abs(w).sum())
        if denom < 1e-15:
            return 0.0
        w /= denom
        x = np.nan_to_num(features_row, nan=0.0)
        return float(w @ x)

    # top_k_absic: equal-weight top-k by |IC|, score = mean feature
    order = np.argsort(-np.abs(ics))
    pick = []
    for j in order:
        if abs(ics[j]) <= 0.0:
            break
        if not np.isfinite(features_row[j]):
            continue
        pick.append(int(j))
        if len(pick) >= TOP_K:
            break
    if not pick:
        return 0.0
    basket = float(np.mean(features_row[pick]))
    # Most ICs are negative (contrarian). Flip so positive score => long ALGO.
    med = float(np.median(ics[pick]))
    if med < 0:
        return -basket
    return basket


def fit_ics_from_prices(
    prices: np.ndarray,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ICs using all completed labels in prices.

    Returns (ics, features, forward) for the return index space.
    """
    if prices.shape[1] < lookback + horizon + 5:
        return np.zeros(prices.shape[0] - 1), np.empty((0, 0)), np.empty(0)
    returns = _log_returns(prices)
    features = _feature_matrix(returns, lookback)
    forward = _forward_algo(returns, horizon)
    # Only indices with complete forward label and finite features.
    n = returns.shape[1]
    # last usable label start: n - horizon
    max_i = n - horizon
    if max_i <= 0:
        return np.zeros(features.shape[1]), features, forward
    train_idx = np.arange(0, max_i)
    ok = np.isfinite(forward[train_idx]) & np.all(
        np.isfinite(features[train_idx]), axis=1
    )
    # soften: allow some NaN features early
    ok = np.isfinite(forward[train_idx])
    train_idx = train_idx[ok]
    ics = estimate_ics(features, forward, train_idx)
    return ics, features, forward


def signal_from_prices(prices: np.ndarray, ics: np.ndarray | None = None) -> float:
    """ALGO trade direction in {-1,0,+1} from price history."""
    if prices.shape[1] < MIN_HISTORY_DAYS:
        return 0.0
    returns = _log_returns(prices)
    features = _feature_matrix(returns, LOOKBACK)
    if features.shape[0] == 0:
        return 0.0
    # Current decision uses the latest return index = n_rets - 1, but the
    # feature at that index is lagged (known). Prefer index n_rets - 1.
    t = features.shape[0] - 1
    row = features[t]
    if ics is None:
        ics, _, _ = fit_ics_from_prices(prices)
    score = build_score(row, ics, MODE)
    if not np.isfinite(score) or abs(score) < 1e-15:
        return 0.0
    return float(np.sign(score))


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """Return share positions; only ALGO is traded."""
    prices = np.asarray(prcSoFar, dtype=float)
    n_days = prices.shape[1]
    pos = np.zeros(N_INST, dtype=int)

    if n_days < MIN_HISTORY_DAYS:
        return pos

    # Cache IC refits every RETRAIN_EVERY new columns.
    cache = _STATE.setdefault("cache", {})
    last_fit = int(cache.get("last_fit_days", -10**9))
    ics = cache.get("ics")
    if ics is None or n_days - last_fit >= RETRAIN_EVERY:
        ics, _, _ = fit_ics_from_prices(prices)
        cache["ics"] = ics
        cache["last_fit_days"] = n_days

    direction = signal_from_prices(prices, ics=ics)
    if direction == 0.0:
        return pos

    px = float(prices[ALGO, -1])
    if not np.isfinite(px) or px <= 0:
        return pos
    dollars = direction * min(DOLLAR_CAP, LIMIT[ALGO])
    pos[ALGO] = int(np.round(dollars / px))
    return pos


def describe() -> dict:
    return {
        "name": "algo_combo_ic_sleeve",
        "mode": MODE,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "top_k": TOP_K,
        "min_abs_ic": MIN_ABS_IC,
        "dollar_cap": DOLLAR_CAP,
        "retrain_every": RETRAIN_EVERY,
        "min_history_days": MIN_HISTORY_DAYS,
    }

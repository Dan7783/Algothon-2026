#!/usr/bin/env python
"""Causal per-day spread-score research across every asset pair.

The analysis deliberately separates:

1. relationship estimation (OLS hedge ratio and residual stationarity),
2. the current trading signal (a past-only spread z-score), and
3. the future outcome (fixed-beta spread reversion after the signal).

All 51 choose 2 = 1,275 unordered combinations are evaluated.  Scores on day
``t`` use observations strictly before day ``t``; forward outcomes are used
only for evaluation.  Nothing in this file changes ``teamName.py``.
"""

from __future__ import annotations

# --- repo path bootstrap: run as `python research/<script>.py` from root ---
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "research"), str(_REPO_ROOT / "archive")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import coint


PRICE_PATH = Path("prices.txt")
OUTPUT_DIR = Path("cointegration_output/all_pair_spread_scores")

MIN_HISTORY = 50
LOCAL_CENTER = 30
EVAL_START = 250              # compare every estimator over the same days
HORIZONS = (1, 3, 5, 10, 20)
Z_ENTRY = 2.0
Z_MAX = 4.0
LIGHT_ADF_T = -2.0            # deliberately a health gate, not a 5% test
HALF_LIFE_MIN = 1.5
HALF_LIFE_MAX = 40.0
BETA_RATIO_MAX = 2.0
RAW_P_ALPHA = 0.05
STRICT_P_ALPHA = 0.001
FDR_Q = 0.05
CONFIRMATIONS = (1, 2, 3, 5, 10)


@dataclass(frozen=True)
class FitSpec:
    name: str
    window: int | None


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    fit_name: str
    center: int | None


FIT_SPECS = (
    FitSpec("w50", 50),
    FitSpec("w100", 100),
    FitSpec("w250", 250),
    FitSpec("expanding", None),
)

SCORE_SPECS = (
    ScoreSpec("50d beta / 30d centre", "w50", LOCAL_CENTER),
    ScoreSpec("100d beta / 30d centre", "w100", LOCAL_CENTER),
    ScoreSpec("250d beta / 30d centre", "w250", LOCAL_CENTER),
    ScoreSpec("expanding beta / 30d centre", "expanding", LOCAL_CENTER),
    ScoreSpec("250d beta / 250d centre", "w250", None),
    ScoreSpec("expanding beta / full centre", "expanding", None),
)


def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+")
    if not np.all(np.isfinite(prices.to_numpy(float))):
        raise ValueError("prices.txt contains non-finite values")
    if np.any(prices.to_numpy(float) <= 0.0):
        raise ValueError("Log-spread analysis requires strictly positive prices")
    return prices


def ols_beta(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorised slope for columns of y ~ constant + beta*x."""
    yc = y - y.mean(axis=0)
    xc = x - x.mean(axis=0)
    return (yc * xc).sum(axis=0) / np.maximum((xc * xc).sum(axis=0), 1e-18)


def adf1_no_constant_tstat(residual: np.ndarray) -> np.ndarray:
    """ADF(1) t-stat used by statsmodels' Engle-Granger residual step.

    For each residual column, regress
        delta e_t = rho e_(t-1) + gamma delta e_(t-1) + noise
    without an intercept, then return the t-statistic on rho.
    """
    ds = np.diff(residual, axis=0)
    y = ds[1:]
    x1 = residual[1:-1]
    x2 = ds[:-1]

    a11 = (x1 * x1).sum(axis=0)
    a12 = (x1 * x2).sum(axis=0)
    a22 = (x2 * x2).sum(axis=0)
    b1 = (x1 * y).sum(axis=0)
    b2 = (x2 * y).sum(axis=0)
    yy = (y * y).sum(axis=0)
    det = a11 * a22 - a12 * a12

    out = np.full(residual.shape[1], np.nan)
    valid = np.isfinite(det) & (det > 1e-20)
    rho = np.zeros_like(det)
    gamma = np.zeros_like(det)
    rho[valid] = (b1[valid] * a22[valid] - b2[valid] * a12[valid]) / det[valid]
    gamma[valid] = (b2[valid] * a11[valid] - b1[valid] * a12[valid]) / det[valid]
    dof = max(y.shape[0] - 2, 1)
    sig2 = (yy - rho * b1 - gamma * b2) / dof
    se2 = sig2 * a22 / np.maximum(det, 1e-300)
    valid &= np.isfinite(se2) & (se2 > 1e-300)
    out[valid] = rho[valid] / np.sqrt(se2[valid])
    return out


def half_life(residual: np.ndarray) -> np.ndarray:
    lag = residual[:-1]
    delta = np.diff(residual, axis=0)
    lag_c = lag - lag.mean(axis=0)
    delta_c = delta - delta.mean(axis=0)
    kappa = (lag_c * delta_c).sum(axis=0) / np.maximum(
        (lag_c * lag_c).sum(axis=0), 1e-18
    )
    return np.where(kappa < 0.0, -np.log(2.0) / kappa, np.inf)


def beta_half_ratio(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    middle = len(y) // 2
    b1 = ols_beta(y[:middle], x[:middle])
    b2 = ols_beta(y[middle:], x[middle:])
    same_positive_sign = (b1 > 0.0) & (b2 > 0.0)
    ratio = np.maximum(np.abs(b1), np.abs(b2)) / np.maximum(
        np.minimum(np.abs(b1), np.abs(b2)), 1e-12
    )
    return np.where(same_positive_sign, ratio, np.inf)


def fit_pair_window(
    log_prices: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    start: int,
    end: int,
):
    """Fit all pairs on [start, end), then score the unseen observation end."""
    y = log_prices[start:end, pair_i]
    x = log_prices[start:end, pair_j]

    beta = ols_beta(y, x)
    alpha = y.mean(axis=0) - beta * x.mean(axis=0)
    residual = y - alpha - beta * x

    reverse_beta = ols_beta(x, y)
    reverse_alpha = x.mean(axis=0) - reverse_beta * y.mean(axis=0)
    reverse_residual = x - reverse_alpha - reverse_beta * y

    t_forward = adf1_no_constant_tstat(residual)
    t_reverse = adf1_no_constant_tstat(reverse_residual)
    conservative_t = np.maximum(t_forward, t_reverse)

    current_residual = (
        log_prices[end, pair_i] - alpha - beta * log_prices[end, pair_j]
    )
    full_sigma = residual.std(axis=0, ddof=1)
    z_full = current_residual / np.maximum(full_sigma, 1e-12)

    center = min(LOCAL_CENTER, len(residual))
    local = residual[-center:]
    z_local = (current_residual - local.mean(axis=0)) / np.maximum(
        local.std(axis=0, ddof=1), 1e-12
    )

    return {
        "beta": beta,
        "adf_t": conservative_t,
        "half_life": half_life(residual),
        "beta_ratio": beta_half_ratio(y, x),
        "z_local": z_local,
        "z_full": z_full,
        "t_forward": t_forward,
        "t_reverse": t_reverse,
    }


def mackinnon_vector(tstats: np.ndarray) -> np.ndarray:
    """Map Engle-Granger residual t-stats to MacKinnon approximate p-values."""
    flat = np.asarray(tstats, dtype=float).ravel()
    values = np.fromiter(
        (
            mackinnonp(float(value), regression="c", N=2)
            if np.isfinite(value)
            else np.nan
            for value in flat
        ),
        dtype=float,
        count=len(flat),
    )
    return values.reshape(np.shape(tstats))


def benjamini_hochberg_mask(pvalues: np.ndarray, q: float) -> np.ndarray:
    result = np.zeros_like(pvalues, dtype=bool)
    valid_idx = np.flatnonzero(np.isfinite(pvalues))
    if not len(valid_idx):
        return result
    order = valid_idx[np.argsort(pvalues[valid_idx])]
    ordered = pvalues[order]
    limits = q * np.arange(1, len(order) + 1) / len(order)
    passed = np.flatnonzero(ordered <= limits)
    if len(passed):
        result[order[: passed[-1] + 1]] = True
    return result


def duration_bucket(duration: int) -> str:
    if duration == 1:
        return "1"
    if duration <= 3:
        return "2-3"
    if duration <= 7:
        return "4-7"
    if duration <= 15:
        return "8-15"
    return "16+"


def split_runs(indices: np.ndarray, signs: np.ndarray | None = None):
    if not len(indices):
        return []
    breaks = np.diff(indices) > 1
    if signs is not None:
        breaks |= signs[1:] != signs[:-1]
    return np.split(indices, np.flatnonzero(breaks) + 1)


def fixed_beta_forward_bps(
    log_prices: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_index: int,
    day_index: int,
    beta: float,
    z: float,
    horizon: int,
) -> float:
    future = day_index + horizon
    if future >= len(log_prices) or not np.isfinite(beta) or not np.isfinite(z):
        return np.nan
    i = pair_i[pair_index]
    j = pair_j[pair_index]
    spread_change = (
        log_prices[future, i]
        - log_prices[day_index, i]
        - beta * (log_prices[future, j] - log_prices[day_index, j])
    )
    return float(-np.sign(z) * spread_change / (1.0 + abs(beta)) * 1e4)


def signal_forward_metrics(
    log_prices,
    pair_i,
    pair_j,
    score_specs,
    fit_lookup,
    z_scores,
    fit_beta,
    fit_adf,
    fit_hl,
    fit_ratio,
):
    rows = []
    last_day = len(log_prices) - max(HORIZONS)
    eval_slice = slice(EVAL_START, last_day)
    period_slices = (
        ("all_days_251_730", slice(None)),
        ("days_251_500", slice(0, 500 - EVAL_START)),
        ("days_501_730", slice(500 - EVAL_START, None)),
    )

    for score_idx, spec in enumerate(score_specs):
        fit_idx = fit_lookup[spec.fit_name]
        z = z_scores[score_idx, eval_slice].astype(float)
        beta = fit_beta[fit_idx, eval_slice].astype(float)
        adf = fit_adf[fit_idx, eval_slice].astype(float)
        hl = fit_hl[fit_idx, eval_slice].astype(float)
        ratio = fit_ratio[fit_idx, eval_slice].astype(float)
        base = (
            np.isfinite(z)
            & np.isfinite(beta)
            & (beta > 0.0)
            & (np.abs(z) >= Z_ENTRY)
            & (np.abs(z) < Z_MAX)
        )
        health = (
            (adf < LIGHT_ADF_T)
            & (hl >= HALF_LIFE_MIN)
            & (hl <= HALF_LIFE_MAX)
            & (ratio <= BETA_RATIO_MAX)
        )

        for gate_name, mask in (("z_only", base), ("light_health", base & health)):
            for horizon in HORIZONS:
                pnl = np.full_like(z, np.nan, dtype=float)
                for local_day, day in enumerate(range(EVAL_START, last_day)):
                    active = mask[local_day]
                    if not active.any():
                        continue
                    dy = log_prices[day + horizon, pair_i] - log_prices[day, pair_i]
                    dx = log_prices[day + horizon, pair_j] - log_prices[day, pair_j]
                    pnl[local_day, active] = (
                        -np.sign(z[local_day, active])
                        * (dy[active] - beta[local_day, active] * dx[active])
                        / (1.0 + np.abs(beta[local_day, active]))
                        * 1e4
                    )
                for period, period_slice in period_slices:
                    period_pnl = pnl[period_slice]
                    period_mask = mask[period_slice]
                    values = period_pnl[period_mask & np.isfinite(period_pnl)]
                    valid_count = np.sum(np.isfinite(period_pnl), axis=1)
                    daily_basket = np.divide(
                        np.nansum(period_pnl, axis=1),
                        valid_count,
                        out=np.full(len(valid_count), np.nan),
                        where=valid_count > 0,
                    )
                    daily_basket = daily_basket[np.isfinite(daily_basket)]
                    rows.append(
                        {
                            "score_method": spec.name,
                            "fit_method": spec.fit_name,
                            "gate": gate_name,
                            "period": period,
                            "horizon": horizon,
                            "signals": int(len(values)),
                            "signal_mean_bps": float(np.mean(values)) if len(values) else np.nan,
                            "signal_median_bps": float(np.median(values)) if len(values) else np.nan,
                            "signal_win_rate": float(np.mean(values > 0.0)) if len(values) else np.nan,
                            "daily_baskets": int(len(daily_basket)),
                            "daily_basket_mean_bps": float(np.mean(daily_basket)) if len(daily_basket) else np.nan,
                            "daily_basket_win_rate": float(np.mean(daily_basket > 0.0)) if len(daily_basket) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def estimator_stability_table(
    score_specs,
    fit_lookup,
    z_scores,
    fit_beta,
    fit_adf,
    fit_hl,
    fit_ratio,
):
    rows = []
    for score_idx, spec in enumerate(score_specs):
        fit_idx = fit_lookup[spec.fit_name]
        z = z_scores[score_idx, EVAL_START:].astype(float)
        beta = fit_beta[fit_idx, EVAL_START:].astype(float)
        adf = fit_adf[fit_idx, EVAL_START:].astype(float)
        hl = fit_hl[fit_idx, EVAL_START:].astype(float)
        ratio = fit_ratio[fit_idx, EVAL_START:].astype(float)
        relative_turnover = np.abs(np.diff(beta, axis=0)) / np.maximum(
            np.abs(beta[:-1]), 0.05
        )
        signal = (
            (np.abs(z) >= Z_ENTRY)
            & (np.abs(z) < Z_MAX)
            & (beta > 0.0)
        )
        health = (
            (adf < LIGHT_ADF_T)
            & (hl >= HALF_LIFE_MIN)
            & (hl <= HALF_LIFE_MAX)
            & (ratio <= BETA_RATIO_MAX)
        )
        rows.append(
            {
                "score_method": spec.name,
                "fit_method": spec.fit_name,
                "median_daily_relative_beta_change": float(
                    np.nanmedian(relative_turnover)
                ),
                "p95_daily_relative_beta_change": float(
                    np.nanpercentile(relative_turnover, 95)
                ),
                "z_standard_deviation": float(np.nanstd(z)),
                "z_signal_count": int(np.sum(signal)),
                "light_health_signal_count": int(np.sum(signal & health)),
                "light_health_fraction": float(np.sum(signal & health) / max(np.sum(signal), 1)),
            }
        )
    return pd.DataFrame(rows)


def build_pvalue_runs(
    pair_names,
    p50,
    bh_mask,
    z,
    beta,
    log_prices,
    pair_i,
    pair_j,
):
    rows = []
    for gate_name, active_matrix in (
        ("raw_p_le_0.05", p50 <= RAW_P_ALPHA),
        ("strict_p_le_0.001", p50 <= STRICT_P_ALPHA),
        ("bh_fdr_q_0.05", bh_mask),
    ):
        for pair_idx, pair in enumerate(pair_names):
            indices = np.flatnonzero(active_matrix[EVAL_START:, pair_idx]) + EVAL_START
            for run in split_runs(indices):
                start = int(run[0])
                end = int(run[-1])
                start_z = float(z[start, pair_idx])
                row = {
                    "gate": gate_name,
                    "pair": pair,
                    "pair_index": pair_idx,
                    "start_day": start + 1,
                    "end_day": end + 1,
                    "duration": int(len(run)),
                    "duration_bucket": duration_bucket(len(run)),
                    "minimum_p": float(np.nanmin(p50[run, pair_idx])),
                    "start_z": start_z,
                    "maximum_abs_z": float(np.nanmax(np.abs(z[run, pair_idx]))),
                    "start_is_extreme": bool(
                        Z_ENTRY <= abs(start_z) < Z_MAX
                        and beta[start, pair_idx] > 0.0
                    ),
                }
                for horizon in HORIZONS:
                    row[f"start_forward_{horizon}d_bps"] = fixed_beta_forward_bps(
                        log_prices,
                        pair_i,
                        pair_j,
                        pair_idx,
                        start,
                        float(beta[start, pair_idx]),
                        start_z,
                        horizon,
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def build_zscore_runs(
    pair_names,
    z,
    beta,
    adf,
    hl,
    ratio,
    p50,
    log_prices,
    pair_i,
    pair_j,
):
    health = (
        (adf < LIGHT_ADF_T)
        & (hl >= HALF_LIFE_MIN)
        & (hl <= HALF_LIFE_MAX)
        & (ratio <= BETA_RATIO_MAX)
        & (beta > 0.0)
    )
    extreme = (np.abs(z) >= Z_ENTRY) & (np.abs(z) < Z_MAX)
    active = health & extreme
    rows = []
    for pair_idx, pair in enumerate(pair_names):
        indices = np.flatnonzero(active[EVAL_START:, pair_idx]) + EVAL_START
        signs = np.sign(z[indices, pair_idx])
        for run in split_runs(indices, signs):
            start = int(run[0])
            end = int(run[-1])
            start_z = float(z[start, pair_idx])
            row = {
                "pair": pair,
                "pair_index": pair_idx,
                "start_day": start + 1,
                "end_day": end + 1,
                "duration": int(len(run)),
                "duration_bucket": duration_bucket(len(run)),
                "side": "short_spread" if start_z > 0 else "long_spread",
                "start_z": start_z,
                "maximum_abs_z": float(np.nanmax(np.abs(z[run, pair_idx]))),
                "minimum_p_during_run": float(np.nanmin(p50[run, pair_idx])),
            }
            for horizon in HORIZONS:
                row[f"start_forward_{horizon}d_bps"] = fixed_beta_forward_bps(
                    log_prices,
                    pair_i,
                    pair_j,
                    pair_idx,
                    start,
                    float(beta[start, pair_idx]),
                    start_z,
                    horizon,
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_run_outcomes(runs: pd.DataFrame, kind: str) -> pd.DataFrame:
    rows = []
    bucket_order = ("1", "2-3", "4-7", "8-15", "16+")
    for gate, gate_group in (
        runs.groupby("gate") if "gate" in runs else [(kind, runs)]
    ):
        periods = (
            ("all_days_251_750", gate_group),
            ("days_251_500", gate_group[gate_group["start_day"] <= 500]),
            ("days_501_750", gate_group[gate_group["start_day"] >= 501]),
        )
        for period, period_group in periods:
            for bucket in bucket_order:
                group = period_group[period_group["duration_bucket"] == bucket]
                if "start_is_extreme" in group:
                    group = group[group["start_is_extreme"]]
                for horizon in HORIZONS:
                    column = f"start_forward_{horizon}d_bps"
                    values = group[column].dropna().to_numpy(float)
                    rows.append(
                        {
                            "kind": kind,
                            "gate": gate,
                            "period": period,
                            "duration_bucket": bucket,
                            "horizon": horizon,
                            "episodes": int(len(values)),
                            "mean_bps": float(np.mean(values)) if len(values) else np.nan,
                            "median_bps": float(np.median(values)) if len(values) else np.nan,
                            "win_rate": float(np.mean(values > 0.0)) if len(values) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def confirmation_summary(
    runs: pd.DataFrame,
    kind: str,
    z,
    beta,
    log_prices,
    pair_i,
    pair_j,
):
    rows = []
    grouped = runs.groupby("gate") if "gate" in runs else [(kind, runs)]
    for gate, gate_runs in grouped:
        periods = (
            ("all_days_251_750", gate_runs),
            ("days_251_500", gate_runs[gate_runs["start_day"] <= 500]),
            ("days_501_750", gate_runs[gate_runs["start_day"] >= 501]),
        )
        for period, period_runs in periods:
            for confirmation in CONFIRMATIONS:
                for horizon in HORIZONS:
                    values = []
                    eligible_runs = period_runs[period_runs["duration"] >= confirmation]
                    for run in eligible_runs.itertuples(index=False):
                        pair_idx = int(run.pair_index)
                        day = int(run.start_day - 1 + confirmation - 1)
                        current_z = float(z[day, pair_idx])
                        current_beta = float(beta[day, pair_idx])
                        if not (
                            Z_ENTRY <= abs(current_z) < Z_MAX
                            and current_beta > 0.0
                        ):
                            continue
                        value = fixed_beta_forward_bps(
                            log_prices,
                            pair_i,
                            pair_j,
                            pair_idx,
                            day,
                            current_beta,
                            current_z,
                            horizon,
                        )
                        if np.isfinite(value):
                            values.append(value)
                    values = np.asarray(values, dtype=float)
                    rows.append(
                        {
                            "kind": kind,
                            "gate": gate,
                            "period": period,
                            "confirmation_days": confirmation,
                            "horizon": horizon,
                            "entries": int(len(values)),
                            "mean_bps": float(np.mean(values)) if len(values) else np.nan,
                            "median_bps": float(np.median(values)) if len(values) else np.nan,
                            "win_rate": float(np.mean(values > 0.0)) if len(values) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def daily_breadth_table(
    names,
    pair_i,
    pair_j,
    p50,
    bh_mask,
    z,
    beta,
    adf,
    hl,
    ratio,
):
    rows = []
    raw = p50 <= RAW_P_ALPHA
    strict = p50 <= STRICT_P_ALPHA
    extreme = (np.abs(z) >= Z_ENTRY) & (np.abs(z) < Z_MAX) & (beta > 0.0)
    light = (
        extreme
        & (adf < LIGHT_ADF_T)
        & (hl >= HALF_LIFE_MIN)
        & (hl <= HALF_LIFE_MAX)
        & (ratio <= BETA_RATIO_MAX)
    )
    for day in range(MIN_HISTORY, len(z)):
        active_pairs = np.flatnonzero(light[day])
        degrees = np.zeros(len(names), dtype=int)
        if len(active_pairs):
            np.add.at(degrees, pair_i[active_pairs], 1)
            np.add.at(degrees, pair_j[active_pairs], 1)
        rows.append(
            {
                "day": day + 1,
                "raw_p_significant_pairs": int(raw[day].sum()),
                "strict_p_significant_pairs": int(strict[day].sum()),
                "bh_significant_pairs": int(bh_mask[day].sum()),
                "extreme_z_pairs": int(extreme[day].sum()),
                "light_health_extreme_pairs": int(light[day].sum()),
                "raw_p_and_extreme_z": int((raw[day] & extreme[day]).sum()),
                "strict_p_and_extreme_z": int((strict[day] & extreme[day]).sum()),
                "bh_and_extreme_z": int((bh_mask[day] & extreme[day]).sum()),
                "active_assets": int(np.sum(degrees > 0)),
                "maximum_asset_degree": int(degrees.max(initial=0)),
                "maximum_degree_asset": names[int(np.argmax(degrees))] if degrees.any() else "",
            }
        )
    return pd.DataFrame(rows)


def plot_daily_breadth(daily: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    axes[0].plot(daily["day"], daily["raw_p_significant_pairs"], label="raw p <= 0.05", lw=1)
    axes[0].plot(daily["day"], daily["strict_p_significant_pairs"], label="raw p <= 0.001", lw=1)
    axes[0].plot(daily["day"], daily["bh_significant_pairs"], label="BH-FDR q=0.05", lw=1)
    axes[0].set_ylabel("relationship-qualified pairs")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(daily["day"], daily["extreme_z_pairs"], label="|z| >= 2", lw=1)
    axes[1].plot(
        daily["day"],
        daily["light_health_extreme_pairs"],
        label="|z| >= 2 + light health gate",
        lw=1,
    )
    axes[1].plot(
        daily["day"],
        daily["bh_and_extreme_z"],
        label="|z| >= 2 + BH relationship gate",
        lw=1,
    )
    axes[1].set_ylabel("spread-score events")
    axes[1].set_xlabel("decision day (all inputs are from this day or earlier)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("All 1,275 pairs: daily relationship breadth versus actual spread entries", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lookback_comparison(metrics, stability, path):
    methods = [spec.name for spec in SCORE_SPECS]
    ten = metrics[
        (metrics["gate"] == "light_health")
        & (metrics["period"] == "all_days_251_730")
        & (metrics["horizon"] == 10)
    ].set_index("score_method").reindex(methods)
    stab = stability.set_index("score_method").reindex(methods)
    labels = [
        "50/30",
        "100/30",
        "250/30",
        "expanding/30",
        "250/250",
        "expanding/full",
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].bar(x, ten["signal_mean_bps"], color="#4c78a8")
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].set_title("10-day gross reversion per signal")
    axes[0, 0].set_ylabel("basis points")

    axes[0, 1].bar(x, ten["signal_win_rate"], color="#59a14f")
    axes[0, 1].axhline(0.5, color="black", lw=0.8, ls="--")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("10-day signal win rate")

    axes[1, 0].bar(x, ten["signals"], color="#f28e2b")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Qualifying pair-day signals")

    axes[1, 1].bar(x, stab["median_daily_relative_beta_change"], color="#b07aa1")
    axes[1, 1].set_title("Median daily relative hedge-ratio change")
    axes[1, 1].set_ylabel("absolute fraction")

    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Does more history improve the causal spread score?", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lookback_period_comparison(metrics, path):
    methods = [spec.name for spec in SCORE_SPECS]
    labels = ["50/30", "100/30", "250/30", "expanding/30", "250/250", "expanding/full"]
    early = metrics[
        (metrics["gate"] == "light_health")
        & (metrics["period"] == "days_251_500")
        & (metrics["horizon"] == 10)
    ].set_index("score_method").reindex(methods)
    later = metrics[
        (metrics["gate"] == "light_health")
        & (metrics["period"] == "days_501_730")
        & (metrics["horizon"] == 10)
    ].set_index("score_method").reindex(methods)

    x = np.arange(len(labels))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    axes[0].bar(x - width / 2, early["signal_mean_bps"], width, label="days 251-500")
    axes[0].bar(x + width / 2, later["signal_mean_bps"], width, label="days 501-730")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("10-day gross bps per signal")
    axes[0].set_title("Forward reversion")

    axes[1].bar(x - width / 2, early["signal_win_rate"], width, label="days 251-500")
    axes[1].bar(x + width / 2, later["signal_win_rate"], width, label="days 501-730")
    axes[1].axhline(0.5, color="black", lw=0.8, ls="--")
    axes[1].set_ylim(0.45, 0.60)
    axes[1].set_ylabel("10-day win rate")
    axes[1].set_title("Chronological stability")

    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("All-pair spread scores: early versus later data", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_persistence(p_runs, p_confirm, z_runs, z_confirm, path):
    buckets = ["1", "2-3", "4-7", "8-15", "16+"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for gate, color in (
        ("raw_p_le_0.05", "#4c78a8"),
        ("strict_p_le_0.001", "#f28e2b"),
        ("bh_fdr_q_0.05", "#e15759"),
    ):
        counts = p_runs[p_runs["gate"] == gate]["duration_bucket"].value_counts().reindex(buckets, fill_value=0)
        axes[0].plot(buckets, counts, marker="o", label=gate.replace("_", " "), color=color)
    axes[0].set_yscale("log")
    axes[0].set_title("Relationship-health episode lengths")
    axes[0].set_xlabel("consecutive decision days")
    axes[0].set_ylabel("episodes (log scale)")
    axes[0].legend(fontsize=8)

    for gate, color in (
        ("raw_p_le_0.05", "#59a14f"),
        ("strict_p_le_0.001", "#e15759"),
    ):
        pc = p_confirm[
            (p_confirm["gate"] == gate)
            & (p_confirm["period"] == "all_days_251_750")
            & (p_confirm["horizon"] == 10)
        ]
        axes[1].plot(
            pc["confirmation_days"].astype(str),
            pc["mean_bps"],
            marker="o",
            color=color,
            label=gate.replace("_", " "),
        )
        for point in pc.itertuples(index=False):
            if np.isfinite(point.mean_bps):
                axes[1].annotate(
                    f"n={point.entries}",
                    (str(point.confirmation_days), point.mean_bps),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    color=color,
                )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Enter only after p-value persistence")
    axes[1].set_xlabel("consecutive significant days required")
    axes[1].set_ylabel("next 10-day gross bps")
    axes[1].legend(fontsize=8)

    zc = z_confirm[
        (z_confirm["period"] == "all_days_251_750")
        & (z_confirm["horizon"] == 10)
    ]
    axes[2].bar(zc["confirmation_days"].astype(str), zc["mean_bps"], color="#f28e2b")
    for point in zc.itertuples(index=False):
        if np.isfinite(point.mean_bps):
            axes[2].annotate(
                f"n={point.entries}",
                (str(point.confirmation_days), point.mean_bps),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Enter only after z-score persistence")
    axes[2].set_xlabel("consecutive extreme-score days required")
    axes[2].set_ylabel("next 10-day gross bps")

    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("One-day noise versus persistent all-pair episodes", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_case_studies(p_runs, pair_names, p50, z, path):
    raw = p_runs[p_runs["gate"] == "raw_p_le_0.05"].copy()
    long = raw.sort_values(["duration", "maximum_abs_z"], ascending=False)
    short = raw[raw["duration"] <= 3].sort_values(["minimum_p", "maximum_abs_z"])
    selected = []
    used = set()
    for frame in (long, short):
        for row in frame.itertuples(index=False):
            if row.pair in used:
                continue
            selected.append(row)
            used.add(row.pair)
            if len(selected) == (3 if frame is long else 6):
                break
    if not selected:
        return

    days = np.arange(1, p50.shape[0] + 1)
    fig, axes = plt.subplots(len(selected), 2, figsize=(17, 3.0 * len(selected)), sharex=True)
    axes = np.atleast_2d(axes)
    for row_idx, row in enumerate(selected):
        pair_idx = int(row.pair_index)
        strength = -np.log10(np.clip(p50[:, pair_idx], 1e-16, 1.0))
        axes[row_idx, 0].plot(days, strength, color="#6a3d9a", lw=1)
        axes[row_idx, 0].axhline(-np.log10(RAW_P_ALPHA), color="#777777", ls="--", lw=0.8)
        axes[row_idx, 0].axvspan(row.start_day, row.end_day, color="#f28e2b", alpha=0.25)
        axes[row_idx, 0].set_ylabel("-log10(p)")
        axes[row_idx, 0].set_title(f"{row.pair}: selected p-run = {row.duration} days", fontsize=10)

        axes[row_idx, 1].plot(days, z[:, pair_idx], color="#4c78a8", lw=1)
        axes[row_idx, 1].axhline(Z_ENTRY, color="#d62728", ls="--", lw=0.8)
        axes[row_idx, 1].axhline(-Z_ENTRY, color="#d62728", ls="--", lw=0.8)
        axes[row_idx, 1].axvspan(row.start_day, row.end_day, color="#f28e2b", alpha=0.25)
        axes[row_idx, 1].set_ylabel("causal spread z")
        axes[row_idx, 1].set_title("50-day beta / 30-day centre", fontsize=10)
        for ax in axes[row_idx]:
            ax.grid(alpha=0.15)
    axes[-1, 0].set_xlabel("day")
    axes[-1, 1].set_xlabel("day")
    fig.suptitle("Long relationship continuations versus 1-3-day statistical spikes", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    names = prices.columns.to_numpy(str)
    log_prices = np.log(prices.to_numpy(float))
    n_days, n_assets = log_prices.shape
    pair_i, pair_j = np.triu_indices(n_assets, k=1)
    pair_names = np.asarray([f"{names[i]}-{names[j]}" for i, j in zip(pair_i, pair_j)])
    n_pairs = len(pair_i)
    if n_pairs != 1275:
        warnings.warn(f"Expected 1,275 pairs but found {n_pairs}")

    fit_lookup = {spec.name: idx for idx, spec in enumerate(FIT_SPECS)}
    score_lookup = {spec.name: idx for idx, spec in enumerate(SCORE_SPECS)}
    score_by_fit = {
        fit.name: [idx for idx, spec in enumerate(SCORE_SPECS) if spec.fit_name == fit.name]
        for fit in FIT_SPECS
    }

    shape_fit = (len(FIT_SPECS), n_days, n_pairs)
    fit_beta = np.full(shape_fit, np.nan, dtype=np.float32)
    fit_adf = np.full(shape_fit, np.nan, dtype=np.float32)
    fit_hl = np.full(shape_fit, np.nan, dtype=np.float32)
    fit_ratio = np.full(shape_fit, np.nan, dtype=np.float32)
    z_scores = np.full((len(SCORE_SPECS), n_days, n_pairs), np.nan, dtype=np.float32)
    w50_t_forward = np.full((n_days, n_pairs), np.nan, dtype=np.float32)
    w50_t_reverse = np.full((n_days, n_pairs), np.nan, dtype=np.float32)

    print(f"{n_assets} assets, {n_pairs} unordered pairs, {n_days} days")
    for day in range(MIN_HISTORY, n_days):
        for fit_idx, fit_spec in enumerate(FIT_SPECS):
            if fit_spec.window is not None and day < fit_spec.window:
                continue
            start = 0 if fit_spec.window is None else day - fit_spec.window
            stats = fit_pair_window(log_prices, pair_i, pair_j, start, day)
            fit_beta[fit_idx, day] = stats["beta"]
            fit_adf[fit_idx, day] = stats["adf_t"]
            fit_hl[fit_idx, day] = stats["half_life"]
            fit_ratio[fit_idx, day] = stats["beta_ratio"]
            for score_idx in score_by_fit[fit_spec.name]:
                center = SCORE_SPECS[score_idx].center
                z_scores[score_idx, day] = stats["z_full"] if center is None else stats["z_local"]
            if fit_spec.name == "w50":
                w50_t_forward[day] = stats["t_forward"]
                w50_t_reverse[day] = stats["t_reverse"]
        if day % 50 == 0 or day == n_days - 1:
            print(f"computed causal scores through day {day + 1}/{n_days}")

    print("mapping all 50-day residual tests to MacKinnon p-values")
    p_forward = mackinnon_vector(w50_t_forward)
    p_reverse = mackinnon_vector(w50_t_reverse)
    p50 = np.maximum(p_forward, p_reverse).astype(np.float32)

    # Verify the vectorised Engle-Granger calculation against statsmodels.
    validation_rows = []
    for day in (100, 300, 600):
        for pair_idx in (0, 100, 1000):
            y = log_prices[day - 50 : day, pair_i[pair_idx]]
            x = log_prices[day - 50 : day, pair_j[pair_idx]]
            exact_forward = float(coint(y, x, maxlag=1, autolag=None)[1])
            exact_reverse = float(coint(x, y, maxlag=1, autolag=None)[1])
            validation_rows.append(
                {
                    "day": day + 1,
                    "pair": pair_names[pair_idx],
                    "vector_forward_p": float(p_forward[day, pair_idx]),
                    "statsmodels_forward_p": exact_forward,
                    "vector_reverse_p": float(p_reverse[day, pair_idx]),
                    "statsmodels_reverse_p": exact_reverse,
                }
            )
    validation = pd.DataFrame(validation_rows)
    validation["maximum_direction_error"] = np.maximum(
        np.abs(validation["vector_forward_p"] - validation["statsmodels_forward_p"]),
        np.abs(validation["vector_reverse_p"] - validation["statsmodels_reverse_p"]),
    )
    validation.to_csv(OUTPUT_DIR / "engle_granger_validation.csv", index=False)
    print(f"maximum validation p-value error: {validation['maximum_direction_error'].max():.3g}")

    bh_mask = np.zeros_like(p50, dtype=bool)
    for day in range(MIN_HISTORY, n_days):
        bh_mask[day] = benjamini_hochberg_mask(p50[day].astype(float), FDR_Q)

    metrics = signal_forward_metrics(
        log_prices,
        pair_i,
        pair_j,
        SCORE_SPECS,
        fit_lookup,
        z_scores,
        fit_beta,
        fit_adf,
        fit_hl,
        fit_ratio,
    )
    metrics.to_csv(OUTPUT_DIR / "lookback_forward_metrics.csv", index=False)

    stability = estimator_stability_table(
        SCORE_SPECS,
        fit_lookup,
        z_scores,
        fit_beta,
        fit_adf,
        fit_hl,
        fit_ratio,
    )
    stability.to_csv(OUTPUT_DIR / "estimator_stability.csv", index=False)

    w50_fit = fit_lookup["w50"]
    w50_score = score_lookup["50d beta / 30d centre"]
    z50 = z_scores[w50_score].astype(float)
    beta50 = fit_beta[w50_fit].astype(float)
    adf50 = fit_adf[w50_fit].astype(float)
    hl50 = fit_hl[w50_fit].astype(float)
    ratio50 = fit_ratio[w50_fit].astype(float)

    p_runs = build_pvalue_runs(
        pair_names,
        p50,
        bh_mask,
        z50,
        beta50,
        log_prices,
        pair_i,
        pair_j,
    )
    p_runs.to_csv(OUTPUT_DIR / "pvalue_episodes.csv", index=False)
    p_outcomes = summarise_run_outcomes(p_runs, "pvalue_episode")
    p_outcomes.to_csv(OUTPUT_DIR / "pvalue_episode_outcomes.csv", index=False)
    p_confirmation = confirmation_summary(
        p_runs,
        "pvalue_persistence",
        z50,
        beta50,
        log_prices,
        pair_i,
        pair_j,
    )
    p_confirmation.to_csv(OUTPUT_DIR / "pvalue_confirmation_outcomes.csv", index=False)

    z_runs = build_zscore_runs(
        pair_names,
        z50,
        beta50,
        adf50,
        hl50,
        ratio50,
        p50,
        log_prices,
        pair_i,
        pair_j,
    )
    z_runs.to_csv(OUTPUT_DIR / "zscore_episodes.csv", index=False)
    z_outcomes = summarise_run_outcomes(z_runs, "zscore_episode")
    z_outcomes.to_csv(OUTPUT_DIR / "zscore_episode_outcomes.csv", index=False)
    z_confirmation = confirmation_summary(
        z_runs,
        "zscore_persistence",
        z50,
        beta50,
        log_prices,
        pair_i,
        pair_j,
    )
    z_confirmation.to_csv(OUTPUT_DIR / "zscore_confirmation_outcomes.csv", index=False)

    daily = daily_breadth_table(
        names,
        pair_i,
        pair_j,
        p50,
        bh_mask,
        z50,
        beta50,
        adf50,
        hl50,
        ratio50,
    )
    daily.to_csv(OUTPUT_DIR / "daily_all_pair_breadth.csv", index=False)

    pd.DataFrame(
        {
            "pair_index": np.arange(n_pairs),
            "pair": pair_names,
            "ticker_a": names[pair_i],
            "ticker_b": names[pair_j],
            "asset_index_a": pair_i,
            "asset_index_b": pair_j,
        }
    ).to_csv(OUTPUT_DIR / "pair_index.csv", index=False)

    np.savez_compressed(
        OUTPUT_DIR / "all_pair_scores_w50.npz",
        days=np.arange(1, n_days + 1),
        pair_names=pair_names,
        pair_i=pair_i,
        pair_j=pair_j,
        z=z50.astype(np.float32),
        beta=beta50.astype(np.float32),
        conservative_p=p50.astype(np.float32),
        conservative_adf_t=adf50.astype(np.float32),
        half_life=hl50.astype(np.float32),
        beta_ratio=ratio50.astype(np.float32),
        bh_fdr_mask=bh_mask,
    )

    plot_daily_breadth(daily, OUTPUT_DIR / "daily_all_pair_breadth.png")
    plot_lookback_comparison(metrics, stability, OUTPUT_DIR / "lookback_comparison.png")
    plot_lookback_period_comparison(
        metrics,
        OUTPUT_DIR / "lookback_period_comparison.png",
    )
    plot_persistence(
        p_runs,
        p_confirmation,
        z_runs,
        z_confirmation,
        OUTPUT_DIR / "persistence_diagnostics.png",
    )
    plot_case_studies(
        p_runs,
        pair_names,
        p50,
        z50,
        OUTPUT_DIR / "persistence_case_studies.png",
    )

    print("\n10-day light-health results")
    print(
        metrics[
            (metrics["gate"] == "light_health")
            & (metrics["period"] == "all_days_251_730")
            & (metrics["horizon"] == 10)
        ][
            ["score_method", "period", "signals", "signal_mean_bps", "signal_win_rate"]
        ].to_string(index=False)
    )
    print("\n50-day p-value episode duration counts")
    print(
        p_runs.groupby(["gate", "duration_bucket"], observed=False)
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

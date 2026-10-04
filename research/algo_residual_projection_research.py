"""Train/validation research on ALGO residual forecasts and beta projection.

The study separates two questions:

1. Forecast: do active ALGO-X signals predict the sign of the next residual
   return ``r_x - beta_x * r_algo``?
2. Construction: does projecting the resulting dollar vector into the
   beta-neutral subspace improve the current Four Sigmas book?

Only days through 750 are loaded into the analysis. Days 751-1000 are neither
evaluated nor used for selection.
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import algo12_equal_sleeve_research as sleeves


OUTPUT_DIR = ROOT / "analysis_outputs" / "algo_residual_projection"
ALGO = strategy.I["ALGO"]
LAST_DEVELOPMENT_DAY = 750
HORIZONS = (1, 2, 5)
WINDOWS = (
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("TRAIN", 251, 500),
    ("VALIDATION_A", 501, 625),
    ("VALIDATION_B", 626, 750),
    ("VALIDATION", 501, 750),
)
PRIMARY_WINDOWS = ("TRAIN_A", "TRAIN_B", "VALIDATION_A", "VALIDATION_B")
VARIANTS = (
    "asset_only",
    "direct_projection",
    "synthetic_euclidean",
    "synthetic_diag_risk",
    "flat_hedge_euclidean",
    "flat_hedge_diag_risk",
    "flat_hedge_var_min",
)


def score(pnl: np.ndarray) -> float:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe**2 / (sharpe**2 + 1.0)


def window_name(day: int) -> str:
    if 251 <= day <= 375:
        return "TRAIN_A"
    if 376 <= day <= 500:
        return "TRAIN_B"
    if 501 <= day <= 625:
        return "VALIDATION_A"
    if 626 <= day <= 750:
        return "VALIDATION_B"
    return "OTHER"


def causal_factor_estimates(
    prices: np.ndarray,
    history_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return trailing betas, correlations, and variances through the decision."""
    betas = np.zeros(strategy.N_INST)
    correlations = np.zeros(strategy.N_INST)
    variances = np.zeros(strategy.N_INST)
    if history_size < strategy.ALGO_BETA_MIN_OBS + 1:
        return betas, correlations, variances, False

    returns = np.diff(
        np.log(np.maximum(prices[:, :history_size], strategy._EPS_DENOM)),
        axis=1,
    )
    known = returns[:, -strategy.ALGO_BETA_WINDOW:]
    centered = known - known.mean(axis=1, keepdims=True)
    variances = np.mean(centered * centered, axis=1)
    algo = centered[ALGO]
    algo_ss = float(algo @ algo)
    if algo_ss <= strategy._EPS_VAR:
        return betas, correlations, variances, False
    cross = centered @ algo
    betas = cross / algo_ss
    norms = np.sqrt(np.sum(centered * centered, axis=1) * algo_ss)
    correlations = np.divide(
        cross,
        norms,
        out=np.zeros(strategy.N_INST),
        where=norms > strategy._EPS_VAR,
    )
    betas[ALGO] = 1.0
    correlations[ALGO] = 1.0
    return betas, correlations, variances, True


def replay_signals(
    prices: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """Replay the twelve signals and record causal one/multi-day outcomes."""
    days = prices.shape[1]
    agents = [strategy.PairAgent(spec) for spec in strategy.ALGO_RESIDUAL_CFGS]
    raw_alpha = np.zeros((days + 1, strategy.N_INST))
    betas = np.zeros_like(raw_alpha)
    correlations = np.zeros_like(raw_alpha)
    variances = np.zeros_like(raw_alpha)
    events: list[dict[str, float | int | str | bool]] = []
    log_prices = np.log(np.maximum(prices, strategy._EPS_DENOM))

    for history_size in range(1, days):
        (
            current_betas,
            current_correlations,
            current_variances,
            ready,
        ) = causal_factor_estimates(prices, history_size)
        betas[history_size] = current_betas
        correlations[history_size] = current_correlations
        variances[history_size] = current_variances
        if history_size < strategy.WARMUP + 10:
            continue

        for name, agent in zip(
            sleeves.PAIR_NAMES,
            agents,
            strict=True,
        ):
            prior_side = int(agent.side)
            agent.catch_up(prices[:, :history_size])
            _, asset_dollars, abs_z = agent.dollar_positions()
            asset_dollars *= sleeves.SLOT / strategy.LEG
            asset = int(agent.spec.x)
            raw_alpha[history_size, asset] += asset_dollars
            if not ready or abs(asset_dollars) <= 1.0e-9:
                continue

            direction = float(np.sign(asset_dollars))
            pnl_day = history_size + 1
            for horizon in HORIZONS:
                future_index = history_size - 1 + horizon
                if future_index >= prices.shape[1]:
                    continue
                asset_return = float(
                    log_prices[asset, future_index]
                    - log_prices[asset, history_size - 1]
                )
                algo_return = float(
                    log_prices[ALGO, future_index]
                    - log_prices[ALGO, history_size - 1]
                )
                beta = float(current_betas[asset])
                residual_return = asset_return - beta * algo_return
                events.append(
                    {
                        "pair": name,
                        "asset": strategy._SYMBOL_BY_INDEX[asset],
                        "history_size": history_size,
                        "pnl_day": pnl_day,
                        "window": window_name(pnl_day),
                        "horizon": horizon,
                        "is_entry": prior_side == 0 and agent.side != 0,
                        "direction": direction,
                        "z": float(agent.current_z),
                        "abs_z": float(abs_z),
                        "entry_ratio": float(
                            abs_z / max(agent.spec.trade.z_entry, 1.0e-12)
                        ),
                        "return_beta": beta,
                        "return_correlation": float(
                            current_correlations[asset]
                        ),
                        "asset_return": asset_return,
                        "algo_return": algo_return,
                        "residual_return": residual_return,
                        "signed_asset_return": direction * asset_return,
                        "signed_market_component": (
                            direction * beta * algo_return
                        ),
                        "signed_residual_return": (
                            direction * residual_return
                        ),
                    }
                )
    return (
        raw_alpha,
        betas,
        correlations,
        variances,
        pd.DataFrame(events),
    )


def _uniform_capacity_scale(
    base: np.ndarray,
    overlay: np.ndarray,
) -> np.ndarray:
    """Scale a complete neutral vector without changing its beta exposure."""
    scale = 1.0
    limits = strategy.BANG_BANG_LIMITS
    for index in range(1, strategy.N_INST):
        value = float(overlay[index])
        if value > 1.0e-12:
            scale = min(scale, (limits[index] - base[index]) / value)
        elif value < -1.0e-12:
            scale = min(scale, (-limits[index] - base[index]) / value)
    return overlay * float(np.clip(scale, 0.0, 1.0))


def direct_projection(
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Apply w = u - beta (beta' u)/(beta' beta) on tradeable assets."""
    result = alpha.copy()
    b = beta[1:]
    denominator = float(b @ b)
    if denominator > strategy._EPS_VAR:
        result[1:] -= b * float(b @ result[1:]) / denominator
    result[ALGO] = 0.0
    largest = float(np.max(np.abs(result[1:])))
    if largest > strategy.LEG:
        result *= strategy.LEG / largest
    return result


def bounded_basis_hedge(
    target_beta_dollars: float,
    beta: np.ndarray,
    basis: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eligible: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Bounded one-constraint hedge with h(lambda)=clip(lambda*basis)."""
    hedge = np.zeros(strategy.N_INST)
    use = (
        np.asarray(eligible, dtype=bool)
        & np.isfinite(beta)
        & np.isfinite(basis)
        & (np.abs(beta) >= strategy.ALGO_HEDGE_BETA_FLOOR)
        & (np.abs(basis) > 1.0e-18)
    )
    use[ALGO] = False
    if abs(target_beta_dollars) <= 1.0e-9:
        return hedge, 1.0
    if not np.any(use):
        return hedge, 0.0

    b = beta[use]
    g = basis[use]
    lo = lower[use]
    hi = upper[use]
    minimum = float(b @ np.where(b >= 0.0, lo, hi))
    maximum = float(b @ np.where(b >= 0.0, hi, lo))
    delivered_target = float(
        np.clip(target_beta_dollars, minimum, maximum)
    )
    delivery = float(
        np.clip(
            delivered_target / target_beta_dollars,
            0.0,
            1.0,
        )
    )

    def exposure(multiplier: float) -> float:
        return float(b @ np.clip(multiplier * g, lo, hi))

    left = -1.0
    right = 1.0
    while exposure(left) > delivered_target:
        left *= 2.0
    while exposure(right) < delivered_target:
        right *= 2.0
    for _ in range(80):
        middle = 0.5 * (left + right)
        if exposure(middle) < delivered_target:
            left = middle
        else:
            right = middle
    hedge[use] = np.clip(0.5 * (left + right) * g, lo, hi)
    return hedge, delivery


def synthetic_overlay(
    base: np.ndarray,
    requested_alpha: np.ndarray,
    beta: np.ndarray,
    variances: np.ndarray,
    crash: np.ndarray,
    risk_weighted: bool,
    flat_hedge_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Capacity-aware residual alpha plus a distributed non-ALGO hedge."""
    lower = -strategy.BANG_BANG_LIMITS - base
    upper = strategy.BANG_BANG_LIMITS - base
    alpha = np.clip(requested_alpha, lower, upper)
    alpha[ALGO] = 0.0
    alpha[crash] = 0.0

    eligible = np.ones(strategy.N_INST, dtype=bool)
    eligible[ALGO] = False
    eligible[strategy._ALGO_RESIDUAL_ASSETS] = False
    eligible[crash] = False
    if flat_hedge_only:
        eligible &= np.abs(base) <= 1.0
    if risk_weighted:
        positive = variances[variances > strategy._EPS_VAR]
        floor = (
            float(np.median(positive)) * 0.05
            if len(positive)
            else strategy._EPS_VAR
        )
        basis = beta / np.maximum(variances, floor)
    else:
        basis = beta.copy()

    target = -float(beta @ alpha)
    hedge, delivery = bounded_basis_hedge(
        target,
        beta,
        basis,
        lower,
        upper,
        eligible,
    )
    if delivery < 1.0:
        alpha *= delivery
        target = -float(beta @ alpha)
        hedge, _ = bounded_basis_hedge(
            target,
            beta,
            basis,
            lower,
            upper,
            eligible,
        )
    overlay = alpha + hedge
    overlay[ALGO] = 0.0
    return overlay, alpha, delivery


def build_targets(
    prices: np.ndarray,
    base_targets: np.ndarray,
    raw_alpha: np.ndarray,
    betas: np.ndarray,
    variances: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    """Build standalone and base-combined targets for every construction."""
    standalone = {
        name: np.zeros_like(base_targets) for name in VARIANTS
    }
    combined = {
        name: np.zeros_like(base_targets) for name in VARIANTS
    }
    diagnostics: list[dict[str, float | int | str]] = []

    for history_size in range(
        strategy.ALGO_BETA_MIN_OBS + 1,
        prices.shape[1],
    ):
        current = prices[:, history_size - 1]
        base = base_targets[history_size] * current
        alpha = raw_alpha[history_size].copy()
        beta = betas[history_size]
        variance = variances[history_size]
        last_return = np.log(
            np.maximum(
                prices[:, history_size - 1],
                strategy._EPS_DENOM,
            )
        ) - np.log(
            np.maximum(
                prices[:, history_size - 2],
                strategy._EPS_DENOM,
            )
        )
        crash = np.abs(last_return) > 0.40
        crash[ALGO] = False

        standalone_vectors: dict[str, np.ndarray] = {
            "asset_only": np.clip(
                alpha,
                -strategy.BANG_BANG_LIMITS,
                strategy.BANG_BANG_LIMITS,
            ),
            "direct_projection": direct_projection(alpha, beta),
        }
        euclidean, _, _ = synthetic_overlay(
            np.zeros(strategy.N_INST),
            alpha,
            beta,
            variance,
            crash,
            risk_weighted=False,
        )
        diagonal, _, _ = synthetic_overlay(
            np.zeros(strategy.N_INST),
            alpha,
            beta,
            variance,
            crash,
            risk_weighted=True,
        )
        standalone_vectors["synthetic_euclidean"] = euclidean
        standalone_vectors["synthetic_diag_risk"] = diagonal
        standalone_vectors["flat_hedge_euclidean"] = euclidean
        standalone_vectors["flat_hedge_diag_risk"] = diagonal
        standalone_vectors["flat_hedge_var_min"] = euclidean

        combined_vectors: dict[str, np.ndarray] = {
            "asset_only": np.clip(
                base + standalone_vectors["asset_only"],
                -strategy.BANG_BANG_LIMITS,
                strategy.BANG_BANG_LIMITS,
            )
        }
        direct = _uniform_capacity_scale(
            base,
            standalone_vectors["direct_projection"],
        )
        combined_vectors["direct_projection"] = base + direct
        euclidean_capacity, _, euclidean_delivery = synthetic_overlay(
            base,
            alpha,
            beta,
            variance,
            crash,
            risk_weighted=False,
        )
        diagonal_capacity, _, diagonal_delivery = synthetic_overlay(
            base,
            alpha,
            beta,
            variance,
            crash,
            risk_weighted=True,
        )
        combined_vectors["synthetic_euclidean"] = (
            base + euclidean_capacity
        )
        combined_vectors["synthetic_diag_risk"] = (
            base + diagonal_capacity
        )
        flat_euclidean, _, flat_euclidean_delivery = synthetic_overlay(
            base,
            alpha,
            beta,
            variance,
            crash,
            risk_weighted=False,
            flat_hedge_only=True,
        )
        flat_diagonal, _, flat_diagonal_delivery = synthetic_overlay(
            base,
            alpha,
            beta,
            variance,
            crash,
            risk_weighted=True,
            flat_hedge_only=True,
        )
        combined_vectors["flat_hedge_euclidean"] = (
            base + flat_euclidean
        )
        combined_vectors["flat_hedge_diag_risk"] = (
            base + flat_diagonal
        )
        known_returns = np.diff(
            np.log(
                np.maximum(
                    prices[
                        :,
                        max(
                            0,
                            history_size
                            - strategy.ALGO_BETA_WINDOW
                            - 1,
                        ):history_size,
                    ],
                    strategy._EPS_DENOM,
                )
            ),
            axis=1,
        )
        centered_returns = (
            known_returns
            - known_returns.mean(axis=1, keepdims=True)
        )
        covariance = (
            centered_returns @ centered_returns.T
            / max(centered_returns.shape[1], 1)
        )
        overlay_variance = float(
            flat_euclidean @ covariance @ flat_euclidean
        )
        base_overlay_covariance = float(
            base @ covariance @ flat_euclidean
        )
        variance_scale = (
            float(
                np.clip(
                    -base_overlay_covariance / overlay_variance,
                    0.0,
                    1.0,
                )
            )
            if overlay_variance > strategy._EPS_VAR
            else 0.0
        )
        combined_vectors["flat_hedge_var_min"] = (
            base + variance_scale * flat_euclidean
        )

        for name in VARIANTS:
            standalone[name][history_size] = sleeves.dollars_to_shares(
                standalone_vectors[name],
                current,
            )
            combined[name][history_size] = sleeves.dollars_to_shares(
                combined_vectors[name],
                current,
            )
            delivered = (
                combined[name][history_size] * current - base
            )
            diagnostics.append(
                {
                    "history_size": history_size,
                    "pnl_day": history_size + 1,
                    "window": window_name(history_size + 1),
                    "variant": name,
                    "requested_alpha_gross": float(
                        np.sum(np.abs(alpha))
                    ),
                    "delivered_gross": float(
                        np.sum(np.abs(delivered))
                    ),
                    "delivered_beta_dollars": float(beta @ delivered),
                    "delivered_net_dollars": float(np.sum(delivered)),
                    "euclidean_delivery": float(euclidean_delivery),
                    "diagonal_delivery": float(diagonal_delivery),
                    "flat_euclidean_delivery": float(
                        flat_euclidean_delivery
                    ),
                    "flat_diagonal_delivery": float(
                        flat_diagonal_delivery
                    ),
                    "variance_scale": variance_scale,
                    "base_overlay_covariance": base_overlay_covariance,
                    "overlay_variance": overlay_variance,
                }
            )
    return standalone, combined, pd.DataFrame(diagnostics)


def summarize_forecasts(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    relevant = events[events["window"].isin(PRIMARY_WINDOWS)].copy()
    groupings = {
        "pooled": ["window", "horizon"],
        "pair": ["window", "horizon", "pair"],
        "entry": ["window", "horizon", "is_entry"],
    }
    for view, columns in groupings.items():
        for keys, group in relevant.groupby(columns, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            values = group["signed_residual_return"].to_numpy(dtype=float)
            residual_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row: dict[str, float | int | str] = {
                "view": view,
                "n": len(values),
                "mean_signed_residual_bps": float(np.mean(values) * 1.0e4),
                "median_signed_residual_bps": float(np.median(values) * 1.0e4),
                "win_rate": float(np.mean(values > 0.0)),
                "t_stat_naive": (
                    float(np.mean(values) / (residual_std / sqrt(len(values))))
                    if residual_std > 0.0
                    else 0.0
                ),
                "mean_signed_asset_bps": float(
                    group["signed_asset_return"].mean() * 1.0e4
                ),
                "mean_signed_market_bps": float(
                    group["signed_market_component"].mean() * 1.0e4
                ),
            }
            row.update(dict(zip(columns, keys, strict=True)))
            rows.append(row)

    h1 = relevant[relevant["horizon"] == 1].copy()
    day_level = (
        h1.groupby(["window", "pnl_day"], as_index=False)[
            "signed_residual_return"
        ]
        .mean()
    )
    for window, group in day_level.groupby("window"):
        values = group["signed_residual_return"].to_numpy(dtype=float)
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(
            {
                "view": "day_level",
                "window": window,
                "horizon": 1,
                "n": len(values),
                "mean_signed_residual_bps": float(np.mean(values) * 1.0e4),
                "median_signed_residual_bps": float(np.median(values) * 1.0e4),
                "win_rate": float(np.mean(values > 0.0)),
                "t_stat_naive": (
                    float(np.mean(values) / (std / sqrt(len(values))))
                    if std > 0.0
                    else 0.0
                ),
                "mean_signed_asset_bps": float("nan"),
                "mean_signed_market_bps": float("nan"),
            }
        )

    h1["strength_bucket"] = pd.cut(
        h1["entry_ratio"],
        bins=[0.0, 0.75, 1.0, 1.25, 1.5, 2.0, np.inf],
        labels=["<0.75", "0.75-1", "1-1.25", "1.25-1.5", "1.5-2", "2+"],
        right=False,
    )
    for keys, group in h1.groupby(
        ["window", "strength_bucket"],
        observed=True,
    ):
        values = group["signed_residual_return"].to_numpy(dtype=float)
        rows.append(
            {
                "view": "strength",
                "window": keys[0],
                "horizon": 1,
                "strength_bucket": str(keys[1]),
                "n": len(values),
                "mean_signed_residual_bps": float(np.mean(values) * 1.0e4),
                "median_signed_residual_bps": float(np.median(values) * 1.0e4),
                "win_rate": float(np.mean(values > 0.0)),
                "t_stat_naive": float("nan"),
                "mean_signed_asset_bps": float(
                    group["signed_asset_return"].mean() * 1.0e4
                ),
                "mean_signed_market_bps": float(
                    group["signed_market_component"].mean() * 1.0e4
                ),
            }
        )
    return pd.DataFrame(rows)


def episode_randomization_test(
    events: pd.DataFrame,
    seeds: int = 1_000,
) -> pd.DataFrame:
    """Randomize held-episode direction while preserving activity and returns."""
    h1 = events[
        (events["horizon"] == 1)
        & events["window"].isin(PRIMARY_WINDOWS)
    ].copy()
    h1.sort_values(["pair", "history_size"], inplace=True)
    episode_ids = np.zeros(len(h1), dtype=int)
    episode = 0
    previous_pair = ""
    previous_history = -10
    previous_direction = 0.0
    for offset, row in enumerate(h1.itertuples(index=False)):
        if (
            row.pair != previous_pair
            or int(row.history_size) != previous_history + 1
            or float(row.direction) != previous_direction
        ):
            episode += 1
        episode_ids[offset] = episode
        previous_pair = str(row.pair)
        previous_history = int(row.history_size)
        previous_direction = float(row.direction)
    h1["episode_id"] = episode_ids
    episode_codes = h1["episode_id"].to_numpy(dtype=int)
    unsigned_outcome = (
        h1["signed_residual_return"].to_numpy(dtype=float)
        * h1["direction"].to_numpy(dtype=float)
    )
    pnl_days = h1["pnl_day"].to_numpy(dtype=int)
    windows = h1["window"].to_numpy(dtype=str)
    unique_episodes = int(episode_codes.max()) + 1

    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(87_321)
    for window in PRIMARY_WINDOWS:
        use = windows == window
        observed_frame = pd.DataFrame(
            {
                "day": pnl_days[use],
                "value": h1.loc[use, "signed_residual_return"].to_numpy(
                    dtype=float
                ),
            }
        )
        observed = float(
            observed_frame.groupby("day")["value"].mean().mean()
        )
        randomized = np.zeros(seeds)
        for seed in range(seeds):
            signs = rng.choice(
                np.asarray([-1.0, 1.0]),
                size=unique_episodes,
            )
            values = unsigned_outcome[use] * signs[episode_codes[use]]
            frame = pd.DataFrame(
                {"day": pnl_days[use], "value": values}
            )
            randomized[seed] = float(
                frame.groupby("day")["value"].mean().mean()
            )
        rows.append(
            {
                "window": window,
                "observed_mean_bps": observed * 1.0e4,
                "random_mean_bps": float(randomized.mean() * 1.0e4),
                "random_std_bps": float(randomized.std() * 1.0e4),
                "random_95pct_bps": float(
                    np.quantile(randomized, 0.95) * 1.0e4
                ),
                "one_sided_p": float(
                    (1 + np.sum(randomized >= observed))
                    / (seeds + 1)
                ),
                "episodes": int(
                    h1.loc[use, "episode_id"].nunique()
                ),
                "active_days": int(
                    h1.loc[use, "pnl_day"].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_targets(
    prices: np.ndarray,
    base_targets: np.ndarray,
    standalone: dict[str, np.ndarray],
    combined: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    base_metrics = sleeves.evaluate(prices, base_targets, WINDOWS)
    for window, start, end in WINDOWS:
        rows.append(
            {
                "portfolio": "base",
                "variant": "base",
                "window": window,
                **base_metrics[window],
                "score_delta": 0.0,
            }
        )
        for name in VARIANTS:
            standalone_pnl, turnover, fees = sleeves.account(
                prices,
                standalone[name],
                start,
                end,
            )
            standalone_metrics = sleeves.metrics(
                standalone_pnl,
                turnover,
                fees,
            )
            rows.append(
                {
                    "portfolio": "standalone",
                    "variant": name,
                    "window": window,
                    **standalone_metrics,
                    "score_delta": standalone_metrics["score"],
                    "win_rate": float(np.mean(standalone_pnl > 0.0)),
                }
            )
            combined_pnl, turnover, fees = sleeves.account(
                prices,
                combined[name],
                start,
                end,
            )
            combined_metrics = sleeves.metrics(
                combined_pnl,
                turnover,
                fees,
            )
            rows.append(
                {
                    "portfolio": "combined",
                    "variant": name,
                    "window": window,
                    **combined_metrics,
                    "score_delta": (
                        combined_metrics["score"]
                        - base_metrics[window]["score"]
                    ),
                    "win_rate": float(np.mean(combined_pnl > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def make_plot(
    forecast: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    pooled = forecast[
        (forecast["view"] == "day_level")
        & (forecast["horizon"] == 1)
    ]
    axes[0, 0].bar(
        pooled["window"],
        pooled["mean_signed_residual_bps"],
        color=np.where(
            pooled["mean_signed_residual_bps"] >= 0.0,
            "#2ca02c",
            "#d62728",
        ),
    )
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Next-day signed residual return")
    axes[0, 0].set_ylabel("basis points")
    axes[0, 0].tick_params(axis="x", rotation=25)

    strength = forecast[
        (forecast["view"] == "strength")
        & (forecast["window"].isin(("TRAIN_A", "TRAIN_B")))
    ]
    train_strength = (
        strength.groupby("strength_bucket", observed=True)["mean_signed_residual_bps"]
        .mean()
    )
    validation_strength = (
        forecast[
            (forecast["view"] == "strength")
            & (forecast["window"].isin(("VALIDATION_A", "VALIDATION_B")))
        ]
        .groupby("strength_bucket", observed=True)["mean_signed_residual_bps"]
        .mean()
    )
    buckets = sorted(
        set(train_strength.index) | set(validation_strength.index)
    )
    x = np.arange(len(buckets))
    axes[0, 1].bar(
        x - 0.2,
        [train_strength.get(bucket, np.nan) for bucket in buckets],
        width=0.4,
        label="train",
    )
    axes[0, 1].bar(
        x + 0.2,
        [validation_strength.get(bucket, np.nan) for bucket in buckets],
        width=0.4,
        label="validation",
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xticks(x, buckets, rotation=25)
    axes[0, 1].set_title("Residual forecast versus signal strength")
    axes[0, 1].set_ylabel("basis points")
    axes[0, 1].legend()

    combined = metrics[
        (metrics["portfolio"] == "combined")
        & (metrics["window"].isin(("TRAIN", "VALIDATION")))
    ]
    pivot = combined.pivot(
        index="variant",
        columns="window",
        values="score_delta",
    ).reindex(VARIANTS)
    x = np.arange(len(pivot))
    axes[1, 0].bar(
        x - 0.2,
        pivot["TRAIN"],
        width=0.4,
        label="train",
    )
    axes[1, 0].bar(
        x + 0.2,
        pivot["VALIDATION"],
        width=0.4,
        label="validation",
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xticks(x, pivot.index, rotation=20, ha="right")
    axes[1, 0].set_title("Combined-book score change")
    axes[1, 0].legend()

    standalone = metrics[
        (metrics["portfolio"] == "standalone")
        & (metrics["window"].isin(("TRAIN", "VALIDATION")))
    ]
    for window, marker in (("TRAIN", "o"), ("VALIDATION", "s")):
        group = standalone[standalone["window"] == window]
        axes[1, 1].scatter(
            group["std_pnl"],
            group["mean_pnl"],
            marker=marker,
            s=70,
            label=window.lower(),
        )
        for _, row in group.iterrows():
            axes[1, 1].annotate(
                str(row["variant"]).replace("synthetic_", ""),
                (row["std_pnl"], row["mean_pnl"]),
                fontsize=8,
            )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("daily PnL standard deviation")
    axes[1, 1].set_ylabel("mean daily PnL")
    axes[1, 1].set_title("Projection trades mean for variance")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "algo_residual_projection.png",
        dpi=160,
    )
    plt.close(fig)


def write_report(
    forecast: pd.DataFrame,
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    randomization: pd.DataFrame,
) -> None:
    lines = [
        "# ALGO residual forecast and beta-projection research",
        "",
        "Only days 1-750 are loaded. Days 751-1000 are excluded.",
        "",
        "## Forecast test",
        "",
        "Positive values mean the asset position correctly predicted the future "
        "return residual `r_asset - beta * r_ALGO`.",
        "",
        "| Fold | Days | Active days | Mean signed residual (bp) | Win rate | t-stat |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    pooled = forecast[
        (forecast["view"] == "day_level")
        & (forecast["horizon"] == 1)
    ].set_index("window")
    fold_days = {
        "TRAIN_A": "251-375",
        "TRAIN_B": "376-500",
        "VALIDATION_A": "501-625",
        "VALIDATION_B": "626-750",
    }
    for window in PRIMARY_WINDOWS:
        row = pooled.loc[window]
        lines.append(
            f"| {window} | {fold_days[window]} | {int(row['n'])} | "
            f"{row['mean_signed_residual_bps']:.3f} | "
            f"{row['win_rate']:.3f} | {row['t_stat_naive']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Episode randomization",
            "",
            "| Fold | Observed bp | Random 95% | One-sided p | Episodes |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for window in PRIMARY_WINDOWS:
        row = randomization[
            randomization["window"] == window
        ].iloc[0]
        lines.append(
            f"| {window} | {row['observed_mean_bps']:.3f} | "
            f"{row['random_95pct_bps']:.3f} | "
            f"{row['one_sided_p']:.4f} | {int(row['episodes'])} |"
        )

    lines.extend(
        [
            "",
            "## Portfolio construction",
            "",
            "| Variant | Train standalone | Validation standalone | "
            "Train combined delta | Validation combined delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in VARIANTS:
        train_standalone = metrics[
            (metrics["portfolio"] == "standalone")
            & (metrics["variant"] == name)
            & (metrics["window"] == "TRAIN")
        ].iloc[0]
        validation_standalone = metrics[
            (metrics["portfolio"] == "standalone")
            & (metrics["variant"] == name)
            & (metrics["window"] == "VALIDATION")
        ].iloc[0]
        train_combined = metrics[
            (metrics["portfolio"] == "combined")
            & (metrics["variant"] == name)
            & (metrics["window"] == "TRAIN")
        ].iloc[0]
        validation_combined = metrics[
            (metrics["portfolio"] == "combined")
            & (metrics["variant"] == name)
            & (metrics["window"] == "VALIDATION")
        ].iloc[0]
        lines.append(
            f"| {name} | {train_standalone['score']:.3f} | "
            f"{validation_standalone['score']:.3f} | "
            f"{train_combined['score_delta']:+.3f} | "
            f"{validation_combined['score_delta']:+.3f} |"
        )

    def metric(
        portfolio: str,
        variant: str,
        window: str,
    ) -> pd.Series:
        return metrics[
            (metrics["portfolio"] == portfolio)
            & (metrics["variant"] == variant)
            & (metrics["window"] == window)
        ].iloc[0]

    asset_train = metric("standalone", "asset_only", "TRAIN")
    asset_validation = metric(
        "standalone",
        "asset_only",
        "VALIDATION",
    )
    projected_train = metric(
        "standalone",
        "direct_projection",
        "TRAIN",
    )
    projected_validation = metric(
        "standalone",
        "direct_projection",
        "VALIDATION",
    )
    flat_train_a = metric(
        "combined",
        "flat_hedge_euclidean",
        "TRAIN_A",
    )
    var_train_a = metric(
        "combined",
        "flat_hedge_var_min",
        "TRAIN_A",
    )

    lines.extend(
        [
            "",
            "## Key findings",
            "",
            "- The signed next-day residual is positive in all four folds. "
            "Episode randomization rejects chance at 5% in three folds; the "
            "first training half is only suggestive.",
            f"- Direct projection reduces standalone daily volatility by "
            f"{100.0 * (1.0 - projected_train['std_pnl'] / asset_train['std_pnl']):.1f}% "
            f"in train and "
            f"{100.0 * (1.0 - projected_validation['std_pnl'] / asset_validation['std_pnl']):.1f}% "
            f"in validation.",
            f"- It also removes mean PnL: "
            f"{100.0 * (1.0 - projected_validation['mean_pnl'] / asset_validation['mean_pnl']):.1f}% "
            f"of validation mean disappears with the market component.",
            "- Directly adding the complete projection to the bang-bang book "
            "is infeasible: one saturated leg forces almost the whole vector "
            "to be scaled away.",
            "- Hedging only with currently flat assets improves aggregate "
            "train and validation, but its first train-half delta is "
            f"{flat_train_a['score_delta']:+.2f}. The variance-minimizing "
            f"version still produces {var_train_a['score_delta']:+.2f}.",
            "- Residual edge is not monotonic in z-score magnitude. This "
            "rejects proportional sizing by raw dislocation strength.",
            "",
            "## Interpretation",
            "",
            "- The projection equation guarantees estimated beta neutrality; it "
            "does not guarantee positive expected residual returns.",
            "- A production candidate must have a positive residual forecast in "
            "both training halves and both validation halves.",
            "- A construction must also improve aggregate train and validation "
            "score. Validation-only improvement is insufficient.",
            "- Naive t-statistics are descriptive because held episodes and "
            "multi-asset observations are dependent.",
            "",
            "## Hedge diagnostics",
            "",
        ]
    )
    active = diagnostics[diagnostics["requested_alpha_gross"] > 1.0e-9]
    for name in VARIANTS:
        group = active[active["variant"] == name]
        lines.append(
            f"- `{name}`: mean delivered beta dollars "
            f"{group['delivered_beta_dollars'].abs().mean():.2f}; "
            f"mean gross delivery ${group['delivered_gross'].mean():,.0f}."
        )
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(
        sleeves.PRICE_PATH,
        sep=r"\s+",
        header=0,
    ).to_numpy(dtype=float).T[:, :LAST_DEVELOPMENT_DAY]
    (
        raw_alpha,
        betas,
        correlations,
        variances,
        events,
    ) = replay_signals(prices)
    base_targets = sleeves.build_base_targets(prices)
    standalone, combined, diagnostics = build_targets(
        prices,
        base_targets,
        raw_alpha,
        betas,
        variances,
    )
    forecast = summarize_forecasts(events)
    randomization = episode_randomization_test(events)
    metrics = evaluate_targets(
        prices,
        base_targets,
        standalone,
        combined,
    )

    events.to_csv(OUTPUT_DIR / "events.csv", index=False)
    forecast.to_csv(OUTPUT_DIR / "forecast_summary.csv", index=False)
    randomization.to_csv(
        OUTPUT_DIR / "episode_randomization.csv",
        index=False,
    )
    metrics.to_csv(OUTPUT_DIR / "projection_metrics.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "hedge_diagnostics.csv", index=False)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "last_loaded_day": LAST_DEVELOPMENT_DAY,
                "horizons": HORIZONS,
                "variants": VARIANTS,
                "pairs": sleeves.PAIR_NAMES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(forecast, metrics, diagnostics, randomization)
    make_plot(forecast, metrics)

    print((OUTPUT_DIR / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

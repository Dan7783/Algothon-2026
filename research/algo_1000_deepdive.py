#!/usr/bin/env python
"""Comprehensive 1,000-day investigation of instrument 0 (ALGO).

Questions:

1. Is ALGO directly predictable from its own history or the other 50 assets?
2. Does ALGO lead returns elsewhere in the universe?
3. Is ALGO useful as a hedge for the current Four Sigmas book?
4. Do any simple, causal ALGO overlays improve the current portfolio?

Protocol:

* model families and features are fixed in this file;
* forecasts are expanding-window and causal;
* days 251-500 are training diagnostics;
* days 501-750 select among train-stable candidates;
* days 751-1000 are already exposed and reported only descriptively.

The script does not modify the production strategy.
"""

from __future__ import annotations

import hashlib
import json
import sys
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy


OUTPUT_DIR = ROOT / "analysis_outputs" / "algo_1000_deepdive"
PRICE_PATH = ROOT / "prices.txt"
BASE_CACHE = OUTPUT_DIR / "base_targets.npz"
ALGO = strategy.I["ALGO"]
ALGO_CAP = 100_000.0
DIRECT_CAP = 25_000.0
COMMISSIONS = np.full(strategy.N_INST, 1.0e-4)
COMMISSIONS[ALGO] = 2.0e-5
LIMITS = np.full(strategy.N_INST, strategy.LEG)
LIMITS[ALGO] = ALGO_CAP

BLOCKS = {
    "train_a_251_375": (251, 375),
    "train_b_376_500": (376, 500),
    "train_251_500": (251, 500),
    "validation_a_501_625": (501, 625),
    "validation_b_626_750": (626, 750),
    "validation_501_750": (501, 750),
    "contaminated_a_751_875": (751, 875),
    "contaminated_b_876_1000": (876, 1000),
    "contaminated_751_1000": (751, 1000),
}
MAIN_WINDOWS = (
    "train_251_500",
    "validation_501_750",
    "contaminated_751_1000",
)


def portfolio_score(pnl: np.ndarray) -> float:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    if mean <= 0.0 or std < 1.0e-12:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe * sharpe / (sharpe * sharpe + 1.0)


def account(
    prices: np.ndarray,
    targets: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, float, float]:
    """Match eval.py's delayed commission and final mark timing."""
    cash = 0.0
    value = 0.0
    prior_fee = 0.0
    current = np.zeros(prices.shape[0])
    pnl: list[float] = []
    turnover = 0.0
    fees = 0.0
    for day in range(start - 1, end + 1):
        current_prices = prices[:, day - 1]
        new = targets[day].astype(float) if day < end else current.copy()
        delta = new - current
        cash -= float(current_prices @ delta) + prior_fee
        volume = current_prices * np.abs(delta)
        prior_fee = float(volume @ COMMISSIONS)
        fees += prior_fee
        turnover += float(volume.sum())
        current = new
        new_value = cash + float(current @ current_prices)
        if day >= start:
            pnl.append(new_value - value)
        value = new_value
    return np.asarray(pnl), turnover, fees


def metrics(
    pnl: np.ndarray,
    turnover: float,
    fees: float,
) -> dict[str, float]:
    mean = float(pnl.mean())
    std = float(pnl.std())
    return {
        "score": portfolio_score(pnl),
        "mean_pnl": mean,
        "std_pnl": std,
        "sharpe": sqrt(250.0) * mean / std if std > 0.0 else 0.0,
        "total_pnl": float(pnl.sum()),
        "positive_day_rate": float(np.mean(pnl > 0.0)),
        "turnover": float(turnover),
        "fees": float(fees),
    }


def source_fingerprint() -> str:
    return hashlib.sha256(
        (ROOT / "four_sigmas.py").read_bytes()
    ).hexdigest()


def build_base_targets(prices: np.ndarray) -> np.ndarray:
    """Replay the current production strategy once."""
    fingerprint = source_fingerprint()
    if BASE_CACHE.exists():
        with np.load(BASE_CACHE, allow_pickle=False) as loaded:
            cached_fingerprint = str(loaded["fingerprint"].item())
            if cached_fingerprint == fingerprint:
                return loaded["targets"]

    strategy.reset()
    n_days = prices.shape[1]
    targets = np.zeros((n_days + 1, prices.shape[0]), dtype=int)
    for history_size in range(1, n_days):
        requested = strategy.getMyPosition(prices[:, :history_size])
        current = prices[:, history_size - 1]
        limits = (LIMITS / current).astype(int)
        targets[history_size] = np.clip(
            requested,
            -limits,
            limits,
        ).astype(int)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BASE_CACHE,
        targets=targets,
        fingerprint=np.asarray(fingerprint),
    )
    return targets


def make_feature_data(
    prices: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create features available immediately before each target return."""
    log_returns = np.diff(np.log(prices), axis=1)
    rows: list[dict[str, float | int]] = []
    cross_rows: list[np.ndarray] = []
    for day in range(126, prices.shape[1]):
        # At history size ``day``, return column day-2 is the latest known
        # movement and column day-1 is the movement being forecast.
        known = log_returns[:, : day - 1]
        last = known[:, -1]
        market = known[1:].mean(axis=0)
        algo = known[ALGO]

        def lag1(values: np.ndarray, window: int) -> float:
            sample = values[-(window + 1) :]
            return float(np.corrcoef(sample[:-1], sample[1:])[0, 1])

        rows.append(
            {
                "day": day,
                "algo_r1": float(algo[-1]),
                "algo_r2": float(algo[-2:].sum()),
                "algo_r5": float(algo[-5:].sum()),
                "algo_r20": float(algo[-20:].sum()),
                "abs_algo_r1": float(abs(algo[-1])),
                "algo_vol5": float(algo[-5:].std()),
                "algo_vol20": float(algo[-20:].std()),
                "algo_vol60": float(algo[-60:].std()),
                "market_r1": float(market[-1]),
                "market_r5": float(market[-5:].sum()),
                "market_r20": float(market[-20:].sum()),
                "breadth_r1": float(np.mean(last[1:] > 0.0)),
                "dispersion_r1": float(last[1:].std()),
                "relative_r1": float(algo[-1] - market[-1]),
                "relative_r5": float(
                    algo[-5:].sum() - market[-5:].sum()
                ),
                "algo_acf60": lag1(algo, 60),
                "algo_acf125": lag1(algo, 125),
                "market_acf60": lag1(market, 60),
                "market_acf125": lag1(market, 125),
                "target_algo": float(log_returns[ALGO, day - 1]),
                "target_market": float(
                    log_returns[1:, day - 1].mean()
                ),
            }
        )
        cross_rows.append(last[1:].copy())
    return pd.DataFrame(rows), np.asarray(cross_rows)


COMPACT_FEATURES = (
    "algo_r1",
    "algo_r2",
    "algo_r5",
    "algo_r20",
    "abs_algo_r1",
    "algo_vol5",
    "algo_vol20",
    "algo_vol60",
    "market_r1",
    "market_r5",
    "market_r20",
    "breadth_r1",
    "dispersion_r1",
    "relative_r1",
    "relative_r5",
    "algo_acf60",
    "algo_acf125",
    "market_acf60",
    "market_acf125",
)


def ridge_one_step(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_current: np.ndarray,
    ridge_lambda: float = 1.0,
) -> float:
    """Standardized ridge with penalty scaled by sample count."""
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0) + 1.0e-12
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) + 1.0e-12
    x = (x_train - x_mean) / x_scale
    y = (y_train - y_mean) / y_scale
    coefficients = np.linalg.solve(
        x.T @ x
        + ridge_lambda * len(x) * np.eye(x.shape[1]),
        x.T @ y,
    )
    standardized_prediction = (
        (x_current - x_mean) / x_scale
    ) @ coefficients
    return float(y_mean + y_scale * standardized_prediction)


def causal_forecasts(
    features: pd.DataFrame,
    cross_features: np.ndarray,
) -> pd.DataFrame:
    """Generate simple and expanding-ridge forecasts for ALGO."""
    result = features[["day", "target_algo"]].copy()
    result["always_long"] = 1.0
    result["algo_momentum_1"] = features["algo_r1"]
    result["algo_reversal_1"] = -features["algo_r1"]
    result["market_lead"] = features["market_r1"]
    result["market_reversal"] = -features["market_r1"]
    result["relative_reversal"] = -features["relative_r1"]
    result["adaptive_algo_acf60"] = (
        np.sign(features["algo_acf60"]) * features["algo_r1"]
    )
    result["adaptive_algo_acf125"] = (
        np.sign(features["algo_acf125"]) * features["algo_r1"]
    )
    result["adaptive_market_acf60"] = (
        np.sign(features["market_acf60"]) * features["market_r1"]
    )
    result["adaptive_market_acf125"] = (
        np.sign(features["market_acf125"]) * features["market_r1"]
    )
    result["shock_reversal"] = 0.0
    result["adaptive_shock_acf60"] = 0.0
    result["adaptive_shock_acf125"] = 0.0
    result["own_ar1"] = 0.0
    result["compact_ridge"] = 0.0
    result["cross_asset_ridge"] = 0.0
    result["full_ridge"] = 0.0

    compact = features.loc[:, COMPACT_FEATURES].to_numpy(float)
    target = features["target_algo"].to_numpy(float)
    days = features["day"].to_numpy(int)
    for row, day in enumerate(days):
        prior = (days < day) & (days >= 126)
        if int(prior.sum()) < 100 or day < 251:
            continue
        threshold = float(
            np.quantile(
                np.abs(features.loc[prior, "algo_r1"]),
                0.90,
            )
        )
        if abs(float(features.loc[row, "algo_r1"])) >= threshold:
            result.loc[row, "shock_reversal"] = -float(
                features.loc[row, "algo_r1"]
            )
            result.loc[row, "adaptive_shock_acf60"] = (
                np.sign(float(features.loc[row, "algo_acf60"]))
                * float(features.loc[row, "algo_r1"])
            )
            result.loc[row, "adaptive_shock_acf125"] = (
                np.sign(float(features.loc[row, "algo_acf125"]))
                * float(features.loc[row, "algo_r1"])
            )

        result.loc[row, "own_ar1"] = ridge_one_step(
            compact[prior, :1],
            target[prior],
            compact[row, :1],
        )
        result.loc[row, "compact_ridge"] = ridge_one_step(
            compact[prior],
            target[prior],
            compact[row],
        )
        result.loc[row, "cross_asset_ridge"] = ridge_one_step(
            cross_features[prior],
            target[prior],
            cross_features[row],
        )
        full_train = np.concatenate(
            [compact[prior], cross_features[prior]],
            axis=1,
        )
        full_current = np.concatenate(
            [compact[row], cross_features[row]],
        )
        result.loc[row, "full_ridge"] = ridge_one_step(
            full_train,
            target[prior],
            full_current,
        )
    return result


def algo_targets_from_forecast(
    prices: np.ndarray,
    base_targets: np.ndarray,
    forecast: pd.Series,
    days: pd.Series,
    cap: float = DIRECT_CAP,
) -> np.ndarray:
    targets = base_targets.copy()
    for day, value in zip(days.to_numpy(int), forecast.to_numpy(float)):
        if day >= prices.shape[1] or value == 0.0:
            continue
        dollars = cap * np.sign(value)
        limit = int(ALGO_CAP / prices[ALGO, day - 1])
        requested = int(np.round(dollars / prices[ALGO, day - 1]))
        targets[day, ALGO] = int(np.clip(requested, -limit, limit))
    return targets


def rolling_factor_hedge_targets(
    prices: np.ndarray,
    base_targets: np.ndarray,
    window: int = 60,
    hedge_ratio: float = 0.5,
    cap: float = 100_000.0,
) -> np.ndarray:
    """Recreate the previously selected ALGO beta hedge on the current book."""
    targets = base_targets.copy()
    log_returns = np.diff(np.log(prices), axis=1)
    for day in range(window + 1, prices.shape[1]):
        known = log_returns[:, day - 1 - window : day - 1]
        algo = known[ALGO] - known[ALGO].mean()
        others = known[1:] - known[1:].mean(axis=1, keepdims=True)
        variance = float(algo @ algo)
        if variance <= 1.0e-14:
            continue
        beta = (others @ algo) / variance
        beta = 0.5 * beta + 0.5
        beta = np.clip(beta, -1.0, 3.0)
        current_dollars = (
            base_targets[day, 1:] * prices[1:, day - 1]
        )
        loading = float(current_dollars @ beta)
        dollars = float(
            np.clip(-hedge_ratio * loading, -cap, cap)
        )
        limit = int(ALGO_CAP / prices[ALGO, day - 1])
        requested = int(np.round(dollars / prices[ALGO, day - 1]))
        targets[day, ALGO] = int(np.clip(requested, -limit, limit))
    return targets


def evaluate_candidates(
    prices: np.ndarray,
    base_targets: np.ndarray,
    forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    candidates: dict[str, np.ndarray] = {"baseline": base_targets}
    for name in forecasts.columns:
        if name in {"day", "target_algo"}:
            continue
        candidates[name] = algo_targets_from_forecast(
            prices,
            base_targets,
            forecasts[name],
            forecasts["day"],
        )
    candidates["factor_hedge_w60_h0.5_c100k"] = (
        rolling_factor_hedge_targets(prices, base_targets)
    )

    rows: list[dict[str, object]] = []
    pnl_cache: dict[str, np.ndarray] = {}
    for name, targets in candidates.items():
        for window, (start, end) in BLOCKS.items():
            pnl, turnover, fees = account(prices, targets, start, end)
            values = metrics(pnl, turnover, fees)
            rows.append(
                {
                    "candidate": name,
                    "window": window,
                    **values,
                }
            )
            if window in MAIN_WINDOWS:
                pnl_cache[f"{name}:{window}"] = pnl
    results = pd.DataFrame(rows)
    baseline = (
        results[results["candidate"] == "baseline"]
        .set_index("window")["score"]
    )
    results["score_delta"] = results.apply(
        lambda row: float(row["score"] - baseline[row["window"]]),
        axis=1,
    )
    return results, pnl_cache


def select_candidate(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    pivot = results.pivot(
        index="candidate",
        columns="window",
        values="score_delta",
    )
    pivot["train_floor"] = pivot[
        ["train_a_251_375", "train_b_376_500"]
    ].min(axis=1)
    pivot["validation_floor"] = pivot[
        ["validation_a_501_625", "validation_b_626_750"]
    ].min(axis=1)
    eligible = pivot[
        (pivot.index != "baseline")
        & (pivot["train_floor"] > 0.0)
        & (pivot["validation_floor"] > 0.0)
    ]
    if eligible.empty:
        return "baseline", pivot.sort_values(
            ["validation_floor", "train_floor"],
            ascending=False,
        )
    selected = str(
        eligible.sort_values(
            ["validation_floor", "train_floor"],
            ascending=False,
        ).index[0]
    )
    return selected, pivot.sort_values(
        ["validation_floor", "train_floor"],
        ascending=False,
    )


def forecast_diagnostics(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in forecasts.columns:
        if name in {"day", "target_algo"}:
            continue
        for window in MAIN_WINDOWS:
            start, end = BLOCKS[window]
            sample = forecasts[
                forecasts["day"].between(start, end - 1)
            ]
            prediction = sample[name].to_numpy(float)
            target = sample["target_algo"].to_numpy(float)
            active = prediction != 0.0
            prediction = prediction[active]
            target = target[active]
            if len(target) == 0:
                continue
            hits = np.sign(prediction) == np.sign(target)
            move_weighted = float(
                np.sum(np.abs(target) * hits) / np.sum(np.abs(target))
            )
            corr = float(np.corrcoef(prediction, target)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
            rows.append(
                {
                    "forecast": name,
                    "window": window,
                    "n": int(len(target)),
                    "hit_rate": float(hits.mean()),
                    "move_weighted_hit_rate": move_weighted,
                    "forecast_return_corr": corr,
                    "mean_signed_return_bps": float(
                        10_000.0
                        * np.mean(np.sign(prediction) * target)
                    ),
                }
            )
    return pd.DataFrame(rows)


def yearly_structure(
    prices: np.ndarray,
) -> pd.DataFrame:
    log_returns = np.diff(np.log(prices), axis=1)
    rows: list[dict[str, object]] = []
    for year, start, end in (
        ("year1", 1, 250),
        ("year2", 251, 500),
        ("year3", 501, 750),
        ("year4_contaminated", 751, 1000),
    ):
        sample = log_returns[:, start - 1 : end - 1]
        algo = sample[ALGO]
        market = sample[1:].mean(axis=0)
        variance = float(np.var(algo))
        beta = (
            float(np.cov(market, algo, ddof=0)[0, 1] / variance)
            if variance > 0.0
            else 0.0
        )
        standardized = sample[1:].T / (
            sample[1:].std(axis=1) + 1.0e-12
        )
        left, singular_values, _ = np.linalg.svd(
            standardized,
            full_matrices=False,
        )
        pc1 = left[:, 0] * singular_values[0]
        pc_corr = float(np.corrcoef(algo, pc1)[0, 1])
        rows.append(
            {
                "window": year,
                "algo_simple_return": float(
                    prices[ALGO, end - 1]
                    / prices[ALGO, start - 1]
                    - 1.0
                ),
                "algo_ann_vol": float(sqrt(250.0) * algo.std()),
                "algo_return_sharpe": float(
                    sqrt(250.0) * algo.mean() / algo.std()
                ),
                "algo_market_corr": float(
                    np.corrcoef(algo, market)[0, 1]
                ),
                "market_beta_to_algo": beta,
                "algo_pc1_abs_corr": abs(pc_corr),
                "pc1_variance_share": float(
                    singular_values[0] ** 2
                    / np.sum(singular_values ** 2)
                ),
                "mean_asset_algo_corr": float(
                    np.mean(
                        [
                            np.corrcoef(algo, sample[index])[0, 1]
                            for index in range(1, strategy.N_INST)
                        ]
                    )
                ),
                "algo_lag1_autocorr": float(
                    np.corrcoef(algo[:-1], algo[1:])[0, 1]
                ),
            }
        )
    return pd.DataFrame(rows)


def cross_asset_leads(
    prices: np.ndarray,
    features: pd.DataFrame,
) -> pd.DataFrame:
    log_returns = np.diff(np.log(prices), axis=1)
    symbols = {index: symbol for symbol, index in strategy.I.items()}
    rows: list[dict[str, object]] = []
    for index in range(1, strategy.N_INST):
        values: dict[str, float] = {}
        for window in MAIN_WINDOWS:
            start, end = BLOCKS[window]
            sample = features[
                features["day"].between(start, end - 1)
            ]
            days = sample["day"].to_numpy(int)
            lagged_algo = sample["algo_r1"].to_numpy(float)
            target = np.asarray(
                [log_returns[index, day - 1] for day in days]
            )
            corr = float(np.corrcoef(lagged_algo, target)[0, 1])
            values[window] = corr if np.isfinite(corr) else 0.0
        train_corr = values["train_251_500"]
        n = BLOCKS["train_251_500"][1] - BLOCKS["train_251_500"][0]
        t_stat = (
            train_corr
            * sqrt((n - 2) / max(1.0 - train_corr**2, 1.0e-12))
        )
        p_value = float(2.0 * stats.t.sf(abs(t_stat), n - 2))
        rows.append(
            {
                "symbol": symbols[index],
                "train_corr": train_corr,
                "validation_corr": values["validation_501_750"],
                "contaminated_corr": values[
                    "contaminated_751_1000"
                ],
                "train_t": t_stat,
                "train_p": p_value,
                "validation_aligned": float(
                    np.sign(train_corr)
                    * values["validation_501_750"]
                ),
                "contaminated_aligned": float(
                    np.sign(train_corr)
                    * values["contaminated_751_1000"]
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        "validation_aligned",
        ascending=False,
    )
    ordered = frame.sort_values("train_p").copy()
    m = len(ordered)
    ordered["bh_q"] = np.minimum.accumulate(
        (ordered["train_p"].to_numpy() * m / np.arange(1, m + 1))[::-1]
    )[::-1]
    return frame.merge(
        ordered[["symbol", "bh_q"]],
        on="symbol",
        how="left",
    )


def portfolio_algo_exposure(
    prices: np.ndarray,
    pnl_cache: dict[str, np.ndarray],
) -> pd.DataFrame:
    log_returns = np.diff(np.log(prices[ALGO]))
    rows: list[dict[str, object]] = []
    for window in MAIN_WINDOWS:
        start, end = BLOCKS[window]
        pnl = pnl_cache[f"baseline:{window}"]
        # Eval PnL observations run from the price movement ending on
        # ``start`` through the movement ending on ``end`` (inclusive).
        algo = log_returns[start - 2 : end - 1]
        variance = float(np.var(algo))
        covariance = float(np.cov(pnl, algo, ddof=0)[0, 1])
        rows.append(
            {
                "window": window,
                "portfolio_algo_corr": float(
                    np.corrcoef(pnl, algo)[0, 1]
                ),
                "equivalent_algo_beta_dollars": (
                    covariance / variance if variance > 0.0 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def random_direction_control(
    prices: np.ndarray,
    base_targets: np.ndarray,
    forecasts: pd.DataFrame,
    selected: str,
    seeds: int = 500,
) -> pd.DataFrame:
    if selected == "baseline" or selected == "factor_hedge_w60_h0.5_c100k":
        return pd.DataFrame()
    source = forecasts[selected].to_numpy(float)
    days = forecasts["day"].to_numpy(int)
    rows: list[dict[str, object]] = []
    observed_targets = algo_targets_from_forecast(
        prices,
        base_targets,
        forecasts[selected],
        forecasts["day"],
    )
    for window in MAIN_WINDOWS:
        start, end = BLOCKS[window]
        baseline_pnl, _, _ = account(
            prices, base_targets, start, end
        )
        baseline_score = portfolio_score(baseline_pnl)
        observed_pnl, _, _ = account(
            prices, observed_targets, start, end
        )
        observed_delta = portfolio_score(observed_pnl) - baseline_score
        random_deltas = []
        for seed in range(seeds):
            rng = np.random.default_rng(20_000 + seed)
            random_sign = np.where(
                rng.random(len(source)) < 0.5,
                -1.0,
                1.0,
            )
            random_values = np.where(source != 0.0, random_sign, 0.0)
            random_targets = algo_targets_from_forecast(
                prices,
                base_targets,
                pd.Series(random_values),
                forecasts["day"],
            )
            random_pnl, _, _ = account(
                prices, random_targets, start, end
            )
            random_deltas.append(
                portfolio_score(random_pnl) - baseline_score
            )
        random_array = np.asarray(random_deltas)
        rows.append(
            {
                "window": window,
                "candidate": selected,
                "observed_delta": observed_delta,
                "random_mean_delta": float(random_array.mean()),
                "random_std_delta": float(random_array.std()),
                "random_95pct": float(np.quantile(random_array, 0.95)),
                "randomization_p": float(
                    (1 + np.sum(random_array >= observed_delta))
                    / (seeds + 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", r"\|")

    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    rows.extend(
        "| "
        + " | ".join(render(value) for value in row)
        + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def make_plot(
    prices: np.ndarray,
    structure: pd.DataFrame,
    results: pd.DataFrame,
    selected: str,
    leads: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(prices[ALGO] / prices[ALGO, 0], color="black")
    for boundary in (250, 500, 750):
        axes[0, 0].axvline(boundary, color="grey", linestyle="--", alpha=0.6)
    axes[0, 0].set_title("ALGO normalized price")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].bar(
        structure["window"],
        structure["algo_market_corr"],
        color="#1f77b4",
    )
    axes[0, 1].set_title("Contemporaneous ALGO / equal-market correlation")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)

    chosen = results[
        results["candidate"].isin(["baseline", selected])
        & results["window"].isin(MAIN_WINDOWS)
    ].pivot(index="window", columns="candidate", values="score")
    chosen.plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("Current book versus selected ALGO overlay")
    axes[1, 0].tick_params(axis="x", rotation=20)

    top = leads.reindex(
        leads["validation_aligned"].abs().sort_values(
            ascending=False
        ).index
    ).head(12)
    axes[1, 1].barh(
        top["symbol"],
        top["validation_aligned"],
        color=np.where(
            top["contaminated_aligned"] > 0.0,
            "#2ca02c",
            "#d62728",
        ),
    )
    axes[1, 1].set_title(
        "ALGO lag → next asset return\n"
        "validation aligned corr; green survives contaminated window"
    )
    axes[1, 1].axvline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "algo_1000_deepdive.png", dpi=160)
    plt.close(fig)


def write_report(
    structure: pd.DataFrame,
    diagnostics: pd.DataFrame,
    results: pd.DataFrame,
    ranking: pd.DataFrame,
    selected: str,
    exposure: pd.DataFrame,
    leads: pd.DataFrame,
    random_control: pd.DataFrame,
) -> None:
    selected_results = results[
        results["candidate"].isin(["baseline", selected])
        & results["window"].isin(BLOCKS)
    ][
        [
            "candidate",
            "window",
            "score",
            "score_delta",
            "mean_pnl",
            "total_pnl",
            "fees",
        ]
    ]
    stable_leads = leads[
        (leads["validation_aligned"] > 0.0)
        & (leads["contaminated_aligned"] > 0.0)
    ].head(15)
    direct_main = diagnostics[
        diagnostics["window"].isin(MAIN_WINDOWS)
    ]
    lines = [
        "# ALGO 1,000-day deep dive",
        "",
        "## Protocol",
        "",
        "- Expanding, causal forecasts only.",
        "- Days 251-500: training stability.",
        "- Days 501-750: validation selection.",
        "- Days 751-1000: already exposed; descriptive only.",
        f"- Direct ALGO overlay size: ${DIRECT_CAP:,.0f}.",
        "",
        "## ALGO as a market/factor instrument",
        "",
        markdown_table(structure),
        "",
        "## Current book exposure to ALGO",
        "",
        markdown_table(exposure),
        "",
        "## Direct ALGO forecast diagnostics",
        "",
        markdown_table(direct_main),
        "",
        "## Candidate ranking",
        "",
        "A candidate must have positive score delta in both training halves "
        "and both validation halves. Selection uses validation floor; the "
        "contaminated final window is not consulted.",
        "",
        f"Selected candidate: `{selected}`",
        "",
        markdown_table(
            ranking.reset_index().rename(columns={"index": "candidate"})
        ),
        "",
        "## Selected overlay versus baseline",
        "",
        markdown_table(selected_results),
        "",
        "## Random-direction control",
        "",
        markdown_table(random_control),
        "",
        "## Does ALGO lead other assets?",
        "",
        "Correlations are between today's ALGO return and the next return of "
        "each asset. Validation/final values are aligned to the sign learned "
        "in training.",
        "",
        markdown_table(stable_leads),
        "",
        "## Interpretation guardrails",
        "",
        "- Contemporaneous correlation makes ALGO a plausible market proxy, "
        "but does not itself create a forecast.",
        "- A normalized or large model output is not a probability.",
        "- Fifty cross-asset lead tests create a multiple-testing burden; BH "
        "q-values are reported.",
        "- Any final-window improvement is not fresh out-of-sample evidence.",
    ]
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(
        PRICE_PATH,
        sep=r"\s+",
        header=0,
    ).to_numpy(float).T
    base_targets = build_base_targets(prices)
    features, cross_features = make_feature_data(prices)
    forecasts = causal_forecasts(features, cross_features)
    results, pnl_cache = evaluate_candidates(
        prices,
        base_targets,
        forecasts,
    )
    selected, ranking = select_candidate(results)
    diagnostics = forecast_diagnostics(forecasts)
    structure = yearly_structure(prices)
    leads = cross_asset_leads(prices, features)
    exposure = portfolio_algo_exposure(prices, pnl_cache)
    random_control = random_direction_control(
        prices,
        base_targets,
        forecasts,
        selected,
    )

    structure.to_csv(OUTPUT_DIR / "yearly_structure.csv", index=False)
    features.to_csv(OUTPUT_DIR / "causal_features.csv", index=False)
    forecasts.to_csv(OUTPUT_DIR / "causal_forecasts.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "forecast_diagnostics.csv", index=False)
    results.to_csv(OUTPUT_DIR / "candidate_results.csv", index=False)
    ranking.reset_index().to_csv(
        OUTPUT_DIR / "candidate_ranking.csv",
        index=False,
    )
    leads.to_csv(OUTPUT_DIR / "algo_cross_asset_leads.csv", index=False)
    exposure.to_csv(OUTPUT_DIR / "portfolio_algo_exposure.csv", index=False)
    random_control.to_csv(
        OUTPUT_DIR / "random_direction_control.csv",
        index=False,
    )
    write_report(
        structure,
        diagnostics,
        results,
        ranking,
        selected,
        exposure,
        leads,
        random_control,
    )
    make_plot(prices, structure, results, selected, leads)

    selected_rows = results[
        results["candidate"] == selected
    ].set_index("window")
    summary = {
        "selected_candidate": selected,
        "selected_score_deltas": {
            window: float(selected_rows.loc[window, "score_delta"])
            for window in BLOCKS
        },
        "stable_cross_asset_leads": int(
            (
                (leads["validation_aligned"] > 0.0)
                & (leads["contaminated_aligned"] > 0.0)
            ).sum()
        ),
        "production_modified": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()

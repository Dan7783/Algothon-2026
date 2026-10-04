"""Audit MMBT lag-1 return as a forecast of ALGO's 30-day return.

Primary causal equation:

    ALGO_log_return[t, t+30] = alpha + beta * MMBT_return[t-1, t] + error

Forward labels are embargoed at each 250-day boundary. Newey-West standard
errors use 29 lags because adjacent 30-day labels overlap heavily. Days
751-1000 are descriptive and are never used to select a specification.
"""

from __future__ import annotations

import json
import sys
from math import erfc, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "mmbt_algo_forward_ols"
HORIZON = 30
HAC_LAGS = HORIZON - 1
ALGO_NOTIONAL = 100_000.0
ALGO_COMMISSION = 2.0e-5
EPS = 1.0e-12

# Inclusive one-based anchor days. The forward endpoint remains inside its
# own chronological region, leaving a horizon-sized embargo at every end.
PRIMARY_WINDOWS = {
    "TRAIN_EARLY": (31, 220),
    "TRAIN_LATE": (251, 470),
    "VALIDATION": (501, 720),
    "EXPOSED": (751, 970),
}
TRAIN_WINDOWS = ("TRAIN_EARLY", "TRAIN_LATE")


def normal_two_sided_p(t_value: float) -> float:
    return erfc(abs(t_value) / sqrt(2.0))


def newey_west_fit(
    x: np.ndarray,
    y: np.ndarray,
    max_lags: int,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack((np.ones(len(x)), x))
    bread = np.linalg.inv(design.T @ design)
    coef = bread @ design.T @ y
    residual = y - design @ coef

    meat = np.zeros((2, 2))
    for t in range(len(x)):
        row = design[t][:, None]
        meat += residual[t] ** 2 * (row @ row.T)
    use_lags = min(max_lags, len(x) - 1)
    for lag in range(1, use_lags + 1):
        weight = 1.0 - lag / (use_lags + 1.0)
        cross = np.zeros((2, 2))
        for t in range(lag, len(x)):
            cross += (
                residual[t]
                * residual[t - lag]
                * np.outer(design[t], design[t - lag])
            )
        meat += weight * (cross + cross.T)
    covariance = bread @ meat @ bread
    covariance *= len(x) / max(len(x) - 2, 1)
    standard_error = np.sqrt(
        np.maximum(np.diag(covariance), 0.0)
    )
    t_values = np.divide(
        coef,
        standard_error,
        out=np.zeros_like(coef),
        where=standard_error > EPS,
    )
    fitted = design @ coef
    sst = float(np.sum((y - y.mean()) ** 2))
    r_squared = (
        1.0 - float(np.sum(residual**2)) / sst
        if sst > EPS
        else 0.0
    )
    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "alpha_se_hac": float(standard_error[0]),
        "beta_se_hac": float(standard_error[1]),
        "alpha_t_hac": float(t_values[0]),
        "beta_t_hac": float(t_values[1]),
        "beta_p_hac": normal_two_sided_p(float(t_values[1])),
        "r_squared": r_squared,
        "residual": residual,
        "fitted": fitted,
    }


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) <= EPS or np.std(y) <= EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def build_frame(
    prices: pd.DataFrame,
    horizon: int,
    feature_symbol: str = "MMBT",
    feature_lag: int = 0,
) -> pd.DataFrame:
    log_prices = np.log(prices)
    one_day = log_prices.diff()
    anchors = np.arange(1 + feature_lag, len(prices) - horizon)
    # At anchor t, lag zero is the latest completed return P[t]/P[t-1].
    feature = one_day[feature_symbol].to_numpy()[
        anchors - feature_lag
    ]
    target_log = (
        log_prices["ALGO"].to_numpy()[anchors + horizon]
        - log_prices["ALGO"].to_numpy()[anchors]
    )
    target_simple = (
        prices["ALGO"].to_numpy()[anchors + horizon]
        / prices["ALGO"].to_numpy()[anchors]
        - 1.0
    )
    return pd.DataFrame(
        {
            "anchor_day": anchors + 1,
            "feature": feature,
            "target_log": target_log,
            "target_simple": target_simple,
        }
    )


def window_sample(
    frame: pd.DataFrame,
    window: str,
) -> pd.DataFrame:
    start, end = PRIMARY_WINDOWS[window]
    return frame[frame["anchor_day"].between(start, end)].copy()


def training_sample(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [window_sample(frame, name) for name in TRAIN_WINDOWS],
        ignore_index=True,
    )


def standardize_from_train(
    train_x: np.ndarray,
    other_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(np.mean(train_x))
    scale = float(np.std(train_x))
    scale = max(scale, EPS)
    return (
        (train_x - mean) / scale,
        (other_x - mean) / scale,
        mean,
        scale,
    )


def evaluate_prediction(
    model: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    train_mean_y: float,
) -> dict[str, float]:
    prediction = model["alpha"] + model["beta"] * x
    error_model = float(np.sum((y - prediction) ** 2))
    error_baseline = float(np.sum((y - train_mean_y) ** 2))
    oos_r2 = (
        1.0 - error_model / error_baseline
        if error_baseline > EPS
        else 0.0
    )
    sign_model = np.sign(prediction)
    sign_baseline = np.sign(train_mean_y)
    feature_component = model["beta"] * x
    return {
        "n": len(y),
        "correlation": correlation(x, y),
        "prediction_target_correlation": correlation(prediction, y),
        "oos_r_squared": oos_r2,
        "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "baseline_rmse": float(
            np.sqrt(np.mean((y - train_mean_y) ** 2))
        ),
        "direction_accuracy": float(np.mean(sign_model == np.sign(y))),
        "baseline_direction_accuracy": float(
            np.mean(sign_baseline == np.sign(y))
        ),
        "prediction_long_share": float(np.mean(sign_model > 0.0)),
        "feature_flips_intercept_share": float(
            np.mean(np.sign(prediction) != np.sign(model["alpha"]))
        ),
        "mean_signed_forward_return": float(
            np.mean(sign_model * y)
        ),
        "mean_feature_only_signed_return": float(
            np.mean(np.sign(feature_component) * y)
        ),
    }


def overlapping_horizon_strategy(
    prices: pd.DataFrame,
    frame: pd.DataFrame,
    model: dict[str, Any],
    feature_mean: float,
    feature_scale: float,
    window: str,
) -> dict[str, float]:
    """Start 1/30 of an ALGO position daily and hold every sleeve 30 days."""
    sample = window_sample(frame, window)
    first = int(sample["anchor_day"].min())
    last_anchor = int(sample["anchor_day"].max())
    last_day = last_anchor + HORIZON
    signals = {
        int(row.anchor_day): float(
            np.sign(
                model["alpha"]
                + model["beta"]
                * ((row.feature - feature_mean) / feature_scale)
            )
        )
        for row in sample.itertuples()
    }
    daily_return = prices["ALGO"].pct_change().to_numpy()
    active_sleeves: list[tuple[int, float]] = []
    prior_position = 0.0
    pnl: list[float] = []
    turnover = 0.0
    for day in range(first, last_day + 1):
        active_sleeves = [
            (expiry, sign)
            for expiry, sign in active_sleeves
            if expiry >= day
        ]
        if day in signals:
            active_sleeves.append((day + HORIZON - 1, signals[day]))
        position = (
            ALGO_NOTIONAL
            * sum(sign for _, sign in active_sleeves)
            / HORIZON
        )
        position = float(
            np.clip(position, -ALGO_NOTIONAL, ALGO_NOTIONAL)
        )
        change = abs(position - prior_position)
        fee = ALGO_COMMISSION * change
        if day > first:
            pnl.append(
                prior_position * daily_return[day - 1] - fee
            )
        turnover += change
        prior_position = position
    pnl_array = np.asarray(pnl)
    equity = np.cumsum(pnl_array)
    high = np.maximum.accumulate(np.insert(equity, 0, 0.0))[1:]
    drawdown = high - equity
    mean = float(np.mean(pnl_array))
    std = float(np.std(pnl_array))
    return {
        "total_pnl": float(np.sum(pnl_array)),
        "annualized_sharpe": (
            sqrt(250.0) * mean / std if std > EPS else 0.0
        ),
        "max_drawdown": float(np.max(drawdown)),
        "daily_hit_rate": float(np.mean(pnl_array > 0.0)),
        "turnover": turnover,
        "fees": ALGO_COMMISSION * turnover,
        "mean_abs_position": float(
            ALGO_NOTIONAL
            * np.mean(
                [
                    abs(sum(
                        sign
                        for anchor, sign in signals.items()
                        if anchor <= day <= anchor + HORIZON - 1
                    ))
                    / HORIZON
                    for day in range(first, last_day + 1)
                ]
            )
        ),
    }


def primary_analysis(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    frame = build_frame(prices, HORIZON)
    train = training_sample(frame)
    train_x = train["feature"].to_numpy(float)
    train_y = train["target_log"].to_numpy(float)
    standardized_train, _, feature_mean, feature_scale = (
        standardize_from_train(train_x, train_x)
    )
    model = newey_west_fit(
        standardized_train,
        train_y,
        HAC_LAGS,
    )
    model_public = {
        key: value
        for key, value in model.items()
        if key not in {"residual", "fitted"}
    }
    model_public["feature_mean"] = feature_mean
    model_public["feature_scale"] = feature_scale
    model_public["beta_per_raw_return"] = (
        model["beta"] / feature_scale
    )

    rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    for window in (*TRAIN_WINDOWS, "VALIDATION", "EXPOSED"):
        sample = window_sample(frame, window)
        x = (
            sample["feature"].to_numpy(float) - feature_mean
        ) / feature_scale
        y = sample["target_log"].to_numpy(float)
        local = newey_west_fit(x, y, HAC_LAGS)
        evaluated = evaluate_prediction(
            model,
            x,
            y,
            float(np.mean(train_y)),
        )
        rows.append(
            {
                "window": window,
                **evaluated,
                "local_beta_standardized": local["beta"],
                "local_beta_t_hac": local["beta_t_hac"],
                "local_beta_p_hac": local["beta_p_hac"],
                "local_r_squared": local["r_squared"],
            }
        )
        if window in {"VALIDATION", "EXPOSED"}:
            economic = overlapping_horizon_strategy(
                prices,
                frame,
                model,
                feature_mean,
                feature_scale,
                window,
            )
            strategy_rows.append({"window": window, **economic})
    return pd.DataFrame(rows), model_public, pd.DataFrame(strategy_rows)


def horizon_sensitivity(prices: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in (1, 5, 10, 20, 30, 40, 60):
        windows = {
            "TRAIN_EARLY": (31, 250 - horizon),
            "TRAIN_LATE": (251, 500 - horizon),
            "VALIDATION": (501, 750 - horizon),
            "EXPOSED": (751, 1000 - horizon),
        }
        frame = build_frame(prices, horizon)
        train = pd.concat(
            [
                frame[
                    frame["anchor_day"].between(
                        *windows["TRAIN_EARLY"]
                    )
                ],
                frame[
                    frame["anchor_day"].between(
                        *windows["TRAIN_LATE"]
                    )
                ],
            ],
            ignore_index=True,
        )
        train_x_raw = train["feature"].to_numpy(float)
        train_y = train["target_log"].to_numpy(float)
        train_x, _, mean_x, scale_x = standardize_from_train(
            train_x_raw,
            train_x_raw,
        )
        model = newey_west_fit(
            train_x,
            train_y,
            max(0, horizon - 1),
        )
        row: dict[str, Any] = {
            "horizon": horizon,
            "train_n": len(train),
            "train_beta_standardized": model["beta"],
            "train_beta_t_hac": model["beta_t_hac"],
            "train_beta_p_hac": model["beta_p_hac"],
            "train_correlation": correlation(train_x, train_y),
            "train_r_squared": model["r_squared"],
        }
        for window in ("VALIDATION", "EXPOSED"):
            start, end = windows[window]
            sample = frame[frame["anchor_day"].between(start, end)]
            x = (
                sample["feature"].to_numpy(float) - mean_x
            ) / scale_x
            y = sample["target_log"].to_numpy(float)
            evaluated = evaluate_prediction(
                model,
                x,
                y,
                float(np.mean(train_y)),
            )
            row[f"{window.lower()}_correlation"] = evaluated[
                "correlation"
            ]
            row[f"{window.lower()}_oos_r_squared"] = evaluated[
                "oos_r_squared"
            ]
            row[f"{window.lower()}_direction_accuracy"] = evaluated[
                "direction_accuracy"
            ]
        rows.append(row)
    return pd.DataFrame(rows)


def feature_lag_sensitivity(prices: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lag in (0, 1, 2, 3, 5):
        frame = build_frame(
            prices,
            HORIZON,
            feature_lag=lag,
        )
        train = training_sample(frame)
        mean_x = float(train["feature"].mean())
        scale_x = max(float(train["feature"].std(ddof=0)), EPS)
        train_x = (
            train["feature"].to_numpy(float) - mean_x
        ) / scale_x
        train_y = train["target_log"].to_numpy(float)
        model = newey_west_fit(train_x, train_y, HAC_LAGS)
        row: dict[str, Any] = {
            "feature_lag": lag,
            "train_beta_standardized": model["beta"],
            "train_beta_t_hac": model["beta_t_hac"],
            "train_beta_p_hac": model["beta_p_hac"],
            "train_correlation": correlation(train_x, train_y),
        }
        for window in ("VALIDATION", "EXPOSED"):
            sample = window_sample(frame, window)
            x = (
                sample["feature"].to_numpy(float) - mean_x
            ) / scale_x
            y = sample["target_log"].to_numpy(float)
            result = evaluate_prediction(
                model,
                x,
                y,
                float(np.mean(train_y)),
            )
            row[f"{window.lower()}_correlation"] = result[
                "correlation"
            ]
            row[f"{window.lower()}_oos_r_squared"] = result[
                "oos_r_squared"
            ]
        rows.append(row)
    return pd.DataFrame(rows)


def circular_shift_test(
    prices: pd.DataFrame,
) -> dict[str, float]:
    """Exact-ish temporal placebo on the contiguous late-training window."""
    frame = build_frame(prices, HORIZON)
    sample = window_sample(frame, "TRAIN_LATE")
    x = sample["feature"].to_numpy(float)
    y = sample["target_log"].to_numpy(float)
    observed = abs(correlation(x, y))
    placebo: list[float] = []
    n = len(x)
    for shift in range(HORIZON + 1, n - HORIZON):
        placebo.append(abs(correlation(np.roll(x, shift), y)))
    values = np.asarray(placebo)
    return {
        "observed_abs_correlation": observed,
        "placebo_count": len(values),
        "circular_shift_p": float(
            (1.0 + np.sum(values >= observed))
            / (1.0 + len(values))
        ),
        "placebo_median_abs_correlation": float(np.median(values)),
        "placebo_95pct_abs_correlation": float(
            np.quantile(values, 0.95)
        ),
    }


def cross_asset_scan(prices: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol in prices.columns:
        frame = build_frame(
            prices,
            HORIZON,
            feature_symbol=symbol,
        )
        train = training_sample(frame)
        mean_x = float(train["feature"].mean())
        scale_x = max(float(train["feature"].std(ddof=0)), EPS)
        x_train = (
            train["feature"].to_numpy(float) - mean_x
        ) / scale_x
        y_train = train["target_log"].to_numpy(float)
        model = newey_west_fit(x_train, y_train, HAC_LAGS)
        row: dict[str, Any] = {
            "symbol": symbol,
            "train_correlation": correlation(x_train, y_train),
            "train_abs_correlation": abs(
                correlation(x_train, y_train)
            ),
            "train_beta_t_hac": model["beta_t_hac"],
            "train_beta_p_hac": model["beta_p_hac"],
            "train_r_squared": model["r_squared"],
        }
        for window in ("VALIDATION", "EXPOSED"):
            sample = window_sample(frame, window)
            x = (
                sample["feature"].to_numpy(float) - mean_x
            ) / scale_x
            y = sample["target_log"].to_numpy(float)
            result = evaluate_prediction(
                model,
                x,
                y,
                float(np.mean(y_train)),
            )
            row[f"{window.lower()}_correlation"] = result[
                "correlation"
            ]
            row[f"{window.lower()}_oos_r_squared"] = result[
                "oos_r_squared"
            ]
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        "train_abs_correlation",
        ascending=False,
    )
    result["train_abs_correlation_rank"] = np.arange(1, len(result) + 1)
    result["bonferroni_p_hac"] = np.minimum(
        1.0,
        result["train_beta_p_hac"] * len(result),
    )
    return result


def write_report(
    primary: pd.DataFrame,
    model: dict[str, Any],
    economic: pd.DataFrame,
    horizons: pd.DataFrame,
    lags: pd.DataFrame,
    placebo: dict[str, float],
    cross_assets: pd.DataFrame,
) -> None:
    mmbt = cross_assets[cross_assets["symbol"] == "MMBT"].iloc[0]
    lines = [
        "# MMBT lag-1 return -> ALGO 30-day forward-return audit",
        "",
        "Primary equation: `ALGO forward log return = alpha + beta * "
        "latest completed MMBT one-day return`. All forward labels end inside "
        "their own chronological window.",
        "",
        "## Training fit",
        "",
        f"- Observations: **{sum(primary[primary.window.isin(TRAIN_WINDOWS)].n):.0f}**.",
        f"- Beta for a one-standard-deviation MMBT move: "
        f"**{model['beta']:+.4%}** ALGO over 30 days.",
        f"- HAC t-stat / p-value: **{model['beta_t_hac']:+.2f} / "
        f"{model['beta_p_hac']:.4f}**.",
        f"- In-sample R-squared: **{model['r_squared']:.3%}**.",
        "",
        "## Chronological stability",
        "",
        "| Window | Correlation | Local beta | HAC t | Local R2 | "
        "Frozen-model OOS R2 | Direction | Baseline direction | "
        "Prediction long |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        lines.append(
            f"| {row.window} | {row.correlation:+.3f} | "
            f"{row.local_beta_standardized:+.3%} | "
            f"{row.local_beta_t_hac:+.2f} | {row.local_r_squared:.2%} | "
            f"{row.oos_r_squared:+.2%} | {row.direction_accuracy:.1%} | "
            f"{row.baseline_direction_accuracy:.1%} | "
            f"{row.prediction_long_share:.1%} |"
        )

    lines.extend(
        [
            "",
            "## 30-day overlapping-sleeve ALGO strategy",
            "",
            "One-thirtieth of the $100k ALGO limit is initiated daily and held "
            "for 30 days, with the supplied ALGO commission.",
            "",
            "| Window | PnL | Sharpe | Max drawdown | Hit rate | Turnover | "
            "Mean absolute position |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in economic.itertuples():
        lines.append(
            f"| {row.window} | ${row.total_pnl:,.0f} | "
            f"{row.annualized_sharpe:+.2f} | ${row.max_drawdown:,.0f} | "
            f"{row.daily_hit_rate:.1%} | ${row.turnover:,.0f} | "
            f"${row.mean_abs_position:,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Robustness controls",
            "",
            f"- Late-train circular-shift p-value: "
            f"**{placebo['circular_shift_p']:.4f}** "
            f"(observed |corr| {placebo['observed_abs_correlation']:.3f}; "
            f"95th-percentile placebo "
            f"{placebo['placebo_95pct_abs_correlation']:.3f}).",
            f"- MMBT ranks **{int(mmbt.train_abs_correlation_rank)}/"
            f"{len(cross_assets)}** assets by absolute training correlation.",
            f"- MMBT HAC p-value after a 51-asset Bonferroni correction: "
            f"**{mmbt.bonferroni_p_hac:.4f}**.",
            "",
            "### Horizon sensitivity",
            "",
            "| Horizon | Train corr | HAC t | HAC p | Validation corr | "
            "Validation OOS R2 | Exposed corr | Exposed OOS R2 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in horizons.itertuples():
        lines.append(
            f"| {row.horizon} | {row.train_correlation:+.3f} | "
            f"{row.train_beta_t_hac:+.2f} | {row.train_beta_p_hac:.4f} | "
            f"{row.validation_correlation:+.3f} | "
            f"{row.validation_oos_r_squared:+.2%} | "
            f"{row.exposed_correlation:+.3f} | "
            f"{row.exposed_oos_r_squared:+.2%} |"
        )

    lines.extend(
        [
            "",
            "### Feature-lag sensitivity at 30 days",
            "",
            "| Extra lag | Train corr | HAC t | HAC p | Validation corr | "
            "Validation OOS R2 | Exposed corr | Exposed OOS R2 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in lags.itertuples():
        lines.append(
            f"| {row.feature_lag} | {row.train_correlation:+.3f} | "
            f"{row.train_beta_t_hac:+.2f} | {row.train_beta_p_hac:.4f} | "
            f"{row.validation_correlation:+.3f} | "
            f"{row.validation_oos_r_squared:+.2%} | "
            f"{row.exposed_correlation:+.3f} | "
            f"{row.exposed_oos_r_squared:+.2%} |"
        )
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+")
    primary, model, economic = primary_analysis(prices)
    horizons = horizon_sensitivity(prices)
    lags = feature_lag_sensitivity(prices)
    placebo = circular_shift_test(prices)
    cross_assets = cross_asset_scan(prices)

    primary.to_csv(OUTPUT_DIR / "primary_windows.csv", index=False)
    economic.to_csv(OUTPUT_DIR / "economic_strategy.csv", index=False)
    horizons.to_csv(OUTPUT_DIR / "horizon_sensitivity.csv", index=False)
    lags.to_csv(OUTPUT_DIR / "lag_sensitivity.csv", index=False)
    cross_assets.to_csv(OUTPUT_DIR / "cross_asset_scan.csv", index=False)
    (OUTPUT_DIR / "model.json").write_text(
        json.dumps(model, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "placebo.json").write_text(
        json.dumps(placebo, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        primary,
        model,
        economic,
        horizons,
        lags,
        placebo,
        cross_assets,
    )
    print(primary.to_string(index=False))
    print("\n", json.dumps(model, indent=2))
    print("\n", json.dumps(placebo, indent=2))
    print(f"\nWrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

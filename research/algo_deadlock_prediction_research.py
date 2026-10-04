"""Can ALGO movements predict pair-vote deadlocks or resolve their direction?

Protocol:
* Train A: days 251-375
* Train B: days 376-500 (first forward check)
* Full train: days 251-500
* Validation: days 501-750 (single final development check)

Days 751-1000 are not loaded.
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
from research import algo_residual_projection_research as residual


OUTPUT_DIR = ROOT / "analysis_outputs" / "algo_deadlock_prediction"
LAST_DEVELOPMENT_DAY = 750
ALGO = strategy.I["ALGO"]
OVERLAY_DOLLARS = (
    strategy.DEADLOCK_RIDGE_OVERLAY_FRACTION * strategy.LEG
)
ROUND_TRIP_RETURN_COST = 2.0e-4
FEATURES = (
    "algo_r1",
    "algo_abs_r1",
    "algo_r5",
    "algo_r20",
    "algo_vol5",
    "algo_vol20",
    "algo_shock",
    "algo_acceleration",
)
OCCURRENCE_MODELS = (
    "constant",
    "persistence",
    "algo",
    "combined",
)
PORTFOLIO_WINDOWS = (
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("TRAIN", 251, 500),
    ("VALIDATION_A", 501, 625),
    ("VALIDATION_B", 626, 750),
    ("VALIDATION", 501, 750),
)


def algo_features(
    prices: np.ndarray,
    history_size: int,
) -> dict[str, float]:
    log_price = np.log(
        np.maximum(
            prices[ALGO, :history_size],
            strategy._EPS_DENOM,
        )
    )
    returns = np.diff(log_price)

    def cumulative(days: int) -> float:
        if len(log_price) <= days:
            return 0.0
        return float(log_price[-1] - log_price[-1 - days])

    def volatility(days: int) -> float:
        values = returns[-days:]
        return float(np.std(values)) if len(values) else 0.0

    r1 = cumulative(1)
    r5 = cumulative(5)
    vol5 = volatility(5)
    vol20 = volatility(20)
    return {
        "algo_r1": r1,
        "algo_abs_r1": abs(r1),
        "algo_r5": r5,
        "algo_r20": cumulative(20),
        "algo_vol5": vol5,
        "algo_vol20": vol20,
        "algo_shock": r1 / max(vol20, 1.0e-12),
        "algo_acceleration": r1 - r5 / 5.0,
    }


def collect_deadlocks(
    prices: np.ndarray,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
]:
    """Replay current logic and collect exact ties before the ridge overlay."""
    days = prices.shape[1]
    no_overlay = np.zeros((days + 1, strategy.N_INST))
    existing = np.zeros_like(no_overlay)
    exact_by_history: dict[int, np.ndarray] = {}
    daily_rows: list[dict[str, float | int | bool]] = []
    event_rows: list[dict[str, float | int | str | bool]] = []
    strategy.reset()

    previous_deadlock_count = 0
    for history_size in range(1, days):
        history = prices[:, :history_size]
        sleeves_now = strategy.sleeve_breakdown(history)
        if sleeves_now is None:
            continue
        core = np.asarray(sleeves_now["core_dollars"], dtype=float)
        ridge_overlay = np.asarray(
            sleeves_now["deadlock_ridge_overlay_dollars"],
            dtype=float,
        )
        no_overlay[history_size] = core - ridge_overlay
        existing[history_size] = core

        pair_active = np.asarray(
            sleeves_now["pair_active_mask"],
            dtype=bool,
        )
        pair_vote = np.asarray(sleeves_now["pair_vote"], dtype=float)
        ridge_active = np.asarray(
            sleeves_now["ridge_active_mask"],
            dtype=bool,
        )
        crash = np.asarray(sleeves_now["crash_mask"], dtype=bool)
        exact = (
            pair_active
            & (np.abs(pair_vote) <= 1.0e-12)
            & ~ridge_active
            & ~crash
        )
        exact[ALGO] = False
        exact_by_history[history_size] = exact

        score_day = history_size + 1
        feature = algo_features(prices, history_size)
        deadlock_count = int(np.sum(exact))
        daily_rows.append(
            {
                "history_size": history_size,
                "score_day": score_day,
                **feature,
                "lag_deadlock_count": previous_deadlock_count,
                "deadlock_count": deadlock_count,
                "any_deadlock": deadlock_count > 0,
            }
        )
        previous_deadlock_count = deadlock_count

        if score_day > days:
            continue
        betas, correlations, _, ready = residual.causal_factor_estimates(
            prices,
            history_size,
        )
        next_returns = np.log(
            np.maximum(prices[:, score_day - 1], strategy._EPS_DENOM)
        ) - np.log(
            np.maximum(prices[:, score_day - 2], strategy._EPS_DENOM)
        )
        for instrument in np.flatnonzero(exact):
            instrument = int(instrument)
            beta = float(betas[instrument]) if ready else 0.0
            raw_return = float(next_returns[instrument])
            algo_return = float(next_returns[ALGO])
            event_rows.append(
                {
                    "history_size": history_size,
                    "score_day": score_day,
                    "instrument": instrument,
                    "symbol": strategy._SYMBOL_BY_INDEX[instrument],
                    **feature,
                    "return_beta": beta,
                    "return_correlation": (
                        float(correlations[instrument])
                        if ready
                        else 0.0
                    ),
                    "next_asset_return": raw_return,
                    "next_algo_return": algo_return,
                    "next_residual_return": (
                        raw_return - beta * algo_return
                    ),
                    "existing_ridge_overlay": float(
                        ridge_overlay[instrument]
                    ),
                }
            )
    strategy.reset()
    return (
        pd.DataFrame(daily_rows),
        pd.DataFrame(event_rows),
        no_overlay,
        existing,
        exact_by_history,
    )


class StandardizedRidge:
    """Small deterministic ridge model with an unpenalized intercept."""

    def __init__(self, logistic: bool = False) -> None:
        self.logistic = logistic
        self.mean = np.asarray([])
        self.scale = np.asarray([])
        self.coefficients = np.asarray([])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "StandardizedRidge":
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0) + 1.0e-12
        z = (x - self.mean) / self.scale
        design = np.column_stack([np.ones(len(z)), z])
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        if not self.logistic:
            self.coefficients = np.linalg.solve(
                design.T @ design + len(design) * penalty,
                design.T @ y,
            )
            return self

        coefficients = np.zeros(design.shape[1])
        for _ in range(50):
            linear = np.clip(design @ coefficients, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-linear))
            weights = np.maximum(
                probability * (1.0 - probability),
                1.0e-6,
            )
            gradient = (
                design.T @ (probability - y)
                + len(design) * penalty @ coefficients
            )
            hessian = (
                design.T @ (weights[:, None] * design)
                + len(design) * penalty
            )
            step = np.linalg.solve(hessian, gradient)
            coefficients -= step
            if float(np.max(np.abs(step))) < 1.0e-10:
                break
        self.coefficients = coefficients
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        design = np.column_stack([np.ones(len(z)), z])
        output = design @ self.coefficients
        if self.logistic:
            output = 1.0 / (
                1.0 + np.exp(-np.clip(output, -30.0, 30.0))
            )
        return output


def auc(y: np.ndarray, probability: np.ndarray) -> float:
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == 0))
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = pd.Series(probability).rank(method="average").to_numpy()
    return float(
        (
            np.sum(ranks[y == 1])
            - positive * (positive + 1) / 2.0
        )
        / (positive * negative)
    )


def occurrence_design(
    frame: pd.DataFrame,
    model: str,
) -> np.ndarray:
    columns: list[str] = []
    if model in ("persistence", "combined"):
        columns.append("lag_deadlock_count")
    if model in ("algo", "combined"):
        columns.extend(FEATURES)
    if not columns:
        return np.zeros((len(frame), 0))
    return frame[columns].to_numpy(dtype=float)


def occurrence_research(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    protocols = (
        ("TRAIN_FORWARD", 251, 375, 376, 500),
        ("VALIDATION", 251, 500, 501, 750),
    )
    for protocol, fit_start, fit_end, test_start, test_end in protocols:
        fit = daily[daily["score_day"].between(fit_start, fit_end)]
        test = daily[daily["score_day"].between(test_start, test_end)]
        count_y = fit["deadlock_count"].to_numpy(dtype=float)
        any_y = fit["any_deadlock"].to_numpy(dtype=float)
        test_count = test["deadlock_count"].to_numpy(dtype=float)
        test_any = test["any_deadlock"].to_numpy(dtype=int)

        for model_name in OCCURRENCE_MODELS:
            if model_name == "constant":
                count_prediction = np.full(
                    len(test),
                    float(np.mean(count_y)),
                )
                probability = np.full(
                    len(test),
                    float(np.mean(any_y)),
                )
            else:
                x_fit = occurrence_design(fit, model_name)
                x_test = occurrence_design(test, model_name)
                count_prediction = StandardizedRidge().fit(
                    x_fit,
                    count_y,
                ).predict(x_test)
                probability = StandardizedRidge(logistic=True).fit(
                    x_fit,
                    any_y,
                ).predict(x_test)
            threshold = float(np.quantile(probability, 0.75))
            top = probability >= threshold
            rows.append(
                {
                    "protocol": protocol,
                    "model": model_name,
                    "n": len(test),
                    "count_rmse": float(
                        np.sqrt(np.mean((count_prediction - test_count) ** 2))
                    ),
                    "count_mae": float(
                        np.mean(np.abs(count_prediction - test_count))
                    ),
                    "count_correlation": float(
                        np.corrcoef(count_prediction, test_count)[0, 1]
                        if np.std(count_prediction) > 1.0e-12
                        else 0.0
                    ),
                    "classification_auc": auc(test_any, probability),
                    "brier": float(
                        np.mean((probability - test_any) ** 2)
                    ),
                    "top_quartile_rate": float(
                        np.mean(test_any[top]) if np.any(top) else 0.0
                    ),
                    "unconditional_rate": float(np.mean(test_any)),
                }
            )
    return pd.DataFrame(rows)


def resolution_design(frame: pd.DataFrame) -> np.ndarray:
    base = frame[list(FEATURES)].to_numpy(dtype=float)
    beta = frame["return_beta"].to_numpy(dtype=float)[:, None]
    correlation = frame[
        "return_correlation"
    ].to_numpy(dtype=float)[:, None]
    interactions = base[:, [0, 2, 3]] * beta
    return np.column_stack([base, beta, correlation, interactions])


def resolution_research(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    protocols = (
        ("TRAIN_FORWARD", 251, 375, 376, 500),
        ("VALIDATION", 251, 500, 501, 750),
    )
    for protocol, fit_start, fit_end, test_start, test_end in protocols:
        fit = events[events["score_day"].between(fit_start, fit_end)]
        test = events[events["score_day"].between(test_start, test_end)]
        x_fit = resolution_design(fit)
        x_test = resolution_design(test)
        raw_model = StandardizedRidge().fit(
            x_fit,
            fit["next_asset_return"].to_numpy(dtype=float),
        )
        residual_model = StandardizedRidge().fit(
            x_fit,
            fit["next_residual_return"].to_numpy(dtype=float),
        )
        raw_forecast = raw_model.predict(x_test)
        residual_forecast = residual_model.predict(x_test)
        predictions = {
            "algo_momentum": np.sign(
                test["algo_r1"].to_numpy(dtype=float)
            ),
            "algo_reversal": -np.sign(
                test["algo_r1"].to_numpy(dtype=float)
            ),
            "algo_ridge_raw": np.sign(raw_forecast),
            "algo_ridge_raw_cost_gate": np.where(
                np.abs(raw_forecast) > ROUND_TRIP_RETURN_COST,
                np.sign(raw_forecast),
                0.0,
            ),
            "algo_ridge_residual": np.sign(
                residual_forecast
            ),
            "existing_ridge": np.sign(
                test["existing_ridge_overlay"].to_numpy(dtype=float)
            ),
        }
        existing_direction = predictions["existing_ridge"]
        raw_direction = predictions["algo_ridge_raw"]
        residual_direction = predictions["algo_ridge_residual"]
        predictions["existing_plus_algo_raw"] = np.where(
            existing_direction != 0.0,
            existing_direction,
            raw_direction,
        )
        predictions["existing_plus_algo_momentum"] = np.where(
            existing_direction != 0.0,
            existing_direction,
            predictions["algo_momentum"],
        )
        predictions["existing_plus_algo_raw_cost_gate"] = np.where(
            existing_direction != 0.0,
            existing_direction,
            predictions["algo_ridge_raw_cost_gate"],
        )
        predictions["existing_plus_algo_residual"] = np.where(
            existing_direction != 0.0,
            existing_direction,
            residual_direction,
        )
        predictions["algo_raw_uncovered_only"] = np.where(
            existing_direction == 0.0,
            raw_direction,
            0.0,
        )
        outcomes = test["next_asset_return"].to_numpy(dtype=float)
        for name, direction in predictions.items():
            active = direction != 0.0
            signed_returns = direction[active] * outcomes[active]
            pnl = (
                OVERLAY_DOLLARS * signed_returns
                - 2.0 * OVERLAY_DOLLARS * 1.0e-4
            )
            if np.any(active):
                observed = float(np.mean(signed_returns))
                rng = np.random.default_rng(
                    91_000
                    + sum(ord(character) for character in name)
                    + (0 if protocol == "TRAIN_FORWARD" else 10_000)
                )
                randomized = np.mean(
                    rng.choice(
                        np.asarray([-1.0, 1.0]),
                        size=(5_000, int(np.sum(active))),
                    )
                    * outcomes[active][None, :],
                    axis=1,
                )
                randomization_p = float(
                    (1 + np.sum(randomized >= observed)) / 5_001
                )
            else:
                randomization_p = 1.0
            rows.append(
                {
                    "protocol": protocol,
                    "model": name,
                    "events": int(np.sum(active)),
                    "coverage": float(np.mean(active)),
                    "win_rate": float(
                        np.mean(
                            direction[active] * outcomes[active] > 0.0
                        )
                        if np.any(active)
                        else 0.0
                    ),
                    "mean_event_pnl": float(
                        np.mean(pnl) if len(pnl) else 0.0
                    ),
                    "total_event_pnl": float(np.sum(pnl)),
                    "mean_signed_return_bps": float(
                        np.mean(signed_returns)
                        * 1.0e4
                        if np.any(active)
                        else 0.0
                    ),
                    "random_direction_p": randomization_p,
                }
            )
    return pd.DataFrame(rows)


def targets_from_dollars(
    prices: np.ndarray,
    dollars: np.ndarray,
) -> np.ndarray:
    targets = np.zeros_like(dollars, dtype=int)
    for history_size in range(1, prices.shape[1]):
        targets[history_size] = sleeves.dollars_to_shares(
            dollars[history_size],
            prices[:, history_size - 1],
        )
    return targets


def build_rule_dollars(
    prices: np.ndarray,
    daily: pd.DataFrame,
    events: pd.DataFrame,
    no_overlay: np.ndarray,
    existing: np.ndarray,
    exact_by_history: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    books = {
        "no_overlay": no_overlay.copy(),
        "existing_ridge": existing.copy(),
        "algo_momentum": no_overlay.copy(),
        "algo_reversal": no_overlay.copy(),
        "algo_ridge_raw": no_overlay.copy(),
        "algo_ridge_raw_cost_gate": no_overlay.copy(),
        "algo_ridge_residual": no_overlay.copy(),
        "existing_plus_algo_raw": existing.copy(),
        "existing_plus_algo_momentum": existing.copy(),
        "existing_plus_algo_raw_cost_gate": existing.copy(),
        "existing_plus_algo_residual": existing.copy(),
    }
    features_by_history = daily.set_index("history_size")
    for history_size, exact in exact_by_history.items():
        if history_size not in features_by_history.index:
            continue
        r1 = float(
            features_by_history.loc[history_size, "algo_r1"]
        )
        books["algo_momentum"][history_size, exact] = (
            OVERLAY_DOLLARS * np.sign(r1)
        )
        books["algo_reversal"][history_size, exact] = (
            -OVERLAY_DOLLARS * np.sign(r1)
        )
        uncovered = (
            exact
            & (
                np.abs(
                    existing[history_size] - no_overlay[history_size]
                )
                <= 1.0e-9
            )
        )
        books["existing_plus_algo_momentum"][
            history_size,
            uncovered,
        ] = OVERLAY_DOLLARS * np.sign(r1)

    train_a = events[events["score_day"].between(251, 375)]
    full_train = events[events["score_day"].between(251, 500)]
    models: dict[str, tuple[StandardizedRidge, StandardizedRidge]] = {}
    for label, fit in (
        ("TRAIN_FORWARD", train_a),
        ("VALIDATION", full_train),
    ):
        x_fit = resolution_design(fit)
        models[label] = (
            StandardizedRidge().fit(
                x_fit,
                fit["next_asset_return"].to_numpy(dtype=float),
            ),
            StandardizedRidge().fit(
                x_fit,
                fit["next_residual_return"].to_numpy(dtype=float),
            ),
        )

    for protocol, start, end in (
        ("TRAIN_FORWARD", 376, 500),
        ("VALIDATION", 501, 750),
    ):
        sample = events[events["score_day"].between(start, end)]
        if len(sample) == 0:
            continue
        x = resolution_design(sample)
        raw_prediction = models[protocol][0].predict(x)
        residual_prediction = models[protocol][1].predict(x)
        for row, raw, residual_prediction_value in zip(
            sample.itertuples(index=False),
            raw_prediction,
            residual_prediction,
            strict=True,
        ):
            books["algo_ridge_raw"][
                int(row.history_size),
                int(row.instrument),
            ] = OVERLAY_DOLLARS * np.sign(raw)
            books["algo_ridge_raw_cost_gate"][
                int(row.history_size),
                int(row.instrument),
            ] = (
                OVERLAY_DOLLARS * np.sign(raw)
                if abs(float(raw)) > ROUND_TRIP_RETURN_COST
                else 0.0
            )
            books["algo_ridge_residual"][
                int(row.history_size),
                int(row.instrument),
            ] = OVERLAY_DOLLARS * np.sign(
                residual_prediction_value
            )
            if abs(float(row.existing_ridge_overlay)) <= 1.0e-9:
                books["existing_plus_algo_raw"][
                    int(row.history_size),
                    int(row.instrument),
                ] = OVERLAY_DOLLARS * np.sign(raw)
                if abs(float(raw)) > ROUND_TRIP_RETURN_COST:
                    books["existing_plus_algo_raw_cost_gate"][
                        int(row.history_size),
                        int(row.instrument),
                    ] = OVERLAY_DOLLARS * np.sign(raw)
                books["existing_plus_algo_residual"][
                    int(row.history_size),
                    int(row.instrument),
                ] = OVERLAY_DOLLARS * np.sign(
                    residual_prediction_value
                )
    return books


def portfolio_research(
    prices: np.ndarray,
    books: dict[str, np.ndarray],
) -> pd.DataFrame:
    targets = {
        name: targets_from_dollars(prices, dollars)
        for name, dollars in books.items()
    }
    rows: list[dict[str, float | int | str]] = []
    baseline = sleeves.evaluate(
        prices,
        targets["no_overlay"],
        PORTFOLIO_WINDOWS,
    )
    for window, start, end in PORTFOLIO_WINDOWS:
        for name, target in targets.items():
            pnl, turnover, fees = sleeves.account(
                prices,
                target,
                start,
                end,
            )
            metrics = sleeves.metrics(pnl, turnover, fees)
            rows.append(
                {
                    "variant": name,
                    "window": window,
                    **metrics,
                    "score_delta": (
                        metrics["score"]
                        - baseline[window]["score"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    occurrence: pd.DataFrame,
    resolution_frame: pd.DataFrame,
    portfolio: pd.DataFrame,
    daily: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    lines = [
        "# ALGO movement and pair-vote deadlocks",
        "",
        "Only days 1-750 are loaded. Days 751-1000 are excluded.",
        "",
        f"- Training deadlock asset-events: "
        f"**{len(events[events['score_day'].between(251, 500)])}**.",
        f"- Validation deadlock asset-events: "
        f"**{len(events[events['score_day'].between(501, 750)])}**.",
        f"- Training days with any deadlock: "
        f"**{daily[daily['score_day'].between(251, 500)]['any_deadlock'].mean():.1%}**.",
        f"- Validation days with any deadlock: "
        f"**{daily[daily['score_day'].between(501, 750)]['any_deadlock'].mean():.1%}**.",
        "",
        "## Can ALGO predict deadlock occurrence?",
        "",
        "| Protocol | Model | Count RMSE | Count corr | AUC | Brier | "
        "Top-quartile rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in occurrence.iterrows():
        lines.append(
            f"| {row['protocol']} | {row['model']} | "
            f"{row['count_rmse']:.3f} | {row['count_correlation']:.3f} | "
            f"{row['classification_auc']:.3f} | {row['brier']:.3f} | "
            f"{row['top_quartile_rate']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Can ALGO resolve deadlock direction?",
            "",
        "| Protocol | Model | Events | Coverage | Win rate | "
            "Mean event PnL | Total PnL | Random p |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in resolution_frame.iterrows():
        lines.append(
            f"| {row['protocol']} | {row['model']} | "
            f"{int(row['events'])} | {row['coverage']:.1%} | "
            f"{row['win_rate']:.1%} | ${row['mean_event_pnl']:.2f} | "
            f"${row['total_event_pnl']:.0f} | "
            f"{row['random_direction_p']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Portfolio score changes versus no deadlock overlay",
            "",
            "| Variant | Train A | Train B | Train | Validation A | "
            "Validation B | Validation |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "existing_ridge",
        "algo_momentum",
        "algo_reversal",
        "algo_ridge_raw",
        "algo_ridge_raw_cost_gate",
        "algo_ridge_residual",
        "existing_plus_algo_raw",
        "existing_plus_algo_momentum",
        "existing_plus_algo_raw_cost_gate",
        "existing_plus_algo_residual",
    ):
        values = portfolio[
            portfolio["variant"] == name
        ].set_index("window")
        lines.append(
            f"| {name} | {values.loc['TRAIN_A', 'score_delta']:+.2f} | "
            f"{values.loc['TRAIN_B', 'score_delta']:+.2f} | "
            f"{values.loc['TRAIN', 'score_delta']:+.2f} | "
            f"{values.loc['VALIDATION_A', 'score_delta']:+.2f} | "
            f"{values.loc['VALIDATION_B', 'score_delta']:+.2f} | "
            f"{values.loc['VALIDATION', 'score_delta']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Acceptance rule",
            "",
            "ALGO must add predictive value beyond the constant/persistence "
            "baseline in the Train-B forward check and validation. A direction "
            "rule must improve both chronological validation halves and must "
            "not be justified by any data after day 750.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def make_plot(
    occurrence: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    validation = occurrence[
        occurrence["protocol"] == "VALIDATION"
    ].set_index("model")
    axes[0].bar(
        validation.index,
        validation["classification_auc"],
    )
    axes[0].axhline(0.5, color="black", linewidth=0.8)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Validation: predicting any next deadlock")
    axes[0].set_ylabel("AUC")
    axes[0].tick_params(axis="x", rotation=20)

    variants = (
        "existing_ridge",
        "algo_momentum",
        "algo_reversal",
        "algo_ridge_raw",
        "algo_ridge_raw_cost_gate",
        "algo_ridge_residual",
        "existing_plus_algo_raw",
        "existing_plus_algo_momentum",
        "existing_plus_algo_raw_cost_gate",
        "existing_plus_algo_residual",
    )
    validation_scores = portfolio[
        (portfolio["window"] == "VALIDATION")
        & portfolio["variant"].isin(variants)
    ].set_index("variant").reindex(variants)
    axes[1].bar(
        validation_scores.index,
        validation_scores["score_delta"],
        color=np.where(
            validation_scores["score_delta"] >= 0.0,
            "#2ca02c",
            "#d62728",
        ),
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Validation deadlock overlay score change")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "algo_deadlock_prediction.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(
        sleeves.PRICE_PATH,
        sep=r"\s+",
        header=0,
    ).to_numpy(dtype=float).T[:, :LAST_DEVELOPMENT_DAY]
    (
        daily,
        events,
        no_overlay,
        existing,
        exact_by_history,
    ) = collect_deadlocks(prices)
    occurrence = occurrence_research(daily)
    resolution_frame = resolution_research(events)
    books = build_rule_dollars(
        prices,
        daily,
        events,
        no_overlay,
        existing,
        exact_by_history,
    )
    portfolio = portfolio_research(prices, books)

    daily.to_csv(OUTPUT_DIR / "daily_deadlocks.csv", index=False)
    events.to_csv(OUTPUT_DIR / "deadlock_events.csv", index=False)
    occurrence.to_csv(
        OUTPUT_DIR / "occurrence_models.csv",
        index=False,
    )
    resolution_frame.to_csv(
        OUTPUT_DIR / "resolution_models.csv",
        index=False,
    )
    portfolio.to_csv(
        OUTPUT_DIR / "portfolio_metrics.csv",
        index=False,
    )
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "last_loaded_day": LAST_DEVELOPMENT_DAY,
                "features": FEATURES,
                "overlay_dollars": OVERLAY_DOLLARS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        occurrence,
        resolution_frame,
        portfolio,
        daily,
        events,
    )
    make_plot(occurrence, portfolio)
    print((OUTPUT_DIR / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

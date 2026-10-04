"""Leakage-safe research for a dedicated, short-horizon ALGO sleeve.

The experiment is deliberately split into two commands:

``select``
    Reads only days 1-750.  Candidate model families, forecast horizons, and
    the economically defined fee hurdle are compared on chronological
    development and validation blocks.  The selected specification is frozen
    to disk before the final data can be read.

``evaluate``
    Loads the frozen specification, refits it through day 750, and evaluates it
    once on days 751-1000.  It also measures the selected ALGO sleeve beside
    the current ``four_sigmas`` book without changing the production strategy.

At a decision made with prices through day ``t``, every feature uses data no
later than day ``t`` and the label is ALGO's log return from day ``t`` to
``t + horizon``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRICE_PATH = ROOT / "prices.txt"
OUT_DIR = ROOT / "analysis_outputs" / "algo_dedicated_horizon"
FROZEN_PATH = OUT_DIR / "frozen_selection.json"
DEVELOPMENT_PATH = OUT_DIR / "development_candidates.csv"
FINAL_PATH = OUT_DIR / "final_results.csv"
EXPOSED_CANDIDATES_PATH = OUT_DIR / "exposed_candidate_diagnostics.csv"
FEE_COUNTERFACTUAL_PATH = OUT_DIR / "fee_counterfactual_candidates.csv"
COEFFICIENT_PATH = OUT_DIR / "selected_coefficient_stability.csv"
REPORT_PATH = OUT_DIR / "report.md"

ALGO = 0
N_INST = 51
ALGO_CAP = 100_000.0
OTHER_CAP = 10_000.0
ALGO_FEE = 2.0e-5
OTHER_FEE = 1.0e-4
MIN_HISTORY = 60

HORIZONS = (1, 2, 3, 5, 10, 20, 30)
MODEL_FAMILIES = (
    "algo_ar1",
    "algo_multiscale",
    "market_compact",
    "cross_asset_lag1",
    "hybrid",
)
RIDGE_LAMBDAS = (0.3, 1.0, 3.0)
HURDLES = ("none", "algo_round_trip")

DEVELOPMENT_WINDOWS = (
    ("development_251_500", 251, 500),
    ("development_a_251_375", 251, 375),
    ("development_b_376_500", 376, 500),
)
VALIDATION_WINDOWS = (
    ("validation_501_750", 501, 750),
    ("validation_a_501_625", 501, 625),
    ("validation_b_626_750", 626, 750),
)
EXPOSED_WINDOWS = (
    ("exposed_751_1000", 751, 1000),
    ("exposed_a_751_875", 751, 875),
    ("exposed_b_876_1000", 876, 1000),
)


@dataclass(frozen=True)
class ModelSpec:
    family: str
    horizon: int
    ridge_lambda: float
    hurdle: str

    @property
    def name(self) -> str:
        return (
            f"{self.family}_h{self.horizon}_l{self.ridge_lambda:g}"
            f"_{self.hurdle}"
        )


@dataclass(frozen=True)
class FittedModel:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    coefficients: np.ndarray


def load_prices(nrows: int | None) -> np.ndarray:
    frame = pd.read_csv(PRICE_PATH, sep=r"\s+", nrows=nrows)
    values = frame.to_numpy(float).T
    if values.shape[0] != N_INST:
        raise ValueError(f"Expected {N_INST} instruments, got {values.shape[0]}")
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("Prices must be finite and strictly positive")
    return values


def score(pnl: np.ndarray) -> float:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe * sharpe / (sharpe * sharpe + 1.0)


def metrics(
    pnl: np.ndarray,
    turnover: float,
    fees: float,
    targets: np.ndarray | None = None,
    prices: np.ndarray | None = None,
    start: int | None = None,
    end: int | None = None,
) -> dict[str, float]:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    result = {
        "score": score(pnl),
        "mean_pnl": mean,
        "total_pnl": float(np.sum(pnl)),
        "std_pnl": std,
        "sharpe": sqrt(250.0) * mean / std if std > 0.0 else 0.0,
        "hit_rate": float(np.mean(pnl > 0.0)),
        "turnover": float(turnover),
        "fees": float(fees),
    }
    if (
        targets is not None
        and prices is not None
        and start is not None
        and end is not None
    ):
        anchors = np.arange(start - 1, end)
        dollars = (
            targets[anchors, ALGO]
            * prices[ALGO, anchors - 1]
        )
        result["algo_mean_abs_dollars"] = float(np.mean(np.abs(dollars)))
        result["algo_long_fraction"] = float(np.mean(dollars > 0.0))
        result["algo_flat_fraction"] = float(np.mean(dollars == 0.0))
        result["algo_direction_changes"] = float(
            np.count_nonzero(np.diff(np.sign(dollars)))
        )
    return result


def _trailing_log_return(
    log_prices: np.ndarray,
    anchors: np.ndarray,
    window: int,
) -> np.ndarray:
    return (
        log_prices[:, anchors - 1]
        - log_prices[:, anchors - 1 - window]
    ).T


def feature_matrix(
    prices: np.ndarray,
    anchors: np.ndarray,
    family: str,
) -> np.ndarray:
    """Return features available after observing each one-indexed anchor day."""
    log_prices = np.log(prices)
    r1 = _trailing_log_return(log_prices, anchors, 1)

    if family == "algo_ar1":
        return r1[:, [ALGO]]

    algo_parts = [
        _trailing_log_return(log_prices, anchors, window)[:, [ALGO]]
        for window in (1, 2, 3, 5, 10, 20)
    ]
    daily = np.diff(log_prices, axis=1).T
    algo_vol10 = np.asarray(
        [daily[a - 11 : a - 1, ALGO].std() for a in anchors]
    )[:, None]
    algo_vol20 = np.asarray(
        [daily[a - 21 : a - 1, ALGO].std() for a in anchors]
    )[:, None]
    own = np.concatenate(algo_parts + [algo_vol10, algo_vol20], axis=1)
    if family == "algo_multiscale":
        return own

    market_parts = []
    relative_parts = []
    for window in (1, 2, 5, 10, 20):
        trailing = _trailing_log_return(log_prices, anchors, window)
        market = trailing[:, 1:].mean(axis=1, keepdims=True)
        market_parts.append(market)
        relative_parts.append(trailing[:, [ALGO]] - market)
    dispersion = r1[:, 1:].std(axis=1, keepdims=True)
    breadth = (r1[:, 1:] > 0.0).mean(axis=1, keepdims=True) - 0.5
    compact = np.concatenate(
        [own] + market_parts + relative_parts + [dispersion, breadth],
        axis=1,
    )
    if family == "market_compact":
        return compact
    if family == "cross_asset_lag1":
        return r1
    if family == "hybrid":
        return np.concatenate([compact, r1[:, 1:]], axis=1)
    raise ValueError(f"Unknown model family: {family}")


def feature_names(family: str) -> list[str]:
    own = [
        "algo_r1",
        "algo_r2",
        "algo_r3",
        "algo_r5",
        "algo_r10",
        "algo_r20",
        "algo_vol10",
        "algo_vol20",
    ]
    compact = (
        own
        + [f"market_r{window}" for window in (1, 2, 5, 10, 20)]
        + [f"algo_relative_r{window}" for window in (1, 2, 5, 10, 20)]
        + ["cross_asset_dispersion_r1", "cross_asset_breadth_r1"]
    )
    symbols = list(pd.read_csv(PRICE_PATH, sep=r"\s+", nrows=0).columns)
    if family == "algo_ar1":
        return ["algo_r1"]
    if family == "algo_multiscale":
        return own
    if family == "market_compact":
        return compact
    if family == "cross_asset_lag1":
        return [f"{symbol}_r1" for symbol in symbols]
    if family == "hybrid":
        return compact + [f"{symbol}_r1" for symbol in symbols[1:]]
    raise ValueError(f"Unknown model family: {family}")


def forward_target(
    prices: np.ndarray,
    anchors: np.ndarray,
    horizon: int,
) -> np.ndarray:
    log_algo = np.log(prices[ALGO])
    return log_algo[anchors - 1 + horizon] - log_algo[anchors - 1]


def fit_model(
    prices: np.ndarray,
    spec: ModelSpec,
    fit_end: int,
) -> FittedModel:
    last_anchor = fit_end - spec.horizon
    anchors = np.arange(MIN_HISTORY + 1, last_anchor + 1)
    x_raw = feature_matrix(prices, anchors, spec.family)
    y_raw = forward_target(prices, anchors, spec.horizon)
    x_mean = x_raw.mean(axis=0)
    x_scale = x_raw.std(axis=0) + 1.0e-12
    y_mean = float(y_raw.mean())
    y_scale = float(y_raw.std()) + 1.0e-12
    x = (x_raw - x_mean) / x_scale
    y = (y_raw - y_mean) / y_scale
    coefficients = np.linalg.solve(
        x.T @ x
        + spec.ridge_lambda * len(x) * np.eye(x.shape[1]),
        x.T @ y,
    )
    return FittedModel(
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        coefficients=coefficients,
    )


def predict(
    prices: np.ndarray,
    spec: ModelSpec,
    model: FittedModel,
    anchors: np.ndarray,
) -> np.ndarray:
    x_raw = feature_matrix(prices, anchors, spec.family)
    standardized = (x_raw - model.x_mean) / model.x_scale
    return model.y_mean + model.y_scale * (standardized @ model.coefficients)


def signal_targets(
    prices: np.ndarray,
    spec: ModelSpec,
    model: FittedModel,
    start: int,
    end: int,
    cap: float = ALGO_CAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Build daily ALGO targets for an evaluator window.

    The first target is chosen at the close immediately before ``start``.
    """
    targets = np.zeros((prices.shape[1] + 1, N_INST), dtype=int)
    anchors = np.arange(start - 1, end)
    forecasts = predict(prices, spec, model, anchors)
    hurdle = 0.0
    if spec.hurdle == "algo_round_trip":
        hurdle = 2.0 * ALGO_FEE
    active = np.abs(forecasts) > hurdle
    dollars = np.zeros(len(anchors))
    dollars[active] = cap * np.sign(forecasts[active])
    current_prices = prices[ALGO, anchors - 1]
    shares = np.round(dollars / current_prices).astype(int)
    limit = np.floor(ALGO_CAP / current_prices).astype(int)
    targets[anchors, ALGO] = np.clip(shares, -limit, limit)
    return targets, forecasts


def account(
    prices: np.ndarray,
    targets: np.ndarray,
    start: int,
    end: int,
    commission: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Match the official evaluator's delayed-fee convention."""
    cash = 0.0
    value = 0.0
    prior_fee = 0.0
    current = np.zeros(N_INST)
    pnl = []
    turnover = 0.0
    fees = 0.0
    for t in range(start - 1, end + 1):
        current_prices = prices[:, t - 1]
        new = targets[t].astype(float) if t < end else current.copy()
        delta = new - current
        cash -= float(current_prices @ delta) + prior_fee
        volumes = current_prices * np.abs(delta)
        prior_fee = float(volumes @ commission)
        turnover += float(volumes.sum())
        fees += prior_fee
        current = new
        new_value = cash + float(current @ current_prices)
        day_pnl = new_value - value
        value = new_value
        if t >= start:
            pnl.append(day_pnl)
    return np.asarray(pnl), turnover, fees


def evaluate_standalone(
    prices: np.ndarray,
    spec: ModelSpec,
    fit_end: int,
    windows: tuple[tuple[str, int, int], ...],
    *,
    cap: float = ALGO_CAP,
    algo_fee: float = ALGO_FEE,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    model = fit_model(prices, spec, fit_end)
    commission = np.full(N_INST, OTHER_FEE)
    commission[ALGO] = algo_fee
    results = {}
    forecast_stats = {}
    for name, start, end in windows:
        targets, forecasts = signal_targets(
            prices, spec, model, start, end, cap=cap
        )
        pnl, turnover, fees = account(
            prices, targets, start, end, commission
        )
        results[name] = metrics(
            pnl, turnover, fees, targets, prices, start, end
        )

        label_anchors = np.arange(start - 1, end - spec.horizon + 1)
        label_forecasts = predict(prices, spec, model, label_anchors)
        actual = forward_target(prices, label_anchors, spec.horizon)
        forecast_stats[name] = {
            "n_labels": int(len(actual)),
            "forecast_target_corr": float(
                np.corrcoef(label_forecasts, actual)[0, 1]
            ),
            "forward_sign_accuracy": float(
                np.mean(np.sign(label_forecasts) == np.sign(actual))
            ),
            "mean_forecast": float(np.mean(label_forecasts)),
            "std_forecast": float(np.std(label_forecasts)),
            "mean_actual_forward_return": float(np.mean(actual)),
        }
    return results, forecast_stats


def candidate_specs() -> tuple[ModelSpec, ...]:
    return tuple(
        ModelSpec(family, horizon, lam, hurdle)
        for family in MODEL_FAMILIES
        for horizon in HORIZONS
        for lam in RIDGE_LAMBDAS
        for hurdle in HURDLES
    )


def _flatten_result(
    spec: ModelSpec,
    results: dict[str, dict[str, float]],
    forecasts: dict[str, dict[str, float]],
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "name": spec.name,
        "family": spec.family,
        "horizon": spec.horizon,
        "ridge_lambda": spec.ridge_lambda,
        "hurdle": spec.hurdle,
    }
    for window, values in results.items():
        for key, value in values.items():
            row[f"{window}_{key}"] = value
    for window, values in forecasts.items():
        for key, value in values.items():
            row[f"{window}_{key}"] = value
    return row


def select() -> None:
    """Select and freeze a specification without reading days 751-1000."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices(750)
    print(f"Selection loaded exactly {prices.shape[1]} days")
    rows = []
    for index, spec in enumerate(candidate_specs(), start=1):
        development, development_forecasts = evaluate_standalone(
            prices, spec, 250, DEVELOPMENT_WINDOWS
        )
        validation, validation_forecasts = evaluate_standalone(
            prices, spec, 500, VALIDATION_WINDOWS
        )
        results = development | validation
        forecast_results = development_forecasts | validation_forecasts
        row = _flatten_result(spec, results, forecast_results)
        half_names = (
            "development_a_251_375",
            "development_b_376_500",
            "validation_a_501_625",
            "validation_b_626_750",
        )
        half_scores = np.asarray(
            [results[name]["score"] for name in half_names]
        )
        half_pnl = np.asarray(
            [results[name]["total_pnl"] for name in half_names]
        )
        row["positive_half_count"] = int(np.sum(half_pnl > 0.0))
        row["half_score_floor"] = float(np.min(half_scores))
        row["half_score_median"] = float(np.median(half_scores))
        row["eligible"] = bool(
            np.all(half_pnl > 0.0)
            and results["development_251_500"]["total_pnl"] > 0.0
            and results["validation_501_750"]["total_pnl"] > 0.0
        )
        rows.append(row)
        if index % 50 == 0:
            print(f"Evaluated {index} candidates")

    table = pd.DataFrame(rows)
    eligible = table.loc[table["eligible"]].copy()
    if eligible.empty:
        ranked = table.sort_values(
            [
                "positive_half_count",
                "half_score_floor",
                "half_score_median",
                "validation_501_750_score",
            ],
            ascending=False,
        )
        selection_rule = "fallback_max_positive_halves"
    else:
        ranked = eligible.sort_values(
            [
                "half_score_floor",
                "half_score_median",
                "validation_501_750_score",
            ],
            ascending=False,
        )
        selection_rule = "positive_all_four_halves_then_max_floor"
    selected_name = str(ranked.iloc[0]["name"])
    spec_lookup = {spec.name: spec for spec in candidate_specs()}
    selected = spec_lookup[selected_name]

    table.sort_values(
        [
            "eligible",
            "positive_half_count",
            "half_score_floor",
            "half_score_median",
        ],
        ascending=False,
    ).to_csv(DEVELOPMENT_PATH, index=False)
    payload = {
        "selection_rule": selection_rule,
        "selected": asdict(selected),
        "selected_name": selected.name,
        "candidate_count": len(table),
        "eligible_count": int(table["eligible"].sum()),
        "data_boundary": 750,
        "test_data_read_during_selection": False,
        "selected_development_row": (
            table.loc[table["name"] == selected.name].iloc[0].to_dict()
        ),
    }
    FROZEN_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def build_base_targets(prices: np.ndarray) -> np.ndarray:
    strategy = importlib.import_module("four_sigmas")
    strategy.reset()
    targets = np.zeros((prices.shape[1] + 1, N_INST), dtype=int)
    limits = np.full(N_INST, OTHER_CAP)
    limits[ALGO] = ALGO_CAP
    for history_size in range(1, prices.shape[1]):
        requested = strategy.getMyPosition(prices[:, :history_size])
        current_prices = prices[:, history_size - 1]
        share_limits = np.floor(limits / current_prices).astype(int)
        targets[history_size] = np.clip(
            requested, -share_limits, share_limits
        ).astype(int)
    return targets


def _evaluate_targets(
    prices: np.ndarray,
    targets: np.ndarray,
    windows: tuple[tuple[str, int, int], ...],
    commission: np.ndarray,
) -> dict[str, dict[str, float]]:
    result = {}
    for name, start, end in windows:
        pnl, turnover, fees = account(
            prices, targets, start, end, commission
        )
        result[name] = metrics(
            pnl, turnover, fees, targets, prices, start, end
        )
    return result


def evaluate() -> None:
    """Evaluate the frozen choice once on the exposed final window."""
    if not FROZEN_PATH.exists():
        raise FileNotFoundError("Run the `select` command first")
    frozen = json.loads(FROZEN_PATH.read_text())
    spec = ModelSpec(**frozen["selected"])
    prices = load_prices(None)
    if prices.shape[1] < 1000:
        raise ValueError("The exposed evaluation requires 1,000 price days")

    standalone, forecast_stats = evaluate_standalone(
        prices, spec, 750, EXPOSED_WINDOWS
    )
    standalone_10k, _ = evaluate_standalone(
        prices,
        spec,
        750,
        EXPOSED_WINDOWS,
        cap=10_000.0,
    )
    standalone_other_fee, _ = evaluate_standalone(
        prices,
        spec,
        750,
        EXPOSED_WINDOWS,
        algo_fee=OTHER_FEE,
    )
    selected_development_other_fee, _ = evaluate_standalone(
        prices,
        spec,
        250,
        (("development_251_500", 251, 500),),
        algo_fee=OTHER_FEE,
    )
    selected_validation_other_fee, _ = evaluate_standalone(
        prices,
        spec,
        500,
        (("validation_501_750", 501, 750),),
        algo_fee=OTHER_FEE,
    )

    base_targets = build_base_targets(prices)
    commission = np.full(N_INST, OTHER_FEE)
    commission[ALGO] = ALGO_FEE
    all_windows = (
        ("development_251_500", 251, 500, 250),
        ("validation_501_750", 501, 750, 500),
        ("exposed_751_1000", 751, 1000, 750),
    )
    base = _evaluate_targets(
        prices,
        base_targets,
        tuple((name, start, end) for name, start, end, _ in all_windows),
        commission,
    )
    combined = {}
    for name, start, end, fit_end in all_windows:
        window_model = fit_model(prices, spec, fit_end)
        algo_targets, _ = signal_targets(
            prices, spec, window_model, start, end, cap=ALGO_CAP
        )
        combined_targets = base_targets.copy()
        combined_targets[:, ALGO] = algo_targets[:, ALGO]
        pnl, turnover, fees = account(
            prices, combined_targets, start, end, commission
        )
        combined[name] = metrics(
            pnl,
            turnover,
            fees,
            combined_targets,
            prices,
            start,
            end,
        )

    rows = []
    for name, values in standalone.items():
        rows.append({"variant": "algo_100k_0.2bp", "window": name, **values})
    for name, values in standalone_10k.items():
        rows.append({"variant": "algo_10k_0.2bp", "window": name, **values})
    for name, values in standalone_other_fee.items():
        rows.append({"variant": "algo_100k_1.0bp", "window": name, **values})
    for name, values in selected_development_other_fee.items():
        rows.append({"variant": "algo_100k_1.0bp", "window": name, **values})
    for name, values in selected_validation_other_fee.items():
        rows.append({"variant": "algo_100k_1.0bp", "window": name, **values})
    for name, values in base.items():
        rows.append({"variant": "base", "window": name, **values})
    for name, values in combined.items():
        rows.append({"variant": "base_plus_algo", "window": name, **values})
    final = pd.DataFrame(rows)
    final.to_csv(FINAL_PATH, index=False)

    # This is a post-selection stability diagnostic only.  It cannot replace
    # the specification frozen above, even if another row looks better here.
    development_table = pd.read_csv(DEVELOPMENT_PATH)
    spec_lookup = {candidate.name: candidate for candidate in candidate_specs()}
    exposed_rows = []
    for _, development_candidate in development_table.iterrows():
        candidate = spec_lookup[str(development_candidate["name"])]
        candidate_result, candidate_forecast = evaluate_standalone(
            prices,
            candidate,
            750,
            (("exposed_751_1000", 751, 1000),),
        )
        values = candidate_result["exposed_751_1000"]
        forecast_values = candidate_forecast["exposed_751_1000"]
        exposed_rows.append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "horizon": candidate.horizon,
                "ridge_lambda": candidate.ridge_lambda,
                "hurdle": candidate.hurdle,
                "was_eligible": bool(development_candidate["eligible"]),
                "development_half_score_floor": float(
                    development_candidate["half_score_floor"]
                ),
                "validation_score": float(
                    development_candidate["validation_501_750_score"]
                ),
                "exposed_score": values["score"],
                "exposed_total_pnl": values["total_pnl"],
                "exposed_sharpe": values["sharpe"],
                "exposed_forecast_target_corr": forecast_values[
                    "forecast_target_corr"
                ],
            }
        )
    exposed_table = pd.DataFrame(exposed_rows)
    exposed_table.to_csv(EXPOSED_CANDIDATES_PATH, index=False)

    # Re-run the *development-only* comparison at the ordinary 1 bp fee.  This
    # directly tests whether the lower ALGO fee changes which horizon clears
    # the same robustness rule.  The frozen ALGO selection is not altered.
    development_prices = prices[:, :750]
    fee_counterfactual_rows = []
    for candidate in candidate_specs():
        candidate_development, _ = evaluate_standalone(
            development_prices,
            candidate,
            250,
            DEVELOPMENT_WINDOWS,
            algo_fee=OTHER_FEE,
        )
        candidate_validation, _ = evaluate_standalone(
            development_prices,
            candidate,
            500,
            VALIDATION_WINDOWS,
            algo_fee=OTHER_FEE,
        )
        candidate_results = candidate_development | candidate_validation
        half_names = (
            "development_a_251_375",
            "development_b_376_500",
            "validation_a_501_625",
            "validation_b_626_750",
        )
        half_scores = np.asarray(
            [candidate_results[name]["score"] for name in half_names]
        )
        half_pnl = np.asarray(
            [candidate_results[name]["total_pnl"] for name in half_names]
        )
        fee_counterfactual_rows.append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "horizon": candidate.horizon,
                "ridge_lambda": candidate.ridge_lambda,
                "hurdle": candidate.hurdle,
                "positive_half_count": int(np.sum(half_pnl > 0.0)),
                "half_score_floor": float(np.min(half_scores)),
                "half_score_median": float(np.median(half_scores)),
                "validation_score": candidate_results[
                    "validation_501_750"
                ]["score"],
                "eligible": bool(
                    np.all(half_pnl > 0.0)
                    and candidate_results[
                        "development_251_500"
                    ]["total_pnl"]
                    > 0.0
                    and candidate_results[
                        "validation_501_750"
                    ]["total_pnl"]
                    > 0.0
                ),
            }
        )
    fee_counterfactual = pd.DataFrame(fee_counterfactual_rows)
    fee_counterfactual.to_csv(FEE_COUNTERFACTUAL_PATH, index=False)
    fee_counterfactual_eligible = fee_counterfactual.loc[
        fee_counterfactual["eligible"]
    ]
    if fee_counterfactual_eligible.empty:
        fee_counterfactual_ranked = fee_counterfactual.sort_values(
            [
                "positive_half_count",
                "half_score_floor",
                "half_score_median",
                "validation_score",
            ],
            ascending=False,
        )
    else:
        fee_counterfactual_ranked = fee_counterfactual_eligible.sort_values(
            ["half_score_floor", "half_score_median", "validation_score"],
            ascending=False,
        )
    ordinary_fee_choice = fee_counterfactual_ranked.iloc[0]

    fitted_by_boundary = {
        boundary: fit_model(prices, spec, boundary)
        for boundary in (250, 500, 750)
    }
    names = feature_names(spec.family)
    coefficient_rows = []
    for feature_index, feature_name in enumerate(names):
        coefficient_rows.append(
            {
                "feature": feature_name,
                **{
                    f"standardized_coefficient_fit_through_{boundary}": (
                        fitted.coefficients[feature_index]
                    )
                    for boundary, fitted in fitted_by_boundary.items()
                },
            }
        )
    coefficient_table = pd.DataFrame(coefficient_rows)
    coefficient_table.to_csv(COEFFICIENT_PATH, index=False)
    coef_500 = fitted_by_boundary[500].coefficients
    coef_750 = fitted_by_boundary[750].coefficients
    coefficient_cosine = float(
        (coef_500 @ coef_750)
        / (
            np.linalg.norm(coef_500)
            * np.linalg.norm(coef_750)
            + 1.0e-12
        )
    )
    largest_500 = coefficient_table.iloc[
        np.argsort(np.abs(coef_500))[-5:][::-1]
    ]
    largest_750 = coefficient_table.iloc[
        np.argsort(np.abs(coef_750))[-5:][::-1]
    ]

    development_row = frozen["selected_development_row"]
    report_lines = [
        "# Dedicated ALGO horizon audit",
        "",
        "## Protocol",
        "",
        "- Candidate selection physically read only days 1-750.",
        "- Development check: refit through day 250, evaluate days 251-500.",
        "- Validation selection: refit through day 500, evaluate days 501-750.",
        "- Final evaluation: refit through day 750, evaluate days 751-1000 once.",
        "- ALGO commission is 0.2 bp one-way and its direct cap is $100,000.",
        "- No production strategy code was changed.",
        "",
        "## Frozen selection",
        "",
        f"- Specification: `{spec.name}`",
        f"- Selection rule: `{frozen['selection_rule']}`",
        f"- Eligible candidates: {frozen['eligible_count']} / "
        f"{frozen['candidate_count']}",
        f"- Forecast horizon: {spec.horizon} day(s)",
        f"- Feature family: `{spec.family}`",
        f"- Ridge lambda: {spec.ridge_lambda:g}",
        f"- Cost hurdle: `{spec.hurdle}`",
        "",
        "## Selected development evidence",
        "",
        "| Window | Score | Total PnL | Sharpe | Forecast/target correlation |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in (
        "development_251_500",
        "development_a_251_375",
        "development_b_376_500",
        "validation_501_750",
        "validation_a_501_625",
        "validation_b_626_750",
    ):
        report_lines.append(
            f"| {name} | "
            f"{development_row[f'{name}_score']:.4f} | "
            f"${development_row[f'{name}_total_pnl']:,.2f} | "
            f"{development_row[f'{name}_sharpe']:.4f} | "
            f"{development_row[f'{name}_forecast_target_corr']:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Exposed standalone result",
            "",
            "| Variant | Window | Score | Total PnL | Sharpe | Turnover | Fees |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in (
        "algo_100k_0.2bp",
        "algo_10k_0.2bp",
        "algo_100k_1.0bp",
    ):
        values = final.loc[
            (final["variant"] == variant)
            & (final["window"] == "exposed_751_1000")
        ].iloc[0]
        report_lines.append(
            f"| {variant} | exposed_751_1000 | "
            f"{values['score']:.4f} | ${values['total_pnl']:,.2f} | "
            f"{values['sharpe']:.4f} | ${values['turnover']:,.0f} | "
            f"${values['fees']:,.2f} |"
        )

    report_lines.extend(
        [
            "",
            "## Current book interaction",
            "",
            "| Window | Base score | Combined score | Score delta | "
            "Base PnL | Combined PnL | PnL delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, _, _, _ in all_windows:
        base_values = base[name]
        combined_values = combined[name]
        report_lines.append(
            f"| {name} | {base_values['score']:.4f} | "
            f"{combined_values['score']:.4f} | "
            f"{combined_values['score'] - base_values['score']:+.4f} | "
            f"${base_values['total_pnl']:,.2f} | "
            f"${combined_values['total_pnl']:,.2f} | "
            f"${combined_values['total_pnl'] - base_values['total_pnl']:+,.2f} |"
        )

    exposed_forecast = forecast_stats["exposed_751_1000"]
    eligible_exposed = exposed_table.loc[exposed_table["was_eligible"]]
    eligible_positive = int(
        (eligible_exposed["exposed_total_pnl"] > 0.0).sum()
    )
    selected_validation_fee_02 = float(
        development_row["validation_501_750_fees"]
    )
    selected_validation_fee_10 = selected_validation_other_fee[
        "validation_501_750"
    ]["fees"]
    selected_validation_pnl_10 = selected_validation_other_fee[
        "validation_501_750"
    ]["total_pnl"]
    report_lines.extend(
        [
            "",
            "## Exposed forecast diagnostic",
            "",
            f"- Forward-return correlation: "
            f"{exposed_forecast['forecast_target_corr']:.4f}",
            f"- Forward-return sign accuracy: "
            f"{exposed_forecast['forward_sign_accuracy']:.4f}",
            f"- Mean forecast: {exposed_forecast['mean_forecast']:.6f}",
            f"- Forecast standard deviation: "
            f"{exposed_forecast['std_forecast']:.6f}",
            "",
            "## Is the result genuinely fee-dependent?",
            "",
            f"- Validation PnL at 0.2 bp: "
            f"${development_row['validation_501_750_total_pnl']:,.2f}.",
            f"- Validation PnL at 1.0 bp using the identical positions: "
            f"${selected_validation_pnl_10:,.2f}.",
            f"- Validation fees rise from ${selected_validation_fee_02:,.2f} "
            f"to ${selected_validation_fee_10:,.2f}.",
            f"- Repeating the complete development-only selection at 1.0 bp "
            f"still chooses `{ordinary_fee_choice['name']}` "
            f"(horizon {int(ordinary_fee_choice['horizon'])}).",
            f"- Development-eligible candidates remaining profitable in the "
            f"exposed window: {eligible_positive} / {len(eligible_exposed)}.",
            f"- Standardized-coefficient cosine similarity between fits "
            f"through days 500 and 750: {coefficient_cosine:.4f}.",
            "",
            "The complete exposed-candidate table is diagnostic only; it was "
            "not used to replace the frozen selection.",
            "",
            "## Largest standardized model coefficients",
            "",
            "| Fit boundary | Feature | Coefficient |",
            "| --- | --- | ---: |",
        ]
    )
    for boundary, largest in ((500, largest_500), (750, largest_750)):
        column = f"standardized_coefficient_fit_through_{boundary}"
        for _, coefficient_row in largest.iterrows():
            report_lines.append(
                f"| {boundary} | {coefficient_row['feature']} | "
                f"{coefficient_row[column]:+.5f} |"
            )
    report_lines.extend(
        [
            "",
            "The exposed result is descriptive because days 751-1000 have "
            "already been seen in this research process.",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "evaluate"))
    args = parser.parse_args()
    if args.command == "select":
        select()
    else:
        evaluate()


if __name__ == "__main__":
    main()

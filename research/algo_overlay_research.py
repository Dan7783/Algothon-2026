"""Leakage-safe ALGO overlay research for the protected barbell strategy.

The selection phase reads exactly the first 750 rows of ``prices.txt`` and
compares:

* ALGO as a rolling factor hedge for the existing book;
* ALGO as a directly predicted ridge sleeve; and
* deliberately simple one-day ALGO momentum/reversal controls.

The selected configuration is written before the held-out phase can run.
The test phase refuses to run unless that frozen selection exists, then
evaluates the unchanged baseline and the frozen overlay on days 751-1000.
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
OUT_DIR = ROOT / "analysis_outputs" / "algo_overlay"
FROZEN_PATH = OUT_DIR / "frozen_selection.json"
SELECTION_REPORT_PATH = OUT_DIR / "development_results.json"
TEST_REPORT_PATH = OUT_DIR / "heldout_results.json"
BASE_MODULE = "testbedv2_tuned_ridge_protectedbarbell"

N_DEVELOPMENT_DAYS = 750
TEST_START = 751
TEST_END = 1000
COMM = np.full(51, 1.0e-4)
COMM[0] = 2.0e-5
LIMIT = np.full(51, 10_000.0)
LIMIT[0] = 100_000.0
FOLDS = (
    ("dev_1", 126, 250),
    ("dev_2", 251, 375),
    ("dev_3", 376, 500),
    ("val_1", 501, 625),
    ("val_2", 626, 750),
)


@dataclass(frozen=True)
class OverlaySpec:
    family: str
    window: int = 0
    hedge_ratio: float = 0.0
    cap: float = 0.0
    beta_shrink: float = 0.0
    ridge_lambda: float = 0.0
    threshold: float = 0.0
    direction: float = 0.0

    @property
    def name(self) -> str:
        parts = [self.family]
        if self.window:
            parts.append(f"w{self.window}")
        if self.hedge_ratio:
            parts.append(f"h{self.hedge_ratio:g}")
        if self.cap:
            parts.append(f"c{self.cap / 1000:g}k")
        if self.beta_shrink:
            parts.append(f"s{self.beta_shrink:g}")
        if self.ridge_lambda:
            parts.append(f"l{self.ridge_lambda:g}")
        if self.threshold:
            parts.append(f"t{self.threshold:g}")
        if self.direction:
            parts.append("mom" if self.direction > 0 else "rev")
        return "_".join(parts)


def load_prices(nrows: int | None) -> np.ndarray:
    """Read no rows beyond the permitted research boundary."""
    frame = pd.read_csv(PRICE_PATH, sep=r"\s+", nrows=nrows)
    values = frame.to_numpy(float).T
    if values.shape[0] != 51:
        raise ValueError(f"Expected 51 instruments, got {values.shape[0]}")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Prices must be finite and strictly positive")
    return values


def load_base():
    return importlib.import_module(BASE_MODULE)


def score(pnl: np.ndarray) -> float:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe * sharpe / (sharpe * sharpe + 1.0)


def metrics(pnl: np.ndarray, turnover: float, fees: float) -> dict[str, float]:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    return {
        "score": score(pnl),
        "mean": mean,
        "std": std,
        "sharpe": sqrt(250.0) * mean / std if std > 0.0 else 0.0,
        "total_pnl": float(np.sum(pnl)),
        "hit_rate": float(np.mean(pnl > 0.0)),
        "turnover": float(turnover),
        "fees": float(fees),
    }


def build_base_targets(prices: np.ndarray) -> np.ndarray:
    """Generate the baseline positions once, causally, for every history size."""
    strategy = load_base()
    strategy.reset()
    nins, ndays = prices.shape
    targets = np.zeros((ndays + 1, nins), dtype=int)
    for history_size in range(1, ndays):
        requested = strategy.getMyPosition(prices[:, :history_size])
        current = prices[:, history_size - 1]
        limits = (LIMIT / current).astype(int)
        targets[history_size] = np.clip(
            requested, -limits, limits
        ).astype(int)
    return targets


def rolling_factor_loading(
    prices: np.ndarray,
    base_targets: np.ndarray,
    window: int,
    beta_shrink: float,
) -> np.ndarray:
    """Estimate the baseline book's dollar beta to ALGO at each decision."""
    ndays = prices.shape[1]
    loading = np.zeros(ndays + 1)
    log_returns = np.diff(np.log(prices), axis=1)
    for history_size in range(window + 1, ndays):
        returns = log_returns[:, history_size - 1 - window : history_size - 1]
        algo = returns[0] - returns[0].mean()
        others = returns[1:] - returns[1:].mean(axis=1, keepdims=True)
        variance = float(algo @ algo)
        if variance <= 1.0e-14:
            continue
        beta = (others @ algo) / variance
        # Shrinking toward unit beta is appropriate because ALGO is a
        # near-equal-weight market proxy, and stabilises short windows.
        beta = (1.0 - beta_shrink) * beta + beta_shrink
        beta = np.clip(beta, -1.0, 3.0)
        current_dollars = (
            base_targets[history_size, 1:]
            * prices[1:, history_size - 1]
        )
        loading[history_size] = float(current_dollars @ beta)
    return loading


def algo_ridge_forecasts(
    prices: np.ndarray,
    window: int,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict ALGO from lagged non-ALGO returns using only available history."""
    ndays = prices.shape[1]
    prediction = np.zeros(ndays + 1)
    strength = np.zeros(ndays + 1)
    returns = np.diff(np.log(prices), axis=1).T
    minimum = max(126, window if window else 126)
    for history_size in range(minimum + 1, ndays):
        history = returns[: history_size - 1]
        if window:
            history = history[-window:]
        x_raw = history[:-1, 1:]
        y_raw = history[1:, 0]
        x_scale = x_raw.std(axis=0) + 1.0e-12
        y_scale = float(y_raw.std()) + 1.0e-12
        x = x_raw / x_scale
        y = y_raw / y_scale
        coefficients = np.linalg.solve(
            x.T @ x
            + ridge_lambda * len(x) * np.eye(x.shape[1]),
            x.T @ y,
        )
        fitted = x @ coefficients
        pred = float((history[-1, 1:] / x_scale) @ coefficients)
        prediction[history_size] = pred
        strength[history_size] = abs(pred) / (
            float(fitted.std()) + 1.0e-12
        )
    return prediction, strength


def overlay_targets(
    prices: np.ndarray,
    base_targets: np.ndarray,
    spec: OverlaySpec,
    factor_cache: dict[tuple[int, float], np.ndarray],
    ridge_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    targets = base_targets.copy()
    ndays = prices.shape[1]
    algo_dollars = np.zeros(ndays + 1)

    if spec.family == "baseline":
        return targets
    if spec.family == "factor_hedge":
        loading = factor_cache[(spec.window, spec.beta_shrink)]
        algo_dollars = np.clip(
            -spec.hedge_ratio * loading,
            -spec.cap,
            spec.cap,
        )
    elif spec.family == "algo_ridge":
        prediction, strength = ridge_cache[
            (spec.window, spec.ridge_lambda)
        ]
        active = strength > spec.threshold
        algo_dollars[active] = (
            spec.cap * np.sign(prediction[active])
        )
    elif spec.family == "algo_lag":
        last_return = np.zeros(ndays + 1)
        last_return[2:ndays] = np.diff(np.log(prices[0]))[: ndays - 2]
        algo_dollars = (
            spec.cap * spec.direction * np.sign(last_return)
        )
    else:
        raise ValueError(f"Unknown overlay family: {spec.family}")

    for history_size in range(1, ndays):
        requested = int(
            np.round(
                algo_dollars[history_size]
                / prices[0, history_size - 1]
            )
        )
        limit = int(
            LIMIT[0] / prices[0, history_size - 1]
        )
        targets[history_size, 0] = int(
            np.clip(requested, -limit, limit)
        )
    return targets


def account(
    prices: np.ndarray,
    targets: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, float, float]:
    """Match the official evaluator's position, fee, and mark timing."""
    cash = 0.0
    value = 0.0
    prior_fee = 0.0
    current = np.zeros(prices.shape[0])
    pnl = []
    turnover = 0.0
    fees = 0.0
    for t in range(start - 1, end + 1):
        current_prices = prices[:, t - 1]
        new = targets[t].astype(float) if t < end else current.copy()
        delta = new - current
        cash -= float(current_prices @ delta) + prior_fee
        volumes = current_prices * np.abs(delta)
        prior_fee = float(volumes @ COMM)
        fees += prior_fee
        turnover += float(volumes.sum())
        current = new
        new_value = cash + float(current @ current_prices)
        day_pnl = new_value - value
        value = new_value
        if t >= start:
            pnl.append(day_pnl)
    return np.asarray(pnl), turnover, fees


def evaluate_spec(
    prices: np.ndarray,
    targets: np.ndarray,
    windows: tuple[tuple[str, int, int], ...],
) -> dict[str, dict[str, float]]:
    result = {}
    for name, start, end in windows:
        pnl, turnover, fees = account(prices, targets, start, end)
        result[name] = metrics(pnl, turnover, fees)
    return result


def development_candidates() -> tuple[OverlaySpec, ...]:
    candidates = [OverlaySpec("baseline")]
    for window in (60, 125, 250):
        for shrink in (0.0, 0.5):
            for ratio in (0.5, 1.0):
                for cap in (50_000.0, 100_000.0):
                    candidates.append(
                        OverlaySpec(
                            "factor_hedge",
                            window=window,
                            hedge_ratio=ratio,
                            cap=cap,
                            beta_shrink=shrink,
                        )
                    )
    for window in (250, 0):
        for ridge_lambda in (0.3, 1.0, 3.0):
            for threshold in (0.75, 1.25):
                for cap in (25_000.0, 50_000.0, 100_000.0):
                    candidates.append(
                        OverlaySpec(
                            "algo_ridge",
                            window=window,
                            ridge_lambda=ridge_lambda,
                            threshold=threshold,
                            cap=cap,
                        )
                    )
    for direction in (-1.0, 1.0):
        for cap in (25_000.0, 50_000.0):
            candidates.append(
                OverlaySpec(
                    "algo_lag",
                    cap=cap,
                    direction=direction,
                )
            )
    return tuple(candidates)


def select() -> None:
    # nrows makes the embargo physical: pandas never reads rows 751-1000.
    prices = load_prices(N_DEVELOPMENT_DAYS)
    print(f"Selection loaded exactly {prices.shape[1]} development days")
    base_targets = build_base_targets(prices)
    candidates = development_candidates()

    factor_keys = {
        (spec.window, spec.beta_shrink)
        for spec in candidates
        if spec.family == "factor_hedge"
    }
    ridge_keys = {
        (spec.window, spec.ridge_lambda)
        for spec in candidates
        if spec.family == "algo_ridge"
    }
    factor_cache = {
        key: rolling_factor_loading(
            prices, base_targets, key[0], key[1]
        )
        for key in sorted(factor_keys)
    }
    ridge_cache = {
        key: algo_ridge_forecasts(prices, key[0], key[1])
        for key in sorted(ridge_keys)
    }

    rows = []
    detailed = {}
    for spec in candidates:
        targets = overlay_targets(
            prices, base_targets, spec, factor_cache, ridge_cache
        )
        fold_metrics = evaluate_spec(prices, targets, FOLDS)
        detailed[spec.name] = {
            "spec": asdict(spec),
            "folds": fold_metrics,
        }
        fold_scores = np.array(
            [fold_metrics[name]["score"] for name, _, _ in FOLDS]
        )
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "median_fold_score": float(np.median(fold_scores)),
                "mean_fold_score": float(np.mean(fold_scores)),
                "min_fold_score": float(np.min(fold_scores)),
            }
        )

    baseline_scores = np.array(
        [
            detailed["baseline"]["folds"][name]["score"]
            for name, _, _ in FOLDS
        ]
    )
    for row in rows:
        candidate_scores = np.array(
            [
                detailed[row["name"]]["folds"][name]["score"]
                for name, _, _ in FOLDS
            ]
        )
        deltas = candidate_scores - baseline_scores
        row["median_fold_delta"] = float(np.median(deltas))
        row["mean_fold_delta"] = float(np.mean(deltas))
        row["min_fold_delta"] = float(np.min(deltas))
        row["positive_folds"] = int(np.sum(deltas > 0.0))

    eligible = [
        row
        for row in rows
        if row["family"] != "baseline"
        and row["positive_folds"] >= 4
        and row["median_fold_delta"] > 0.0
    ]
    # Robustness first: positive-fold count, median improvement, then mean.
    eligible.sort(
        key=lambda row: (
            row["positive_folds"],
            row["median_fold_delta"],
            row["mean_fold_delta"],
        ),
        reverse=True,
    )
    selected_row = eligible[0] if eligible else next(
        row for row in rows if row["name"] == "baseline"
    )
    selected_spec = detailed[selected_row["name"]]["spec"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "data_embargo": {
            "selection_rows_loaded": int(prices.shape[1]),
            "last_selection_day": N_DEVELOPMENT_DAYS,
            "heldout_days": [TEST_START, TEST_END],
        },
        "folds": [
            {"name": name, "start": start, "end": end}
            for name, start, end in FOLDS
        ],
        "ranking": sorted(
            rows,
            key=lambda row: (
                row["positive_folds"],
                row["median_fold_delta"],
                row["mean_fold_delta"],
            ),
            reverse=True,
        ),
        "details": detailed,
    }
    SELECTION_REPORT_PATH.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    frozen = {
        "base_module": BASE_MODULE,
        "selection_rule": (
            "At least 4/5 positive fold deltas; maximize positive folds, "
            "median fold delta, then mean fold delta"
        ),
        "selected_name": selected_row["name"],
        "selected_spec": selected_spec,
        "selected_development_summary": selected_row,
        "selection_data_last_day": N_DEVELOPMENT_DAYS,
        "heldout_window": [TEST_START, TEST_END],
        "heldout_evaluated": False,
    }
    FROZEN_PATH.write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    print("Frozen selection:", selected_row)
    print("Wrote", FROZEN_PATH)


def test() -> None:
    if not FROZEN_PATH.exists():
        raise RuntimeError("Run --phase select before accessing held-out data")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    if frozen.get("selection_data_last_day") != N_DEVELOPMENT_DAYS:
        raise RuntimeError("Frozen selection has an invalid development boundary")
    spec = OverlaySpec(**frozen["selected_spec"])

    prices = load_prices(TEST_END)
    if prices.shape[1] < TEST_END:
        raise ValueError("Held-out data is incomplete")
    base_targets = build_base_targets(prices)
    factor_cache = {}
    ridge_cache = {}
    if spec.family == "factor_hedge":
        key = (spec.window, spec.beta_shrink)
        factor_cache[key] = rolling_factor_loading(
            prices, base_targets, spec.window, spec.beta_shrink
        )
    elif spec.family == "algo_ridge":
        key = (spec.window, spec.ridge_lambda)
        ridge_cache[key] = algo_ridge_forecasts(
            prices, spec.window, spec.ridge_lambda
        )

    baseline_targets = overlay_targets(
        prices,
        base_targets,
        OverlaySpec("baseline"),
        factor_cache,
        ridge_cache,
    )
    selected_targets = overlay_targets(
        prices, base_targets, spec, factor_cache, ridge_cache
    )
    windows = (
        ("heldout", TEST_START, TEST_END),
        ("heldout_751_849", 751, 849),
        ("heldout_850_950", 850, 950),
        ("heldout_951_1000", 951, 1000),
    )
    baseline = evaluate_spec(prices, baseline_targets, windows)
    selected = evaluate_spec(prices, selected_targets, windows)
    result = {
        "frozen_selection": frozen,
        "baseline": baseline,
        "selected": selected,
        "score_delta": {
            name: selected[name]["score"] - baseline[name]["score"]
            for name, _, _ in windows
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_REPORT_PATH.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    frozen["heldout_evaluated"] = True
    frozen["heldout_result_path"] = str(TEST_REPORT_PATH.relative_to(ROOT))
    FROZEN_PATH.write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print("Wrote", TEST_REPORT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("select", "test"), required=True
    )
    args = parser.parse_args()
    if args.phase == "select":
        select()
    else:
        test()


if __name__ == "__main__":
    main()

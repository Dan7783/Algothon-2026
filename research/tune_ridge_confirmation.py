"""Tune the independent ridge conflict-confirmation model.

The development phase is deliberately unable to see observations after day
750.  It performs a coarse grid search followed by a seeded differential-
evolution refinement and locks one candidate to JSON.  The held-out phase
loads that lock, replays the unchanged production strategy through day 1000,
and compares the candidate with the current confirmation parameters.

Only ``RIDGE_CONFIRM_LAM`` and ``RIDGE_CONFIRM_THETA`` vary.  The primary
ridge model, pair agents, pair voting, deadlock overlay, ALGO residual basket,
position limits, share rounding, and commission model remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import algo12_equal_sleeve_research as evaluator


PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "ridge_confirmation_tune"
LOCK_PATH = OUTPUT_DIR / "locked_candidate.json"

BASELINE_LAM = float(strategy.RIDGE_CONFIRM_LAM)
BASELINE_THETA = float(strategy.RIDGE_CONFIRM_THETA)
DEVELOPMENT_END = 750
MATERIAL_HALF_TOLERANCE = 5.0

DEVELOPMENT_WINDOWS = (
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("TRAIN", 251, 500),
    ("VALIDATION_A", 501, 625),
    ("VALIDATION_B", 626, 750),
    ("VALIDATION", 501, 750),
)
HELDOUT_WINDOWS = (
    ("TEST_A", 751, 875),
    ("TEST_B", 876, 1000),
    ("TEST", 751, 1000),
)
HALF_WINDOWS = (
    "TRAIN_A",
    "TRAIN_B",
    "VALIDATION_A",
    "VALIDATION_B",
)
GRID_LAMBDAS = (
    0.03,
    0.06,
    0.10,
    0.20,
    0.35,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
    5.00,
    8.00,
    12.00,
)
GRID_THETAS = (
    0.10,
    0.20,
    0.31,
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
    1.10,
    1.35,
    1.70,
    2.20,
)


@dataclass
class ReplayCache:
    """All production state that is independent of confirmation parameters."""

    prices: np.ndarray
    pair_vote: np.ndarray
    pair_active: np.ndarray
    ridge_sign: np.ndarray
    ridge_strength: np.ndarray
    ridge_active: np.ndarray
    ridge_latent_sign: np.ndarray
    ridge_gate: np.ndarray
    crash: np.ndarray
    algo_ready: np.ndarray
    algo_requested: np.ndarray
    algo_betas: np.ndarray
    production_targets: np.ndarray


def load_prices(last_day: int | None = None) -> np.ndarray:
    """Load prices, optionally making later observations inaccessible."""
    frame = pd.read_csv(PRICE_PATH, sep=r"\s+")
    prices = frame.to_numpy(dtype=float).T
    if last_day is not None:
        prices = prices[:, :last_day].copy()
    return prices


def dollars_to_shares(
    dollars: np.ndarray,
    current_prices: np.ndarray,
) -> np.ndarray:
    safe = np.maximum(current_prices, strategy._EPS_DENOM)
    limits = np.floor(strategy.BANG_BANG_LIMITS / safe).astype(int)
    return np.clip(
        np.round(dollars / safe),
        -limits,
        limits,
    ).astype(int)


def build_replay_cache(prices: np.ndarray) -> ReplayCache:
    """Run the stateful pair book once and cache confirmation-free state."""
    shape = (prices.shape[1] + 1, strategy.N_INST)
    pair_vote = np.zeros(shape)
    pair_active = np.zeros(shape, dtype=bool)
    ridge_sign = np.zeros(shape)
    ridge_strength = np.zeros(shape)
    ridge_active = np.zeros(shape, dtype=bool)
    ridge_latent_sign = np.zeros(shape)
    ridge_gate = np.zeros(prices.shape[1] + 1, dtype=bool)
    crash = np.zeros(shape, dtype=bool)
    algo_ready = np.zeros(prices.shape[1] + 1, dtype=bool)
    algo_requested = np.zeros(shape)
    algo_betas = np.zeros(shape)
    production_targets = np.zeros(shape, dtype=int)

    strategy.reset()
    for history_size in range(1, prices.shape[1]):
        known = prices[:, :history_size]
        sleeves = strategy.sleeve_breakdown(known)
        if sleeves is None:
            continue
        pair_vote[history_size] = np.asarray(
            sleeves["pair_vote"],
            dtype=float,
        )
        pair_active[history_size] = np.asarray(
            sleeves["pair_active_mask"],
            dtype=bool,
        )
        ridge_sign[history_size] = np.sign(
            np.asarray(sleeves["ridge_raw"], dtype=float)
        )
        ridge_strength[history_size] = np.asarray(
            sleeves["ridge_strength"],
            dtype=float,
        )
        ridge_active[history_size] = np.asarray(
            sleeves["ridge_active_mask"],
            dtype=bool,
        )
        ridge_latent_sign[history_size] = np.asarray(
            sleeves["ridge_latent_sign"],
            dtype=float,
        )
        ridge_gate[history_size] = bool(sleeves["ridge_gate_pass"])
        crash[history_size] = np.asarray(
            sleeves["crash_mask"],
            dtype=bool,
        )
        algo_ready[history_size] = bool(sleeves["algo_beta_ready"])
        algo_requested[history_size] = np.asarray(
            sleeves["algo_residual_requested"],
            dtype=float,
        )
        algo_betas[history_size] = np.asarray(
            sleeves["algo_return_betas"],
            dtype=float,
        )
        production_targets[history_size] = dollars_to_shares(
            np.asarray(sleeves["dollars"], dtype=float),
            known[:, -1],
        )
    strategy.reset()
    return ReplayCache(
        prices=prices,
        pair_vote=pair_vote,
        pair_active=pair_active,
        ridge_sign=ridge_sign,
        ridge_strength=ridge_strength,
        ridge_active=ridge_active,
        ridge_latent_sign=ridge_latent_sign,
        ridge_gate=ridge_gate,
        crash=crash,
        algo_ready=algo_ready,
        algo_requested=algo_requested,
        algo_betas=algo_betas,
        production_targets=production_targets,
    )


class ConfirmationEvaluator:
    """Evaluate confirmation parameters against one immutable replay cache."""

    def __init__(
        self,
        replay: ReplayCache,
        windows: tuple[tuple[str, int, int], ...],
    ) -> None:
        self.replay = replay
        self.windows = windows
        self._paths: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._results: dict[
            tuple[float, float],
            dict[str, dict[str, float]],
        ] = {}

    @staticmethod
    def lambda_key(value: float) -> float:
        return round(float(value), 10)

    @staticmethod
    def parameter_key(lam: float, theta: float) -> tuple[float, float]:
        return round(float(lam), 10), round(float(theta), 10)

    def confirmation_path(
        self,
        lam: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return latent signs and strengths; theta is applied later."""
        key = self.lambda_key(lam)
        if key in self._paths:
            return self._paths[key]
        prices = self.replay.prices
        signs = np.zeros_like(self.replay.ridge_sign)
        strengths = np.zeros_like(self.replay.ridge_strength)
        log_prices = np.log(np.maximum(prices, strategy._EPS_DENOM))
        all_returns = np.diff(log_prices, axis=1)
        for history_size in range(
            strategy.RIDGE_MIN_T,
            prices.shape[1],
        ):
            _, strength, latent_sign, gate = (
                strategy._ridge_forecast_details(
                    all_returns[:, : history_size - 1],
                    ridge_lam=key,
                    ridge_theta=0.0,
                )
            )
            if gate:
                signs[history_size] = latent_sign
                strengths[history_size] = strength
        self._paths[key] = signs, strengths
        return signs, strengths

    def targets(self, lam: float, theta: float) -> np.ndarray:
        replay = self.replay
        confirm_sign, confirm_strength = self.confirmation_path(lam)
        confirm_active = confirm_strength > float(theta)
        targets = np.zeros_like(replay.production_targets)
        for history_size in range(
            strategy.WARMUP + 10,
            replay.prices.shape[1],
        ):
            (
                _,
                direction,
                _,
                _,
            ) = strategy._resolve_pair_ridge_direction(
                replay.pair_vote[history_size],
                replay.pair_active[history_size],
                replay.ridge_sign[history_size],
                replay.ridge_strength[history_size],
                replay.ridge_active[history_size],
                confirm_sign[history_size],
                confirm_active[history_size],
            )
            day_crash = replay.crash[history_size]
            direction[day_crash] = 0.0
            dollars = strategy.BANG_BANG_LIMITS * direction
            deadlock = (
                strategy.USE_RIDGE_CONFLICT_VOTES
                & replay.ridge_gate[history_size]
                & replay.pair_active[history_size]
                & (
                    np.abs(replay.pair_vote[history_size])
                    < 1.0e-12
                )
                & ~replay.ridge_active[history_size]
                & (replay.ridge_strength[history_size] > 0.0)
                & (
                    replay.ridge_strength[history_size]
                    <= strategy.RIDGE_THETA
                )
                & (replay.ridge_latent_sign[history_size] != 0.0)
                & ~day_crash
            )
            dollars[deadlock] = (
                strategy.DEADLOCK_RIDGE_OVERLAY_FRACTION
                * strategy.BANG_BANG_LIMITS[deadlock]
                * replay.ridge_latent_sign[history_size, deadlock]
            )
            if (
                strategy.USE_ALGO_RESIDUAL_BASKET
                and replay.algo_ready[history_size]
            ):
                dollars, _, _, _ = strategy._algo_beta_neutral_overlay(
                    dollars,
                    replay.algo_requested[history_size],
                    replay.algo_betas[history_size],
                    day_crash,
                )
            targets[history_size] = dollars_to_shares(
                dollars,
                replay.prices[:, history_size - 1],
            )
        return targets

    def evaluate(
        self,
        lam: float,
        theta: float,
    ) -> dict[str, dict[str, float]]:
        key = self.parameter_key(lam, theta)
        if key not in self._results:
            self._results[key] = evaluator.evaluate(
                self.replay.prices,
                self.targets(*key),
                self.windows,
            )
        return self._results[key]


def flatten_result(
    source: str,
    lam: float,
    theta: float,
    result: dict[str, dict[str, float]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": source,
        "lambda": float(lam),
        "theta": float(theta),
    }
    for window, metrics in result.items():
        for metric, value in metrics.items():
            row[f"{window}_{metric}"] = float(value)
    return row


def add_selection_fields(
    frame: pd.DataFrame,
    baseline: pd.Series,
) -> pd.DataFrame:
    result = frame.copy()
    for window in (
        "TRAIN_A",
        "TRAIN_B",
        "TRAIN",
        "VALIDATION_A",
        "VALIDATION_B",
        "VALIDATION",
    ):
        result[f"{window}_delta"] = (
            result[f"{window}_score"] - float(baseline[window])
        )
    result["worst_half_delta"] = result[
        [f"{window}_delta" for window in HALF_WINDOWS]
    ].min(axis=1)
    result["positive_halves"] = (
        result[[f"{window}_delta" for window in HALF_WINDOWS]] > 0.0
    ).sum(axis=1)
    result["development_score_sum"] = (
        result["TRAIN_score"] + result["VALIDATION_score"]
    )
    result["passes"] = (
        (result["TRAIN_delta"] > 0.0)
        & (result["VALIDATION_delta"] > 0.0)
        & (
            result["worst_half_delta"]
            >= -MATERIAL_HALF_TOLERANCE
        )
    )
    return result


def select_best(frame: pd.DataFrame) -> pd.Series | None:
    eligible = frame[frame["passes"]].copy()
    if eligible.empty:
        return None
    return eligible.sort_values(
        [
            "development_score_sum",
            "worst_half_delta",
            "VALIDATION_score",
            "TRAIN_score",
        ],
        ascending=False,
    ).iloc[0]


def development() -> None:
    """Search using a prices object physically truncated at day 750."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices(DEVELOPMENT_END)
    replay = build_replay_cache(prices)
    search = ConfirmationEvaluator(replay, DEVELOPMENT_WINDOWS)

    baseline_result = search.evaluate(BASELINE_LAM, BASELINE_THETA)
    baseline_scores = pd.Series(
        {
            window: metrics["score"]
            for window, metrics in baseline_result.items()
        }
    )
    reconstructed = search.targets(BASELINE_LAM, BASELINE_THETA)
    difference = reconstructed - replay.production_targets
    cache_check = {
        "days_available": int(prices.shape[1]),
        "different_target_cells": int(np.count_nonzero(difference)),
        "maximum_share_difference": int(np.max(np.abs(difference))),
        "baseline_lambda": BASELINE_LAM,
        "baseline_theta": BASELINE_THETA,
        "baseline_scores": baseline_scores.to_dict(),
    }
    (OUTPUT_DIR / "cache_check.json").write_text(
        json.dumps(cache_check, indent=2),
        encoding="utf-8",
    )
    if cache_check["different_target_cells"] != 0:
        raise RuntimeError(
            "Cached evaluator does not exactly reproduce production"
        )

    grid_rows: list[dict[str, Any]] = []
    for lam in GRID_LAMBDAS:
        for theta in GRID_THETAS:
            grid_rows.append(
                flatten_result(
                    "grid",
                    lam,
                    theta,
                    search.evaluate(lam, theta),
                )
            )
    grid = add_selection_fields(
        pd.DataFrame(grid_rows),
        baseline_scores,
    )
    grid.to_csv(OUTPUT_DIR / "grid_results.csv", index=False)
    grid_best = select_best(grid)

    optimizer_rows: list[dict[str, Any]] = []

    def objective(values: np.ndarray) -> float:
        lam = float(np.exp(values[0]))
        theta = float(values[1])
        raw = flatten_result(
            "optimizer",
            lam,
            theta,
            search.evaluate(lam, theta),
        )
        scored = add_selection_fields(
            pd.DataFrame([raw]),
            baseline_scores,
        ).iloc[0]
        optimizer_rows.append(scored.to_dict())
        penalty = 0.0
        penalty += 20.0 * max(0.0, -float(scored["TRAIN_delta"]))
        penalty += 20.0 * max(
            0.0,
            -float(scored["VALIDATION_delta"]),
        )
        penalty += 20.0 * max(
            0.0,
            -MATERIAL_HALF_TOLERANCE
            - float(scored["worst_half_delta"]),
        )
        return -float(scored["development_score_sum"]) + penalty

    differential_evolution(
        objective,
        bounds=((np.log(0.02), np.log(15.0)), (0.05, 2.50)),
        seed=20260730,
        maxiter=7,
        popsize=5,
        tol=0.005,
        polish=True,
        workers=1,
        updating="immediate",
    )
    optimized = pd.DataFrame(optimizer_rows).drop_duplicates(
        subset=["lambda", "theta"],
        keep="last",
    )
    optimized.to_csv(
        OUTPUT_DIR / "optimizer_results.csv",
        index=False,
    )
    optimizer_best = select_best(optimized)

    candidates = [
        row
        for row in (grid_best, optimizer_best)
        if row is not None
    ]
    if not candidates:
        selected_lam = BASELINE_LAM
        selected_theta = BASELINE_THETA
        selected_source = "baseline_fallback"
        selected_result = baseline_result
        selected_scored = add_selection_fields(
            pd.DataFrame(
                [
                    flatten_result(
                        selected_source,
                        selected_lam,
                        selected_theta,
                        selected_result,
                    )
                ]
            ),
            baseline_scores,
        ).iloc[0]
    else:
        selected_scored = (
            pd.DataFrame(candidates)
            .sort_values(
                [
                    "development_score_sum",
                    "worst_half_delta",
                    "VALIDATION_score",
                    "TRAIN_score",
                ],
                ascending=False,
            )
            .iloc[0]
        )
        selected_lam = float(selected_scored["lambda"])
        selected_theta = float(selected_scored["theta"])
        selected_source = str(selected_scored["source"])
        selected_result = search.evaluate(
            selected_lam,
            selected_theta,
        )

    comparison_rows = [
        flatten_result(
            "baseline",
            BASELINE_LAM,
            BASELINE_THETA,
            baseline_result,
        ),
        flatten_result(
            "locked_candidate",
            selected_lam,
            selected_theta,
            selected_result,
        ),
    ]
    comparison = add_selection_fields(
        pd.DataFrame(comparison_rows),
        baseline_scores,
    )
    comparison.to_csv(
        OUTPUT_DIR / "development_comparison.csv",
        index=False,
    )
    lock = {
        "selection_data_max_day": DEVELOPMENT_END,
        "selection_rule": (
            "maximize TRAIN+VALIDATION score, require both aggregate "
            "deltas > 0 and every 125-day half delta >= -5"
        ),
        "baseline": {
            "lambda": BASELINE_LAM,
            "theta": BASELINE_THETA,
            "scores": baseline_scores.to_dict(),
        },
        "candidate": {
            "lambda": selected_lam,
            "theta": selected_theta,
            "source": selected_source,
            "passes": bool(selected_scored["passes"]),
            "scores": {
                window: float(selected_result[window]["score"])
                for window, _, _ in DEVELOPMENT_WINDOWS
            },
            "development_score_sum": float(
                selected_scored["development_score_sum"]
            ),
            "worst_half_delta": float(
                selected_scored["worst_half_delta"]
            ),
        },
    }
    LOCK_PATH.write_text(
        json.dumps(lock, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(lock, indent=2))


def heldout() -> None:
    """Evaluate the locked development candidate once on days 751-1000."""
    if not LOCK_PATH.exists():
        raise FileNotFoundError(
            "Run --phase development before held-out evaluation"
        )
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    candidate = lock["candidate"]
    if not bool(candidate["passes"]):
        raise RuntimeError(
            "No development candidate passed; held-out data not opened"
        )
    prices = load_prices()
    replay = build_replay_cache(prices)
    comparison = ConfirmationEvaluator(replay, HELDOUT_WINDOWS)
    rows = []
    for name, lam, theta in (
        ("baseline", BASELINE_LAM, BASELINE_THETA),
        (
            "locked_candidate",
            float(candidate["lambda"]),
            float(candidate["theta"]),
        ),
    ):
        result = comparison.evaluate(lam, theta)
        rows.append(flatten_result(name, lam, theta, result))
    heldout_frame = pd.DataFrame(rows)
    baseline_score = float(
        heldout_frame.loc[
            heldout_frame["source"] == "baseline",
            "TEST_score",
        ].iloc[0]
    )
    candidate_score = float(
        heldout_frame.loc[
            heldout_frame["source"] == "locked_candidate",
            "TEST_score",
        ].iloc[0]
    )
    heldout_frame["TEST_delta"] = (
        heldout_frame["TEST_score"] - baseline_score
    )
    heldout_frame.to_csv(
        OUTPUT_DIR / "heldout_comparison.csv",
        index=False,
    )
    decision = {
        "candidate_lambda": float(candidate["lambda"]),
        "candidate_theta": float(candidate["theta"]),
        "baseline_test_score": baseline_score,
        "candidate_test_score": candidate_score,
        "test_delta": candidate_score - baseline_score,
        "ship": bool(candidate_score > baseline_score),
        "rule": "ship only when locked candidate TEST score > baseline",
    }
    (OUTPUT_DIR / "ship_decision.json").write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("development", "heldout"),
        required=True,
    )
    args = parser.parse_args()
    if args.phase == "development":
        development()
    else:
        heldout()


if __name__ == "__main__":
    main()

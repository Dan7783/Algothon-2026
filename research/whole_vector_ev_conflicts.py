#!/usr/bin/env python
"""Train/validate/test study for whole-vector EV signal combination.

Protocol
--------
* Calibration labels and parameters use realised PnL from days 1-500 only.
* A small, predeclared shrinkage/edge grid is selected on days 501-750.
* ``--phase develop`` never loads prices after day 750.
* ``--phase test`` consumes the frozen selection and evaluates 751-1000 once.

Every active pair is represented by its complete beta-scaled two-leg dollar
vector. Every active ridge forecast is represented by a one-asset vector.
The calibrator estimates gross next-day PnL, forecast variance, and isolated
round-trip costs for each proposal. Positive net-EV proposals are combined
with mean-variance weights (EV / variance); instrument limits are imposed
only after the complete proposal vectors have been accumulated.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy


OUTPUT_DIR = ROOT / "analysis_outputs" / "whole_vector_ev"
PRICE_PATH = ROOT / "prices.txt"
LOCK_PATH = OUTPUT_DIR / "locked_selection.json"

TRAIN_END = 500
VALIDATION_START = 501
VALIDATION_END = 750
TEST_START = 751
TEST_END = 1000

COMMISSION = np.full(strategy.N_INST, 1.0e-4)
COMMISSION[strategy.I["ALGO"]] = 2.0e-5
POSITION_LIMIT = np.full(strategy.N_INST, strategy.LEG)
POSITION_LIMIT[strategy.I["ALGO"]] = 100_000.0

# Deliberately small grid. It is declared in source before validation is read.
PRIOR_OBSERVATIONS = (25.0, 75.0, 200.0)
MIN_EDGE_Z = (0.0, 0.01, 0.025, 0.05)


@dataclass
class SignalCache:
    """Decision-time proposals and their next-day isolated outcomes."""

    prices: np.ndarray
    pair_y: np.ndarray
    pair_x: np.ndarray
    pair_pos_y: np.ndarray
    pair_pos_x: np.ndarray
    pair_strength: np.ndarray
    pair_cost: np.ndarray
    pair_gross_pnl: np.ndarray
    ridge_pos: np.ndarray
    ridge_strength: np.ndarray
    ridge_cost: np.ndarray
    ridge_gross_pnl: np.ndarray
    crash: np.ndarray
    baseline_dollars: np.ndarray


@dataclass
class FamilyCalibration:
    """Shrunk identity intercepts around one monotone family calibration."""

    center: float
    lower: float
    upper: float
    global_mean: float
    slope: float
    offsets: np.ndarray
    variances: np.ndarray
    counts: np.ndarray

    def predict(self, strength: np.ndarray) -> np.ndarray:
        clipped = np.clip(strength, self.lower, self.upper)
        return (
            self.global_mean
            + self.slope * (clipped - self.center)
            + self.offsets[None, :]
        )


def load_prices(max_day: int) -> np.ndarray:
    """Load no more data than the requested protocol phase permits."""
    values = pd.read_csv(PRICE_PATH, sep=r"\s+").to_numpy(dtype=float).T
    if values.shape[1] < max_day:
        raise RuntimeError(
            f"Need {max_day} price days, found {values.shape[1]}"
        )
    return values[:, :max_day]


def _score(pnl: np.ndarray) -> float:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe * sharpe / (sharpe * sharpe + 1.0)


def _metrics(pnl: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    return {
        "score": _score(pnl),
        "mean_pnl": mean,
        "std_pnl": std,
        "sharpe": sqrt(250.0) * mean / std if std > 0.0 else 0.0,
        "total_pnl": float(np.sum(pnl)),
    }


def _target_shares(dollars: np.ndarray, prices: np.ndarray) -> np.ndarray:
    limits = (POSITION_LIMIT / prices).astype(int)
    requested = np.round(
        dollars / np.maximum(prices, strategy._EPS_DENOM)
    ).astype(int)
    return np.clip(requested, -limits, limits).astype(int)


def simulate(
    prices: np.ndarray,
    dollars_by_history: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """Match eval.py accounting and its one-day commission timing."""
    cash = 0.0
    value = 0.0
    prior_commission = 0.0
    current = np.zeros(strategy.N_INST)
    pnl: list[float] = []

    for t in range(start - 1, end + 1):
        current_prices = prices[:, t - 1]
        if t < end:
            new_position = _target_shares(
                dollars_by_history[t],
                current_prices,
            ).astype(float)
        else:
            new_position = current.copy()

        delta = new_position - current
        cash -= float(current_prices @ delta) + prior_commission
        volume = current_prices * np.abs(delta)
        prior_commission = float(volume @ COMMISSION)
        current = new_position

        new_value = cash + float(current @ current_prices)
        daily_pnl = new_value - value
        value = new_value
        if t >= start:
            pnl.append(daily_pnl)

    return np.asarray(pnl)


def collect_signals(prices: np.ndarray) -> SignalCache:
    """Replay the live pair and ridge signal generators exactly once."""
    n_history = prices.shape[1]
    n_pairs = len(strategy.PAIR_CFGS)
    shape_pair = (n_history + 1, n_pairs)
    shape_asset = (n_history + 1, strategy.N_INST)

    pair_y = np.asarray([spec.y for spec in strategy.PAIR_CFGS], dtype=int)
    pair_x = np.asarray([spec.x for spec in strategy.PAIR_CFGS], dtype=int)
    pair_pos_y = np.zeros(shape_pair)
    pair_pos_x = np.zeros(shape_pair)
    pair_strength = np.zeros(shape_pair)
    pair_cost = np.zeros(shape_pair)
    pair_gross_pnl = np.zeros(shape_pair)

    ridge_pos = np.zeros(shape_asset)
    ridge_strength = np.zeros(shape_asset)
    ridge_cost = np.zeros(shape_asset)
    ridge_gross_pnl = np.zeros(shape_asset)
    crash = np.zeros(shape_asset, dtype=bool)
    baseline_dollars = np.zeros(shape_asset)

    agents = [strategy.PairAgent(spec) for spec in strategy.PAIR_CFGS]
    previous_pair_y_shares = np.zeros(n_pairs)
    previous_pair_x_shares = np.zeros(n_pairs)
    previous_ridge_shares = np.zeros(strategy.N_INST)

    for history_size in range(1, n_history):
        history = prices[:, :history_size]
        current = prices[:, history_size - 1]
        following = prices[:, history_size]

        for agent in agents:
            agent.catch_up(history)

        candidates: list[
            tuple[float, strategy.PairAgent, float, float]
        ] = []
        if history_size >= strategy.WARMUP + 10:
            for pair_index, agent in enumerate(agents):
                pos_y, pos_x, abs_z = agent.dollar_positions()
                pair_pos_y[history_size, pair_index] = pos_y
                pair_pos_x[history_size, pair_index] = pos_x
                if abs(pos_y) <= 1.0e-9:
                    continue

                y_index = pair_y[pair_index]
                x_index = pair_x[pair_index]
                y_shares = pos_y / current[y_index]
                x_shares = pos_x / current[x_index]
                entry_cost = (
                    abs(y_shares - previous_pair_y_shares[pair_index])
                    * current[y_index]
                    * COMMISSION[y_index]
                    + abs(x_shares - previous_pair_x_shares[pair_index])
                    * current[x_index]
                    * COMMISSION[x_index]
                )
                # A live proposal must eventually be liquidated. Including a
                # one-way exit reserve prevents held positions from appearing
                # artificially costless relative to new entries.
                exit_reserve = (
                    abs(pos_y) * COMMISSION[y_index]
                    + abs(pos_x) * COMMISSION[x_index]
                )
                pair_cost[history_size, pair_index] = (
                    entry_cost + exit_reserve
                )
                pair_gross_pnl[history_size, pair_index] = (
                    y_shares * (following[y_index] - current[y_index])
                    + x_shares * (following[x_index] - current[x_index])
                )
                kappa = max(float(agent.current_kappa), 0.0)
                sigma = max(float(agent.current_sigma), 0.0)
                pair_strength[history_size, pair_index] = (
                    abs(pos_y)
                    * (1.0 - np.exp(-kappa))
                    * sigma
                    * abs_z
                )
                priority = kappa * abs_z
                candidates.append((priority, agent, pos_y, pos_x))

        current_pair_y_shares = (
            pair_pos_y[history_size]
            / current[pair_y]
        )
        current_pair_x_shares = (
            pair_pos_x[history_size]
            / current[pair_x]
        )
        previous_pair_y_shares = current_pair_y_shares
        previous_pair_x_shares = current_pair_x_shares

        returns = np.diff(
            np.log(np.maximum(history, strategy._EPS_DENOM)),
            axis=1,
        )
        ridge_raw, current_ridge_strength = strategy._ridge_forecast(
            returns
        )
        ridge_pos[history_size] = ridge_raw
        ridge_strength[history_size] = current_ridge_strength
        current_ridge_shares = ridge_raw / current
        ridge_cost[history_size] = (
            np.abs(current_ridge_shares - previous_ridge_shares)
            * current
            * COMMISSION
            + np.abs(ridge_raw) * COMMISSION
        )
        ridge_gross_pnl[history_size] = (
            current_ridge_shares * (following - current)
        )
        previous_ridge_shares = current_ridge_shares

        pair_vote, pair_active, _ = strategy._pair_conflict_consensus(
            candidates
        )
        ridge_sign = np.sign(ridge_raw)
        active_ridge = ridge_sign != 0.0
        ridge_vote = (
            strategy.RIDGE_CONFLICT_WEIGHT
            * ridge_sign
            * current_ridge_strength
        )
        combined = pair_vote + ridge_vote
        active = pair_active | active_ridge
        direction = np.sign(combined)
        tied = active & (direction == 0.0) & active_ridge
        direction[tied] = ridge_sign[tied]

        if history_size >= 2:
            last_return = np.log(current) - np.log(
                np.maximum(
                    prices[:, history_size - 2],
                    strategy._EPS_DENOM,
                )
            )
            crash[history_size] = np.abs(last_return) > 0.40
            crash[history_size, strategy.I["ALGO"]] = False
            direction[crash[history_size]] = 0.0

        baseline_dollars[history_size] = (
            strategy.BANG_BANG_LIMITS * direction
        )

    return SignalCache(
        prices=prices,
        pair_y=pair_y,
        pair_x=pair_x,
        pair_pos_y=pair_pos_y,
        pair_pos_x=pair_pos_x,
        pair_strength=pair_strength,
        pair_cost=pair_cost,
        pair_gross_pnl=pair_gross_pnl,
        ridge_pos=ridge_pos,
        ridge_strength=ridge_strength,
        ridge_cost=ridge_cost,
        ridge_gross_pnl=ridge_gross_pnl,
        crash=crash,
        baseline_dollars=baseline_dollars,
    )


def fit_family(
    strength: np.ndarray,
    gross_pnl: np.ndarray,
    active: np.ndarray,
    train_rows: np.ndarray,
    prior_observations: float,
) -> FamilyCalibration:
    """Fit a monotone global slope plus shrunk identity effects."""
    mask = active & train_rows[:, None]
    observed_strength = strength[mask]
    observed_pnl = gross_pnl[mask]
    if len(observed_pnl) == 0:
        raise RuntimeError("No active calibration observations")

    lower, upper = np.quantile(observed_strength, [0.01, 0.99])
    clipped = np.clip(observed_strength, lower, upper)
    center = float(np.mean(clipped))
    global_mean = float(np.mean(observed_pnl))
    centered = clipped - center
    denominator = float(centered @ centered)
    # Ten-percent denominator inflation is a fixed slope shrinkage, not a
    # validation-tuned parameter. Negative slopes are rejected because signal
    # magnitude is required to map monotonically to expected gross edge.
    slope = max(
        0.0,
        float(centered @ (observed_pnl - global_mean))
        / max(1.1 * denominator, 1.0e-12),
    )

    base_prediction = (
        global_mean
        + slope * (
            np.clip(strength, lower, upper) - center
        )
    )
    residual = gross_pnl - base_prediction
    n_identity = strength.shape[1]
    offsets = np.zeros(n_identity)
    counts = np.zeros(n_identity)

    for identity in range(n_identity):
        identity_mask = mask[:, identity]
        count = float(np.sum(identity_mask))
        counts[identity] = count
        if count:
            offsets[identity] = float(
                np.sum(residual[identity_mask, identity])
                / (count + prior_observations)
            )

    fitted = base_prediction + offsets[None, :]
    global_variance = float(
        np.mean((observed_pnl - (
            global_mean + slope * (clipped - center)
        )) ** 2)
    )
    global_variance = max(global_variance, 1.0e-6)
    variances = np.full(n_identity, global_variance)
    for identity in range(n_identity):
        identity_mask = mask[:, identity]
        count = counts[identity]
        if count:
            squared = (
                gross_pnl[identity_mask, identity]
                - fitted[identity_mask, identity]
            ) ** 2
            variances[identity] = (
                float(np.sum(squared))
                + prior_observations * global_variance
            ) / (count + prior_observations)

    return FamilyCalibration(
        center=center,
        lower=float(lower),
        upper=float(upper),
        global_mean=global_mean,
        slope=slope,
        offsets=offsets,
        variances=np.maximum(variances, 1.0e-6),
        counts=counts,
    )


def fit_calibrations(
    cache: SignalCache,
    prior_observations: float,
) -> tuple[FamilyCalibration, FamilyCalibration]:
    n_history = cache.prices.shape[1]
    # A proposal formed with history size h earns the PnL labelled h+1 by
    # eval.py. Consequently h <= 499 is the final permissible training row.
    train_rows = np.arange(n_history + 1) + 1 <= TRAIN_END
    pair_active = np.abs(cache.pair_pos_y) > 1.0e-9
    ridge_active = np.abs(cache.ridge_pos) > 1.0e-9
    pair_calibration = fit_family(
        cache.pair_strength,
        cache.pair_gross_pnl,
        pair_active,
        train_rows,
        prior_observations,
    )
    ridge_calibration = fit_family(
        cache.ridge_strength,
        cache.ridge_gross_pnl,
        ridge_active,
        train_rows,
        prior_observations,
    )
    return pair_calibration, ridge_calibration


def ev_dollars(
    cache: SignalCache,
    pair_calibration: FamilyCalibration,
    ridge_calibration: FamilyCalibration,
    min_edge_z: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Combine complete proposals using positive net EV / variance weights."""
    n_history = cache.prices.shape[1]
    aggregate = np.zeros((n_history + 1, strategy.N_INST))

    pair_active = np.abs(cache.pair_pos_y) > 1.0e-9
    pair_expected_gross = pair_calibration.predict(
        cache.pair_strength
    )
    pair_ev = pair_expected_gross - cache.pair_cost
    pair_std = np.sqrt(pair_calibration.variances)[None, :]
    pair_accepted = (
        pair_active
        & (pair_ev > min_edge_z * pair_std)
    )
    pair_weight = np.where(
        pair_accepted,
        pair_ev / pair_calibration.variances[None, :],
        0.0,
    )

    for pair_index, (y_index, x_index) in enumerate(
        zip(cache.pair_y, cache.pair_x, strict=True)
    ):
        aggregate[:, y_index] += (
            pair_weight[:, pair_index]
            * cache.pair_pos_y[:, pair_index]
        )
        aggregate[:, x_index] += (
            pair_weight[:, pair_index]
            * cache.pair_pos_x[:, pair_index]
        )

    ridge_active = np.abs(cache.ridge_pos) > 1.0e-9
    ridge_expected_gross = ridge_calibration.predict(
        cache.ridge_strength
    )
    ridge_ev = ridge_expected_gross - cache.ridge_cost
    ridge_std = np.sqrt(ridge_calibration.variances)[None, :]
    ridge_accepted = (
        ridge_active
        & (ridge_ev > min_edge_z * ridge_std)
    )
    ridge_weight = np.where(
        ridge_accepted,
        ridge_ev / ridge_calibration.variances[None, :],
        0.0,
    )
    aggregate += ridge_weight * cache.ridge_pos

    direction = np.sign(aggregate)
    direction[cache.crash] = 0.0
    dollars = strategy.BANG_BANG_LIMITS[None, :] * direction

    diagnostics = {
        "pair_acceptance": float(
            np.sum(pair_accepted) / max(np.sum(pair_active), 1)
        ),
        "ridge_acceptance": float(
            np.sum(ridge_accepted) / max(np.sum(ridge_active), 1)
        ),
        "mean_active_assets": float(
            np.mean(np.sum(direction != 0.0, axis=1))
        ),
    }
    return dollars, diagnostics


def region_metrics(
    prices: np.ndarray,
    dollars: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    return _metrics(simulate(prices, dollars, start, end))


def development_phase() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices(VALIDATION_END)
    cache = collect_signals(prices)

    validation_windows = (
        ("VAL1", 501, 625),
        ("VAL2", 626, 750),
        ("VALIDATION", 501, 750),
    )
    baseline = {
        name: region_metrics(
            prices,
            cache.baseline_dollars,
            start,
            end,
        )
        for name, start, end in validation_windows
    }

    rows: list[dict[str, float]] = []
    calibrations: dict[
        float,
        tuple[FamilyCalibration, FamilyCalibration],
    ] = {}
    for prior in PRIOR_OBSERVATIONS:
        calibrations[prior] = fit_calibrations(cache, prior)
        for threshold in MIN_EDGE_Z:
            pair_calibration, ridge_calibration = calibrations[prior]
            dollars, diagnostics = ev_dollars(
                cache,
                pair_calibration,
                ridge_calibration,
                threshold,
            )
            record: dict[str, float] = {
                "prior_observations": prior,
                "min_edge_z": threshold,
                **diagnostics,
            }
            for name, start, end in validation_windows:
                result = region_metrics(prices, dollars, start, end)
                record[f"{name.lower()}_score"] = result["score"]
                record[f"{name.lower()}_delta"] = (
                    result["score"] - baseline[name]["score"]
                )
            record["robust_delta"] = min(
                record["val1_delta"],
                record["val2_delta"],
            )
            rows.append(record)

    results = pd.DataFrame(rows).sort_values(
        ["robust_delta", "validation_delta"],
        ascending=False,
    )
    results.to_csv(OUTPUT_DIR / "development_grid.csv", index=False)
    winner = results.iloc[0]
    locked = {
        "protocol": {
            "calibration_days": [1, TRAIN_END],
            "selection_days": [VALIDATION_START, VALIDATION_END],
            "heldout_days": [TEST_START, TEST_END],
            "selection_rule": (
                "maximize the minimum score delta across VAL1 and VAL2; "
                "break ties with full validation delta"
            ),
        },
        "selected": {
            "prior_observations": float(winner["prior_observations"]),
            "min_edge_z": float(winner["min_edge_z"]),
        },
        "baseline": baseline,
        "winner_validation": {
            key: float(winner[key])
            for key in winner.index
            if key not in {"prior_observations", "min_edge_z"}
        },
        "heldout_evaluated": False,
    }
    LOCK_PATH.write_text(json.dumps(locked, indent=2), encoding="utf-8")

    print("Validation baseline:")
    for name in ("VAL1", "VAL2", "VALIDATION"):
        print(f"  {name:<10} {baseline[name]['score']:10.3f}")
    print("\nTop validation-selected configurations:")
    print(
        results[
            [
                "prior_observations",
                "min_edge_z",
                "val1_score",
                "val2_score",
                "validation_score",
                "robust_delta",
                "validation_delta",
                "pair_acceptance",
                "ridge_acceptance",
            ]
        ].head(12).to_string(index=False)
    )
    print(f"\nLocked selection: {LOCK_PATH}")
    print("No prices after day 750 were loaded.")


def heldout_phase() -> None:
    if not LOCK_PATH.exists():
        raise RuntimeError("Run --phase develop before held-out evaluation")
    locked = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if locked.get("heldout_evaluated"):
        raise RuntimeError(
            "Held-out result already exists; refusing a second evaluation"
        )

    selected = locked["selected"]
    prices = load_prices(TEST_END)
    cache = collect_signals(prices)
    pair_calibration, ridge_calibration = fit_calibrations(
        cache,
        float(selected["prior_observations"]),
    )
    dollars, diagnostics = ev_dollars(
        cache,
        pair_calibration,
        ridge_calibration,
        float(selected["min_edge_z"]),
    )

    baseline = region_metrics(
        prices,
        cache.baseline_dollars,
        TEST_START,
        TEST_END,
    )
    ev_result = region_metrics(
        prices,
        dollars,
        TEST_START,
        TEST_END,
    )
    heldout = {
        "selected": selected,
        "baseline": baseline,
        "whole_vector_ev": ev_result,
        "score_delta": ev_result["score"] - baseline["score"],
        "diagnostics": diagnostics,
    }
    (OUTPUT_DIR / "heldout_results.json").write_text(
        json.dumps(heldout, indent=2),
        encoding="utf-8",
    )
    locked["heldout_evaluated"] = True
    locked["heldout_result_file"] = "heldout_results.json"
    LOCK_PATH.write_text(json.dumps(locked, indent=2), encoding="utf-8")

    # Frozen calibration constants are needed by the submission-safe runtime.
    calibration = {
        "pair": {
            "center": pair_calibration.center,
            "lower": pair_calibration.lower,
            "upper": pair_calibration.upper,
            "global_mean": pair_calibration.global_mean,
            "slope": pair_calibration.slope,
            "offsets": pair_calibration.offsets.tolist(),
            "variances": pair_calibration.variances.tolist(),
            "counts": pair_calibration.counts.tolist(),
        },
        "ridge": {
            "center": ridge_calibration.center,
            "lower": ridge_calibration.lower,
            "upper": ridge_calibration.upper,
            "global_mean": ridge_calibration.global_mean,
            "slope": ridge_calibration.slope,
            "offsets": ridge_calibration.offsets.tolist(),
            "variances": ridge_calibration.variances.tolist(),
            "counts": ridge_calibration.counts.tolist(),
        },
        "selected": selected,
    }
    (OUTPUT_DIR / "frozen_calibration.json").write_text(
        json.dumps(calibration, indent=2),
        encoding="utf-8",
    )

    print("Held-out evaluation (performed once):")
    print(f"  baseline         {baseline['score']:10.3f}")
    print(f"  whole-vector EV  {ev_result['score']:10.3f}")
    print(f"  delta            {heldout['score_delta']:+10.3f}")
    print(f"  results: {OUTPUT_DIR / 'heldout_results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("develop", "test"),
        required=True,
    )
    args = parser.parse_args()
    if args.phase == "develop":
        development_phase()
    else:
        heldout_phase()


if __name__ == "__main__":
    main()

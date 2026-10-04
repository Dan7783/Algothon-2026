#!/usr/bin/env python
"""Adaptive whole-vector EV using pair-filter-style regime handling.

The static EV model failed because its pair/ridge family edge estimates were
frozen while realised family profitability changed. This experiment replaces
139 identity calibrations with two causal states: one for pair proposals and
one for ridge proposals.

Each state:
* observes strictly prior realised proposal PnL;
* tracks family mean edge and proposal forecast variance by EWMA;
* uses current within-family signal strength only as a relative multiplier;
* runs a two-sided CUSUM on daily family-mean forecast errors;
* resets the stale mean and inflates uncertainty on an alarm.

Protocol:
* ``develop`` loads days 1-750 only and selects the two adaptation settings;
* ``test`` consumes the frozen lock and reports 751-1000 once;
* held-out results never promote or alter a configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import whole_vector_ev_conflicts as study


OUTPUT_DIR = ROOT / "analysis_outputs" / "adaptive_ev_pair_filter"
LOCK_PATH = OUTPUT_DIR / "locked_selection.json"

# Small declared grid, evaluated only in the develop phase.
HALF_LIFE_GRID = (10.0, 20.0, 40.0, 80.0, 160.0)
CUSUM_H_GRID = (4.0, 8.0, float("inf"))

# Reuse pair-filter structural defaults rather than tune more knobs.
CUSUM_K = 0.5
UNCERTAINTY_INFLATION = 25.0
MIN_STATE_DAYS = 20

WINDOWS_DEVELOP = (
    ("TRAIN1", 251, 375),
    ("TRAIN2", 376, 500),
    ("VAL1", 501, 625),
    ("VAL2", 626, 750),
    ("TRAIN", 251, 500),
    ("VALIDATION", 501, 750),
)


@dataclass(frozen=True)
class AdaptiveConfig:
    half_life: float
    cusum_h: float


class AdaptiveFamilyState:
    """Two-moment causal state with pair-filter-style CUSUM resets."""

    def __init__(self, config: AdaptiveConfig) -> None:
        self.config = config
        self.alpha = 1.0 - 0.5 ** (1.0 / config.half_life)
        self.mean_edge = 0.0
        self.proposal_variance = 1.0
        self.daily_mean_variance = 1.0
        self.cusum_up = 0.0
        self.cusum_down = 0.0
        self.days_seen = 0
        self.alarms = 0

    @staticmethod
    def _relative_strength(
        strength: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        relative = np.zeros_like(strength)
        if np.any(active):
            scale = float(np.mean(strength[active]))
            if scale > 1.0e-12:
                relative[active] = strength[active] / scale
            else:
                relative[active] = 1.0
        return relative

    def forecast(
        self,
        strength: np.ndarray,
        active: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        relative = self._relative_strength(strength, active)
        expected = self.mean_edge * relative
        return expected, max(self.proposal_variance, 1.0e-6)

    def update(
        self,
        strength: np.ndarray,
        realised_gross: np.ndarray,
        active: np.ndarray,
    ) -> None:
        if not np.any(active):
            return

        relative = self._relative_strength(strength, active)
        observed = realised_gross[active]
        observed_mean = float(np.mean(observed))
        prior_prediction = self.mean_edge * relative[active]
        squared_error = float(
            np.mean((observed - prior_prediction) ** 2)
        )

        if self.days_seen < MIN_STATE_DAYS:
            rate = 1.0 / (self.days_seen + 1.0)
            innovation = observed_mean - self.mean_edge
            self.mean_edge += rate * innovation
            self.daily_mean_variance = (
                (1.0 - rate) * self.daily_mean_variance
                + rate * innovation * innovation
            )
            self.proposal_variance = (
                (1.0 - rate) * self.proposal_variance
                + rate * squared_error
            )
            self.days_seen += 1
            return

        innovation = observed_mean - self.mean_edge
        innovation_scale = sqrt(
            max(self.daily_mean_variance, 1.0e-6)
        )
        standardized = innovation / innovation_scale
        self.cusum_up = max(
            0.0,
            self.cusum_up + standardized - CUSUM_K,
        )
        self.cusum_down = max(
            0.0,
            self.cusum_down - standardized - CUSUM_K,
        )
        alarm = (
            self.cusum_up > self.config.cusum_h
            or self.cusum_down > self.config.cusum_h
        )

        if alarm:
            self.mean_edge = observed_mean
            self.proposal_variance *= UNCERTAINTY_INFLATION
            self.cusum_up = 0.0
            self.cusum_down = 0.0
            self.alarms += 1
        else:
            self.mean_edge = (
                (1.0 - self.alpha) * self.mean_edge
                + self.alpha * observed_mean
            )

        self.daily_mean_variance = (
            (1.0 - self.alpha) * self.daily_mean_variance
            + self.alpha * innovation * innovation
        )
        self.proposal_variance = (
            (1.0 - self.alpha) * self.proposal_variance
            + self.alpha * squared_error
        )
        self.days_seen += 1


def adaptive_targets(
    cache: study.SignalCache,
    config: AdaptiveConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate positions with every forecast based on strictly-prior PnL."""
    n_history = cache.prices.shape[1]
    dollars = np.zeros((n_history + 1, strategy.N_INST))
    pair_state = AdaptiveFamilyState(config)
    ridge_state = AdaptiveFamilyState(config)
    records: list[dict[str, float]] = []

    for history_size in range(1, n_history):
        # The outcome of proposal h-1 becomes observable at price h and may
        # update the state before proposal h is formed.
        previous = history_size - 1
        if previous >= 1:
            prior_pair_active = (
                np.abs(cache.pair_pos_y[previous]) > 1.0e-9
            )
            pair_state.update(
                cache.pair_strength[previous],
                cache.pair_gross_pnl[previous],
                prior_pair_active,
            )
            prior_ridge_active = (
                np.abs(cache.ridge_pos[previous]) > 1.0e-9
            )
            ridge_state.update(
                cache.ridge_strength[previous],
                cache.ridge_gross_pnl[previous],
                prior_ridge_active,
            )

        pair_active = (
            np.abs(cache.pair_pos_y[history_size]) > 1.0e-9
        )
        pair_expected, pair_variance = pair_state.forecast(
            cache.pair_strength[history_size],
            pair_active,
        )
        pair_net = (
            pair_expected - cache.pair_cost[history_size]
        )
        pair_weight = np.where(
            pair_active & (pair_net > 0.0),
            pair_net / pair_variance,
            0.0,
        )

        combined = np.zeros(strategy.N_INST)
        for pair_index, (y_index, x_index) in enumerate(
            zip(cache.pair_y, cache.pair_x, strict=True)
        ):
            weight = pair_weight[pair_index]
            if weight == 0.0:
                continue
            combined[y_index] += (
                weight * cache.pair_pos_y[history_size, pair_index]
            )
            combined[x_index] += (
                weight * cache.pair_pos_x[history_size, pair_index]
            )

        ridge_active = (
            np.abs(cache.ridge_pos[history_size]) > 1.0e-9
        )
        ridge_expected, ridge_variance = ridge_state.forecast(
            cache.ridge_strength[history_size],
            ridge_active,
        )
        ridge_net = (
            ridge_expected - cache.ridge_cost[history_size]
        )
        ridge_weight = np.where(
            ridge_active & (ridge_net > 0.0),
            ridge_net / ridge_variance,
            0.0,
        )
        combined += (
            ridge_weight * cache.ridge_pos[history_size]
        )

        direction = np.sign(combined)
        direction[cache.crash[history_size]] = 0.0
        dollars[history_size] = (
            strategy.BANG_BANG_LIMITS * direction
        )
        records.append(
            {
                "history_size": float(history_size),
                "pair_mean_edge": pair_state.mean_edge,
                "ridge_mean_edge": ridge_state.mean_edge,
                "pair_variance": pair_state.proposal_variance,
                "ridge_variance": ridge_state.proposal_variance,
                "pair_alarms": float(pair_state.alarms),
                "ridge_alarms": float(ridge_state.alarms),
                "active_assets": float(np.sum(direction != 0.0)),
            }
        )

    return dollars, pd.DataFrame(records)


def evaluate_windows(
    cache: study.SignalCache,
    dollars: np.ndarray,
    windows: tuple[tuple[str, int, int], ...],
) -> dict[str, dict[str, float]]:
    return {
        name: study.region_metrics(
            cache.prices,
            dollars,
            start,
            end,
        )
        for name, start, end in windows
    }


def develop() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = study.load_prices(750)
    cache = study.collect_signals(prices)
    baseline = evaluate_windows(
        cache,
        cache.baseline_dollars,
        WINDOWS_DEVELOP,
    )

    rows = []
    for half_life in HALF_LIFE_GRID:
        for cusum_h in CUSUM_H_GRID:
            config = AdaptiveConfig(half_life, cusum_h)
            dollars, diagnostics = adaptive_targets(cache, config)
            result = evaluate_windows(
                cache,
                dollars,
                WINDOWS_DEVELOP,
            )
            record = {
                "half_life": half_life,
                "cusum_h": cusum_h,
            }
            for name, _, _ in WINDOWS_DEVELOP:
                key = name.lower()
                record[f"{key}_score"] = result[name]["score"]
                record[f"{key}_delta"] = (
                    result[name]["score"] - baseline[name]["score"]
                )
            fold_deltas = [
                record["train1_delta"],
                record["train2_delta"],
                record["val1_delta"],
                record["val2_delta"],
            ]
            record["robust_delta"] = min(fold_deltas)
            record["positive_folds"] = sum(
                value > 0.0 for value in fold_deltas
            )
            final = diagnostics.iloc[-1]
            record["pair_alarms"] = final["pair_alarms"]
            record["ridge_alarms"] = final["ridge_alarms"]
            record["mean_active_assets"] = float(
                diagnostics["active_assets"].mean()
            )
            rows.append(record)

    frame = pd.DataFrame(rows).sort_values(
        [
            "positive_folds",
            "robust_delta",
            "validation_delta",
        ],
        ascending=False,
    )
    frame.to_csv(OUTPUT_DIR / "development_grid.csv", index=False)
    winner = frame.iloc[0]
    accepted = bool(
        winner["train1_delta"] > 0.0
        and winner["train2_delta"] > 0.0
        and winner["val1_delta"] > 0.0
        and winner["val2_delta"] > 0.0
    )
    lock = {
        "protocol": {
            "development_days": [1, 500],
            "validation_days": [501, 750],
            "heldout_days": [751, 1000],
            "selection_rule": (
                "maximize number of positive 125-day train/validation "
                "folds, then worst-fold delta, then full validation delta"
            ),
            "promotion_rule": "all four fixed folds must improve",
        },
        "selected": {
            "half_life": float(winner["half_life"]),
            "cusum_h": (
                None
                if not isfinite(float(winner["cusum_h"]))
                else float(winner["cusum_h"])
            ),
        },
        "accepted_on_development": accepted,
        "winner": {
            key: float(winner[key])
            for key in winner.index
        },
        "heldout_evaluated": False,
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    columns = [
        "half_life",
        "cusum_h",
        "train1_delta",
        "train2_delta",
        "val1_delta",
        "val2_delta",
        "validation_delta",
        "robust_delta",
        "positive_folds",
        "pair_alarms",
        "ridge_alarms",
    ]
    print(frame[columns].head(15).to_string(index=False))
    print(f"\nAccepted on development: {accepted}")
    print(f"Locked: {LOCK_PATH}")
    print("No prices after day 750 were loaded.")


def test() -> None:
    if not LOCK_PATH.exists():
        raise RuntimeError("Run develop before test")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("heldout_evaluated"):
        raise RuntimeError("Held-out result already recorded")

    selected = lock["selected"]
    cusum_h = (
        float("inf")
        if selected["cusum_h"] is None
        else float(selected["cusum_h"])
    )
    config = AdaptiveConfig(
        float(selected["half_life"]),
        cusum_h,
    )
    prices = study.load_prices(1000)
    cache = study.collect_signals(prices)
    dollars, diagnostics = adaptive_targets(cache, config)
    windows = (
        ("OOS1", 751, 875),
        ("OOS2", 876, 1000),
        ("YEAR4_OOS", 751, 1000),
    )
    baseline = evaluate_windows(
        cache,
        cache.baseline_dollars,
        windows,
    )
    adaptive = evaluate_windows(cache, dollars, windows)
    result = {
        "selected": selected,
        "accepted_on_development": lock[
            "accepted_on_development"
        ],
        "baseline": baseline,
        "adaptive": adaptive,
        "score_delta": {
            name: adaptive[name]["score"] - baseline[name]["score"]
            for name, _, _ in windows
        },
        "final_state": diagnostics.iloc[-1].to_dict(),
    }
    (OUTPUT_DIR / "heldout_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    diagnostics.to_csv(
        OUTPUT_DIR / "adaptive_state_history.csv",
        index=False,
    )
    lock["heldout_evaluated"] = True
    LOCK_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    for name, _, _ in windows:
        print(
            f"{name:<10} baseline {baseline[name]['score']:9.3f} "
            f"adaptive {adaptive[name]['score']:9.3f} "
            f"delta {result['score_delta'][name]:+9.3f}"
        )
    print(
        "\nHeld-out result is diagnostic only and cannot promote "
        "a development-rejected model."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=("develop", "test"),
    )
    args = parser.parse_args()
    if args.phase == "develop":
        develop()
    else:
        test()


if __name__ == "__main__":
    main()

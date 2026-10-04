#!/usr/bin/env python
"""Per-signal adaptive EV states with shared EWMA/CUSUM controls.

This is the closest EV analogue to the live pair filters: every pair and
every ridge asset owns an independent causal edge/uncertainty state, but all
states share one adaptation half-life and one CUSUM threshold. Sparse states
are deterministically shrunk toward the corresponding family state.

Selection uses days 1-750 only. OOS is unavailable unless one setting improves
all four fixed 125-day development/validation folds.
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
from research import adaptive_ev_pair_filter as family_model
from research import whole_vector_ev_conflicts as study


OUTPUT_DIR = (
    ROOT / "analysis_outputs" / "adaptive_identity_ev_pair_filter"
)
LOCK_PATH = OUTPUT_DIR / "locked_selection.json"
SHRINK_OBSERVATIONS = 20.0


class ScalarEdgeState:
    """Causal scalar mean/variance with shared CUSUM controls."""

    def __init__(self, config: family_model.AdaptiveConfig) -> None:
        self.config = config
        self.alpha = 1.0 - 0.5 ** (1.0 / config.half_life)
        self.mean_edge = 0.0
        self.variance = 1.0
        self.innovation_variance = 1.0
        self.cusum_up = 0.0
        self.cusum_down = 0.0
        self.observations = 0
        self.alarms = 0

    def update(self, observed: float) -> None:
        innovation = observed - self.mean_edge
        if self.observations < family_model.MIN_STATE_DAYS:
            rate = 1.0 / (self.observations + 1.0)
            self.mean_edge += rate * innovation
            self.variance = (
                (1.0 - rate) * self.variance
                + rate * innovation * innovation
            )
            self.innovation_variance = self.variance
            self.observations += 1
            return

        scale = sqrt(max(self.innovation_variance, 1.0e-6))
        standardized = innovation / scale
        self.cusum_up = max(
            0.0,
            self.cusum_up
            + standardized
            - family_model.CUSUM_K,
        )
        self.cusum_down = max(
            0.0,
            self.cusum_down
            - standardized
            - family_model.CUSUM_K,
        )
        alarm = (
            self.cusum_up > self.config.cusum_h
            or self.cusum_down > self.config.cusum_h
        )
        if alarm:
            self.mean_edge = observed
            self.variance *= family_model.UNCERTAINTY_INFLATION
            self.cusum_up = 0.0
            self.cusum_down = 0.0
            self.alarms += 1
        else:
            self.mean_edge = (
                (1.0 - self.alpha) * self.mean_edge
                + self.alpha * observed
            )

        self.variance = (
            (1.0 - self.alpha) * self.variance
            + self.alpha * innovation * innovation
        )
        self.innovation_variance = self.variance
        self.observations += 1

    def blended_forecast(
        self,
        family_expected: float,
        family_variance: float,
    ) -> tuple[float, float]:
        weight = (
            self.observations
            / (self.observations + SHRINK_OBSERVATIONS)
        )
        expected = (
            weight * self.mean_edge
            + (1.0 - weight) * family_expected
        )
        variance = (
            weight * max(self.variance, 1.0e-6)
            + (1.0 - weight) * max(family_variance, 1.0e-6)
        )
        return expected, variance


def adaptive_identity_targets(
    cache: study.SignalCache,
    config: family_model.AdaptiveConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    n_history = cache.prices.shape[1]
    dollars = np.zeros((n_history + 1, strategy.N_INST))
    pair_family = family_model.AdaptiveFamilyState(config)
    ridge_family = family_model.AdaptiveFamilyState(config)
    pair_states = [
        ScalarEdgeState(config)
        for _ in range(cache.pair_pos_y.shape[1])
    ]
    ridge_states = [
        ScalarEdgeState(config)
        for _ in range(strategy.N_INST)
    ]
    records = []

    for history_size in range(1, n_history):
        previous = history_size - 1
        if previous >= 1:
            prior_pair_active = (
                np.abs(cache.pair_pos_y[previous]) > 1.0e-9
            )
            pair_family.update(
                cache.pair_strength[previous],
                cache.pair_gross_pnl[previous],
                prior_pair_active,
            )
            for identity in np.flatnonzero(prior_pair_active):
                pair_states[int(identity)].update(
                    float(cache.pair_gross_pnl[previous, identity])
                )

            prior_ridge_active = (
                np.abs(cache.ridge_pos[previous]) > 1.0e-9
            )
            ridge_family.update(
                cache.ridge_strength[previous],
                cache.ridge_gross_pnl[previous],
                prior_ridge_active,
            )
            for identity in np.flatnonzero(prior_ridge_active):
                ridge_states[int(identity)].update(
                    float(cache.ridge_gross_pnl[previous, identity])
                )

        pair_active = (
            np.abs(cache.pair_pos_y[history_size]) > 1.0e-9
        )
        family_pair_expected, family_pair_variance = (
            pair_family.forecast(
                cache.pair_strength[history_size],
                pair_active,
            )
        )
        combined = np.zeros(strategy.N_INST)
        for pair_index in np.flatnonzero(pair_active):
            pair_index = int(pair_index)
            expected, variance = pair_states[
                pair_index
            ].blended_forecast(
                float(family_pair_expected[pair_index]),
                family_pair_variance,
            )
            net = expected - cache.pair_cost[
                history_size,
                pair_index,
            ]
            if net <= 0.0:
                continue
            weight = net / variance
            y_index = cache.pair_y[pair_index]
            x_index = cache.pair_x[pair_index]
            combined[y_index] += (
                weight
                * cache.pair_pos_y[history_size, pair_index]
            )
            combined[x_index] += (
                weight
                * cache.pair_pos_x[history_size, pair_index]
            )

        ridge_active = (
            np.abs(cache.ridge_pos[history_size]) > 1.0e-9
        )
        family_ridge_expected, family_ridge_variance = (
            ridge_family.forecast(
                cache.ridge_strength[history_size],
                ridge_active,
            )
        )
        for instrument in np.flatnonzero(ridge_active):
            instrument = int(instrument)
            expected, variance = ridge_states[
                instrument
            ].blended_forecast(
                float(family_ridge_expected[instrument]),
                family_ridge_variance,
            )
            net = expected - cache.ridge_cost[
                history_size,
                instrument,
            ]
            if net > 0.0:
                combined[instrument] += (
                    net
                    / variance
                    * cache.ridge_pos[history_size, instrument]
                )

        direction = np.sign(combined)
        direction[cache.crash[history_size]] = 0.0
        dollars[history_size] = (
            strategy.BANG_BANG_LIMITS * direction
        )
        records.append(
            {
                "history_size": float(history_size),
                "pair_family_edge": pair_family.mean_edge,
                "ridge_family_edge": ridge_family.mean_edge,
                "pair_family_alarms": float(pair_family.alarms),
                "ridge_family_alarms": float(ridge_family.alarms),
                "pair_identity_alarms": float(
                    sum(state.alarms for state in pair_states)
                ),
                "ridge_identity_alarms": float(
                    sum(state.alarms for state in ridge_states)
                ),
                "active_assets": float(np.sum(direction != 0.0)),
            }
        )

    return dollars, pd.DataFrame(records)


def develop() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = study.load_prices(750)
    cache = study.collect_signals(prices)
    baseline = family_model.evaluate_windows(
        cache,
        cache.baseline_dollars,
        family_model.WINDOWS_DEVELOP,
    )
    rows = []

    for half_life in family_model.HALF_LIFE_GRID:
        for cusum_h in family_model.CUSUM_H_GRID:
            config = family_model.AdaptiveConfig(
                half_life,
                cusum_h,
            )
            dollars, diagnostics = adaptive_identity_targets(
                cache,
                config,
            )
            result = family_model.evaluate_windows(
                cache,
                dollars,
                family_model.WINDOWS_DEVELOP,
            )
            record = {
                "half_life": half_life,
                "cusum_h": cusum_h,
            }
            for name, _, _ in family_model.WINDOWS_DEVELOP:
                key = name.lower()
                record[f"{key}_score"] = result[name]["score"]
                record[f"{key}_delta"] = (
                    result[name]["score"]
                    - baseline[name]["score"]
                )
            folds = [
                record["train1_delta"],
                record["train2_delta"],
                record["val1_delta"],
                record["val2_delta"],
            ]
            record["positive_folds"] = sum(v > 0.0 for v in folds)
            record["robust_delta"] = min(folds)
            last = diagnostics.iloc[-1]
            record["pair_family_alarms"] = last[
                "pair_family_alarms"
            ]
            record["ridge_family_alarms"] = last[
                "ridge_family_alarms"
            ]
            record["pair_identity_alarms"] = last[
                "pair_identity_alarms"
            ]
            record["ridge_identity_alarms"] = last[
                "ridge_identity_alarms"
            ]
            rows.append(record)

    frame = pd.DataFrame(rows).sort_values(
        ["positive_folds", "robust_delta", "validation_delta"],
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
            "promotion_rule": "all four 125-day folds improve",
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
        "pair_identity_alarms",
        "ridge_identity_alarms",
    ]
    print(frame[columns].head(15).to_string(index=False))
    print(f"\nAccepted on development: {accepted}")
    print("No prices after day 750 were loaded.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("develop",),
        required=True,
    )
    parser.parse_args()
    develop()


if __name__ == "__main__":
    main()

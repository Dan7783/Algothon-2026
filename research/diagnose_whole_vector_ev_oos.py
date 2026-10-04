#!/usr/bin/env python
"""Post-test diagnosis of the frozen whole-vector EV model.

This script never selects or changes a setting. It explains the already
observed held-out result by comparing frozen forecasts, realised proposal
PnL, position coverage and changed asset decisions across fixed windows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import whole_vector_ev_conflicts as study


OUTPUT_DIR = study.OUTPUT_DIR
WINDOWS = (
    ("TRAIN1", 251, 375),
    ("TRAIN2", 376, 500),
    ("VAL1", 501, 625),
    ("VAL2", 626, 750),
    ("OOS1", 751, 875),
    ("OOS2", 876, 1000),
)


def account_detail(
    prices: np.ndarray,
    dollars: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    cash = 0.0
    value = 0.0
    prior_fee = 0.0
    current = np.zeros(strategy.N_INST)
    pnl: list[float] = []
    turnover = 0.0
    fees = 0.0

    for t in range(start - 1, end + 1):
        current_prices = prices[:, t - 1]
        new_position = (
            study._target_shares(dollars[t], current_prices).astype(float)
            if t < end
            else current.copy()
        )
        delta = new_position - current
        cash -= float(current_prices @ delta) + prior_fee
        volume = current_prices * np.abs(delta)
        prior_fee = float(volume @ study.COMMISSION)
        turnover += float(np.sum(volume))
        fees += prior_fee
        current = new_position
        new_value = cash + float(current @ current_prices)
        daily_pnl = new_value - value
        value = new_value
        if t >= start:
            pnl.append(daily_pnl)

    values = np.asarray(pnl)
    result = study._metrics(values)
    result["turnover"] = turnover
    result["fees"] = fees
    return result


def decision_diagnostics(
    cache: study.SignalCache,
    ev_dollars: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    counts = {
        "same": 0,
        "dropped": 0,
        "flipped": 0,
        "added": 0,
    }
    delta = {name: 0.0 for name in counts}
    base_active = []
    ev_active = []

    # Decision history h produces the PnL labelled h+1 by eval.py.
    for history_size in range(start - 1, end):
        current = cache.prices[:, history_size - 1]
        following = cache.prices[:, history_size]
        base = cache.baseline_dollars[history_size]
        ev = ev_dollars[history_size]
        base_sign = np.sign(base)
        ev_sign = np.sign(ev)
        base_active.append(int(np.sum(base_sign != 0.0)))
        ev_active.append(int(np.sum(ev_sign != 0.0)))

        base_gross = (
            base / current * (following - current)
        )
        ev_gross = (
            ev / current * (following - current)
        )
        masks = {
            "same": (
                (base_sign != 0.0)
                & (ev_sign == base_sign)
            ),
            "dropped": (
                (base_sign != 0.0)
                & (ev_sign == 0.0)
            ),
            "flipped": (
                (base_sign != 0.0)
                & (ev_sign == -base_sign)
            ),
            "added": (
                (base_sign == 0.0)
                & (ev_sign != 0.0)
            ),
        }
        for name, mask in masks.items():
            counts[name] += int(np.sum(mask))
            delta[name] += float(
                np.sum(ev_gross[mask] - base_gross[mask])
            )

    observations = max((end - start + 1) * strategy.N_INST, 1)
    return {
        "base_active_assets": float(np.mean(base_active)),
        "ev_active_assets": float(np.mean(ev_active)),
        **{
            f"{name}_share": counts[name] / observations
            for name in counts
        },
        **{
            f"{name}_gross_delta": delta[name]
            for name in delta
        },
        "changed_gross_delta": float(
            delta["dropped"] + delta["flipped"] + delta["added"]
        ),
    }


def empirical_slope(
    strength: np.ndarray,
    pnl: np.ndarray,
) -> float:
    if len(strength) < 2:
        return 0.0
    centered = strength - np.mean(strength)
    denominator = float(centered @ centered)
    if denominator < 1.0e-12:
        return 0.0
    return float(centered @ (pnl - np.mean(pnl)) / denominator)


def family_diagnostics(
    cache: study.SignalCache,
    calibration: study.FamilyCalibration,
    family: str,
    start: int,
    end: int,
    min_edge_z: float,
) -> dict[str, float]:
    if family == "pair":
        position = cache.pair_pos_y
        strength = cache.pair_strength
        gross = cache.pair_gross_pnl
        cost = cache.pair_cost
    else:
        position = cache.ridge_pos
        strength = cache.ridge_strength
        gross = cache.ridge_gross_pnl
        cost = cache.ridge_cost

    rows = np.zeros(position.shape[0], dtype=bool)
    rows[start - 1:end] = True
    active = (np.abs(position) > 1.0e-9) & rows[:, None]
    predicted_gross = calibration.predict(strength)
    predicted_net = predicted_gross - cost
    actual_net = gross - cost
    std = np.sqrt(calibration.variances)[None, :]
    accepted = active & (predicted_net > min_edge_z * std)

    s = strength[active]
    g = gross[active]
    net = actual_net[active]
    pred = predicted_gross[active]
    accept_net = actual_net[accepted]
    rejected = active & ~accepted
    reject_net = actual_net[rejected]

    correlation = (
        float(np.corrcoef(pred, g)[0, 1])
        if len(g) > 1 and np.std(pred) > 1.0e-12
        else 0.0
    )
    return {
        "observations": int(len(g)),
        "acceptance": float(np.sum(accepted) / max(np.sum(active), 1)),
        "predicted_gross_mean": float(np.mean(pred)),
        "actual_gross_mean": float(np.mean(g)),
        "calibration_error": float(np.mean(g - pred)),
        "actual_net_mean": float(np.mean(net)),
        "actual_hit_rate": float(np.mean(net > 0.0)),
        "prediction_realized_corr": correlation,
        "empirical_strength_slope": empirical_slope(s, g),
        "accepted_actual_net_mean": (
            float(np.mean(accept_net)) if len(accept_net) else 0.0
        ),
        "rejected_actual_net_mean": (
            float(np.mean(reject_net)) if len(reject_net) else 0.0
        ),
    }


def main() -> None:
    frozen = json.loads(
        (OUTPUT_DIR / "frozen_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    prices = study.load_prices(study.TEST_END)
    cache = study.collect_signals(prices)
    pair_calibration, ridge_calibration = study.fit_calibrations(
        cache,
        float(frozen["selected"]["prior_observations"]),
    )
    ev_dollars, _ = study.ev_dollars(
        cache,
        pair_calibration,
        ridge_calibration,
        float(frozen["selected"]["min_edge_z"]),
    )

    portfolio_rows = []
    family_rows = []
    for name, start, end in WINDOWS:
        baseline = account_detail(
            prices,
            cache.baseline_dollars,
            start,
            end,
        )
        ev = account_detail(prices, ev_dollars, start, end)
        decisions = decision_diagnostics(
            cache,
            ev_dollars,
            start,
            end,
        )
        portfolio_rows.append(
            {
                "window": name,
                "start": start,
                "end": end,
                "baseline_score": baseline["score"],
                "ev_score": ev["score"],
                "score_delta": ev["score"] - baseline["score"],
                "baseline_mean": baseline["mean_pnl"],
                "ev_mean": ev["mean_pnl"],
                "baseline_std": baseline["std_pnl"],
                "ev_std": ev["std_pnl"],
                "baseline_turnover": baseline["turnover"],
                "ev_turnover": ev["turnover"],
                "baseline_fees": baseline["fees"],
                "ev_fees": ev["fees"],
                **decisions,
            }
        )
        for family, calibration in (
            ("pair", pair_calibration),
            ("ridge", ridge_calibration),
        ):
            family_rows.append(
                {
                    "window": name,
                    "family": family,
                    **family_diagnostics(
                        cache,
                        calibration,
                        family,
                        start,
                        end,
                        float(frozen["selected"]["min_edge_z"]),
                    ),
                }
            )

    portfolio = pd.DataFrame(portfolio_rows)
    families = pd.DataFrame(family_rows)
    portfolio.to_csv(
        OUTPUT_DIR / "oos_portfolio_diagnosis.csv",
        index=False,
    )
    families.to_csv(
        OUTPUT_DIR / "oos_calibration_diagnosis.csv",
        index=False,
    )

    print("Portfolio diagnosis:")
    print(
        portfolio[
            [
                "window",
                "baseline_score",
                "ev_score",
                "score_delta",
                "baseline_mean",
                "ev_mean",
                "base_active_assets",
                "ev_active_assets",
                "changed_gross_delta",
            ]
        ].to_string(index=False)
    )
    print("\nCalibration diagnosis:")
    print(
        families[
            [
                "window",
                "family",
                "observations",
                "acceptance",
                "predicted_gross_mean",
                "actual_gross_mean",
                "calibration_error",
                "prediction_realized_corr",
                "empirical_strength_slope",
                "accepted_actual_net_mean",
                "rejected_actual_net_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

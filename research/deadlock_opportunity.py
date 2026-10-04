"""Quantify true zero-position deadlocks caused by overlapping pair legs.

This is a diagnostic, not a tuning routine.  It replays the current strategy
over all four 250-day periods and compares:

* unchanged production allocation;
* a causal, identity-free moneyness tie-breaker; and
* a hindsight oracle upper bound.
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


OUTPUT_DIR = ROOT / "analysis_outputs" / "deadlock_opportunity"
PERIODS = (
    ("YEAR1", 31, 250),
    ("YEAR2", 251, 500),
    ("VALIDATION", 501, 750),
    ("OOS", 751, 1000),
)


def pair_moneyness_vote(
    candidates: list[tuple[float, strategy.PairAgent, float, float]],
) -> np.ndarray:
    """Expected OU movement aggregated without pair-name preferences."""
    vote = np.zeros(strategy.N_INST)
    for _, agent, pos_y, pos_x in candidates:
        abs_z = abs(float(agent.current_z))
        kappa = max(float(agent.current_kappa), 0.0)
        sigma = max(float(agent.current_sigma), 0.0)
        expected_move = abs_z * sigma * (1.0 - np.exp(-kappa))
        for instrument, position in (
            (int(agent.spec.y), float(pos_y)),
            (int(agent.spec.x), float(pos_x)),
        ):
            if abs(position) <= 1.0e-9:
                continue
            leg_scale = abs(position) / max(strategy.LEG, 1.0e-12)
            vote[instrument] += (
                expected_move * leg_scale * np.sign(position)
            )
    return vote


def replay(prices: np.ndarray) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
]:
    n_history = prices.shape[1]
    baseline = np.zeros((n_history + 1, strategy.N_INST))
    moneyness_breaker = np.zeros_like(baseline)
    hindsight_oracle = np.zeros_like(baseline)
    events: list[dict[str, float | int | str | bool]] = []
    daily_counts: list[dict[str, int]] = []
    atlas = strategy.AtlasBook()

    for history_size in range(1, n_history):
        history = prices[:, :history_size]
        candidates = atlas.candidates(history)
        if history_size < strategy.WARMUP + 10:
            continue

        pair_vote, pair_active, contributors = (
            strategy._pair_conflict_consensus(candidates)
        )
        returns = np.diff(
            np.log(np.maximum(history, strategy._EPS_DENOM)),
            axis=1,
        )
        ridge_raw, ridge_strength = strategy._ridge_forecast(returns)
        ridge_sign = np.sign(ridge_raw)
        ridge_active = ridge_sign != 0.0
        combined = (
            pair_vote
            + strategy.RIDGE_CONFLICT_WEIGHT
            * ridge_sign
            * ridge_strength
        )
        active = pair_active | ridge_active
        direction = np.sign(combined)
        tied = active & (direction == 0.0) & ridge_active
        direction[tied] = ridge_sign[tied]
        last_return = np.diff(
            np.log(np.maximum(history[:, -2:], strategy._EPS_DENOM)),
            axis=1,
        )[:, 0]
        crash = np.abs(last_return) > 0.40
        crash[0] = False
        direction[crash] = 0.0
        baseline[history_size] = (
            strategy.BANG_BANG_LIMITS * direction
        )
        moneyness_breaker[history_size] = baseline[history_size]
        hindsight_oracle[history_size] = baseline[history_size]

        # A genuine deadlock is flat specifically because multiple pair legs
        # exactly cancel. Ridge-resolved ties and crash-filtered positions do
        # not count.
        deadlock = (
            pair_active
            & (contributors >= 2)
            & (np.abs(pair_vote) <= 1.0e-12)
            & ~ridge_active
            & ~crash
            & (strategy.BANG_BANG_LIMITS > 0.0)
        )
        daily_counts.append(
            {
                "score_day": int(history_size + 1),
                "active_pair_assets": int(np.sum(pair_active)),
                "overlap_asset_days": int(
                    np.sum(pair_active & (contributors >= 2))
                ),
                "deadlock_asset_days": int(np.sum(deadlock)),
            }
        )
        if not np.any(deadlock):
            continue

        money_vote = pair_moneyness_vote(candidates)
        money_direction = np.sign(money_vote)
        forward_return = (
            prices[:, history_size]
            / np.maximum(
                prices[:, history_size - 1],
                strategy._EPS_DENOM,
            )
            - 1.0
        )
        oracle_direction = np.sign(forward_return)
        for instrument in np.flatnonzero(deadlock):
            chosen = float(money_direction[instrument])
            if chosen != 0.0:
                moneyness_breaker[history_size, instrument] = (
                    strategy.BANG_BANG_LIMITS[instrument] * chosen
                )
            hindsight_oracle[history_size, instrument] = (
                strategy.BANG_BANG_LIMITS[instrument]
                * oracle_direction[instrument]
            )
            symbol = strategy._SYMBOL_BY_INDEX[int(instrument)]
            events.append(
                {
                    "history_size": int(history_size),
                    "score_day": int(history_size + 1),
                    "instrument": int(instrument),
                    "symbol": symbol,
                    "contributors": int(contributors[instrument]),
                    "pair_vote": float(pair_vote[instrument]),
                    "moneyness_vote": float(money_vote[instrument]),
                    "moneyness_direction": int(chosen),
                    "forward_return": float(forward_return[instrument]),
                    "moneyness_correct": bool(
                        chosen != 0.0
                        and chosen == oracle_direction[instrument]
                    ),
                    "gross_oracle_dollars": float(
                        strategy.BANG_BANG_LIMITS[instrument]
                        * abs(forward_return[instrument])
                    ),
                    "gross_moneyness_dollars": float(
                        strategy.BANG_BANG_LIMITS[instrument]
                        * chosen
                        * forward_return[instrument]
                    ),
                }
            )
    return (
        baseline,
        moneyness_breaker,
        hindsight_oracle,
        pd.DataFrame(events),
        pd.DataFrame(daily_counts),
    )


def window_metrics(
    prices: np.ndarray,
    dollars: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        name: study.region_metrics(prices, dollars, start, end)
        for name, start, end in PERIODS
    }


def event_summary(
    events: pd.DataFrame,
    daily_counts: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for name, start, end in PERIODS:
        frame = events[
            (events["score_day"] >= start)
            & (events["score_day"] <= end)
        ]
        counts = daily_counts[
            (daily_counts["score_day"] >= start)
            & (daily_counts["score_day"] <= end)
        ]
        overlap_asset_days = int(counts["overlap_asset_days"].sum())
        output[name] = {
            "deadlock_asset_days": int(len(frame)),
            "deadlock_calendar_days": int(frame["score_day"].nunique()),
            "overlap_asset_days": overlap_asset_days,
            "deadlock_share_of_overlaps": float(
                len(frame) / max(overlap_asset_days, 1)
            ),
            "gross_oracle_opportunity": float(
                frame["gross_oracle_dollars"].sum()
            ),
            "gross_moneyness_pnl_before_costs": float(
                frame["gross_moneyness_dollars"].sum()
            ),
            "moneyness_decision_rate": float(
                np.mean(frame["moneyness_direction"] != 0)
                if len(frame)
                else 0.0
            ),
            "moneyness_directional_accuracy": float(
                frame.loc[
                    frame["moneyness_direction"] != 0,
                    "moneyness_correct",
                ].mean()
                if np.any(frame["moneyness_direction"] != 0)
                else 0.0
            ),
        }
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = study.load_prices(1000)
    (
        baseline,
        moneyness_breaker,
        hindsight_oracle,
        events,
        daily_counts,
    ) = replay(prices)
    events.to_csv(OUTPUT_DIR / "deadlock_events.csv", index=False)

    baseline_metrics = window_metrics(prices, baseline)
    moneyness_metrics = window_metrics(prices, moneyness_breaker)
    oracle_metrics = window_metrics(prices, hindsight_oracle)
    summary = {
        "definition": (
            "two or more active pair legs exactly cancel to zero, ridge is "
            "inactive, and the crash filter is inactive"
        ),
        "total_deadlock_asset_days": int(len(events)),
        "total_deadlock_calendar_days": int(events["score_day"].nunique()),
        "events": event_summary(events, daily_counts),
        "baseline": baseline_metrics,
        "moneyness_breaker": moneyness_metrics,
        "hindsight_oracle": oracle_metrics,
        "moneyness_incremental": {},
        "oracle_incremental": {},
    }
    for name, _, _ in PERIODS:
        summary["moneyness_incremental"][name] = {
            "score": (
                moneyness_metrics[name]["score"]
                - baseline_metrics[name]["score"]
            ),
            "total_pnl": (
                moneyness_metrics[name]["total_pnl"]
                - baseline_metrics[name]["total_pnl"]
            ),
        }
        summary["oracle_incremental"][name] = {
            "score": (
                oracle_metrics[name]["score"]
                - baseline_metrics[name]["score"]
            ),
            "total_pnl": (
                oracle_metrics[name]["total_pnl"]
                - baseline_metrics[name]["total_pnl"]
            ),
        }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Pair-overlap deadlock opportunity",
        "",
        f"- Deadlocked asset-days: **{len(events)}**",
        f"- Calendar days containing a deadlock: "
        f"**{events['score_day'].nunique()}**",
        "",
        "| Period | Events | Oracle movement | Moneyness PnL | "
        "Moneyness total-PnL delta | Moneyness score delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, _, _ in PERIODS:
        event = summary["events"][name]
        incremental = summary["moneyness_incremental"][name]
        lines.append(
            f"| {name} | {event['deadlock_asset_days']} | "
            f"{event['gross_oracle_opportunity']:.2f} | "
            f"{event['gross_moneyness_pnl_before_costs']:.2f} | "
            f"{incremental['total_pnl']:.2f} | "
            f"{incremental['score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "`Oracle movement` is an impossible hindsight upper bound: it",
            "assumes the correct direction is known before the next return.",
            "`Moneyness total-PnL delta` is the causal counterfactual after",
            "share rounding, position transitions, and commissions.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Test complete dynamic-beta pair execution on consensus-winning sides.

The live strategy forms exact pair vectors inside ``PairAgent`` and then
reduces them to independent signed instrument votes.  This research harness
tests the alternative implied by those votes:

1. retain only pair vectors whose *two* intended legs agree with the winning
   pair-consensus direction for their respective instruments;
2. allocate the complete dynamic-beta vector under the existing per-instrument
   dollar limits; and
3. optionally let ridge use either unused instruments or residual capacity.

No signal parameter is tuned here.  Days 251-500 are training, days 501-750
are validation, and days 751-1000 are opened only after a development
candidate has been frozen to ``locked_candidate.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import algo12_equal_sleeve_research as evaluator


PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "winning_pair_hedge"
LOCK_PATH = OUTPUT_DIR / "locked_candidate.json"
ALGO = strategy.I["ALGO"]
EPS = 1.0e-9

DEVELOPMENT_WINDOWS = (
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("TRAIN", 251, 500),
    ("VALIDATION_A", 501, 625),
    ("VALIDATION_B", 626, 750),
    ("VALIDATION", 501, 750),
)
HELDOUT_WINDOWS = (("EXPOSED_YEAR4", 751, 1000),)

ALLOCATORS = ("greedy", "lp_equal", "lp_priority")
RIDGE_POLICIES = ("pairs", "ridge_flat", "ridge_capacity")
ALTERNATIVES = tuple(
    f"winner_{allocator}_{policy}"
    for allocator in ALLOCATORS
    for policy in RIDGE_POLICIES
)
ALL_VARIANTS = ("current_live",) + ALTERNATIVES


def allocator_from_variant(variant: str) -> str:
    for allocator in ALLOCATORS:
        if variant.startswith(f"winner_{allocator}_"):
            return allocator
    raise ValueError(f"Cannot infer allocator from variant: {variant}")


def limits() -> np.ndarray:
    result = strategy.BANG_BANG_LIMITS.copy()
    result[ALGO] = 0.0
    return result


def to_shares(dollars: np.ndarray, prices: np.ndarray) -> np.ndarray:
    safe = np.maximum(prices, strategy._EPS_DENOM)
    share_limits = np.floor(limits() / safe).astype(int)
    requested = np.round(dollars / safe)
    return np.clip(requested, -share_limits, share_limits).astype(int)


def pair_weight(detail: dict[str, Any]) -> float:
    boost = (
        strategy.PROTECTED_VOTE_BOOST
        if str(detail["pair"]) in strategy.PROTECTED_PAIR_NAMES
        else 1.0
    )
    return boost * max(float(detail["priority"]), 1.0e-12)


def winning_candidates(
    details: list[dict[str, Any]],
    pair_vote: np.ndarray,
    crash: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Return complete pairs whose two legs agree with consensus winners."""
    winners = np.sign(pair_vote)
    result: list[dict[str, Any]] = []
    win_counts: dict[int, int] = {}
    for event_id, detail in enumerate(details):
        y = int(detail["y"])
        x = int(detail["x"])
        pos_y = float(detail["pos_y"])
        pos_x = float(detail["pos_x"])
        count = int(winners[y] == np.sign(pos_y)) + int(
            winners[x] == np.sign(pos_x)
        )
        win_counts[event_id] = count
        if (
            count == 2
            and not crash[y]
            and not crash[x]
            and y != ALGO
            and x != ALGO
        ):
            candidate = dict(detail)
            candidate["event_id"] = event_id
            result.append(candidate)
    return result, win_counts


def greedy_allocate(
    candidates: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[int, float]]:
    """Greedily fill complete vectors in signal-priority order."""
    cap = limits()
    dollars = np.zeros(strategy.N_INST)
    fills: dict[int, float] = {}
    for detail in sorted(candidates, key=pair_weight, reverse=True):
        y = int(detail["y"])
        x = int(detail["x"])
        pos_y = float(detail["pos_y"])
        pos_x = float(detail["pos_x"])
        fill = 1.0
        for instrument, delta in ((y, pos_y), (x, pos_x)):
            if delta > EPS:
                fill = min(
                    fill,
                    max(0.0, (cap[instrument] - dollars[instrument]) / delta),
                )
            elif delta < -EPS:
                fill = min(
                    fill,
                    max(
                        0.0,
                        (-cap[instrument] - dollars[instrument]) / delta,
                    ),
                )
        fill = float(np.clip(fill, 0.0, 1.0))
        dollars[y] += fill * pos_y
        dollars[x] += fill * pos_x
        fills[int(detail["event_id"])] = fill
    return dollars, fills


def lp_allocate(
    candidates: list[dict[str, Any]],
    priority_weighted: bool,
) -> tuple[np.ndarray, dict[int, float]]:
    """Maximize complete-vector fills subject to instrument dollar limits."""
    if not candidates:
        return np.zeros(strategy.N_INST), {}

    vectors = np.zeros((strategy.N_INST, len(candidates)))
    weights = np.ones(len(candidates))
    for column, detail in enumerate(candidates):
        vectors[int(detail["y"]), column] = float(detail["pos_y"])
        vectors[int(detail["x"]), column] = float(detail["pos_x"])
        if priority_weighted:
            weights[column] = pair_weight(detail)
    weights /= max(float(weights.max()), EPS)
    cap = limits()
    solution = linprog(
        c=-weights,
        A_ub=np.vstack((vectors, -vectors)),
        b_ub=np.concatenate((cap, cap)),
        bounds=[(0.0, 1.0)] * len(candidates),
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(
            f"Complete-pair allocation LP failed: {solution.message}"
        )
    fill_values = np.clip(solution.x, 0.0, 1.0)
    dollars = vectors @ fill_values
    fills = {
        int(detail["event_id"]): float(fill)
        for detail, fill in zip(candidates, fill_values, strict=True)
    }
    return dollars, fills


def desired_ridge(sleeves: dict[str, Any]) -> np.ndarray:
    """Recreate ridge's own desired vector, including the deadlock sleeve."""
    result = np.asarray(sleeves["ridge_raw"], dtype=float).copy()
    deadlock = np.asarray(
        sleeves["deadlock_ridge_overlay_mask"],
        dtype=bool,
    )
    overlay = np.asarray(
        sleeves["deadlock_ridge_overlay_dollars"],
        dtype=float,
    )
    result[deadlock] = overlay[deadlock]
    result[np.asarray(sleeves["crash_mask"], dtype=bool)] = 0.0
    result[ALGO] = 0.0
    return result


def add_ridge(
    pair_dollars: np.ndarray,
    ridge_dollars: np.ndarray,
    policy: str,
) -> np.ndarray:
    if policy == "pairs":
        return pair_dollars.copy()
    ridge = ridge_dollars.copy()
    if policy == "ridge_flat":
        ridge[np.abs(pair_dollars) > EPS] = 0.0
    elif policy != "ridge_capacity":
        raise ValueError(f"Unknown ridge policy: {policy}")
    lower = -limits() - pair_dollars
    upper = limits() - pair_dollars
    result = pair_dollars + np.clip(ridge, lower, upper)
    result[ALGO] = 0.0
    return result


def apply_algo_overlay(
    core_dollars: np.ndarray,
    sleeves: dict[str, Any],
) -> np.ndarray:
    """Keep the production ALGO-residual overlay constant in every variant."""
    if not bool(sleeves["algo_beta_ready"]):
        return core_dollars.copy()
    final, _, _, _ = strategy._algo_beta_neutral_overlay(
        core_dollars,
        np.asarray(sleeves["algo_residual_requested"], dtype=float),
        np.asarray(sleeves["algo_return_betas"], dtype=float),
        np.asarray(sleeves["crash_mask"], dtype=bool),
    )
    return final


def allocate_all(
    candidates: list[dict[str, Any]],
) -> dict[str, tuple[np.ndarray, dict[int, float]]]:
    greedy, greedy_fills = greedy_allocate(candidates)
    equal, equal_fills = lp_allocate(candidates, priority_weighted=False)
    priority, priority_fills = lp_allocate(
        candidates,
        priority_weighted=True,
    )
    return {
        "greedy": (greedy, greedy_fills),
        "lp_equal": (equal, equal_fills),
        "lp_priority": (priority, priority_fills),
    }


def execution_status(
    intended_y: float,
    intended_x: float,
    actual_y: float,
    actual_x: float,
) -> str:
    if abs(actual_y) <= EPS or abs(actual_x) <= EPS:
        return "missing_leg"
    if (
        np.sign(actual_y) != np.sign(intended_y)
        or np.sign(actual_x) != np.sign(intended_x)
    ):
        return "flipped_leg"
    return "both_signs_match"


def hedge_error(
    beta: float,
    y_dollars: float,
    x_dollars: float,
) -> float:
    numerator = abs(beta * y_dollars + x_dollars)
    denominator = abs(beta * y_dollars) + abs(x_dollars)
    return numerator / denominator if denominator > EPS else 1.0


def build_targets(
    prices: np.ndarray,
    variants: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    targets = {
        variant: np.zeros(
            (prices.shape[1] + 1, strategy.N_INST),
            dtype=int,
        )
        for variant in variants
    }
    event_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    strategy.reset()

    for history_size in range(1, prices.shape[1]):
        if history_size < strategy.WARMUP + 10:
            continue
        known = prices[:, :history_size]
        sleeves = strategy.sleeve_breakdown(known)
        if sleeves is None:
            continue

        pair_vote = np.asarray(sleeves["pair_vote"], dtype=float)
        crash = np.asarray(sleeves["crash_mask"], dtype=bool)
        details = list(sleeves["pair_contributor_details"])
        compatible, win_counts = winning_candidates(
            details,
            pair_vote,
            crash,
        )
        allocations = allocate_all(compatible)
        ridge = desired_ridge(sleeves)
        values: dict[str, np.ndarray] = {
            "current_live": np.asarray(sleeves["dollars"], dtype=float),
        }
        for allocator, (pair_dollars, _) in allocations.items():
            for policy in RIDGE_POLICIES:
                variant = f"winner_{allocator}_{policy}"
                core = add_ridge(pair_dollars, ridge, policy)
                values[variant] = apply_algo_overlay(core, sleeves)

        for variant in variants:
            dollars = values[variant]
            targets[variant][history_size] = to_shares(
                dollars,
                known[:, -1],
            )
            daily_rows.append(
                {
                    "history_size": history_size,
                    "pnl_day": history_size + 1,
                    "variant": variant,
                    "gross_dollars": float(np.abs(dollars).sum()),
                    "active_instruments": int(
                        np.sum(np.abs(dollars) > EPS)
                    ),
                }
            )

        contributors = np.asarray(
            sleeves["pair_contributors"],
            dtype=int,
        )
        current_core = np.asarray(sleeves["core_dollars"], dtype=float)
        current_final = np.asarray(sleeves["dollars"], dtype=float)
        fill_maps = {
            allocator: fills
            for allocator, (_, fills) in allocations.items()
        }
        for event_id, detail in enumerate(details):
            y = int(detail["y"])
            x = int(detail["x"])
            pos_y = float(detail["pos_y"])
            pos_x = float(detail["pos_x"])
            beta = -pos_x / pos_y
            event_rows.append(
                {
                    "history_size": history_size,
                    "pnl_day": history_size + 1,
                    "pair": str(detail["pair"]),
                    "y": y,
                    "x": x,
                    "priority": float(detail["priority"]),
                    "beta": beta,
                    "winning_legs": win_counts[event_id],
                    "strictly_compatible": win_counts[event_id] == 2,
                    "isolated": (
                        contributors[y] == 1 and contributors[x] == 1
                    ),
                    "intended_hedge_error": hedge_error(
                        beta,
                        pos_y,
                        pos_x,
                    ),
                    "current_core_status": execution_status(
                        pos_y,
                        pos_x,
                        current_core[y],
                        current_core[x],
                    ),
                    "current_core_hedge_error": hedge_error(
                        beta,
                        current_core[y],
                        current_core[x],
                    ),
                    "current_final_status": execution_status(
                        pos_y,
                        pos_x,
                        current_final[y],
                        current_final[x],
                    ),
                    "current_final_hedge_error": hedge_error(
                        beta,
                        current_final[y],
                        current_final[x],
                    ),
                    **{
                        f"{allocator}_fill": fill_maps[allocator].get(
                            event_id,
                            0.0,
                        )
                        for allocator in ALLOCATORS
                    },
                }
            )
    return targets, pd.DataFrame(event_rows), pd.DataFrame(daily_rows)


def evaluate_targets(
    prices: np.ndarray,
    targets: dict[str, np.ndarray],
    windows: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, target in targets.items():
        results = evaluator.evaluate(prices, target, windows)
        for window, metrics in results.items():
            rows.append(
                {
                    "variant": variant,
                    "window": window,
                    **metrics,
                }
            )
    frame = pd.DataFrame(rows)
    baseline = (
        frame[frame["variant"] == "current_live"]
        .set_index("window")["score"]
    )
    frame["score_delta"] = frame.apply(
        lambda row: float(row["score"] - baseline[row["window"]]),
        axis=1,
    )
    return frame


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    periods = (
        ("TRAIN", 251, 500),
        ("VALIDATION", 501, 750),
        ("EXPOSED_YEAR4", 751, 1000),
    )
    rows: list[dict[str, Any]] = []
    for period, start, end in periods:
        sample = events[events["pnl_day"].between(start, end)]
        if sample.empty:
            continue
        for layer in ("current_core", "current_final"):
            status = sample[f"{layer}_status"]
            error = sample[f"{layer}_hedge_error"]
            isolated = sample["isolated"]
            rows.append(
                {
                    "period": period,
                    "layer": layer,
                    "pair_events": len(sample),
                    "two_winning_legs": float(
                        (sample["winning_legs"] == 2).mean()
                    ),
                    "one_winning_leg": float(
                        (sample["winning_legs"] == 1).mean()
                    ),
                    "zero_winning_legs": float(
                        (sample["winning_legs"] == 0).mean()
                    ),
                    "both_signs_match": float(
                        (status == "both_signs_match").mean()
                    ),
                    "missing_leg": float(
                        (status == "missing_leg").mean()
                    ),
                    "flipped_leg": float(
                        (status == "flipped_leg").mean()
                    ),
                    "median_hedge_error": float(error.median()),
                    "isolated_events": int(isolated.sum()),
                    "isolated_median_hedge_error": float(
                        error[isolated].median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_fills(events: pd.DataFrame) -> pd.DataFrame:
    periods = (
        ("TRAIN", 251, 500),
        ("VALIDATION", 501, 750),
        ("EXPOSED_YEAR4", 751, 1000),
    )
    rows: list[dict[str, Any]] = []
    for period, start, end in periods:
        sample = events[events["pnl_day"].between(start, end)]
        if sample.empty:
            continue
        for allocator in ALLOCATORS:
            fills = sample[f"{allocator}_fill"].to_numpy(float)
            compatible = sample["strictly_compatible"].to_numpy(bool)
            rows.append(
                {
                    "period": period,
                    "allocator": allocator,
                    "active_pair_events": len(sample),
                    "compatible_pair_events": int(compatible.sum()),
                    "compatible_share": float(compatible.mean()),
                    "fully_filled_share_all": float(
                        np.mean(fills >= 1.0 - EPS)
                    ),
                    "partially_filled_share_all": float(
                        np.mean((fills > EPS) & (fills < 1.0 - EPS))
                    ),
                    "unfilled_share_all": float(
                        np.mean(fills <= EPS)
                    ),
                    "mean_fill_compatible": float(
                        fills[compatible].mean()
                    )
                    if compatible.any()
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_capital(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    periods = (
        ("TRAIN", 251, 500),
        ("VALIDATION", 501, 750),
        ("EXPOSED_YEAR4", 751, 1000),
    )
    rows: list[dict[str, Any]] = []
    for period, start, end in periods:
        sample = daily[daily["pnl_day"].between(start, end)]
        for variant, group in sample.groupby("variant"):
            rows.append(
                {
                    "period": period,
                    "variant": variant,
                    "mean_gross_dollars": float(
                        group["gross_dollars"].mean()
                    ),
                    "mean_active_instruments": float(
                        group["active_instruments"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def choose_candidate(performance: pd.DataFrame) -> dict[str, Any]:
    scores = performance.pivot(
        index="variant",
        columns="window",
        values="score",
    )
    baseline = scores.loc["current_live"]
    rows: list[dict[str, Any]] = []
    for variant in ALTERNATIVES:
        deltas = scores.loc[variant] - baseline
        eligible = bool(
            deltas["TRAIN"] >= 0.0
            and deltas["VALIDATION"] >= 0.0
            and deltas["TRAIN_A"] >= -5.0
            and deltas["TRAIN_B"] >= -5.0
            and deltas["VALIDATION_A"] >= -5.0
            and deltas["VALIDATION_B"] >= -5.0
        )
        rows.append(
            {
                "variant": variant,
                "eligible": eligible,
                "worst_main_delta": float(
                    min(deltas["TRAIN"], deltas["VALIDATION"])
                ),
                "combined_main_delta": float(
                    deltas["TRAIN"] + deltas["VALIDATION"]
                ),
                **{
                    f"{window.lower()}_score": float(
                        scores.loc[variant, window]
                    )
                    for window in scores.columns
                },
                **{
                    f"{window.lower()}_delta": float(deltas[window])
                    for window in scores.columns
                },
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["eligible", "worst_main_delta", "combined_main_delta"],
        ascending=False,
    )
    ranking.to_csv(OUTPUT_DIR / "development_ranking.csv", index=False)
    winner = ranking.iloc[0].to_dict()
    lock = {
        "selection_data_end": 750,
        "selected_variant": str(winner["variant"]),
        "eligible_for_production": bool(winner["eligible"]),
        "selection_rule": (
            "require nonnegative TRAIN and VALIDATION deltas and no "
            "chronological half below live by more than 5 score; rank by "
            "worst main-window delta, then combined main-window delta"
        ),
        "development_metrics": {
            key: (
                bool(value)
                if isinstance(value, (bool, np.bool_))
                else float(value)
                if isinstance(value, (float, np.floating))
                else value
            )
            for key, value in winner.items()
        },
    }
    LOCK_PATH.write_text(
        json.dumps(lock, indent=2) + "\n",
        encoding="utf-8",
    )
    return lock


def write_development_report(
    performance: pd.DataFrame,
    event_summary: pd.DataFrame,
    fill_summary: pd.DataFrame,
    capital: pd.DataFrame,
    lock: dict[str, Any],
) -> None:
    pivot = performance.pivot(
        index="variant",
        columns="window",
        values="score",
    )
    selected = str(lock["selected_variant"])
    variants = ("current_live", selected)
    lines = [
        "# Winning-side complete-pair hedge audit",
        "",
        "The production pair agents already calculate exact dynamic-beta "
        "vectors. The tested allocator keeps a pair only when both legs agree "
        "with the pair-consensus winning side, then scales both legs by the "
        "same fill.",
        "",
        "## Development scores",
        "",
        "| Variant | Train A | Train B | Train | Validation A | "
        "Validation B | Validation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        row = pivot.loc[variant]
        lines.append(
            f"| {variant} | {row.TRAIN_A:+.2f} | {row.TRAIN_B:+.2f} | "
            f"{row.TRAIN:+.2f} | {row.VALIDATION_A:+.2f} | "
            f"{row.VALIDATION_B:+.2f} | {row.VALIDATION:+.2f} |"
        )
    lines.extend(
        [
            "",
            f"Frozen development winner: **{selected}**. Production "
            f"eligibility: **{lock['eligible_for_production']}**.",
            "",
            "## Current execution integrity",
            "",
            "| Period | Layer | Events | Two winning legs | One winning leg | "
            "Both signs match | Missing | Flipped | Median hedge error | "
            "Isolated error |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in event_summary.iterrows():
        lines.append(
            f"| {row.period} | {row.layer} | {int(row.pair_events):,} | "
            f"{row.two_winning_legs:.1%} | {row.one_winning_leg:.1%} | "
            f"{row.both_signs_match:.1%} | {row.missing_leg:.1%} | "
            f"{row.flipped_leg:.1%} | {row.median_hedge_error:.1%} | "
            f"{row.isolated_median_hedge_error:.1%} |"
        )
    lines.extend(
        [
            "",
            "A pair component accepted by a complete-pair allocator has zero "
            "pre-rounding hedge error by construction. The trade-off is how "
            "many active pair-events survive both-leg consensus and capacity.",
            "",
            "## Capacity of frozen allocator",
            "",
        ]
    )
    selected_allocator = allocator_from_variant(selected)
    selected_fills = fill_summary[
        fill_summary["allocator"] == selected_allocator
    ]
    for _, row in selected_fills.iterrows():
        lines.append(
            f"- {row.period}: {row.compatible_share:.1%} of active "
            f"pair-events have two winning legs; "
            f"{row.fully_filled_share_all:.1%} are fully filled, "
            f"{row.partially_filled_share_all:.1%} partially filled, and "
            f"{row.unfilled_share_all:.1%} unfilled."
        )
    lines.extend(
        [
            "",
            "## Capital use",
            "",
            "| Period | Variant | Mean gross dollars | Mean active assets |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in capital[
        capital["variant"].isin(variants)
    ].iterrows():
        lines.append(
            f"| {row.period} | {row.variant} | "
            f"${row.mean_gross_dollars:,.0f} | "
            f"{row.mean_active_instruments:.1f} |"
        )
    (OUTPUT_DIR / "development_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_heldout_report(
    performance: pd.DataFrame,
    event_summary: pd.DataFrame,
    fill_summary: pd.DataFrame,
    capital: pd.DataFrame,
    lock: dict[str, Any],
) -> None:
    selected = str(lock["selected_variant"])
    scores = performance.set_index("variant")
    baseline = scores.loc["current_live"]
    candidate = scores.loc[selected]
    selected_allocator = allocator_from_variant(selected)
    fill = fill_summary[
        (fill_summary["period"] == "EXPOSED_YEAR4")
        & (fill_summary["allocator"] == selected_allocator)
    ].iloc[0]
    current = event_summary[
        (event_summary["period"] == "EXPOSED_YEAR4")
        & (event_summary["layer"] == "current_final")
    ].iloc[0]
    cap = capital[
        (capital["period"] == "EXPOSED_YEAR4")
        & capital["variant"].isin(("current_live", selected))
    ]
    cap_map = cap.set_index("variant")
    lines = [
        "# Frozen held-out result: complete-pair winning-side execution",
        "",
        f"The development-only selection froze **{selected}** before days "
        "751-1000 were evaluated.",
        "",
        "| Variant | Score | Mean PnL | Total PnL | Fees | "
        "Mean gross dollars |",
        "|---|---:|---:|---:|---:|---:|",
        f"| current_live | {baseline.score:+.2f} | "
        f"${baseline.mean_pnl:,.2f} | ${baseline.total_pnl:,.0f} | "
        f"${baseline.fees:,.0f} | "
        f"${cap_map.loc['current_live', 'mean_gross_dollars']:,.0f} |",
        f"| {selected} | {candidate.score:+.2f} | "
        f"${candidate.mean_pnl:,.2f} | ${candidate.total_pnl:,.0f} | "
        f"${candidate.fees:,.0f} | "
        f"${cap_map.loc[selected, 'mean_gross_dollars']:,.0f} |",
        "",
        f"Score delta: **{candidate.score - baseline.score:+.2f}**.",
        "",
        "## What the mechanics show",
        "",
        f"- Current final execution matches both intended pair-leg signs in "
        f"{current.both_signs_match:.1%} of active pair-events; "
        f"{current.missing_leg:.1%} have a missing leg and "
        f"{current.flipped_leg:.1%} have a flipped leg.",
        f"- The current isolated-pair median beta-hedge error is "
        f"{current.isolated_median_hedge_error:.1%}.",
        f"- Only {fill.compatible_share:.1%} of active pair-events have both "
        f"legs on their consensus-winning sides; "
        f"{fill.unfilled_share_all:.1%} of all pair-events are therefore "
        "unfilled after consensus and capacity.",
        "",
        "The held-out result is descriptive only. The production strategy is "
        "unchanged by this audit.",
    ]
    (OUTPUT_DIR / "heldout_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_development(prices: np.ndarray) -> None:
    development_prices = prices[:, :750]
    targets, events, daily = build_targets(
        development_prices,
        ALL_VARIANTS,
    )
    performance = evaluate_targets(
        development_prices,
        targets,
        DEVELOPMENT_WINDOWS,
    )
    event_summary = summarize_events(events)
    fill_summary = summarize_fills(events)
    capital = summarize_capital(daily)
    lock = choose_candidate(performance)

    performance.to_csv(
        OUTPUT_DIR / "development_performance.csv",
        index=False,
    )
    events.to_csv(OUTPUT_DIR / "development_pair_events.csv", index=False)
    event_summary.to_csv(
        OUTPUT_DIR / "development_execution_summary.csv",
        index=False,
    )
    fill_summary.to_csv(
        OUTPUT_DIR / "development_fill_summary.csv",
        index=False,
    )
    capital.to_csv(
        OUTPUT_DIR / "development_capital.csv",
        index=False,
    )
    write_development_report(
        performance,
        event_summary,
        fill_summary,
        capital,
        lock,
    )
    selected = str(lock["selected_variant"])
    print(
        performance[
            performance["variant"].isin(("current_live", selected))
        ][["variant", "window", "score", "score_delta"]].to_string(
            index=False
        )
    )
    print(json.dumps(lock, indent=2))


def run_heldout(prices: np.ndarray) -> None:
    if not LOCK_PATH.exists():
        raise RuntimeError(
            "Run --phase development first to freeze a candidate."
        )
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    selected = str(lock["selected_variant"])
    targets, events, daily = build_targets(
        prices,
        ("current_live", selected),
    )
    performance = evaluate_targets(
        prices,
        targets,
        HELDOUT_WINDOWS,
    )
    event_summary = summarize_events(events)
    fill_summary = summarize_fills(events)
    capital = summarize_capital(daily)
    performance.to_csv(
        OUTPUT_DIR / "heldout_performance.csv",
        index=False,
    )
    event_summary.to_csv(
        OUTPUT_DIR / "heldout_execution_summary.csv",
        index=False,
    )
    fill_summary.to_csv(
        OUTPUT_DIR / "heldout_fill_summary.csv",
        index=False,
    )
    capital.to_csv(
        OUTPUT_DIR / "heldout_capital.csv",
        index=False,
    )
    write_heldout_report(
        performance,
        event_summary,
        fill_summary,
        capital,
        lock,
    )
    print(
        performance[
            ["variant", "window", "score", "score_delta", "total_pnl"]
        ].to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("development", "heldout"),
        required=True,
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+").to_numpy().T
    if args.phase == "development":
        run_development(prices)
    else:
        run_heldout(prices)
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

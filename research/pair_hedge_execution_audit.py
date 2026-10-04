"""Audit whether Four Sigmas executes its pair signals as hedged spreads.

Each PairAgent correctly proposes a dynamic-beta vector

    (d_y, d_x) = (s, -beta * s).

The live allocator subsequently converts each leg to a signed vote and uses a
bang-bang instrument position.  This script measures the resulting loss of
pair integrity and tests simple hedge-preserving portfolio constructions.

Days 251-500 are training, 501-750 are validation, and 751-1000 are an
already-exposed descriptive period.  No production setting is changed here.
"""

from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import algo12_equal_sleeve_research as evaluator


PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "pair_hedge_execution"
ALGO = strategy.I["ALGO"]
WINDOWS = (
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("TRAIN", 251, 500),
    ("VALIDATION_A", 501, 625),
    ("VALIDATION_B", 626, 750),
    ("VALIDATION", 501, 750),
    ("EXPOSED_YEAR4", 751, 1000),
)
VARIANTS = (
    "current_pair_vote",
    "greedy_hedged_pair",
    "lp_equal_hedged_pair",
    "lp_priority_hedged_pair",
    "global_scaled_hedged_pair",
    "clipped_vector_pair",
    "current_live",
    "greedy_hedged_plus_ridge",
    "greedy_hedged_plus_ridge_flat_only",
    "lp_equal_hedged_plus_ridge",
    "lp_equal_hedged_plus_ridge_flat_only",
    "lp_priority_hedged_plus_ridge",
    "lp_priority_hedged_plus_ridge_flat_only",
)


def score(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe**2 / (sharpe**2 + 1.0)


def crash_mask(prices: np.ndarray) -> np.ndarray:
    last_return = np.diff(
        np.log(np.maximum(prices[:, -2:], strategy._EPS_DENOM)),
        axis=1,
    )[:, 0]
    result = np.abs(last_return) > 0.40
    result[ALGO] = False
    return result


def dollar_limits() -> np.ndarray:
    limits = strategy.BANG_BANG_LIMITS.copy()
    limits[ALGO] = 0.0
    return limits


def to_shares(dollars: np.ndarray, prices: np.ndarray) -> np.ndarray:
    safe = np.maximum(prices, strategy._EPS_DENOM)
    limits = np.floor(dollar_limits() / safe).astype(int)
    requested = np.round(dollars / safe)
    return np.clip(requested, -limits, limits).astype(int)


def current_pair_dollars(
    pair_vote: np.ndarray,
    pair_active: np.ndarray,
    crash: np.ndarray,
) -> np.ndarray:
    direction = np.sign(pair_vote)
    direction[~pair_active] = 0.0
    direction[crash] = 0.0
    result = dollar_limits() * direction
    result[ALGO] = 0.0
    return result


def current_live_dollars(
    pair_vote: np.ndarray,
    pair_active: np.ndarray,
    ridge_raw: np.ndarray,
    ridge_strength: np.ndarray,
    ridge_latent_sign: np.ndarray,
    ridge_gate_pass: bool,
    crash: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ridge_sign = np.sign(ridge_raw)
    ridge_active = ridge_sign != 0.0
    ridge_vote = (
        strategy.RIDGE_CONFLICT_WEIGHT
        * ridge_sign
        * ridge_strength
    )
    combined = pair_vote + ridge_vote
    active = pair_active | ridge_active
    direction = np.sign(combined)
    tied = active & (direction == 0.0) & ridge_active
    direction[tied] = ridge_sign[tied]
    direction[crash] = 0.0
    limits = dollar_limits()
    result = limits * direction
    deadlock = (
        ridge_gate_pass
        & pair_active
        & (np.abs(pair_vote) < 1.0e-12)
        & ~ridge_active
        & (ridge_strength > 0.0)
        & (ridge_strength <= strategy.RIDGE_THETA)
        & (ridge_latent_sign != 0.0)
        & ~crash
    )
    result[deadlock] = (
        strategy.DEADLOCK_RIDGE_OVERLAY_FRACTION
        * limits[deadlock]
        * ridge_latent_sign[deadlock]
    )
    result[ALGO] = 0.0
    desired_ridge = ridge_raw.copy()
    desired_ridge[deadlock] = (
        strategy.DEADLOCK_RIDGE_OVERLAY_FRACTION
        * limits[deadlock]
        * ridge_latent_sign[deadlock]
    )
    desired_ridge[crash] = 0.0
    return result, desired_ridge


def candidate_weight(
    priority: float,
    agent: strategy.PairAgent,
) -> float:
    boost = (
        strategy.PROTECTED_VOTE_BOOST
        if strategy._pair_name(agent.spec)
        in strategy.PROTECTED_PAIR_NAMES
        else 1.0
    )
    return boost * max(float(priority), 1.0e-12)


def greedy_hedged_dollars(
    candidates: list[
        tuple[float, strategy.PairAgent, float, float]
    ],
    crash: np.ndarray,
) -> tuple[np.ndarray, dict[int, float]]:
    """Fill complete pair vectors proportionally under instrument limits."""
    limits = dollar_limits()
    result = np.zeros(strategy.N_INST)
    fills: dict[int, float] = {}
    ordered = sorted(
        candidates,
        key=lambda item: candidate_weight(item[0], item[1]),
        reverse=True,
    )
    for _, agent, pos_y, pos_x in ordered:
        y = int(agent.spec.y)
        x = int(agent.spec.x)
        if crash[y] or crash[x] or y == ALGO or x == ALGO:
            fills[id(agent)] = 0.0
            continue
        fill = 1.0
        for instrument, delta in ((y, pos_y), (x, pos_x)):
            if delta > 0.0:
                available = limits[instrument] - result[instrument]
                fill = min(fill, max(0.0, available / delta))
            elif delta < 0.0:
                available = -limits[instrument] - result[instrument]
                fill = min(fill, max(0.0, available / delta))
        fill = float(np.clip(fill, 0.0, 1.0))
        result[y] += fill * pos_y
        result[x] += fill * pos_x
        fills[id(agent)] = fill
    result[crash] = 0.0
    result[ALGO] = 0.0
    return result, fills


def lp_hedged_dollars(
    candidates: list[
        tuple[float, strategy.PairAgent, float, float]
    ],
    crash: np.ndarray,
    priority_weighted: bool,
) -> tuple[np.ndarray, dict[int, float]]:
    """Maximize complete pair-vector fills under all instrument limits."""
    if not candidates:
        return np.zeros(strategy.N_INST), {}
    vectors = np.zeros((strategy.N_INST, len(candidates)))
    weights = np.ones(len(candidates))
    bounds: list[tuple[float, float]] = []
    for column, (priority, agent, pos_y, pos_x) in enumerate(candidates):
        y = int(agent.spec.y)
        x = int(agent.spec.x)
        vectors[y, column] = pos_y
        vectors[x, column] = pos_x
        if priority_weighted:
            weights[column] = candidate_weight(priority, agent)
        blocked = crash[y] or crash[x] or y == ALGO or x == ALGO
        bounds.append((0.0, 0.0 if blocked else 1.0))
    weights /= max(float(weights.max()), 1.0e-12)
    limits = dollar_limits()
    result = linprog(
        c=-weights,
        A_ub=np.vstack((vectors, -vectors)),
        b_ub=np.concatenate((limits, limits)),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Hedge allocation LP failed: {result.message}")
    fills = np.clip(result.x, 0.0, 1.0)
    dollars = vectors @ fills
    dollars[ALGO] = 0.0
    return (
        dollars,
        {
            id(candidate[1]): float(fill)
            for candidate, fill in zip(candidates, fills, strict=True)
        },
    )


def aggregate_hedged_dollars(
    candidates: list[
        tuple[float, strategy.PairAgent, float, float]
    ],
    crash: np.ndarray,
    global_scale: bool,
) -> np.ndarray:
    """Sum complete pair vectors, then globally scale or instrument-clip."""
    raw = np.zeros(strategy.N_INST)
    for _, agent, pos_y, pos_x in candidates:
        boost = (
            strategy.PROTECTED_VOTE_BOOST
            if strategy._pair_name(agent.spec)
            in strategy.PROTECTED_PAIR_NAMES
            else 1.0
        )
        raw[int(agent.spec.y)] += boost * pos_y
        raw[int(agent.spec.x)] += boost * pos_x
    raw[crash] = 0.0
    raw[ALGO] = 0.0
    limits = dollar_limits()
    if global_scale:
        use = np.abs(raw) > 1.0e-12
        scale = (
            min(1.0, float(np.min(limits[use] / np.abs(raw[use]))))
            if use.any()
            else 1.0
        )
        return raw * scale
    return np.clip(raw, -limits, limits)


def add_ridge_in_capacity(
    pair_dollars: np.ndarray,
    desired_ridge: np.ndarray,
    flat_only: bool,
) -> np.ndarray:
    limits = dollar_limits()
    ridge = desired_ridge.copy()
    if flat_only:
        ridge[np.abs(pair_dollars) > 1.0e-12] = 0.0
    lower = -limits - pair_dollars
    upper = limits - pair_dollars
    ridge = np.clip(ridge, lower, upper)
    result = pair_dollars + ridge
    result[ALGO] = 0.0
    return result


def hedge_error(
    beta: float,
    y_dollars: float,
    x_dollars: float,
) -> float:
    numerator = abs(beta * y_dollars + x_dollars)
    denominator = abs(beta * y_dollars) + abs(x_dollars)
    return numerator / denominator if denominator > 1.0e-12 else 1.0


def execution_status(
    intended_y: float,
    intended_x: float,
    actual_y: float,
    actual_x: float,
) -> str:
    if abs(actual_y) <= 1.0e-12 or abs(actual_x) <= 1.0e-12:
        return "missing_leg"
    if (
        np.sign(actual_y) != np.sign(intended_y)
        or np.sign(actual_x) != np.sign(intended_x)
    ):
        return "flipped_leg"
    return "both_signs_match"


def build_all_targets(
    prices: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
]:
    targets = {
        name: np.zeros(
            (prices.shape[1] + 1, strategy.N_INST),
            dtype=int,
        )
        for name in VARIANTS
    }
    audit_rows: list[dict[str, float | int | str | bool]] = []
    daily_rows: list[dict[str, float | int]] = []
    book = strategy.AtlasBook()

    for history_size in range(1, prices.shape[1]):
        known = prices[:, :history_size]
        if history_size < strategy.WARMUP + 10:
            continue
        candidates = book.candidates(known)
        pair_vote, pair_active, contributors = (
            strategy._pair_conflict_consensus(candidates)
        )
        crash = crash_mask(known)
        returns = np.diff(np.log(known), axis=1)
        (
            ridge_raw,
            ridge_strength,
            ridge_latent_sign,
            ridge_gate_pass,
        ) = strategy._ridge_forecast_details(returns)

        pair_current = current_pair_dollars(
            pair_vote,
            pair_active,
            crash,
        )
        live_current, desired_ridge = current_live_dollars(
            pair_vote,
            pair_active,
            ridge_raw,
            ridge_strength,
            ridge_latent_sign,
            ridge_gate_pass,
            crash,
        )
        greedy, fills = greedy_hedged_dollars(candidates, crash)
        lp_equal, lp_equal_fills = lp_hedged_dollars(
            candidates,
            crash,
            priority_weighted=False,
        )
        lp_priority, lp_priority_fills = lp_hedged_dollars(
            candidates,
            crash,
            priority_weighted=True,
        )
        global_scaled = aggregate_hedged_dollars(
            candidates,
            crash,
            global_scale=True,
        )
        clipped_vector = aggregate_hedged_dollars(
            candidates,
            crash,
            global_scale=False,
        )
        values = {
            "current_pair_vote": pair_current,
            "greedy_hedged_pair": greedy,
            "lp_equal_hedged_pair": lp_equal,
            "lp_priority_hedged_pair": lp_priority,
            "global_scaled_hedged_pair": global_scaled,
            "clipped_vector_pair": clipped_vector,
            "current_live": live_current,
            "greedy_hedged_plus_ridge": add_ridge_in_capacity(
                greedy,
                desired_ridge,
                flat_only=False,
            ),
            "greedy_hedged_plus_ridge_flat_only": add_ridge_in_capacity(
                greedy,
                desired_ridge,
                flat_only=True,
            ),
            "lp_equal_hedged_plus_ridge": add_ridge_in_capacity(
                lp_equal,
                desired_ridge,
                flat_only=False,
            ),
            "lp_equal_hedged_plus_ridge_flat_only": add_ridge_in_capacity(
                lp_equal,
                desired_ridge,
                flat_only=True,
            ),
            "lp_priority_hedged_plus_ridge": add_ridge_in_capacity(
                lp_priority,
                desired_ridge,
                flat_only=False,
            ),
            "lp_priority_hedged_plus_ridge_flat_only": add_ridge_in_capacity(
                lp_priority,
                desired_ridge,
                flat_only=True,
            ),
        }
        for name, dollars in values.items():
            targets[name][history_size] = to_shares(
                dollars,
                known[:, -1],
            )

        fill_values = np.asarray(list(fills.values()), dtype=float)
        daily_rows.append(
            {
                "history_size": history_size,
                "pnl_day": history_size + 1,
                "active_pairs": len(candidates),
                "mean_greedy_fill": (
                    float(fill_values.mean())
                    if len(fill_values)
                    else 0.0
                ),
                "fully_filled_pairs": int(
                    np.sum(fill_values >= 1.0 - 1.0e-9)
                ),
                "partially_filled_pairs": int(
                    np.sum(
                        (fill_values > 1.0e-9)
                        & (fill_values < 1.0 - 1.0e-9)
                    )
                ),
                "unfilled_pairs": int(
                    np.sum(fill_values <= 1.0e-9)
                ),
                "lp_equal_mean_fill": (
                    float(np.mean(list(lp_equal_fills.values())))
                    if lp_equal_fills
                    else 0.0
                ),
                "lp_priority_mean_fill": (
                    float(np.mean(list(lp_priority_fills.values())))
                    if lp_priority_fills
                    else 0.0
                ),
            }
        )

        for priority, agent, pos_y, pos_x in candidates:
            y = int(agent.spec.y)
            x = int(agent.spec.x)
            beta = float(agent.filt.beta)
            isolated = contributors[y] == 1 and contributors[x] == 1
            row: dict[str, float | int | str | bool] = {
                "history_size": history_size,
                "pnl_day": history_size + 1,
                "pair": strategy._pair_name(agent.spec),
                "y": y,
                "x": x,
                "beta": beta,
                "priority": float(priority),
                "intended_y": pos_y,
                "intended_x": pos_x,
                "intended_hedge_error": hedge_error(
                    beta,
                    pos_y,
                    pos_x,
                ),
                "isolated": isolated,
                "greedy_fill": fills.get(id(agent), 0.0),
            }
            for label, dollars in (
                ("pair_vote", pair_current),
                ("live", live_current),
                ("greedy_net", greedy),
            ):
                actual_y = float(dollars[y])
                actual_x = float(dollars[x])
                row[f"{label}_status"] = execution_status(
                    pos_y,
                    pos_x,
                    actual_y,
                    actual_x,
                )
                row[f"{label}_hedge_error"] = hedge_error(
                    beta,
                    actual_y,
                    actual_x,
                )
            audit_rows.append(row)
    return (
        targets,
        pd.DataFrame(audit_rows),
        pd.DataFrame(daily_rows),
    )


def evaluate_variants(
    prices: np.ndarray,
    targets: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for variant, variant_targets in targets.items():
        result = evaluator.evaluate(prices, variant_targets, WINDOWS)
        for window, metrics in result.items():
            rows.append(
                {
                    "variant": variant,
                    "window": window,
                    **metrics,
                }
            )
    frame = pd.DataFrame(rows)
    baselines = (
        frame[frame["variant"] == "current_live"]
        .set_index("window")["score"]
    )
    frame["live_score_delta"] = frame.apply(
        lambda row: row["score"] - baselines[row["window"]],
        axis=1,
    )
    pair_baselines = (
        frame[frame["variant"] == "current_pair_vote"]
        .set_index("window")["score"]
    )
    frame["pair_score_delta"] = frame.apply(
        lambda row: row["score"] - pair_baselines[row["window"]],
        axis=1,
    )
    return frame


def summarize_audit(audit: pd.DataFrame) -> pd.DataFrame:
    periods = (
        ("TRAIN", 251, 500),
        ("VALIDATION", 501, 750),
        ("EXPOSED_YEAR4", 751, 1000),
    )
    rows: list[dict[str, float | int | str]] = []
    for period, start, end in periods:
        sample = audit[audit["pnl_day"].between(start, end)]
        # Only current execution layers belong in this comparison.  A net
        # portfolio assembled from several exact pair components cannot be
        # compared back with every component's beta simultaneously; doing so
        # would incorrectly label ordinary broker-level netting as hedge loss.
        for layer in ("pair_vote", "live"):
            status = sample[f"{layer}_status"]
            errors = sample[f"{layer}_hedge_error"]
            isolated = sample["isolated"]
            rows.append(
                {
                    "period": period,
                    "layer": layer,
                    "candidate_events": len(sample),
                    "both_signs_match": float(
                        (status == "both_signs_match").mean()
                    ),
                    "missing_leg": float(
                        (status == "missing_leg").mean()
                    ),
                    "flipped_leg": float(
                        (status == "flipped_leg").mean()
                    ),
                    "median_hedge_error": float(errors.median()),
                    "mean_hedge_error": float(errors.mean()),
                    "isolated_events": int(isolated.sum()),
                    "isolated_median_hedge_error": float(
                        errors[isolated].median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_plot(
    performance: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    windows = ("TRAIN", "VALIDATION", "EXPOSED_YEAR4")
    selected = performance[
        performance["window"].isin(windows)
        & performance["variant"].isin(
            (
                "current_pair_vote",
                "greedy_hedged_pair",
                "current_live",
                "greedy_hedged_plus_ridge",
            )
        )
    ]
    pivot = selected.pivot(
        index="window",
        columns="variant",
        values="score",
    ).reindex(windows)
    pivot.plot.bar(ax=axes[0])
    axes[0].set_title("Hedge-preserving construction performance")
    axes[0].set_ylabel("official score")
    axes[0].tick_params(axis="x", rotation=0)

    validation = summary[summary["period"] == "VALIDATION"].set_index(
        "layer"
    )
    labels = ("both_signs_match", "missing_leg", "flipped_leg")
    validation[list(labels)].plot.bar(stacked=True, ax=axes[1])
    axes[1].set_title("Validation candidate execution status")
    axes[1].set_ylabel("share of active pair-events")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(
        ["both signs match", "missing leg", "flipped leg"],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pair_hedge_audit.png", dpi=160)
    plt.close(fig)


def write_report(
    performance: pd.DataFrame,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    primary = performance[
        performance["window"].isin(
            ("TRAIN_A", "TRAIN_B", "TRAIN", "VALIDATION_A",
             "VALIDATION_B", "VALIDATION", "EXPOSED_YEAR4")
        )
    ]
    pivot = primary.pivot(
        index="variant",
        columns="window",
        values="score",
    )
    daily_development = daily[
        daily["pnl_day"].between(251, 750)
    ]

    lines = [
        "# Pair hedge execution audit",
        "",
        "Each PairAgent's intended vector is exactly dynamic-beta hedged. "
        "The audit tests whether portfolio construction preserves it.",
        "",
        "## Execution integrity",
        "",
        "| Period | Layer | Pair-events | Both signs | Missing leg | "
        "Flipped leg | Median hedge error | Isolated hedge error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.period} | {row.layer} | "
            f"{int(row.candidate_events):,} | "
            f"{row.both_signs_match:.1%} | {row.missing_leg:.1%} | "
            f"{row.flipped_leg:.1%} | "
            f"{row.median_hedge_error:.1%} | "
            f"{row.isolated_median_hedge_error:.1%} |"
        )

    lines.extend(
        [
            "",
            "The normalized hedge error is "
            "`abs(beta*d_y + d_x) / (abs(beta*d_y) + abs(d_x))`; "
            "zero is an exact pair hedge and one is completely unhedged.",
            "",
            "## Portfolio scores",
            "",
            "| Variant | Train A | Train B | Train | Validation A | "
            "Validation B | Validation | Exposed Year 4 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant, row in pivot.iterrows():
        lines.append(
            f"| {variant} | {row.TRAIN_A:+.2f} | {row.TRAIN_B:+.2f} | "
            f"{row.TRAIN:+.2f} | {row.VALIDATION_A:+.2f} | "
            f"{row.VALIDATION_B:+.2f} | {row.VALIDATION:+.2f} | "
            f"{row.EXPOSED_YEAR4:+.2f} |"
        )

    active_pairs = daily_development["active_pairs"].sum()
    fully_filled = daily_development["fully_filled_pairs"].sum()
    partially_filled = daily_development[
        "partially_filled_pairs"
    ].sum()
    unfilled = daily_development["unfilled_pairs"].sum()
    lines.extend(
        [
            "",
            "## Greedy hedge-preserving capacity",
            "",
            f"- Development active pair-events: **{int(active_pairs):,}**.",
            f"- Fully filled: **{fully_filled / active_pairs:.1%}**.",
            f"- Partially filled: **{partially_filled / active_pairs:.1%}**.",
            f"- Unfilled: **{unfilled / active_pairs:.1%}**.",
            "",
            "A production correction must improve aggregate train and validation "
            "without a materially negative chronological half. Exposed Year 4 "
            "is descriptive and is not used for selection.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+").to_numpy().T
    targets, audit, daily = build_all_targets(prices)
    performance = evaluate_variants(prices, targets)
    summary = summarize_audit(audit)
    audit.to_csv(OUTPUT_DIR / "pair_event_audit.csv", index=False)
    daily.to_csv(OUTPUT_DIR / "daily_capacity.csv", index=False)
    performance.to_csv(
        OUTPUT_DIR / "portfolio_performance.csv",
        index=False,
    )
    summary.to_csv(OUTPUT_DIR / "execution_summary.csv", index=False)
    make_plot(performance, summary)
    write_report(performance, summary, daily)
    print(summary.to_string(index=False))
    print("\nPortfolio performance")
    print(
        performance[
            performance["window"].isin(
                ("TRAIN", "VALIDATION", "EXPOSED_YEAR4")
            )
        ][
            ["variant", "window", "score", "mean_pnl", "total_pnl"]
        ].to_string(index=False)
    )
    print(f"\nWrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

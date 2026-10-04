"""Select a simple conflict policy for the new standalone ridge parameters.

Selection uses days 251-500 and confirmation uses 501-750.  Candidate rules
are categorical and parameter-free:

* current normalized voting;
* pair priority whenever a nonzero pair vote opposes ridge;
* pair priority with quarter-sized deadlocks;
* pair priority with deadlocks left flat; and
* leave every true pair/ridge conflict flat.

The selected rule is frozen before the already-exposed 751-1000 window is
reported.  No production source is mutated by this research script.
"""

from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import algo12_equal_sleeve_research as evaluator
from research import pair_hedge_execution_audit as hedge_audit


PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "ridge_conflict_policy"
NEW_LAMBDA = 1.52
NEW_THETA = 0.31
OLD_LAMBDA = 1.0
OLD_THETA = 0.75
DEVELOPMENT_WINDOWS = (
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("TRAIN", 251, 500),
    ("VALIDATION_A", 501, 625),
    ("VALIDATION_B", 626, 750),
    ("VALIDATION", 501, 750),
)
EXPOSED_WINDOWS = (
    ("EXPOSED_A", 751, 875),
    ("EXPOSED_B", 876, 1000),
    ("EXPOSED", 751, 1000),
)
BLOCK_WINDOWS = tuple(
    (
        f"BLOCK_{start}_{start + 49}",
        start,
        start + 49,
    )
    for start in range(251, 1000, 50)
)
CANDIDATES = (
    "new_current_vote",
    "new_pair_priority_full_deadlock",
    "new_pair_priority_quarter_deadlock",
    "new_pair_priority_no_deadlock",
    "new_conflict_flat_full_deadlock",
    "new_conflict_old_model",
    "new_conflict_dual_sign_confirm",
    "new_conflict_dual_min_strength",
    "new_conflict_dual_average_vote",
)
BASELINES = ("pair_only", "old_current_vote")


def score(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe**2 / (sharpe**2 + 1.0)


def ridge_details(
    returns: np.ndarray,
    ridge_lambda: float,
    ridge_theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    original_lambda = strategy.RIDGE_LAM
    original_theta = strategy.RIDGE_THETA
    strategy.RIDGE_LAM = ridge_lambda
    strategy.RIDGE_THETA = ridge_theta
    try:
        return strategy._ridge_forecast_details(returns)
    finally:
        strategy.RIDGE_LAM = original_lambda
        strategy.RIDGE_THETA = original_theta


def current_live_for_parameters(
    pair_vote: np.ndarray,
    pair_active: np.ndarray,
    details: tuple[np.ndarray, np.ndarray, np.ndarray, bool],
    crash: np.ndarray,
    ridge_theta: float,
) -> np.ndarray:
    original_theta = strategy.RIDGE_THETA
    strategy.RIDGE_THETA = ridge_theta
    try:
        result, _ = hedge_audit.current_live_dollars(
            pair_vote,
            pair_active,
            *details,
            crash,
        )
        return result
    finally:
        strategy.RIDGE_THETA = original_theta


def pair_priority_dollars(
    pair_dollars: np.ndarray,
    pair_vote: np.ndarray,
    pair_active: np.ndarray,
    details: tuple[np.ndarray, np.ndarray, np.ndarray, bool],
    crash: np.ndarray,
    deadlock_fraction: float,
    flatten_conflicts: bool,
) -> np.ndarray:
    ridge_raw, ridge_strength, ridge_latent_sign, ridge_gate = details
    result = pair_dollars.copy()
    ridge_active = ridge_raw != 0.0
    pair_direction = np.sign(pair_vote)
    ridge_direction = np.sign(ridge_raw)
    pair_deadlock = pair_active & (np.abs(pair_vote) <= 1.0e-12)
    pair_flat = ~pair_active
    conflicts = (
        pair_active
        & ~pair_deadlock
        & ridge_active
        & (pair_direction != ridge_direction)
    )

    result[pair_flat & ridge_active] = ridge_raw[
        pair_flat & ridge_active
    ]
    active_deadlock = pair_deadlock & ridge_active
    result[active_deadlock] = (
        deadlock_fraction * ridge_raw[active_deadlock]
    )
    subthreshold_deadlock = (
        pair_deadlock
        & ~ridge_active
        & ridge_gate
        & (ridge_strength > 0.0)
        & (ridge_strength <= NEW_THETA)
        & (ridge_latent_sign != 0.0)
    )
    result[subthreshold_deadlock] = (
        deadlock_fraction
        * strategy.BANG_BANG_LIMITS[subthreshold_deadlock]
        * ridge_latent_sign[subthreshold_deadlock]
    )
    if flatten_conflicts:
        result[conflicts] = 0.0
    result[crash] = 0.0
    result[strategy.I["ALGO"]] = 0.0
    return result


def dual_model_conflict_dollars(
    pair_dollars: np.ndarray,
    pair_vote: np.ndarray,
    pair_active: np.ndarray,
    new_details: tuple[np.ndarray, np.ndarray, np.ndarray, bool],
    old_details: tuple[np.ndarray, np.ndarray, np.ndarray, bool],
    crash: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Use new ridge off-conflict and a two-model rule on true conflicts."""
    result = pair_priority_dollars(
        pair_dollars,
        pair_vote,
        pair_active,
        new_details,
        crash,
        deadlock_fraction=1.0,
        flatten_conflicts=False,
    )
    new_raw, new_strength, _, _ = new_details
    old_raw, old_strength, _, _ = old_details
    new_sign = np.sign(new_raw)
    old_sign = np.sign(old_raw)
    pair_sign = np.sign(pair_vote)
    new_active = new_raw != 0.0
    old_active = old_raw != 0.0
    pair_deadlock = pair_active & (np.abs(pair_vote) <= 1.0e-12)
    conflict = (
        pair_active
        & ~pair_deadlock
        & new_active
        & (new_sign != pair_sign)
        & ~crash
    )
    final_sign = pair_sign.copy()

    if mode == "old_model":
        old_vote = (
            pair_vote
            + strategy.RIDGE_CONFLICT_WEIGHT
            * old_sign
            * old_strength
        )
        use_old = conflict & old_active
        final_sign[use_old] = np.sign(old_vote[use_old])
    elif mode == "dual_sign_confirm":
        new_vote = (
            pair_vote
            + strategy.RIDGE_CONFLICT_WEIGHT
            * new_sign
            * new_strength
        )
        confirmed = (
            conflict
            & old_active
            & (old_sign == new_sign)
            & (np.sign(new_vote) == new_sign)
        )
        final_sign[confirmed] = new_sign[confirmed]
    elif mode == "dual_min_strength":
        confirmed = conflict & old_active & (old_sign == new_sign)
        conservative_vote = (
            pair_vote
            + strategy.RIDGE_CONFLICT_WEIGHT
            * new_sign
            * np.minimum(new_strength, old_strength)
        )
        ridge_wins = (
            confirmed
            & (np.sign(conservative_vote) == new_sign)
        )
        final_sign[ridge_wins] = new_sign[ridge_wins]
    elif mode == "dual_average_vote":
        average_ridge_vote = (
            0.5
            * strategy.RIDGE_CONFLICT_WEIGHT
            * (
                new_sign * new_strength
                + old_sign * old_strength
            )
        )
        average_vote = pair_vote + average_ridge_vote
        use = conflict & (np.sign(average_vote) != 0.0)
        final_sign[use] = np.sign(average_vote[use])
    else:
        raise ValueError(f"Unknown dual conflict mode: {mode}")

    result[conflict] = (
        strategy.BANG_BANG_LIMITS[conflict]
        * final_sign[conflict]
    )
    result[crash] = 0.0
    result[strategy.I["ALGO"]] = 0.0
    return result


def build_targets(
    prices: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    names = BASELINES + CANDIDATES
    targets = {
        name: np.zeros(
            (prices.shape[1] + 1, strategy.N_INST),
            dtype=int,
        )
        for name in names
    }
    rows: list[dict[str, float | int]] = []
    book = strategy.AtlasBook()
    for history_size in range(1, prices.shape[1]):
        known = prices[:, :history_size]
        if history_size < strategy.WARMUP + 10:
            continue
        candidates = book.candidates(known)
        pair_vote, pair_active, _ = (
            strategy._pair_conflict_consensus(candidates)
        )
        crash = hedge_audit.crash_mask(known)
        pair_dollars = hedge_audit.current_pair_dollars(
            pair_vote,
            pair_active,
            crash,
        )
        returns = np.diff(np.log(known), axis=1)
        old_details = ridge_details(
            returns,
            OLD_LAMBDA,
            OLD_THETA,
        )
        new_details = ridge_details(
            returns,
            NEW_LAMBDA,
            NEW_THETA,
        )
        values = {
            "pair_only": pair_dollars,
            "old_current_vote": current_live_for_parameters(
                pair_vote,
                pair_active,
                old_details,
                crash,
                OLD_THETA,
            ),
            "new_current_vote": current_live_for_parameters(
                pair_vote,
                pair_active,
                new_details,
                crash,
                NEW_THETA,
            ),
            "new_pair_priority_full_deadlock": pair_priority_dollars(
                pair_dollars,
                pair_vote,
                pair_active,
                new_details,
                crash,
                deadlock_fraction=1.0,
                flatten_conflicts=False,
            ),
            "new_pair_priority_quarter_deadlock": pair_priority_dollars(
                pair_dollars,
                pair_vote,
                pair_active,
                new_details,
                crash,
                deadlock_fraction=0.25,
                flatten_conflicts=False,
            ),
            "new_pair_priority_no_deadlock": pair_priority_dollars(
                pair_dollars,
                pair_vote,
                pair_active,
                new_details,
                crash,
                deadlock_fraction=0.0,
                flatten_conflicts=False,
            ),
            "new_conflict_flat_full_deadlock": pair_priority_dollars(
                pair_dollars,
                pair_vote,
                pair_active,
                new_details,
                crash,
                deadlock_fraction=1.0,
                flatten_conflicts=True,
            ),
            "new_conflict_old_model": dual_model_conflict_dollars(
                pair_dollars,
                pair_vote,
                pair_active,
                new_details,
                old_details,
                crash,
                mode="old_model",
            ),
            "new_conflict_dual_sign_confirm": dual_model_conflict_dollars(
                pair_dollars,
                pair_vote,
                pair_active,
                new_details,
                old_details,
                crash,
                mode="dual_sign_confirm",
            ),
            "new_conflict_dual_min_strength": dual_model_conflict_dollars(
                pair_dollars,
                pair_vote,
                pair_active,
                new_details,
                old_details,
                crash,
                mode="dual_min_strength",
            ),
            "new_conflict_dual_average_vote": dual_model_conflict_dollars(
                pair_dollars,
                pair_vote,
                pair_active,
                new_details,
                old_details,
                crash,
                mode="dual_average_vote",
            ),
        }
        for name, dollars in values.items():
            targets[name][history_size] = hedge_audit.to_shares(
                dollars,
                known[:, -1],
            )

        ridge_raw, ridge_strength, _, _ = new_details
        ridge_active = ridge_raw != 0.0
        pair_deadlock = pair_active & (np.abs(pair_vote) <= 1.0e-12)
        pair_direction = np.sign(pair_vote)
        conflict = (
            pair_active
            & ~pair_deadlock
            & ridge_active
            & (pair_direction != np.sign(ridge_raw))
        )
        rows.append(
            {
                "history_size": history_size,
                "pnl_day": history_size + 1,
                "pair_active_assets": int(pair_active.sum()),
                "ridge_active_assets": int(ridge_active.sum()),
                "pair_deadlocks": int(pair_deadlock.sum()),
                "true_conflicts": int(conflict.sum()),
                "pair_flat_ridge_assets": int(
                    (ridge_active & ~pair_active).sum()
                ),
                "mean_ridge_strength": (
                    float(ridge_strength[ridge_active].mean())
                    if ridge_active.any()
                    else 0.0
                ),
            }
        )
    return targets, pd.DataFrame(rows)


def evaluate(
    prices: np.ndarray,
    targets: dict[str, np.ndarray],
    windows: tuple[tuple[str, int, int], ...],
    variants: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    use_variants = variants or tuple(targets)
    rows: list[dict[str, float | str]] = []
    for variant in use_variants:
        result = evaluator.evaluate(
            prices,
            targets[variant],
            windows,
        )
        for window, metrics in result.items():
            rows.append(
                {
                    "variant": variant,
                    "window": window,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def select_policy(development: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    baseline = (
        development[
            development["variant"] == "old_current_vote"
        ]
        .set_index("window")["score"]
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for candidate in CANDIDATES:
        values = (
            development[development["variant"] == candidate]
            .set_index("window")["score"]
        )
        deltas = values - baseline
        four = deltas[
            ["TRAIN_A", "TRAIN_B", "VALIDATION_A", "VALIDATION_B"]
        ]
        rows.append(
            {
                "variant": candidate,
                "train_delta": float(deltas["TRAIN"]),
                "validation_delta": float(deltas["VALIDATION"]),
                "train_a_delta": float(deltas["TRAIN_A"]),
                "train_b_delta": float(deltas["TRAIN_B"]),
                "validation_a_delta": float(deltas["VALIDATION_A"]),
                "validation_b_delta": float(deltas["VALIDATION_B"]),
                "positive_halves": int((four > 0.0).sum()),
                "worst_half_delta": float(four.min()),
                "passes": bool(
                    deltas["TRAIN"] > 0.0
                    and deltas["VALIDATION"] > 0.0
                    and (four >= 0.0).all()
                ),
            }
        )
    ranking = pd.DataFrame(rows)
    eligible = ranking[ranking["passes"]]
    if eligible.empty:
        return "new_current_vote", ranking
    selected = (
        eligible.sort_values(
            ["worst_half_delta", "validation_delta", "train_delta"],
            ascending=False,
        )
        .iloc[0]["variant"]
    )
    return str(selected), ranking


def make_plot(
    development: pd.DataFrame,
    exposed: pd.DataFrame,
    selected: str,
) -> None:
    variants = ("old_current_vote", "new_current_vote", selected)
    combined = pd.concat((development, exposed), ignore_index=True)
    primary = combined[
        combined["window"].isin(("TRAIN", "VALIDATION", "EXPOSED"))
        & combined["variant"].isin(variants)
    ]
    pivot = primary.pivot(
        index="window",
        columns="variant",
        values="score",
    ).reindex(("TRAIN", "VALIDATION", "EXPOSED"))
    ax = pivot.plot.bar(figsize=(12, 5.5))
    ax.set_title("New-ridge conflict policy")
    ax.set_ylabel("official score")
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(
        OUTPUT_DIR / "ridge_conflict_policy.png",
        dpi=160,
    )
    plt.close(ax.figure)


def write_report(
    development: pd.DataFrame,
    ranking: pd.DataFrame,
    exposed: pd.DataFrame,
    blocks: pd.DataFrame,
    selected: str,
    diagnostics: pd.DataFrame,
) -> None:
    lines = [
        "# New-ridge conflict policy",
        "",
        f"Selected before exposed evaluation: **{selected}**.",
        "",
        "## Development policy deltas versus old production voting",
        "",
        "| Policy | Train | Validation | Train A | Train B | Val A | "
        "Val B | Positive halves | Worst half | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            f"| {row.variant} | {row.train_delta:+.2f} | "
            f"{row.validation_delta:+.2f} | "
            f"{row.train_a_delta:+.2f} | {row.train_b_delta:+.2f} | "
            f"{row.validation_a_delta:+.2f} | "
            f"{row.validation_b_delta:+.2f} | "
            f"{int(row.positive_halves)}/4 | "
            f"{row.worst_half_delta:+.2f} | "
            f"{'yes' if row.passes else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Frozen comparison",
            "",
            "| Variant | Train | Validation | Exposed A | Exposed B | Exposed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    combined = pd.concat((development, exposed), ignore_index=True)
    use = combined[
        combined["variant"].isin(
            ("pair_only", "old_current_vote", "new_current_vote", selected)
        )
    ]
    pivot = use.pivot(
        index="variant",
        columns="window",
        values="score",
    )
    for variant, row in pivot.iterrows():
        lines.append(
            f"| {variant} | {row.get('TRAIN', float('nan')):+.2f} | "
            f"{row.get('VALIDATION', float('nan')):+.2f} | "
            f"{row.get('EXPOSED_A', float('nan')):+.2f} | "
            f"{row.get('EXPOSED_B', float('nan')):+.2f} | "
            f"{row.get('EXPOSED', float('nan')):+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Selected 50-day blocks",
            "",
            "| Days | Score | Mean/day | Total PnL | Sharpe |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    selected_blocks = blocks[blocks["variant"] == selected]
    for _, row in selected_blocks.iterrows():
        _, start, end = row.window.split("_")
        lines.append(
            f"| {start}-{end} | {row.score:+.2f} | "
            f"${row.mean_pnl:+,.0f} | ${row.total_pnl:+,.0f} | "
            f"{row.sharpe:+.2f} |"
        )

    development_diag = diagnostics[
        diagnostics["pnl_day"].between(251, 750)
    ]
    exposed_diag = diagnostics[
        diagnostics["pnl_day"].between(751, 1000)
    ]
    lines.extend(
        [
            "",
            "## Deadlock and overlap frequency",
            "",
            f"- Development mean deadlocks/day: "
            f"**{development_diag['pair_deadlocks'].mean():.2f}**.",
            f"- Exposed mean deadlocks/day: "
            f"**{exposed_diag['pair_deadlocks'].mean():.2f}**.",
            f"- Development mean true conflicts/day: "
            f"**{development_diag['true_conflicts'].mean():.2f}**.",
            f"- Exposed mean true conflicts/day: "
            f"**{exposed_diag['true_conflicts'].mean():.2f}**.",
            "",
            "Days 751-1000 were not used to choose the policy.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+").to_numpy().T
    targets, diagnostics = build_targets(prices)
    development = evaluate(
        prices,
        targets,
        DEVELOPMENT_WINDOWS,
    )
    selected, ranking = select_policy(development)
    exposed_variants = tuple(
        dict.fromkeys(
            ("pair_only", "old_current_vote", "new_current_vote", selected)
        )
    )
    exposed = evaluate(
        prices,
        targets,
        EXPOSED_WINDOWS,
        exposed_variants,
    )
    blocks = evaluate(
        prices,
        targets,
        BLOCK_WINDOWS,
        (selected,),
    )
    development.to_csv(
        OUTPUT_DIR / "development_performance.csv",
        index=False,
    )
    ranking.to_csv(OUTPUT_DIR / "policy_ranking.csv", index=False)
    exposed.to_csv(
        OUTPUT_DIR / "exposed_performance.csv",
        index=False,
    )
    blocks.to_csv(OUTPUT_DIR / "block_performance.csv", index=False)
    diagnostics.to_csv(
        OUTPUT_DIR / "daily_diagnostics.csv",
        index=False,
    )
    (OUTPUT_DIR / "selected_policy.txt").write_text(
        selected + "\n",
        encoding="utf-8",
    )
    make_plot(development, exposed, selected)
    write_report(
        development,
        ranking,
        exposed,
        blocks,
        selected,
        diagnostics,
    )
    print(ranking.to_string(index=False))
    print(f"\nSelected: {selected}")
    print("\nExposed")
    print(exposed.to_string(index=False))
    print(f"\nWrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

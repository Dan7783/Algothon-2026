"""Audit the current four_sigmas execution layer without changing production.

The audit separates signal formation from the mechanics that turn proposals
into positions:

* asset-level pair/ridge arbitration and pair hedge integrity;
* bang-bang saturation, share rounding, turnover, and commissions;
* below-threshold ridge fills on exact pair deadlocks;
* the ALGO-residual overlay and its capacity-aware beta hedge;
* crash-leg flattening and ridge-confirmation dependence.

Days 751-1000 are descriptive only.  No result from this script is used to
select or alter a production rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import algo12_equal_sleeve_research as evaluator


PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "current_execution_audit"
WINDOWS = (
    ("TRAIN", 251, 500),
    ("VALIDATION", 501, 750),
    ("EXPOSED", 751, 1000),
)


def markdown_table(frame: pd.DataFrame, digits: int = 2) -> str:
    columns = [str(column) for column in frame.columns]

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def dollars_to_shares(
    dollars: np.ndarray,
    prices: np.ndarray,
) -> np.ndarray:
    safe = np.maximum(prices, strategy._EPS_DENOM)
    limits = np.floor(strategy.BANG_BANG_LIMITS / safe).astype(int)
    return np.clip(
        np.round(dollars / safe),
        -limits,
        limits,
    ).astype(int)


def apply_algo_overlay(
    core: np.ndarray,
    sleeves: dict[str, object],
) -> np.ndarray:
    if (
        not strategy.USE_ALGO_RESIDUAL_BASKET
        or not bool(sleeves["algo_beta_ready"])
    ):
        return core.copy()
    final, _, _, _ = strategy._algo_beta_neutral_overlay(
        core,
        np.asarray(sleeves["algo_residual_requested"], dtype=float),
        np.asarray(sleeves["algo_return_betas"], dtype=float),
        np.asarray(sleeves["crash_mask"], dtype=bool),
    )
    return final


def collect(
    prices: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
]:
    targets: dict[str, np.ndarray] = {
        "production": np.zeros(
            (prices.shape[1] + 1, strategy.N_INST),
            dtype=float,
        ),
        "core_without_algo_overlay": np.zeros(
            (prices.shape[1] + 1, strategy.N_INST),
            dtype=float,
        ),
        "no_subthreshold_deadlock_fill": np.zeros(
            (prices.shape[1] + 1, strategy.N_INST),
            dtype=float,
        ),
        "fractional_share_upper_bound": np.zeros(
            (prices.shape[1] + 1, strategy.N_INST),
            dtype=float,
        ),
    }
    daily_rows: list[dict[str, float | int | bool]] = []
    pair_rows: list[dict[str, float | int | str | bool]] = []

    strategy.reset()
    for history_size in range(1, prices.shape[1]):
        if history_size < strategy.WARMUP + 10:
            continue
        known = prices[:, :history_size]
        sleeves = strategy.sleeve_breakdown(known)
        if sleeves is None:
            continue

        current_prices = known[:, -1]
        requested = np.asarray(sleeves["dollars"], dtype=float)
        core = np.asarray(sleeves["core_dollars"], dtype=float)
        production_shares = strategy.getMyPosition(known).astype(float)
        actual_dollars = production_shares * current_prices
        core_shares = dollars_to_shares(core, current_prices).astype(float)

        no_subthreshold_core = core.copy()
        subthreshold = np.asarray(
            sleeves["deadlock_ridge_overlay_mask"],
            dtype=bool,
        )
        no_subthreshold_core[subthreshold] = 0.0
        no_subthreshold = apply_algo_overlay(
            no_subthreshold_core,
            sleeves,
        )

        targets["production"][history_size] = production_shares
        targets["core_without_algo_overlay"][history_size] = core_shares
        targets["no_subthreshold_deadlock_fill"][
            history_size
        ] = dollars_to_shares(no_subthreshold, current_prices)
        targets["fractional_share_upper_bound"][
            history_size
        ] = requested / np.maximum(current_prices, strategy._EPS_DENOM)

        pair_active = np.asarray(
            sleeves["pair_active_mask"],
            dtype=bool,
        )
        pair_vote = np.asarray(sleeves["pair_vote"], dtype=float)
        ridge_active = np.asarray(
            sleeves["ridge_active_mask"],
            dtype=bool,
        )
        ridge_sign = np.sign(
            np.asarray(sleeves["ridge_raw"], dtype=float)
        )
        confirm_active = np.asarray(
            sleeves["ridge_confirmation_active_mask"],
            dtype=bool,
        )
        confirm_sign = np.asarray(
            sleeves["ridge_confirmation_sign"],
            dtype=float,
        )
        conflict = np.asarray(sleeves["conflict_mask"], dtype=bool)
        confirmed = np.asarray(
            sleeves["ridge_conflict_confirmed_mask"],
            dtype=bool,
        )
        crash = np.asarray(sleeves["crash_mask"], dtype=bool)
        combined_vote = np.asarray(
            sleeves["combined_vote"],
            dtype=float,
        )
        active_decision = pair_active | ridge_active
        non_algo = np.arange(strategy.N_INST) != strategy.I["ALGO"]
        requested_active = np.abs(requested) > 1.0e-9
        saturated = (
            np.abs(actual_dollars)
            >= 0.95 * strategy.BANG_BANG_LIMITS
        ) & requested_active
        overlay = requested - core
        overlay_changes = np.abs(overlay) > 1.0e-9
        overlay_sign_flip = (
            (np.sign(core) != 0.0)
            & (np.sign(requested) != 0.0)
            & (np.sign(core) != np.sign(requested))
        )
        overlay_fill = (np.sign(core) == 0.0) & (
            np.sign(requested) != 0.0
        )
        confirmation_comparable = ridge_active & confirm_active & non_algo
        deadlock = (
            pair_active
            & (np.abs(pair_vote) < 1.0e-12)
            & non_algo
            & ~crash
        )

        daily_rows.append(
            {
                "history_size": history_size,
                "pnl_day": history_size + 1,
                "gross_dollars": float(np.sum(np.abs(actual_dollars))),
                "net_dollars": float(np.sum(actual_dollars)),
                "active_assets": int(np.sum(np.abs(actual_dollars) > 1.0e-9)),
                "saturated_assets": int(np.sum(saturated)),
                "rounding_shortfall": float(
                    np.sum(np.abs(requested - actual_dollars))
                ),
                "pair_active_assets": int(np.sum(pair_active & non_algo)),
                "ridge_active_assets": int(np.sum(ridge_active & non_algo)),
                "conflicts": int(np.sum(conflict & non_algo)),
                "ridge_confirmed_overrides": int(
                    np.sum(confirmed & non_algo)
                ),
                "raw_deadlocks": int(np.sum(deadlock)),
                "subthreshold_deadlock_fills": int(np.sum(subthreshold)),
                "near_zero_full_positions_005": int(
                    np.sum(
                        active_decision
                        & (np.abs(combined_vote) > 0.0)
                        & (np.abs(combined_vote) < 0.05)
                        & (np.abs(core) > 0.95 * strategy.BANG_BANG_LIMITS)
                        & non_algo
                    )
                ),
                "near_zero_full_positions_010": int(
                    np.sum(
                        active_decision
                        & (np.abs(combined_vote) > 0.0)
                        & (np.abs(combined_vote) < 0.10)
                        & (np.abs(core) > 0.95 * strategy.BANG_BANG_LIMITS)
                        & non_algo
                    )
                ),
                "ridge_confirmation_comparable": int(
                    np.sum(confirmation_comparable)
                ),
                "ridge_confirmation_same_sign": int(
                    np.sum(
                        confirmation_comparable
                        & (ridge_sign == confirm_sign)
                    )
                ),
                "crash_assets": int(np.sum(crash)),
                "algo_beta_ready": bool(sleeves["algo_beta_ready"]),
                "algo_requested_gross": float(
                    np.sum(
                        np.abs(
                            np.asarray(
                                sleeves["algo_residual_requested"],
                                dtype=float,
                            )
                        )
                    )
                ),
                "algo_alpha_gross": float(
                    np.sum(
                        np.abs(
                            np.asarray(
                                sleeves["algo_residual_alpha"],
                                dtype=float,
                            )
                        )
                    )
                ),
                "algo_hedge_gross": float(
                    np.sum(
                        np.abs(
                            np.asarray(
                                sleeves["algo_residual_hedge"],
                                dtype=float,
                            )
                        )
                    )
                ),
                "algo_overlay_gross": float(np.sum(np.abs(overlay))),
                "algo_delivery": float(sleeves["algo_residual_delivery"]),
                "algo_beta_error_dollars": float(
                    sleeves["algo_residual_beta_dollars"]
                ),
                "algo_overlay_changed_assets": int(
                    np.sum(overlay_changes & non_algo)
                ),
                "algo_overlay_filled_assets": int(
                    np.sum(overlay_fill & non_algo)
                ),
                "algo_overlay_sign_flips": int(
                    np.sum(overlay_sign_flip & non_algo)
                ),
            }
        )

        for detail in sleeves["pair_contributor_details"]:
            y = int(detail["y"])
            x = int(detail["x"])
            requested_y = int(np.sign(float(detail["pos_y"])))
            requested_x = int(np.sign(float(detail["pos_x"])))
            actual_y = int(np.sign(core[y]))
            actual_x = int(np.sign(core[x]))
            match_y = actual_y == requested_y
            match_x = actual_x == requested_x
            both_nonzero = actual_y != 0 and actual_x != 0
            both_flipped = (
                actual_y == -requested_y
                and actual_x == -requested_x
            )
            mixed = (
                both_nonzero
                and match_y != match_x
            )
            intended_ratio = abs(
                float(detail["pos_x"]) / float(detail["pos_y"])
            )
            actual_ratio = (
                abs(core[x] / core[y])
                if abs(core[y]) > 1.0e-9
                else np.nan
            )
            ratio_error = (
                abs(actual_ratio - intended_ratio)
                / max(intended_ratio, 1.0e-12)
                if both_nonzero
                else np.nan
            )
            pair_rows.append(
                {
                    "history_size": history_size,
                    "pnl_day": history_size + 1,
                    "pair": str(detail["pair"]),
                    "both_match": bool(match_y and match_x),
                    "one_or_both_flat": bool(
                        actual_y == 0 or actual_x == 0
                    ),
                    "both_flipped": bool(both_flipped),
                    "mixed_direction": bool(mixed),
                    "both_nonzero": bool(both_nonzero),
                    "hedge_ratio_error": float(ratio_error),
                    "one_leg_crash": bool(crash[y] != crash[x]),
                }
            )

    strategy.reset()
    return targets, pd.DataFrame(daily_rows), pd.DataFrame(pair_rows)


def performance_table(
    prices: np.ndarray,
    targets: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for variant, positions in targets.items():
        results = evaluator.evaluate(prices, positions, WINDOWS)
        for window, metrics in results.items():
            rows.append(
                {
                    "variant": variant,
                    "window": window,
                    **metrics,
                }
            )
    original_commission = evaluator.COMMISSION.copy()
    try:
        cost_scenarios = (
            ("production_zero_commission", 0.0),
            ("production_5bp_cost_sensitivity", 5.0e-4),
        )
        for variant, rate in cost_scenarios:
            evaluator.COMMISSION[:] = rate
            results = evaluator.evaluate(
                prices,
                targets["production"],
                WINDOWS,
            )
            for window, metrics in results.items():
                rows.append(
                    {
                        "variant": variant,
                        "window": window,
                        **metrics,
                    }
                )
    finally:
        evaluator.COMMISSION[:] = original_commission
    frame = pd.DataFrame(rows)
    baseline = (
        frame[frame["variant"] == "production"]
        .set_index("window")
    )
    frame["score_delta"] = frame.apply(
        lambda row: float(
            row["score"] - baseline.loc[row["window"], "score"]
        ),
        axis=1,
    )
    frame["pnl_delta"] = frame.apply(
        lambda row: float(
            row["total_pnl"]
            - baseline.loc[row["window"], "total_pnl"]
        ),
        axis=1,
    )
    return frame


def daily_summary(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    additive = (
        "conflicts",
        "ridge_confirmed_overrides",
        "raw_deadlocks",
        "subthreshold_deadlock_fills",
        "near_zero_full_positions_005",
        "near_zero_full_positions_010",
        "crash_assets",
        "algo_overlay_sign_flips",
    )
    for window, start, end in WINDOWS:
        use = daily[daily["pnl_day"].between(start, end)]
        comparable = int(use["ridge_confirmation_comparable"].sum())
        same = int(use["ridge_confirmation_same_sign"].sum())
        row: dict[str, float | int | str] = {
            "window": window,
            "avg_gross_dollars": float(use["gross_dollars"].mean()),
            "avg_net_dollars": float(use["net_dollars"].mean()),
            "avg_active_assets": float(use["active_assets"].mean()),
            "avg_saturated_assets": float(
                use["saturated_assets"].mean()
            ),
            "avg_rounding_shortfall": float(
                use["rounding_shortfall"].mean()
            ),
            "ridge_confirmation_same_sign_rate": (
                same / comparable if comparable else np.nan
            ),
            "avg_algo_overlay_gross": float(
                use["algo_overlay_gross"].mean()
            ),
            "avg_algo_delivery": float(use["algo_delivery"].mean()),
            "avg_abs_algo_beta_error": float(
                use["algo_beta_error_dollars"].abs().mean()
            ),
        }
        for column in additive:
            row[column] = int(use[column].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def pair_integrity_summary(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for window, start, end in WINDOWS:
        use = pairs[pairs["pnl_day"].between(start, end)]
        finite_error = use["hedge_ratio_error"].dropna()
        rows.append(
            {
                "window": window,
                "active_pair_days": len(use),
                "both_legs_match": int(use["both_match"].sum()),
                "both_legs_match_rate": float(use["both_match"].mean()),
                "one_or_both_legs_flat": int(
                    use["one_or_both_flat"].sum()
                ),
                "one_or_both_legs_flat_rate": float(
                    use["one_or_both_flat"].mean()
                ),
                "mixed_direction": int(use["mixed_direction"].sum()),
                "mixed_direction_rate": float(
                    use["mixed_direction"].mean()
                ),
                "both_flipped": int(use["both_flipped"].sum()),
                "both_flipped_rate": float(
                    use["both_flipped"].mean()
                ),
                "median_hedge_ratio_error": float(
                    finite_error.median()
                ),
                "one_leg_crash_events": int(
                    use["one_leg_crash"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def turnover_breakdown(
    prices: np.ndarray,
    positions: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for window, start, end in WINDOWS:
        current = np.zeros(strategy.N_INST)
        volume = {
            "entry": 0.0,
            "exit": 0.0,
            "flip": 0.0,
            "resize": 0.0,
        }
        counts = {key: 0 for key in volume}
        for t in range(start - 1, end):
            current_prices = prices[:, t - 1]
            new = positions[t].astype(float)
            prior_sign = np.sign(current)
            new_sign = np.sign(new)
            categories = {
                "entry": (prior_sign == 0.0) & (new_sign != 0.0),
                "exit": (prior_sign != 0.0) & (new_sign == 0.0),
                "flip": (prior_sign != 0.0)
                & (new_sign != 0.0)
                & (prior_sign != new_sign),
                "resize": (prior_sign != 0.0)
                & (prior_sign == new_sign)
                & (new != current),
            }
            delta_volume = current_prices * np.abs(new - current)
            for label, mask in categories.items():
                volume[label] += float(np.sum(delta_volume[mask]))
                counts[label] += int(np.sum(mask))
            current = new
        total = float(sum(volume.values()))
        rows.append(
            {
                "window": window,
                "turnover": total,
                "entry_turnover": volume["entry"],
                "exit_turnover": volume["exit"],
                "flip_turnover": volume["flip"],
                "resize_turnover": volume["resize"],
                "flip_turnover_fraction": (
                    volume["flip"] / total if total else 0.0
                ),
                "entries": counts["entry"],
                "exits": counts["exit"],
                "full_sign_flips": counts["flip"],
                "same_sign_resizes": counts["resize"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    price_frame = pd.read_csv(PRICE_PATH, sep=r"\s+")
    prices = price_frame.to_numpy(dtype=float).T
    targets, daily, pairs = collect(prices)
    performance = performance_table(prices, targets)
    daily_aggregate = daily_summary(daily)
    pair_aggregate = pair_integrity_summary(pairs)
    turnover = turnover_breakdown(prices, targets["production"])

    performance.to_csv(OUTPUT_DIR / "performance.csv", index=False)
    daily.to_csv(OUTPUT_DIR / "daily.csv", index=False)
    daily_aggregate.to_csv(
        OUTPUT_DIR / "daily_summary.csv",
        index=False,
    )
    pairs.to_csv(OUTPUT_DIR / "pair_integrity_events.csv", index=False)
    pair_aggregate.to_csv(
        OUTPUT_DIR / "pair_integrity_summary.csv",
        index=False,
    )
    turnover.to_csv(OUTPUT_DIR / "turnover.csv", index=False)

    display_performance = performance[
        [
            "window",
            "variant",
            "score",
            "total_pnl",
            "turnover",
            "fees",
            "score_delta",
            "pnl_delta",
        ]
    ]
    report = [
        "# Current execution audit",
        "",
        "No production rule is changed. Days 751-1000 are descriptive only.",
        "",
        "## Exact counterfactual accounting",
        "",
        markdown_table(display_performance, digits=2),
        "",
        "The fractional-share result is an infeasible upper-bound diagnostic;",
        "the evaluator requires integer shares.",
        "The 5bp row is a cost sensitivity, not a production assumption.",
        "",
        "## Daily execution state",
        "",
        markdown_table(daily_aggregate, digits=3),
        "",
        "## Pair hedge integrity after asset-level arbitration",
        "",
        markdown_table(pair_aggregate, digits=3),
        "",
        "## Turnover decomposition",
        "",
        markdown_table(turnover, digits=2),
        "",
        "A pair can be statistically valid at proposal time while ceasing to",
        "be a hedged pair after its two legs are arbitrated independently.",
    ]
    text = "\n".join(report) + "\n"
    (OUTPUT_DIR / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

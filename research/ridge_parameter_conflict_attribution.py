"""Attribute ridge parameter performance and pair-conflict delivery.

The study compares the previous ``lambda=1.0, theta=0.75, gate=35`` ridge
with the proposed ``lambda=1.52, theta=0.31, gate=35`` ridge.  A factorial
design isolates lambda and theta, while an event audit classifies every active
ridge forecast by its relationship with the current pair vote.

Days 251-500 are train, 501-750 are validation, and 751-1000 are already
exposed descriptive data.  This script never changes production constants.
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
OUTPUT_DIR = ROOT / "analysis_outputs" / "ridge_parameter_conflicts"
WINDOWS = (
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("TRAIN", 251, 500),
    ("VALIDATION_A", 501, 625),
    ("VALIDATION_B", 626, 750),
    ("VALIDATION", 501, 750),
    ("EXPOSED_YEAR4_A", 751, 875),
    ("EXPOSED_YEAR4_B", 876, 1000),
    ("EXPOSED_YEAR4", 751, 1000),
)
PRIMARY_WINDOWS = ("TRAIN", "VALIDATION", "EXPOSED_YEAR4")
CONFIGURATIONS = (
    ("old_l1_t075", 1.0, 0.75),
    ("lambda_only_l152_t075", 1.52, 0.75),
    ("theta_only_l1_t031", 1.0, 0.31),
    ("new_l152_t031", 1.52, 0.31),
)
ALGO = strategy.I["ALGO"]


def score(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe**2 / (sharpe**2 + 1.0)


def window_for_day(day: int) -> str | None:
    if 251 <= day <= 500:
        return "TRAIN"
    if 501 <= day <= 750:
        return "VALIDATION"
    if 751 <= day <= 1000:
        return "EXPOSED_YEAR4"
    return None


def strength_band(strength: float) -> str:
    if strength < 0.50:
        return "0.31-0.50"
    if strength < 0.75:
        return "0.50-0.75"
    if strength < 1.00:
        return "0.75-1.00"
    if strength < 1.50:
        return "1.00-1.50"
    return "1.50+"


def replay_configuration(
    prices: np.ndarray,
    name: str,
    ridge_lambda: float,
    ridge_theta: float,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    original_lambda = strategy.RIDGE_LAM
    original_theta = strategy.RIDGE_THETA
    strategy.RIDGE_LAM = ridge_lambda
    strategy.RIDGE_THETA = ridge_theta
    try:
        targets = {
            sleeve: np.zeros(
                (prices.shape[1] + 1, strategy.N_INST),
                dtype=int,
            )
            for sleeve in ("pair_only", "standalone", "combined")
        }
        rows: list[dict[str, float | int | str | bool]] = []
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
            returns = np.diff(np.log(known), axis=1)
            (
                ridge_raw,
                ridge_strength,
                ridge_latent_sign,
                ridge_gate_pass,
            ) = strategy._ridge_forecast_details(returns)
            pair_dollars = hedge_audit.current_pair_dollars(
                pair_vote,
                pair_active,
                crash,
            )
            combined_dollars, _ = hedge_audit.current_live_dollars(
                pair_vote,
                pair_active,
                ridge_raw,
                ridge_strength,
                ridge_latent_sign,
                ridge_gate_pass,
                crash,
            )
            current_prices = known[:, -1]
            targets["pair_only"][history_size] = hedge_audit.to_shares(
                pair_dollars,
                current_prices,
            )
            targets["standalone"][history_size] = hedge_audit.to_shares(
                ridge_raw,
                current_prices,
            )
            targets["combined"][history_size] = hedge_audit.to_shares(
                combined_dollars,
                current_prices,
            )

            pnl_day = history_size + 1
            if pnl_day > prices.shape[1]:
                continue
            next_prices = prices[:, history_size]
            standalone_shares = targets["standalone"][history_size]
            instrument_gross = standalone_shares * (
                next_prices - current_prices
            )
            ridge_active = standalone_shares != 0
            ridge_sign = np.sign(ridge_raw)
            pair_direction = np.sign(pair_vote)
            final_direction = np.sign(combined_dollars)

            for instrument in np.flatnonzero(ridge_active):
                instrument = int(instrument)
                if not pair_active[instrument]:
                    category = "pair_flat"
                elif abs(pair_vote[instrument]) <= 1.0e-12:
                    category = "pair_deadlock"
                elif pair_direction[instrument] == ridge_sign[instrument]:
                    category = "agreement"
                elif final_direction[instrument] == ridge_sign[instrument]:
                    category = "conflict_ridge_wins"
                else:
                    category = "conflict_pair_wins"
                gross = float(instrument_gross[instrument])
                strength = float(ridge_strength[instrument])
                rows.append(
                    {
                        "configuration": name,
                        "lambda": ridge_lambda,
                        "theta": ridge_theta,
                        "history_size": history_size,
                        "pnl_day": pnl_day,
                        "window": window_for_day(pnl_day),
                        "instrument": instrument,
                        "symbol": strategy._SYMBOL_BY_INDEX[instrument],
                        "strength": strength,
                        "strength_band": strength_band(strength),
                        "category": category,
                        "gross_pnl": gross,
                        "hit": gross > 0.0,
                        "pair_vote": float(pair_vote[instrument]),
                        "ridge_final_match": bool(
                            final_direction[instrument]
                            == ridge_sign[instrument]
                        ),
                    }
                )
        return targets, pd.DataFrame(rows)
    finally:
        strategy.RIDGE_LAM = original_lambda
        strategy.RIDGE_THETA = original_theta


def evaluate_all(
    prices: np.ndarray,
    all_targets: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for configuration, targets in all_targets.items():
        for sleeve, sleeve_targets in targets.items():
            result = evaluator.evaluate(prices, sleeve_targets, WINDOWS)
            for window, metrics in result.items():
                rows.append(
                    {
                        "configuration": configuration,
                        "sleeve": sleeve,
                        "window": window,
                        **metrics,
                    }
                )
    frame = pd.DataFrame(rows)
    pair = (
        frame[frame["sleeve"] == "pair_only"]
        .drop_duplicates("window")
        .set_index("window")
    )
    for column in ("score", "total_pnl", "mean_pnl"):
        frame[f"incremental_{column}"] = frame.apply(
            lambda row: (
                row[column] - pair.loc[row["window"], column]
                if row["sleeve"] == "combined"
                else row[column]
            ),
            axis=1,
        )
    return frame


def summarize_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = events[events["window"].notna()].copy()
    category = (
        selected.groupby(
            ["configuration", "window", "category"],
            as_index=False,
        )
        .agg(
            events=("gross_pnl", "size"),
            hit_rate=("hit", "mean"),
            gross_pnl=("gross_pnl", "sum"),
            mean_gross_pnl=("gross_pnl", "mean"),
            mean_strength=("strength", "mean"),
            ridge_final_match=("ridge_final_match", "mean"),
        )
    )
    band = (
        selected.groupby(
            ["configuration", "window", "strength_band"],
            as_index=False,
        )
        .agg(
            events=("gross_pnl", "size"),
            hit_rate=("hit", "mean"),
            gross_pnl=("gross_pnl", "sum"),
            mean_gross_pnl=("gross_pnl", "mean"),
        )
    )
    return category, band


def capture_table(performance: pd.DataFrame) -> pd.DataFrame:
    primary = performance[
        performance["window"].isin(PRIMARY_WINDOWS)
    ]
    standalone = (
        primary[primary["sleeve"] == "standalone"]
        .set_index(["configuration", "window"])
    )
    combined = (
        primary[primary["sleeve"] == "combined"]
        .set_index(["configuration", "window"])
    )
    rows: list[dict[str, float | str]] = []
    for index, standalone_row in standalone.iterrows():
        combined_row = combined.loc[index]
        incremental_pnl = float(combined_row["incremental_total_pnl"])
        standalone_pnl = float(standalone_row["total_pnl"])
        rows.append(
            {
                "configuration": index[0],
                "window": index[1],
                "standalone_score": float(standalone_row["score"]),
                "standalone_total_pnl": standalone_pnl,
                "combined_score": float(combined_row["score"]),
                "combined_incremental_score": float(
                    combined_row["incremental_score"]
                ),
                "combined_incremental_pnl": incremental_pnl,
                "pnl_capture_ratio": (
                    incremental_pnl / standalone_pnl
                    if abs(standalone_pnl) > 1.0e-12
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def make_plot(
    capture: pd.DataFrame,
    bands: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    standalone = capture.pivot(
        index="window",
        columns="configuration",
        values="standalone_score",
    ).reindex(PRIMARY_WINDOWS)
    standalone.plot.bar(ax=axes[0])
    axes[0].set_title("Standalone ridge score")
    axes[0].set_ylabel("official score")
    axes[0].tick_params(axis="x", rotation=0)

    new_bands = bands[
        bands["configuration"] == "new_l152_t031"
    ]
    band_pivot = new_bands.pivot(
        index="strength_band",
        columns="window",
        values="mean_gross_pnl",
    )
    order = ["0.31-0.50", "0.50-0.75", "0.75-1.00", "1.00-1.50", "1.50+"]
    band_pivot.reindex(order).plot.bar(ax=axes[1])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("New ridge: gross PnL per asset-event")
    axes[1].set_ylabel("gross dollars")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ridge_parameter_conflicts.png", dpi=160)
    plt.close(fig)


def write_report(
    performance: pd.DataFrame,
    capture: pd.DataFrame,
    categories: pd.DataFrame,
    bands: pd.DataFrame,
) -> None:
    selected = performance[
        performance["window"].isin(
            (
                "TRAIN_A",
                "TRAIN_B",
                "TRAIN",
                "VALIDATION_A",
                "VALIDATION_B",
                "VALIDATION",
                "EXPOSED_YEAR4_A",
                "EXPOSED_YEAR4_B",
                "EXPOSED_YEAR4",
            )
        )
    ]
    lines = [
        "# Ridge parameter and conflict attribution",
        "",
        "The production strategy is unchanged. Days 751-1000 are already exposed "
        "and are used only for descriptive attribution.",
        "",
        "## Standalone scores",
        "",
        "| Configuration | Train A | Train B | Train | Validation A | "
        "Validation B | Validation | Exposed A | Exposed B | Exposed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    standalone = selected[selected["sleeve"] == "standalone"].pivot(
        index="configuration",
        columns="window",
        values="score",
    )
    for configuration, row in standalone.iterrows():
        lines.append(
            f"| {configuration} | {row.TRAIN_A:+.2f} | "
            f"{row.TRAIN_B:+.2f} | {row.TRAIN:+.2f} | "
            f"{row.VALIDATION_A:+.2f} | {row.VALIDATION_B:+.2f} | "
            f"{row.VALIDATION:+.2f} | {row.EXPOSED_YEAR4_A:+.2f} | "
            f"{row.EXPOSED_YEAR4_B:+.2f} | "
            f"{row.EXPOSED_YEAR4:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Combined pair-plus-ridge scores",
            "",
            "| Configuration | Train A | Train B | Train | Validation A | "
            "Validation B | Validation | Exposed A | Exposed B | Exposed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    combined = selected[selected["sleeve"] == "combined"].pivot(
        index="configuration",
        columns="window",
        values="score",
    )
    for configuration, row in combined.iterrows():
        lines.append(
            f"| {configuration} | {row.TRAIN_A:+.2f} | "
            f"{row.TRAIN_B:+.2f} | {row.TRAIN:+.2f} | "
            f"{row.VALIDATION_A:+.2f} | {row.VALIDATION_B:+.2f} | "
            f"{row.VALIDATION:+.2f} | {row.EXPOSED_YEAR4_A:+.2f} | "
            f"{row.EXPOSED_YEAR4_B:+.2f} | "
            f"{row.EXPOSED_YEAR4:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Standalone PnL delivered into the combined book",
            "",
            "| Configuration | Window | Standalone PnL | Combined incremental "
            "PnL | PnL capture | Standalone score | Combined score delta |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in capture.iterrows():
        lines.append(
            f"| {row.configuration} | {row.window} | "
            f"${row.standalone_total_pnl:+,.0f} | "
            f"${row.combined_incremental_pnl:+,.0f} | "
            f"{row.pnl_capture_ratio:.1%} | "
            f"{row.standalone_score:+.2f} | "
            f"{row.combined_incremental_score:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## New ridge event attribution by conflict state",
            "",
            "| Window | State | Events | Hit rate | Gross PnL | "
            "Mean/event | Final matches ridge |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    new_categories = categories[
        categories["configuration"] == "new_l152_t031"
    ]
    for _, row in new_categories.iterrows():
        lines.append(
            f"| {row.window} | {row.category} | {int(row.events):,} | "
            f"{row.hit_rate:.1%} | ${row.gross_pnl:+,.0f} | "
            f"${row.mean_gross_pnl:+.2f} | "
            f"{row.ridge_final_match:.1%} |"
        )

    lines.extend(
        [
            "",
            "## New ridge strength bands",
            "",
            "| Window | Strength | Events | Hit rate | Gross PnL | Mean/event |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    new_bands = bands[bands["configuration"] == "new_l152_t031"]
    for _, row in new_bands.iterrows():
        lines.append(
            f"| {row.window} | {row.strength_band} | "
            f"{int(row.events):,} | {row.hit_rate:.1%} | "
            f"${row.gross_pnl:+,.0f} | ${row.mean_gross_pnl:+.2f} |"
        )
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+").to_numpy().T
    all_targets: dict[str, dict[str, np.ndarray]] = {}
    event_frames: list[pd.DataFrame] = []
    for name, ridge_lambda, ridge_theta in CONFIGURATIONS:
        print(f"Replaying {name}")
        targets, events = replay_configuration(
            prices,
            name,
            ridge_lambda,
            ridge_theta,
        )
        all_targets[name] = targets
        event_frames.append(events)
    events = pd.concat(event_frames, ignore_index=True)
    performance = evaluate_all(prices, all_targets)
    categories, bands = summarize_events(events)
    capture = capture_table(performance)
    performance.to_csv(OUTPUT_DIR / "performance.csv", index=False)
    events.to_csv(OUTPUT_DIR / "ridge_events.csv", index=False)
    categories.to_csv(
        OUTPUT_DIR / "conflict_categories.csv",
        index=False,
    )
    bands.to_csv(OUTPUT_DIR / "strength_bands.csv", index=False)
    capture.to_csv(OUTPUT_DIR / "capture.csv", index=False)
    make_plot(capture, bands)
    write_report(performance, capture, categories, bands)
    print("\nCapture")
    print(capture.to_string(index=False))
    print(f"\nWrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

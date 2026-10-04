"""Count and value deadlocks under the current production strategy.

A raw deadlock is an asset with active pair contributors whose normalized pair
vote is exactly zero.  It is then classified as:

* active ridge fill;
* subthreshold ridge fill; or
* unresolved after ridge.

The counterfactual removes only ridge positions placed into raw-deadlock slots,
then re-runs the existing ALGO beta-neutral overlay, share rounding, position
limits, turnover, and commissions.
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
OUTPUT_DIR = ROOT / "analysis_outputs" / "current_deadlocks"
WINDOWS = (
    ("TRAIN", 251, 500),
    ("VALIDATION", 501, 750),
    ("EXPOSED", 751, 1000),
)
BLOCKS = tuple(
    (
        f"{start}-{start + 49}",
        start,
        start + 49,
    )
    for start in range(251, 1000, 50)
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
    current_prices: np.ndarray,
) -> np.ndarray:
    safe = np.maximum(current_prices, strategy._EPS_DENOM)
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
        return core
    result, _, _, _ = strategy._algo_beta_neutral_overlay(
        core,
        np.asarray(
            sleeves["algo_residual_requested"],
            dtype=float,
        ),
        np.asarray(sleeves["algo_return_betas"], dtype=float),
        np.asarray(sleeves["crash_mask"], dtype=bool),
    )
    return result


def collect(
    prices: np.ndarray,
    symbols: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    names = (
        "production",
        "no_deadlock_ridge",
        "production_core",
        "no_deadlock_ridge_core",
    )
    targets = {
        name: np.zeros(
            (prices.shape[1] + 1, strategy.N_INST),
            dtype=int,
        )
        for name in names
    }
    daily_rows: list[dict[str, int | bool]] = []
    event_rows: list[dict[str, float | int | str | bool]] = []

    strategy.reset()
    for history_size in range(1, prices.shape[1]):
        known = prices[:, :history_size]
        if history_size < strategy.WARMUP + 10:
            continue
        sleeves = strategy.sleeve_breakdown(known)
        if sleeves is None:
            continue
        current_prices = known[:, -1]
        pair_active = np.asarray(
            sleeves["pair_active_mask"],
            dtype=bool,
        )
        pair_vote = np.asarray(sleeves["pair_vote"], dtype=float)
        crash = np.asarray(sleeves["crash_mask"], dtype=bool)
        ridge_active = np.asarray(
            sleeves["ridge_active_mask"],
            dtype=bool,
        )
        subthreshold = np.asarray(
            sleeves["deadlock_ridge_overlay_mask"],
            dtype=bool,
        )
        core = np.asarray(sleeves["core_dollars"], dtype=float)

        raw = (
            pair_active
            & (np.abs(pair_vote) < 1.0e-12)
            & ~crash
        )
        raw[strategy.I["ALGO"]] = False
        active_fill = raw & ridge_active
        subthreshold_fill = raw & subthreshold
        filled = active_fill | subthreshold_fill
        unresolved = raw & ~filled

        no_deadlock_core = core.copy()
        no_deadlock_core[filled] = 0.0
        no_deadlock_full = apply_algo_overlay(
            no_deadlock_core,
            sleeves,
        )
        targets["production"][history_size] = strategy.getMyPosition(
            known
        )
        targets["production_core"][history_size] = dollars_to_shares(
            core,
            current_prices,
        )
        targets[
            "no_deadlock_ridge_core"
        ][history_size] = dollars_to_shares(
            no_deadlock_core,
            current_prices,
        )
        targets["no_deadlock_ridge"][history_size] = dollars_to_shares(
            no_deadlock_full,
            current_prices,
        )

        pnl_day = history_size + 1
        daily_rows.append(
            {
                "history_size": history_size,
                "pnl_day": pnl_day,
                "raw_events": int(raw.sum()),
                "active_ridge_fills": int(active_fill.sum()),
                "subthreshold_ridge_fills": int(
                    subthreshold_fill.sum()
                ),
                "filled_events": int(filled.sum()),
                "unresolved_events": int(unresolved.sum()),
                "any_raw_deadlock": bool(raw.any()),
                "any_filled_deadlock": bool(filled.any()),
                "any_unresolved_deadlock": bool(unresolved.any()),
            }
        )

        if pnl_day > prices.shape[1]:
            continue
        next_returns = (
            prices[:, history_size] / current_prices - 1.0
        )
        for label, mask in (
            ("active_ridge", active_fill),
            ("subthreshold_ridge", subthreshold_fill),
            ("unresolved", unresolved),
        ):
            for instrument in np.flatnonzero(mask):
                gross_pnl = float(
                    core[instrument] * next_returns[instrument]
                )
                event_rows.append(
                    {
                        "history_size": history_size,
                        "pnl_day": pnl_day,
                        "instrument": int(instrument),
                        "symbol": symbols[instrument],
                        "resolution": label,
                        "position_dollars": float(core[instrument]),
                        "gross_pnl": gross_pnl,
                        "hit": bool(gross_pnl > 0.0),
                    }
                )
    strategy.reset()
    return targets, pd.DataFrame(daily_rows), pd.DataFrame(event_rows)


def evaluate_targets(
    prices: np.ndarray,
    targets: dict[str, np.ndarray],
    windows: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for variant, target in targets.items():
        result = evaluator.evaluate(prices, target, windows)
        for window, metrics in result.items():
            rows.append(
                {
                    "variant": variant,
                    "window": window,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def occurrence_summary(
    daily: pd.DataFrame,
    windows: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for window, start, end in windows:
        use = daily[daily["pnl_day"].between(start, end)]
        rows.append(
            {
                "window": window,
                "days": len(use),
                "days_with_raw": int(use["any_raw_deadlock"].sum()),
                "raw_day_rate": float(
                    use["any_raw_deadlock"].mean()
                ),
                "raw_events": int(use["raw_events"].sum()),
                "raw_events_per_day": float(
                    use["raw_events"].mean()
                ),
                "filled_events": int(use["filled_events"].sum()),
                "filled_fraction": float(
                    use["filled_events"].sum()
                    / max(1, use["raw_events"].sum())
                ),
                "days_with_unresolved": int(
                    use["any_unresolved_deadlock"].sum()
                ),
                "unresolved_events": int(
                    use["unresolved_events"].sum()
                ),
                "max_events_one_day": int(
                    use["raw_events"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def pnl_deltas(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in performance["window"].unique():
        use = performance[performance["window"] == window].set_index(
            "variant"
        )
        for book, baseline in (
            ("production", "no_deadlock_ridge"),
            ("production_core", "no_deadlock_ridge_core"),
        ):
            rows.append(
                {
                    "window": window,
                    "book": book,
                    "score": float(use.loc[book, "score"]),
                    "pnl": float(use.loc[book, "total_pnl"]),
                    "score_delta_from_ridge_fill": float(
                        use.loc[book, "score"]
                        - use.loc[baseline, "score"]
                    ),
                    "pnl_delta_from_ridge_fill": float(
                        use.loc[book, "total_pnl"]
                        - use.loc[baseline, "total_pnl"]
                    ),
                    "turnover_delta_from_ridge_fill": float(
                        use.loc[book, "turnover"]
                        - use.loc[baseline, "turnover"]
                    ),
                    "fee_delta_from_ridge_fill": float(
                        use.loc[book, "fees"]
                        - use.loc[baseline, "fees"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def event_summary(
    events: pd.DataFrame,
    windows: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    rows = []
    for window, start, end in windows:
        use = events[
            events["pnl_day"].between(start, end)
            & (events["resolution"] != "unresolved")
        ]
        for resolution, frame in use.groupby("resolution"):
            rows.append(
                {
                    "window": window,
                    "resolution": resolution,
                    "events": len(frame),
                    "hit_rate": float(frame["hit"].mean()),
                    "gross_pnl": float(frame["gross_pnl"].sum()),
                    "mean_gross_pnl": float(
                        frame["gross_pnl"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def win_loss_summary(
    events: pd.DataFrame,
    windows: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    rows = []
    for window, start, end in windows:
        use = events[
            events["pnl_day"].between(start, end)
            & (events["resolution"] != "unresolved")
        ]
        winners = use[use["gross_pnl"] > 0.0]
        losers = use[use["gross_pnl"] <= 0.0]
        rows.append(
            {
                "window": window,
                "events": len(use),
                "winning_events": len(winners),
                "losing_events": len(losers),
                "positive_gross_pnl": float(
                    winners["gross_pnl"].sum()
                ),
                "negative_gross_pnl": float(
                    losers["gross_pnl"].sum()
                ),
                "net_gross_pnl": float(use["gross_pnl"].sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    price_frame = pd.read_csv(PRICE_PATH, sep=r"\s+")
    symbols = tuple(str(column) for column in price_frame.columns)
    prices = price_frame.to_numpy().T
    targets, daily, events = collect(prices, symbols)
    performance = evaluate_targets(
        prices,
        targets,
        WINDOWS + BLOCKS,
    )
    occurrence = occurrence_summary(daily, WINDOWS + BLOCKS)
    deltas = pnl_deltas(performance)
    event_metrics = event_summary(events, WINDOWS + BLOCKS)
    win_loss = win_loss_summary(events, WINDOWS + BLOCKS)

    daily.to_csv(OUTPUT_DIR / "daily_counts.csv", index=False)
    events.to_csv(OUTPUT_DIR / "events.csv", index=False)
    performance.to_csv(OUTPUT_DIR / "performance.csv", index=False)
    occurrence.to_csv(OUTPUT_DIR / "occurrence.csv", index=False)
    deltas.to_csv(OUTPUT_DIR / "ridge_fill_deltas.csv", index=False)
    event_metrics.to_csv(
        OUTPUT_DIR / "event_performance.csv",
        index=False,
    )
    win_loss.to_csv(
        OUTPUT_DIR / "event_win_loss.csv",
        index=False,
    )

    lines = [
        "# Current deadlock accounting",
        "",
        "A raw deadlock is an exact zero pair vote on an asset with active",
        "pair contributors. Ridge fills and unresolved ties are separated.",
        "",
        "## Occurrence",
        "",
        markdown_table(
            occurrence[occurrence["window"].isin(("TRAIN", "VALIDATION", "EXPOSED"))],
            3,
        ),
        "",
        "## Incremental effect of current ridge deadlock fills",
        "",
        markdown_table(
            deltas[
                deltas["window"].isin(("TRAIN", "VALIDATION", "EXPOSED"))
            ],
            2,
        ),
        "",
        "## Filled-event directional performance",
        "",
        markdown_table(
            event_metrics[
                event_metrics["window"].isin(
                    ("TRAIN", "VALIDATION", "EXPOSED")
                )
            ],
            3,
        ),
        "",
        "## Winning and losing filled events",
        "",
        markdown_table(
            win_loss[
                win_loss["window"].isin(
                    ("TRAIN", "VALIDATION", "EXPOSED")
                )
            ],
            2,
        ),
        "",
        "## Exposed 50-day occurrence",
        "",
        markdown_table(
            occurrence[
                occurrence["window"].isin(
                    ("751-800", "801-850", "851-900", "901-950", "951-1000")
                )
            ],
            3,
        ),
        "",
        "## Exposed 50-day full-book ridge-fill deltas",
        "",
        markdown_table(
            deltas[
                (deltas["book"] == "production")
                & deltas["window"].isin(
                    ("751-800", "801-850", "851-900", "901-950", "951-1000")
                )
            ],
            2,
        ),
        "",
        "Unresolved deadlocks have no causal direction and therefore no",
        "realized PnL. Treating their next movement as a loss would be an",
        "oracle calculation, not a tradable counterfactual.",
    ]
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()

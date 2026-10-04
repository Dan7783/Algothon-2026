"""Describe how the frozen Four Sigmas ridge sleeve changes through time.

This is an attribution study, not a tuning script.  It replays the current
standalone ridge in consecutive 50-day blocks and compares it with:

* the same model estimated from a rolling 250-day history; and
* coefficients frozen at day 500 or day 750.

The frozen models answer whether later performance required newly learned
coefficients.  All reported 751-1000 results are descriptive because that
period has already been exposed elsewhere in this repository.
"""

from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy


PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "ridge_time_trend"
START_DAY = 251
END_DAY = 1000
BLOCK_DAYS = 50
COMMISSION = np.full(strategy.N_INST, 1.0e-4)
COMMISSION[strategy.I["ALGO"]] = 2.0e-5


def score(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe**2 / (sharpe**2 + 1.0)


def link_diagnostics(
    returns: np.ndarray,
) -> tuple[int, float, float, bool]:
    """Reproduce the causal rolling lag-link gate and expose its diagnostics."""
    if returns.shape[1] + 1 < strategy.RIDGE_MIN_T:
        return 0, 0.0, 0.0, False
    recent = returns[1:, -250:].T
    standardized = (
        recent - recent.mean(axis=0)
    ) / (recent.std(axis=0) + 1.0e-12)
    links = (
        standardized[:-1].T @ standardized[1:]
    ) / (len(recent) - 1)
    np.fill_diagonal(links, 0.0)
    threshold = 2.5 / sqrt(len(recent) - 1)
    significant = np.abs(links) > threshold
    count = int(significant.sum())
    mean_abs = float(np.mean(np.abs(links)))
    significant_mean_abs = (
        float(np.mean(np.abs(links[significant])))
        if count
        else 0.0
    )
    return (
        count,
        mean_abs,
        significant_mean_abs,
        count >= strategy.RIDGE_GATE_N,
    )


def fit_coefficients(
    prices: np.ndarray,
    cutoff: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the exact expanding ridge coefficients using history through cutoff."""
    known = prices[:, :cutoff]
    returns = np.diff(
        np.log(np.maximum(known, strategy._EPS_DENOM)),
        axis=1,
    )
    history = returns[1:].T
    scale = history.std(axis=0) + 1.0e-12
    predictors = history[:-1] / scale
    targets = history[1:] / scale
    coefficients = np.linalg.solve(
        predictors.T @ predictors
        + strategy.RIDGE_LAM
        * len(predictors)
        * np.eye(strategy.N_INST - 1),
        predictors.T @ targets,
    )
    prediction_scale = (predictors @ coefficients).std(axis=0)
    return coefficients, scale, prediction_scale


def positions_from_forecast(
    prediction: np.ndarray,
    prediction_scale: np.ndarray,
    gate_open: bool,
) -> tuple[np.ndarray, np.ndarray]:
    strengths = np.abs(prediction) / np.maximum(
        prediction_scale,
        1.0e-12,
    )
    active = strengths > strategy.RIDGE_THETA
    active_indices = np.flatnonzero(active)
    if len(active_indices) > strategy.RIDGE_MAX_ACTIVE:
        keep = active_indices[
            np.argsort(strengths[active_indices])[
                -strategy.RIDGE_MAX_ACTIVE:
            ]
        ]
        active[:] = False
        active[keep] = True
    dollars = np.zeros(strategy.N_INST)
    if gate_open:
        dollars[1:] = (
            strategy.RIDGE_LEG * np.sign(prediction) * active
        )
    full_strength = np.zeros(strategy.N_INST)
    full_strength[1:] = strengths
    return dollars, full_strength


def shares_from_dollars(
    dollars: np.ndarray,
    prices: np.ndarray,
    last_return: np.ndarray,
) -> np.ndarray:
    safe_prices = np.maximum(prices, strategy._EPS_DENOM)
    requested = dollars.copy()
    crash = np.abs(last_return) > 0.40
    crash[strategy.I["ALGO"]] = False
    requested[crash] = 0.0
    requested[0] = 0.0
    requested[1:] = np.clip(
        requested[1:],
        -strategy.LEG,
        strategy.LEG,
    )
    limits = np.floor(
        strategy.BANG_BANG_LIMITS / safe_prices
    ).astype(int)
    shares = np.round(requested / safe_prices)
    return np.clip(shares, -limits, limits).astype(int)


def build_targets(
    prices: np.ndarray,
    mode: str,
    cutoff: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build causal targets for expanding, rolling, or frozen coefficients."""
    targets = np.zeros(
        (prices.shape[1] + 1, strategy.N_INST),
        dtype=int,
    )
    diagnostics: list[dict[str, float | int | bool | str]] = []
    fixed = fit_coefficients(prices, cutoff) if cutoff else None
    original_lookback = strategy.RIDGE_LOOKBACK
    try:
        strategy.RIDGE_LOOKBACK = 250 if mode == "rolling_250" else None
        for history_size in range(2, prices.shape[1]):
            known = prices[:, :history_size]
            returns = np.diff(
                np.log(np.maximum(known, strategy._EPS_DENOM)),
                axis=1,
            )
            (
                significant_links,
                mean_abs_link,
                significant_mean_abs,
                gate_open,
            ) = link_diagnostics(returns)

            if (
                mode.startswith("frozen")
                and cutoff is not None
                and history_size >= cutoff
            ):
                coefficients, scale, prediction_scale = fixed
                prediction = (
                    returns[1:, -1] / scale
                ) @ coefficients
                dollars, strengths = positions_from_forecast(
                    prediction,
                    prediction_scale,
                    gate_open,
                )
            else:
                (
                    dollars,
                    strengths,
                    _,
                    exact_gate,
                ) = strategy._ridge_forecast_details(returns)
                gate_open = exact_gate

            last_return = returns[:, -1]
            targets[history_size] = shares_from_dollars(
                dollars,
                known[:, -1],
                last_return,
            )
            active = targets[history_size] != 0
            diagnostics.append(
                {
                    "mode": mode,
                    "history_size": history_size,
                    "pnl_day": history_size + 1,
                    "gate_open": gate_open,
                    "significant_links": significant_links,
                    "mean_abs_link": mean_abs_link,
                    "significant_mean_abs_link": significant_mean_abs,
                    "active_assets": int(active.sum()),
                    "mean_active_strength": (
                        float(np.mean(strengths[active]))
                        if active.any()
                        else 0.0
                    ),
                }
            )
    finally:
        strategy.RIDGE_LOOKBACK = original_lookback
    return targets, pd.DataFrame(diagnostics)


def account(
    prices: np.ndarray,
    targets: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact evaluator accounting with per-instrument attribution."""
    current = np.zeros(strategy.N_INST)
    cash = np.zeros(strategy.N_INST)
    value = np.zeros(strategy.N_INST)
    prior_fee = np.zeros(strategy.N_INST)
    daily_rows: list[dict[str, float | int]] = []
    instrument_rows: list[dict[str, float | int | str]] = []

    for day in range(START_DAY - 1, END_DAY + 1):
        current_prices = prices[:, day - 1]
        held_before_trade = current.copy()
        new_position = (
            targets[day].astype(float)
            if day < END_DAY
            else current.copy()
        )
        delta = new_position - current
        charged_fee = prior_fee.copy()
        cash -= current_prices * delta + charged_fee
        volume = current_prices * np.abs(delta)
        prior_fee = volume * COMMISSION
        current = new_position
        new_value = cash + current * current_prices
        pnl_by_instrument = new_value - value
        value = new_value

        if day < START_DAY:
            continue
        previous_prices = prices[:, day - 2]
        holding_gross = held_before_trade * (
            current_prices - previous_prices
        )
        active = held_before_trade != 0
        daily_rows.append(
            {
                "day": day,
                "net_pnl": float(pnl_by_instrument.sum()),
                "gross_pnl": float(holding_gross.sum()),
                "fees_charged": float(charged_fee.sum()),
                "turnover": float(volume.sum()),
                "active_assets": int(active.sum()),
                "signal_hits": int(
                    np.sum(holding_gross[active] > 0.0)
                ),
                "signal_events": int(active.sum()),
                "signal_abs_pnl": float(
                    np.sum(np.abs(holding_gross[active]))
                ),
                "signal_positive_pnl": float(
                    np.sum(
                        np.maximum(holding_gross[active], 0.0)
                    )
                ),
            }
        )
        for instrument in range(1, strategy.N_INST):
            instrument_rows.append(
                {
                    "day": day,
                    "instrument": instrument,
                    "symbol": strategy._SYMBOL_BY_INDEX[instrument],
                    "net_pnl": float(pnl_by_instrument[instrument]),
                    "gross_pnl": float(holding_gross[instrument]),
                    "fee": float(charged_fee[instrument]),
                    "active": bool(active[instrument]),
                    "hit": bool(
                        active[instrument]
                        and holding_gross[instrument] > 0.0
                    ),
                }
            )
    return pd.DataFrame(daily_rows), pd.DataFrame(instrument_rows)


def block_number(day: pd.Series) -> pd.Series:
    return ((day - START_DAY) // BLOCK_DAYS).astype(int)


def summarize_blocks(
    mode: str,
    daily: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    data = daily.copy()
    data["block"] = block_number(data["day"])
    diag = diagnostics[
        diagnostics["pnl_day"].between(START_DAY, END_DAY)
    ].copy()
    diag["block"] = block_number(diag["pnl_day"])
    diag_summary = diag.groupby("block").agg(
        gate_rate=("gate_open", "mean"),
        avg_significant_links=("significant_links", "mean"),
        avg_abs_link=("mean_abs_link", "mean"),
        avg_significant_abs_link=(
            "significant_mean_abs_link",
            "mean",
        ),
        avg_forecast_active_assets=("active_assets", "mean"),
        avg_active_strength=("mean_active_strength", "mean"),
    )

    rows: list[dict[str, float | int | str]] = []
    for block, group in data.groupby("block"):
        values = group["net_pnl"].to_numpy()
        mean = float(values.mean())
        std = float(values.std())
        wins = group.loc[
            group["net_pnl"] > 0.0,
            "net_pnl",
        ]
        losses = group.loc[
            group["net_pnl"] < 0.0,
            "net_pnl",
        ]
        signal_events = int(group["signal_events"].sum())
        absolute_signal_pnl = float(group["signal_abs_pnl"].sum())
        diag_row = diag_summary.loc[block]
        rows.append(
            {
                "mode": mode,
                "block": int(block),
                "start_day": int(group["day"].min()),
                "end_day": int(group["day"].max()),
                "score": score(values),
                "mean_pnl": mean,
                "std_pnl": std,
                "sharpe": (
                    sqrt(250.0) * mean / std
                    if std > 0.0
                    else 0.0
                ),
                "total_pnl": float(values.sum()),
                "daily_win_rate": float((values > 0.0).mean()),
                "daily_payoff_ratio": (
                    float(wins.mean() / abs(losses.mean()))
                    if len(wins) and len(losses)
                    else 0.0
                ),
                "fees": float(group["fees_charged"].sum()),
                "turnover": float(group["turnover"].sum()),
                "signal_events": signal_events,
                "signal_hit_rate": (
                    float(group["signal_hits"].sum() / signal_events)
                    if signal_events
                    else 0.0
                ),
                "positive_move_share": (
                    float(
                        group["signal_positive_pnl"].sum()
                        / absolute_signal_pnl
                    )
                    if absolute_signal_pnl
                    else 0.0
                ),
                **{
                    name: float(diag_row[name])
                    for name in diag_summary.columns
                },
            }
        )
    return pd.DataFrame(rows)


def period_attribution(
    instruments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = (
        ("YEAR2", 251, 500),
        ("YEAR3", 501, 750),
        ("EXPOSED_YEAR4", 751, 1000),
    )
    detailed: list[pd.DataFrame] = []
    breadth_rows: list[dict[str, float | int | str]] = []
    for name, start, end in periods:
        selected = instruments[
            instruments["day"].between(start, end)
        ]
        grouped = (
            selected.groupby(["instrument", "symbol"], as_index=False)
            .agg(
                net_pnl=("net_pnl", "sum"),
                gross_pnl=("gross_pnl", "sum"),
                fees=("fee", "sum"),
                active_days=("active", "sum"),
                hits=("hit", "sum"),
            )
        )
        grouped["period"] = name
        grouped["hit_rate"] = np.divide(
            grouped["hits"],
            grouped["active_days"],
            out=np.zeros(len(grouped), dtype=float),
            where=grouped["active_days"] > 0,
        )
        detailed.append(grouped)
        positive = grouped.loc[grouped["net_pnl"] > 0.0, "net_pnl"]
        total_positive = float(positive.sum())
        breadth_rows.append(
            {
                "period": name,
                "positive_instruments": int(
                    (grouped["net_pnl"] > 0.0).sum()
                ),
                "negative_instruments": int(
                    (grouped["net_pnl"] < 0.0).sum()
                ),
                "top5_positive_share": (
                    float(
                        positive.nlargest(5).sum() / total_positive
                    )
                    if total_positive > 0.0
                    else 0.0
                ),
                "total_net_pnl": float(grouped["net_pnl"].sum()),
            }
        )
    return pd.concat(detailed, ignore_index=True), pd.DataFrame(breadth_rows)


def make_plot(blocks: pd.DataFrame) -> None:
    expanding = blocks[blocks["mode"] == "expanding"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    colors = np.where(
        expanding["score"] >= 0.0,
        "#2ca02c",
        "#d62728",
    )
    axes[0].bar(expanding["start_day"], expanding["score"], width=42, color=colors)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("official score")
    axes[0].set_title("Standalone ridge: consecutive 50-day blocks")

    for mode, group in blocks.groupby("mode"):
        axes[1].plot(
            group["start_day"],
            group["signal_hit_rate"],
            marker="o",
            label=mode,
        )
    axes[1].axhline(0.5, color="black", linewidth=0.8)
    axes[1].set_ylabel("instrument-event hit rate")
    axes[1].legend(ncol=2)

    axes[2].plot(
        expanding["start_day"],
        expanding["avg_significant_links"],
        marker="o",
        label="significant lag links",
    )
    second = axes[2].twinx()
    second.plot(
        expanding["start_day"],
        expanding["avg_forecast_active_assets"],
        color="#ff7f0e",
        marker="s",
        label="active assets",
    )
    axes[2].axhline(
        strategy.RIDGE_GATE_N,
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    axes[2].set_ylabel("lag-link count")
    second.set_ylabel("active assets")
    axes[2].set_xlabel("block start day")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ridge_50day_blocks.png", dpi=160)
    plt.close(fig)


def write_report(
    blocks: pd.DataFrame,
    attribution: pd.DataFrame,
    breadth: pd.DataFrame,
) -> None:
    expanding = blocks[blocks["mode"] == "expanding"].copy()
    variables = (
        "signal_hit_rate",
        "positive_move_share",
        "avg_significant_links",
        "avg_forecast_active_assets",
        "avg_active_strength",
        "fees",
    )
    correlations = {
        variable: float(expanding["score"].corr(expanding[variable]))
        for variable in variables
    }
    trend_windows = (
        ("All blocks", expanding),
        (
            "Development 251-750",
            expanding[expanding["end_day"] <= 750],
        ),
        (
            "Exposed 751-1000",
            expanding[expanding["start_day"] >= 751],
        ),
    )
    trends: list[tuple[str, float, float, float, float]] = []
    for name, sample in trend_windows:
        linear = sps.linregress(sample["block"], sample["score"])
        rank = sps.spearmanr(sample["block"], sample["score"])
        trends.append(
            (
                name,
                float(linear.slope),
                float(linear.rvalue),
                float(linear.pvalue),
                float(rank.statistic),
            )
        )
    y4 = attribution[attribution["period"] == "EXPOSED_YEAR4"]
    top_y4 = y4.nlargest(8, "net_pnl")
    worst_y4 = y4.nsmallest(5, "net_pnl")

    lines = [
        "# Standalone ridge time-trend attribution",
        "",
        "The strategy is frozen. Days 751-1000 are descriptive and were not used "
        "to alter any model or threshold.",
        "",
        "## Exact expanding-ridge 50-day blocks",
        "",
        "| Days | Score | Mean/day | Total PnL | Sharpe | Daily win | "
        "Signal hit | Positive move share | Gate | Active assets | Links |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in expanding.iterrows():
        lines.append(
            f"| {int(row.start_day)}-{int(row.end_day)} | "
            f"{row.score:+.2f} | ${row.mean_pnl:+,.0f} | "
            f"${row.total_pnl:+,.0f} | {row.sharpe:+.2f} | "
            f"{row.daily_win_rate:.1%} | {row.signal_hit_rate:.1%} | "
            f"{row.positive_move_share:.1%} | {row.gate_rate:.1%} | "
            f"{row.avg_forecast_active_assets:.1f} | "
            f"{row.avg_significant_links:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Counterfactual score by block",
            "",
            "| Days | Expanding | Rolling 250 | Frozen day 500 | Frozen day 750 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    pivot = blocks.pivot(
        index=["start_day", "end_day"],
        columns="mode",
        values="score",
    )
    for (start, end), row in pivot.iterrows():
        lines.append(
            f"| {start}-{end} | {row.get('expanding', float('nan')):+.2f} | "
            f"{row.get('rolling_250', float('nan')):+.2f} | "
            f"{row.get('frozen_500', float('nan')):+.2f} | "
            f"{row.get('frozen_750', float('nan')):+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Cross-block score correlations",
            "",
        ]
    )
    for name, value in correlations.items():
        lines.append(f"- `{name}`: **{value:+.3f}**")

    lines.extend(
        [
            "",
            "## Is the time trend monotonic?",
            "",
            "| Window | Score slope per block | Pearson r | Linear p | Spearman rho |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, slope, pearson, p_value, spearman in trends:
        lines.append(
            f"| {name} | {slope:+.2f} | {pearson:+.3f} | "
            f"{p_value:.4f} | {spearman:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Instrument breadth",
            "",
            "| Period | Positive | Negative | Top-five share of positive PnL | Net PnL |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in breadth.iterrows():
        lines.append(
            f"| {row.period} | {int(row.positive_instruments)} | "
            f"{int(row.negative_instruments)} | "
            f"{row.top5_positive_share:.1%} | "
            f"${row.total_net_pnl:+,.0f} |"
        )

    lines.extend(["", "### Top exposed-Year-4 contributors", ""])
    for _, row in top_y4.iterrows():
        lines.append(
            f"- {row.symbol}: **${row.net_pnl:+,.0f}**, "
            f"{int(row.active_days)} active days, {row.hit_rate:.1%} hit rate."
        )
    lines.extend(["", "### Worst exposed-Year-4 contributors", ""])
    for _, row in worst_y4.iterrows():
        lines.append(
            f"- {row.symbol}: **${row.net_pnl:+,.0f}**, "
            f"{int(row.active_days)} active days, {row.hit_rate:.1%} hit rate."
        )

    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+").to_numpy().T
    configurations = (
        ("expanding", None),
        ("rolling_250", None),
        ("frozen_500", 500),
        ("frozen_750", 750),
    )
    all_blocks: list[pd.DataFrame] = []
    expanding_instruments: pd.DataFrame | None = None
    for mode, cutoff in configurations:
        targets, diagnostics = build_targets(
            prices,
            mode=mode,
            cutoff=cutoff,
        )
        daily, instruments = account(prices, targets)
        all_blocks.append(
            summarize_blocks(mode, daily, diagnostics)
        )
        if mode == "expanding":
            daily.to_csv(OUTPUT_DIR / "daily_ridge.csv", index=False)
            diagnostics.to_csv(
                OUTPUT_DIR / "daily_diagnostics.csv",
                index=False,
            )
            expanding_instruments = instruments

    blocks = pd.concat(all_blocks, ignore_index=True)
    blocks.to_csv(OUTPUT_DIR / "block_metrics.csv", index=False)
    assert expanding_instruments is not None
    attribution, breadth = period_attribution(expanding_instruments)
    attribution.to_csv(
        OUTPUT_DIR / "instrument_attribution.csv",
        index=False,
    )
    breadth.to_csv(OUTPUT_DIR / "instrument_breadth.csv", index=False)
    make_plot(blocks)
    write_report(blocks, attribution, breadth)
    print(blocks[blocks["mode"] == "expanding"].to_string(index=False))
    print("\nInstrument breadth")
    print(breadth.to_string(index=False))
    print(f"\nWrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

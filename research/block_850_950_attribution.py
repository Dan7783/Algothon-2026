"""Attribute the weak days 851-950 block in the current production strategy.

This is descriptive analysis of an already-exposed interval.  It does not
select parameters or mutate the strategy.
"""

from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy


PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "block_850_950_attribution"
WINDOWS = (
    ("751-800", 751, 800),
    ("801-850", 801, 850),
    ("851-900", 851, 900),
    ("901-950", 901, 950),
    ("951-1000", 951, 1000),
    ("851-950", 851, 950),
)
COMMISSION = np.full(strategy.N_INST, 0.0001)
COMMISSION[strategy.I["ALGO"]] = 0.00002


def score(pnl: np.ndarray) -> float:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe**2 / (sharpe**2 + 1.0)


def markdown_table(frame: pd.DataFrame, digits: int = 2) -> str:
    """Render a small DataFrame without pandas' optional tabulate package."""
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
    raw = dollars / safe
    return np.clip(np.round(raw), -limits, limits).astype(int)


def signal_categories(sleeves: dict[str, object]) -> np.ndarray:
    pair_vote = np.asarray(sleeves["pair_vote"], dtype=float)
    pair_active = np.asarray(
        sleeves["pair_active_mask"],
        dtype=bool,
    )
    ridge_raw = np.asarray(sleeves["ridge_raw"], dtype=float)
    ridge_active = np.asarray(
        sleeves["ridge_active_mask"],
        dtype=bool,
    )
    conflict = np.asarray(sleeves["conflict_mask"], dtype=bool)
    confirmed = np.asarray(
        sleeves["ridge_conflict_confirmed_mask"],
        dtype=bool,
    )
    subthreshold = np.asarray(
        sleeves["deadlock_ridge_overlay_mask"],
        dtype=bool,
    )
    pair_sign = np.sign(pair_vote)
    ridge_sign = np.sign(ridge_raw)
    pair_nonzero = pair_active & (pair_sign != 0.0)
    deadlock = pair_active & ~pair_nonzero

    result = np.full(strategy.N_INST, "flat", dtype=object)
    result[pair_nonzero & ~ridge_active] = "pair_only"
    result[~pair_active & ridge_active] = "ridge_free_slot"
    result[
        pair_nonzero
        & ridge_active
        & (pair_sign == ridge_sign)
    ] = "pair_ridge_agreement"
    result[deadlock & ridge_active] = "active_ridge_deadlock"
    result[deadlock & ~ridge_active] = "unfilled_pair_deadlock"
    result[subthreshold] = "subthreshold_ridge_deadlock"
    result[conflict & ~confirmed] = "conflict_pair_kept"
    result[confirmed] = "conflict_ridge_confirmed"
    return result


def build_targets_and_diagnostics(
    prices: np.ndarray,
    symbols: tuple[str, ...],
) -> tuple[
    dict[str, np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[int, np.ndarray],
]:
    targets = {
        name: np.zeros(
            (prices.shape[1] + 1, strategy.N_INST),
            dtype=int,
        )
        for name in ("full", "core", "pair_only", "ridge_only")
    }
    daily_rows: list[dict[str, float | int]] = []
    pair_rows: list[dict[str, float | int | str | bool]] = []
    ridge_rows: list[dict[str, float | int | str | bool]] = []
    categories: dict[int, np.ndarray] = {}

    strategy.reset()
    for history_size in range(749, prices.shape[1]):
        known = prices[:, :history_size]
        sleeves = strategy.sleeve_breakdown(known)
        if sleeves is None:
            continue
        current_prices = known[:, -1]
        targets["full"][history_size] = strategy.getMyPosition(known)
        targets["core"][history_size] = dollars_to_shares(
            np.asarray(sleeves["core_dollars"], dtype=float),
            current_prices,
        )

        crash = np.asarray(sleeves["crash_mask"], dtype=bool)
        pair_vote = np.asarray(sleeves["pair_vote"], dtype=float)
        pair_dollars = (
            strategy.BANG_BANG_LIMITS * np.sign(pair_vote)
        )
        pair_dollars[crash] = 0.0
        pair_dollars[strategy.I["ALGO"]] = 0.0
        targets["pair_only"][history_size] = dollars_to_shares(
            pair_dollars,
            current_prices,
        )

        ridge_raw = np.asarray(sleeves["ridge_raw"], dtype=float)
        ridge_dollars = (
            strategy.BANG_BANG_LIMITS * np.sign(ridge_raw)
        )
        ridge_dollars[crash] = 0.0
        ridge_dollars[strategy.I["ALGO"]] = 0.0
        targets["ridge_only"][history_size] = dollars_to_shares(
            ridge_dollars,
            current_prices,
        )

        categories[history_size] = signal_categories(sleeves)
        pnl_day = history_size + 1
        pair_active = np.asarray(
            sleeves["pair_active_mask"],
            dtype=bool,
        )
        ridge_active = np.asarray(
            sleeves["ridge_active_mask"],
            dtype=bool,
        )
        daily_rows.append(
            {
                "history_size": history_size,
                "pnl_day": pnl_day,
                "pair_active_assets": int(pair_active.sum()),
                "ridge_active_assets": int(ridge_active.sum()),
                "pair_deadlocks": int(
                    (
                        pair_active
                        & (np.abs(pair_vote) < 1.0e-12)
                    ).sum()
                ),
                "true_conflicts": int(
                    np.asarray(
                        sleeves["conflict_mask"],
                        dtype=bool,
                    ).sum()
                ),
                "ridge_confirmed_conflicts": int(
                    np.asarray(
                        sleeves["ridge_conflict_confirmed_mask"],
                        dtype=bool,
                    ).sum()
                ),
                "algo_delivery": float(
                    sleeves["algo_residual_delivery"]
                ),
            }
        )

        if history_size >= prices.shape[1]:
            continue
        next_returns = (
            prices[:, history_size] / current_prices - 1.0
        )
        ridge_strength = np.asarray(
            sleeves["ridge_strength"],
            dtype=float,
        )
        for instrument in np.flatnonzero(ridge_raw != 0.0):
            gross = (
                strategy.BANG_BANG_LIMITS[instrument]
                * np.sign(ridge_raw[instrument])
                * next_returns[instrument]
            )
            ridge_rows.append(
                {
                    "history_size": history_size,
                    "pnl_day": pnl_day,
                    "instrument": int(instrument),
                    "symbol": symbols[instrument],
                    "strength": float(ridge_strength[instrument]),
                    "gross_pnl": float(gross),
                    "hit": bool(gross > 0.0),
                }
            )
        for detail in sleeves["pair_contributor_details"]:
            y = int(detail["y"])
            x = int(detail["x"])
            gross = (
                float(detail["pos_y"]) * next_returns[y]
                + float(detail["pos_x"]) * next_returns[x]
            )
            pair_rows.append(
                {
                    "history_size": history_size,
                    "pnl_day": pnl_day,
                    "pair": str(detail["pair"]),
                    "y": symbols[y],
                    "x": symbols[x],
                    "gross_pnl": gross,
                    "hit": gross > 0.0,
                    "abs_z": float(detail["abs_z"]),
                    "kappa": float(detail["kappa"]),
                    "cut": bool(detail["cut"]),
                    "adapting": bool(detail["adapting"]),
                    "vol_shift": bool(detail["vol_shift"]),
                }
            )
    strategy.reset()
    return (
        targets,
        pd.DataFrame(daily_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(ridge_rows),
        categories,
    )


def account(
    prices: np.ndarray,
    targets: np.ndarray,
    start: int,
    end: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    cash = np.zeros(strategy.N_INST)
    value = np.zeros(strategy.N_INST)
    prior_fee = np.zeros(strategy.N_INST)
    current = np.zeros(strategy.N_INST)
    daily: list[dict[str, float | int]] = []
    asset_pnl = np.zeros(strategy.N_INST)
    turnover = 0.0
    fees = 0.0

    for day in range(start - 1, end + 1):
        current_prices = prices[:, day - 1]
        new_position = (
            targets[day].astype(float)
            if day < end
            else current.copy()
        )
        delta = new_position - current
        cash -= current_prices * delta + prior_fee
        volume = current_prices * np.abs(delta)
        prior_fee = volume * COMMISSION
        turnover += float(volume.sum())
        fees += float(prior_fee.sum())
        current = new_position
        new_value = cash + current * current_prices
        pnl_by_asset = new_value - value
        value = new_value
        if day >= start:
            asset_pnl += pnl_by_asset
            daily.append(
                {
                    "day": day,
                    "pnl": float(pnl_by_asset.sum()),
                    "gross_before_prior_fee": float(
                        pnl_by_asset.sum() + prior_fee.sum()
                    ),
                }
            )
    assets = pd.DataFrame(
        {
            "instrument": np.arange(strategy.N_INST),
            "total_pnl": asset_pnl,
        }
    )
    return pd.DataFrame(daily), assets, turnover, fees


def category_attribution(
    prices: np.ndarray,
    targets: np.ndarray,
    categories: dict[int, np.ndarray],
    symbols: tuple[str, ...],
    start: int,
    end: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    prior = np.zeros(strategy.N_INST)
    for history_size in range(start - 1, end):
        position = targets[history_size].astype(float)
        current_prices = prices[:, history_size - 1]
        next_prices = prices[:, history_size]
        delta = position - prior
        fee = current_prices * np.abs(delta) * COMMISSION
        pnl = position * (next_prices - current_prices) - fee
        labels = categories[history_size]
        for instrument in range(strategy.N_INST):
            rows.append(
                {
                    "pnl_day": history_size + 1,
                    "instrument": instrument,
                    "symbol": symbols[instrument],
                    "category": str(labels[instrument]),
                    "pnl": float(pnl[instrument]),
                    "active": bool(position[instrument] != 0.0),
                }
            )
        prior = position
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("category", as_index=False)
        .agg(
            total_pnl=("pnl", "sum"),
            asset_days=("active", "sum"),
            mean_pnl=("pnl", "mean"),
        )
        .sort_values("total_pnl")
    )


def metrics_row(
    name: str,
    variant: str,
    daily: pd.DataFrame,
    turnover: float,
    fees: float,
) -> dict[str, float | str]:
    pnl = daily["pnl"].to_numpy(dtype=float)
    mean = float(pnl.mean())
    std = float(pnl.std())
    return {
        "window": name,
        "variant": variant,
        "score": score(pnl),
        "mean_pnl": mean,
        "std_pnl": std,
        "sharpe": sqrt(250.0) * mean / std if std > 0 else 0.0,
        "total_pnl": float(pnl.sum()),
        "turnover": turnover,
        "fees": fees,
        "positive_days": int((pnl > 0.0).sum()),
        "negative_days": int((pnl < 0.0).sum()),
    }


def block_regime(
    prices: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    returns = prices[:, start - 1 : end] / prices[
        :, start - 2 : end - 1
    ] - 1.0
    return {
        "mean_cross_sectional_abs_return": float(
            np.mean(np.abs(returns))
        ),
        "mean_cross_sectional_dispersion": float(
            np.mean(np.std(returns, axis=0))
        ),
        "mean_asset_realized_vol": float(
            np.mean(np.std(returns, axis=1))
        ),
    }


def maximum_drawdown(daily: pd.DataFrame) -> dict[str, float | int]:
    pnl = daily["pnl"].to_numpy(dtype=float)
    cumulative = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(
        np.concatenate(([0.0], cumulative))
    )[1:]
    drawdown = cumulative - running_peak
    trough_index = int(np.argmin(drawdown))
    peak_candidates = np.concatenate(([0.0], cumulative))[
        : trough_index + 1
    ]
    peak_index = int(np.argmax(peak_candidates))
    start_day = (
        int(daily.iloc[0]["day"]) - 1
        if peak_index == 0
        else int(daily.iloc[peak_index - 1]["day"])
    )
    return {
        "max_drawdown": float(drawdown[trough_index]),
        "peak_day": start_day,
        "trough_day": int(daily.iloc[trough_index]["day"]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    price_frame = pd.read_csv(PRICE_PATH, sep=r"\s+")
    symbols = tuple(str(column) for column in price_frame.columns)
    prices = price_frame.to_numpy().T
    targets, diagnostics, pair_events, ridge_events, categories = (
        build_targets_and_diagnostics(prices, symbols)
    )

    performance_rows: list[dict[str, float | str]] = []
    asset_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    regime_rows: list[dict[str, float | str]] = []
    for window, start, end in WINDOWS:
        for variant, variant_targets in targets.items():
            daily, assets, turnover, fees = account(
                prices,
                variant_targets,
                start,
                end,
            )
            performance_rows.append(
                metrics_row(
                    window,
                    variant,
                    daily,
                    turnover,
                    fees,
                )
            )
            daily["window"] = window
            daily["variant"] = variant
            daily_frames.append(daily)
            assets["window"] = window
            assets["variant"] = variant
            assets["symbol"] = [
                symbols[index]
                for index in assets["instrument"].astype(int)
            ]
            asset_frames.append(assets)
        regime_rows.append(
            {
                "window": window,
                **block_regime(prices, start, end),
                **{
                    column: float(
                        diagnostics.loc[
                            diagnostics["pnl_day"].between(start, end),
                            column,
                        ].mean()
                    )
                    for column in (
                        "pair_active_assets",
                        "ridge_active_assets",
                        "pair_deadlocks",
                        "true_conflicts",
                        "ridge_confirmed_conflicts",
                        "algo_delivery",
                    )
                },
            }
        )

    performance = pd.DataFrame(performance_rows)
    assets = pd.concat(asset_frames, ignore_index=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    regime = pd.DataFrame(regime_rows)
    category_frames = []
    pair_frames = []
    for window, start, end in (
        ("851-900", 851, 900),
        ("901-950", 901, 950),
        ("851-950", 851, 950),
    ):
        category_frame = category_attribution(
            prices,
            targets["core"],
            categories,
            symbols,
            start,
            end,
        )
        category_frame.insert(0, "window", window)
        category_frames.append(category_frame)

        pair_window = pair_events[
            pair_events["pnl_day"].between(start, end)
        ]
        pair_frame = (
            pair_window.groupby("pair", as_index=False)
            .agg(
                gross_pnl=("gross_pnl", "sum"),
                active_days=("pnl_day", "count"),
                hit_rate=("hit", "mean"),
                mean_abs_z=("abs_z", "mean"),
                cut_rate=("cut", "mean"),
                adapting_rate=("adapting", "mean"),
                vol_shift_rate=("vol_shift", "mean"),
            )
            .sort_values("gross_pnl")
        )
        pair_frame.insert(0, "window", window)
        pair_frames.append(pair_frame)
    category_summary = pd.concat(category_frames, ignore_index=True)
    pair_summary = pd.concat(pair_frames, ignore_index=True)
    pair_summary_weak = pair_summary[
        pair_summary["window"] == "851-950"
    ].sort_values("gross_pnl")
    ridge_frames = []
    for window, start, end in (
        ("801-850", 801, 850),
        ("851-900", 851, 900),
        ("901-950", 901, 950),
        ("951-1000", 951, 1000),
    ):
        ridge_window = ridge_events[
            ridge_events["pnl_day"].between(start, end)
        ]
        ridge_frames.append(
            pd.DataFrame(
                [
                    {
                        "window": window,
                        "events": len(ridge_window),
                        "hit_rate": float(ridge_window["hit"].mean()),
                        "gross_pnl": float(
                            ridge_window["gross_pnl"].sum()
                        ),
                        "mean_strength": float(
                            ridge_window["strength"].mean()
                        ),
                    }
                ]
            )
        )
    ridge_summary = pd.concat(ridge_frames, ignore_index=True)
    ridge_901 = (
        ridge_events[
            ridge_events["pnl_day"].between(901, 950)
        ]
        .groupby("symbol", as_index=False)
        .agg(
            events=("pnl_day", "count"),
            hit_rate=("hit", "mean"),
            gross_pnl=("gross_pnl", "sum"),
            mean_strength=("strength", "mean"),
        )
        .sort_values("gross_pnl")
    )
    ridge_901_events = ridge_events[
        ridge_events["pnl_day"].between(901, 950)
    ].copy()
    ridge_901_events["strength_band"] = pd.cut(
        ridge_901_events["strength"],
        bins=(
            strategy.RIDGE_THETA,
            0.50,
            0.75,
            1.00,
            1.50,
            np.inf,
        ),
        labels=(
            "0.31-0.50",
            "0.50-0.75",
            "0.75-1.00",
            "1.00-1.50",
            "1.50+",
        ),
        include_lowest=True,
    )
    ridge_strength_bands = (
        ridge_901_events.groupby(
            "strength_band",
            observed=True,
            as_index=False,
        )
        .agg(
            events=("pnl_day", "count"),
            hit_rate=("hit", "mean"),
            gross_pnl=("gross_pnl", "sum"),
            mean_gross_pnl=("gross_pnl", "mean"),
        )
    )
    weak_daily = daily[
        (daily["window"] == "851-950")
        & (daily["variant"] == "full")
    ].copy()
    drawdown = maximum_drawdown(weak_daily)
    worst_days = weak_daily.nsmallest(12, "pnl")

    performance.to_csv(OUTPUT_DIR / "performance.csv", index=False)
    assets.to_csv(OUTPUT_DIR / "asset_attribution.csv", index=False)
    daily.to_csv(OUTPUT_DIR / "daily_pnl.csv", index=False)
    regime.to_csv(OUTPUT_DIR / "regime.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "signal_counts.csv", index=False)
    pair_events.to_csv(
        OUTPUT_DIR / "independent_pair_events.csv",
        index=False,
    )
    ridge_events.to_csv(
        OUTPUT_DIR / "ridge_events.csv",
        index=False,
    )
    ridge_summary.to_csv(
        OUTPUT_DIR / "ridge_calibration.csv",
        index=False,
    )
    ridge_strength_bands.to_csv(
        OUTPUT_DIR / "ridge_strength_bands_901_950.csv",
        index=False,
    )
    pair_summary.to_csv(
        OUTPUT_DIR / "independent_pair_attribution.csv",
        index=False,
    )
    category_summary.to_csv(
        OUTPUT_DIR / "core_signal_category_attribution.csv",
        index=False,
    )

    weak_assets = assets[
        (assets["window"] == "851-950")
        & (assets["variant"] == "full")
    ].sort_values("total_pnl")
    selected_perf = performance[
        performance["window"].isin(
            ("801-850", "851-900", "901-950", "951-1000", "851-950")
        )
        & performance["variant"].isin(
            ("full", "core", "pair_only", "ridge_only")
        )
    ]
    lines = [
        "# Days 851-950 attribution",
        "",
        "Descriptive only: this exposed interval is not used for tuning.",
        "",
        "## Performance",
        "",
        markdown_table(selected_perf),
        "",
        "## Regime and signal counts",
        "",
        markdown_table(
            regime[
                regime["window"].isin(
                    ("801-850", "851-900", "901-950", "951-1000")
                )
            ],
            4,
        ),
        "",
        "## Core signal-state attribution, days 851-950",
        "",
        markdown_table(category_summary),
        "",
        "## Ridge forecast calibration",
        "",
        markdown_table(ridge_summary, 4),
        "",
        "## Worst ridge assets, days 901-950",
        "",
        markdown_table(ridge_901.head(15), 3),
        "",
        "## Ridge strength bands, days 901-950",
        "",
        markdown_table(ridge_strength_bands, 3),
        "",
        "## Worst full-book assets, days 851-950",
        "",
        markdown_table(weak_assets.head(12)),
        "",
        "## Best full-book assets, days 851-950",
        "",
        markdown_table(
            weak_assets.tail(12).sort_values(
                "total_pnl",
                ascending=False,
            )
        ),
        "",
        "## Worst independent pair proposals, days 851-950",
        "",
        markdown_table(pair_summary_weak.head(15), 3),
        "",
        "## Best independent pair proposals, days 851-950",
        "",
        markdown_table(
            pair_summary_weak.tail(15).sort_values(
                "gross_pnl",
                ascending=False,
            ),
            3,
        ),
        "",
        "## Worst days",
        "",
        markdown_table(worst_days),
        "",
        "## Drawdown",
        "",
        f"- Maximum drawdown: ${drawdown['max_drawdown']:,.2f}.",
        f"- Prior peak day: {drawdown['peak_day']}.",
        f"- Trough day: {drawdown['trough_day']}.",
        "",
        "Independent pair PnL deliberately ignores overlap and caps; it is a",
        "diagnostic of raw pair signal quality, not additive portfolio PnL.",
    ]
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()

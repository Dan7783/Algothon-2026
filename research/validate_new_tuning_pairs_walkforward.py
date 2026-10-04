#!/usr/bin/env python
"""Correlation audit and singular-pair walk-forward replay.

This analysis is deliberately narrower than a portfolio backtest:

* only the seven profiles restored from ``tuning_data.db`` are tested;
* each profile trades by itself, with ridge and pair-conflict voting disabled;
* parameters stay fixed at their current values in ``four_sigmas.py``;
* every forward fold starts flat, but the adaptive filter is warmed causally
  on all observations available before that fold;
* the four folds ending at day 750 are development-stability diagnostics,
  while days 751-1000 remain separately labelled as the true holdout.

The earlier folds are not claimed to be unbiased parameter-selection tests:
the current profiles were selected using development data through day 750.
They show chronological stability. Only the final two folds are genuinely
out of sample with respect to that selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy


TARGET_PAIRS = (
    "MTNS-MHRM",
    "FWWG-NGTE",
    "NGTE-FCSG",
    "MTNS-ULXY",
    "GARI-FWWG",
    "CCNS-NAYO",
    "ULXY-FCSG",
)

OUTPUT_DIR = ROOT / "analysis_outputs" / "new_tuning_pairs_walkforward"
LEG = 10_000.0
COMMISSION = 0.0001
ROLLING_CORR_WINDOW = 60

# A 125-observation Pearson correlation becomes conventionally significant
# near |r|=0.18 under IID assumptions. We use 0.20 as a simple, pre-declared
# persistence floor and require it in five of six forward folds.
MIN_ABS_FORWARD_CORR = 0.20
MIN_CORR_PASS_FOLDS = 5
MIN_SIGN_CONSISTENT_FOLDS = 5
MIN_ROLLING_CORR_COVERAGE = 0.50

# Six non-overlapping forward folds after the initial 250-day history.
# Indices are zero-based, half-open; displayed day numbers are one-based.
FOLDS = (
    ("DEV1", 250, 375),
    ("DEV2", 375, 500),
    ("DEV3", 500, 625),
    ("DEV4", 625, 750),
    ("OOS1", 750, 875),
    ("OOS2", 875, 1000),
)


def algothon_metrics(pnl: np.ndarray) -> dict[str, float]:
    """Return the official mean/std/Sharpe/score transformation."""
    values = np.asarray(pnl, dtype=float)
    if values.size == 0:
        return {
            "mean_pnl": 0.0,
            "std_pnl": 0.0,
            "sharpe": 0.0,
            "score": 0.0,
            "total_pnl": 0.0,
        }
    mean = float(np.mean(values))
    std = float(np.std(values))
    sharpe = float(np.sqrt(250.0) * mean / std) if std > 1.0e-10 else 0.0
    if mean <= 0.0 or std < 1.0e-10:
        score = mean
    else:
        score = mean * sharpe**2 / (sharpe**2 + 1.0)
    return {
        "mean_pnl": mean,
        "std_pnl": std,
        "sharpe": sharpe,
        "score": float(score),
        "total_pnl": float(np.sum(values)),
    }


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    """Finite Pearson correlation, or zero for degenerate samples."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if (
        left.size < 3
        or float(np.std(left)) < 1.0e-12
        or float(np.std(right)) < 1.0e-12
    ):
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def pair_name(spec: strategy.PairCfg) -> str:
    symbols = {index: symbol for symbol, index in strategy.I.items()}
    return f"{symbols[int(spec.y)]}-{symbols[int(spec.x)]}"


def target_specs() -> dict[str, strategy.PairCfg]:
    configured = {pair_name(spec): spec for spec in strategy.PAIR_CFGS}
    missing = sorted(set(TARGET_PAIRS) - set(configured))
    if missing:
        raise RuntimeError(f"Target profiles are not active: {missing}")
    return {name: configured[name] for name in TARGET_PAIRS}


def correlation_window(
    log_prices: np.ndarray,
    spec: strategy.PairCfg,
    start: int,
    end: int,
    *,
    include_leading_return: bool = False,
) -> tuple[float, float]:
    """Return log-return and log-level correlation in a price window."""
    y = log_prices[int(spec.y), start:end]
    x = log_prices[int(spec.x), start:end]
    if include_leading_return and start > 0:
        y = log_prices[int(spec.y), start - 1 : end]
        x = log_prices[int(spec.x), start - 1 : end]
    return safe_corr(np.diff(y), np.diff(x)), safe_corr(y, x)


def shares_from_agent(
    agent: strategy.PairAgent,
    signal_prices: np.ndarray,
    previous_prices: np.ndarray | None,
) -> tuple[int, int]:
    """Apply the current strategy's single-pair bang-bang allocation."""
    pos_y, pos_x, _ = agent.dollar_positions()
    dollars_y = LEG * np.sign(pos_y)
    dollars_x = LEG * np.sign(pos_x)

    if previous_prices is not None:
        for index, is_y in ((int(agent.spec.y), True), (int(agent.spec.x), False)):
            move = abs(
                np.log(
                    max(float(signal_prices[index]), strategy._EPS_DENOM)
                    / max(float(previous_prices[index]), strategy._EPS_DENOM)
                )
            )
            if move > 0.40:
                if is_y:
                    dollars_y = 0.0
                else:
                    dollars_x = 0.0

    y_price = max(float(signal_prices[int(agent.spec.y)]), strategy._EPS_DENOM)
    x_price = max(float(signal_prices[int(agent.spec.x)]), strategy._EPS_DENOM)
    y_shares = int(np.round(dollars_y / y_price))
    x_shares = int(np.round(dollars_x / x_price))

    # Mirror the evaluator's integer-share $10k leg caps.
    y_limit = int(LEG / y_price)
    x_limit = int(LEG / x_price)
    return (
        int(np.clip(y_shares, -y_limit, y_limit)),
        int(np.clip(x_shares, -x_limit, x_limit)),
    )


def simulate_fold(
    prices: np.ndarray,
    spec: strategy.PairCfg,
    start: int,
    end: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Replay one fixed pair causally over [start, end), starting flat."""
    agent = strategy.PairAgent(spec)

    # Warm through the bar before the first signal bar, then flatten only the
    # execution state. Filter, CUSUM, beta and OU estimates remain causal.
    warm_end = max(0, start - 1)
    if warm_end:
        agent.catch_up(prices[:, :warm_end])
    agent.side = 0

    held_y = 0
    held_x = 0
    previous_side = 0
    entries = 0
    exits = 0
    turnover = 0.0
    records: list[dict[str, Any]] = []

    for target_index in range(start, end):
        signal_index = target_index - 1
        agent.catch_up(prices[:, : signal_index + 1])

        signal_prices = prices[:, signal_index]
        previous_prices = prices[:, signal_index - 1] if signal_index > 0 else None
        target_y, target_x = shares_from_agent(
            agent,
            signal_prices,
            previous_prices,
        )

        y_idx, x_idx = int(spec.y), int(spec.x)
        delta_y = target_y - held_y
        delta_x = target_x - held_x
        day_turnover = (
            abs(delta_y) * float(signal_prices[y_idx])
            + abs(delta_x) * float(signal_prices[x_idx])
        )
        fee = COMMISSION * day_turnover
        pnl = (
            target_y
            * (float(prices[y_idx, target_index]) - float(signal_prices[y_idx]))
            + target_x
            * (float(prices[x_idx, target_index]) - float(signal_prices[x_idx]))
            - fee
        )

        if previous_side == 0 and agent.side != 0:
            entries += 1
        if previous_side != 0 and agent.side == 0:
            exits += 1
        previous_side = int(agent.side)
        held_y, held_x = target_y, target_x
        turnover += day_turnover
        records.append(
            {
                "day": target_index + 1,
                "pnl": float(pnl),
                "cumulative_pnl": 0.0,
                "side": int(agent.side),
                "z": float(agent.current_z),
                "beta": float(agent.filt.beta),
                "y_shares": target_y,
                "x_shares": target_x,
                "turnover": float(day_turnover),
            }
        )

    detail = pd.DataFrame(records)
    if not detail.empty:
        detail["cumulative_pnl"] = detail["pnl"].cumsum()
    metrics = algothon_metrics(detail["pnl"].to_numpy(dtype=float))
    metrics.update(
        {
            "entries": float(entries),
            "exits": float(exits),
            "active_days": float((detail["side"] != 0).sum()),
            "turnover": float(turnover),
        }
    )
    return metrics, detail


def rolling_correlation(
    log_prices: np.ndarray,
    spec: strategy.PairCfg,
) -> pd.DataFrame:
    y_returns = np.diff(log_prices[int(spec.y)])
    x_returns = np.diff(log_prices[int(spec.x)])
    corr = (
        pd.Series(y_returns)
        .rolling(ROLLING_CORR_WINDOW, min_periods=ROLLING_CORR_WINDOW)
        .corr(pd.Series(x_returns))
    )
    return pd.DataFrame(
        {
            "day": np.arange(2, log_prices.shape[1] + 1),
            "rolling_return_corr": corr.to_numpy(dtype=float),
        }
    )


def dominant_sign(values: list[float]) -> int:
    signs = np.sign(np.asarray(values, dtype=float))
    positive = int((signs > 0).sum())
    negative = int((signs < 0).sum())
    return 1 if positive >= negative else -1


def build_report(
    summary: pd.DataFrame,
    fold_detail: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Seven restored pairs: correlation and singular walk-forward validation",
        "",
        "## Method",
        "",
        "- Exact fixed profiles currently active in `four_sigmas.py`.",
        "- Six sequential 125-day forward folds after an initial 250 days.",
        "- Each fold starts flat; adaptive state is warmed using past data only.",
        "- Pair trades alone with current $10k bang-bang legs and 1bp commission.",
        "- Ridge, other pairs, conflict voting and portfolio netting are disabled.",
        "- DEV1-DEV4 are chronological stability replays, not unbiased parameter "
        "tests because the profiles were selected from data through day 750.",
        "- OOS1/OOS2 (days 751-1000) are the untouched holdout assessment.",
        "",
        "Correlation PASS requires |forward return correlation| >= "
        f"{MIN_ABS_FORWARD_CORR:.2f} in at least {MIN_CORR_PASS_FOLDS}/6 folds, "
        f"a common sign in at least {MIN_SIGN_CONSISTENT_FOLDS}/6 folds, and "
        f"at least {MIN_ROLLING_CORR_COVERAGE:.0%} of 60-day rolling windows "
        f"at |correlation| >= {MIN_ABS_FORWARD_CORR:.2f}.",
        "",
        "## Summary",
        "",
        "| Pair | Corr | Year corr (Y1/Y2/Y3/Y4) | Forward corr (6 folds) | "
        "60d coverage | Positive folds | DEV score | OOS score | OOS decay | "
        "OOS folds + | Full score | Result |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        correlations = "/".join(
            f"{value:.2f}"
            for value in fold_detail.loc[
                fold_detail["pair"] == row.pair,
                "forward_return_corr",
            ]
        )
        lines.append(
            f"| {row.pair} | {row.correlation_status} | "
            f"{row.year1_return_corr:.2f}/{row.year2_return_corr:.2f}/"
            f"{row.year3_return_corr:.2f}/{row.year4_return_corr:.2f} | "
            f"{correlations} | {row.rolling_abs_corr_coverage:.0%} | "
            f"{row.positive_folds}/6 | {row.dev_score:.2f} | "
            f"{row.oos_score:.2f} | {row.oos_score_change_pct:+.0f}% | "
            f"{row.oos_positive_folds}/2 | {row.full_score:.2f} | "
            f"{row.overall_status} |"
        )

    lines.extend(
        [
            "",
            "## Fold detail",
            "",
            "| Pair | Fold | Days | Trailing corr | Forward corr | Level corr | "
            "Score | Mean PnL | Sharpe | Entries | Active days |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fold_detail.itertuples(index=False):
        lines.append(
            f"| {row.pair} | {row.fold} | {row.forward_start_day}-"
            f"{row.forward_end_day} | {row.trailing_return_corr:.3f} | "
            f"{row.forward_return_corr:.3f} | {row.forward_level_corr:.3f} | "
            f"{row.score:.2f} | {row.mean_pnl:.2f} | {row.sharpe:.2f} | "
            f"{int(row.entries)} | {int(row.active_days)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Return correlation is the primary co-movement check. Log-price level "
            "correlation is shown because it is intuitive, but it can be spurious.",
            "- A profitable pair can fail correlation, which means it is behaving "
            "more like an idiosyncratic mean-reversion trade than a reliable pair.",
            "- No pair is removed automatically by this diagnostic.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def make_plot(
    summary: pd.DataFrame,
    rolling: pd.DataFrame,
    daily: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(TARGET_PAIRS), 2, figsize=(15, 3.2 * len(TARGET_PAIRS)))
    colors = {
        "PASS": "#2a9d8f",
        "FAIL": "#e76f51",
    }
    for row_index, pair in enumerate(TARGET_PAIRS):
        corr_ax, pnl_ax = axes[row_index]
        pair_corr = rolling.loc[rolling["pair"] == pair]
        corr_ax.plot(
            pair_corr["day"],
            pair_corr["rolling_return_corr"],
            color="#457b9d",
            linewidth=1.2,
        )
        corr_ax.axhline(MIN_ABS_FORWARD_CORR, color="grey", linestyle="--", linewidth=0.8)
        corr_ax.axhline(-MIN_ABS_FORWARD_CORR, color="grey", linestyle="--", linewidth=0.8)
        corr_ax.axhline(0.0, color="black", linewidth=0.7)
        corr_ax.axvline(750, color="#e63946", linestyle=":", linewidth=1.1)
        corr_ax.set_ylim(-1.0, 1.0)
        corr_ax.set_ylabel("60d return corr")

        result = summary.loc[summary["pair"] == pair].iloc[0]
        corr_ax.set_title(
            f"{pair}: correlation {result['correlation_status']}",
            color=colors[result["correlation_status"]],
        )

        pair_daily = daily.loc[daily["pair"] == pair].copy()
        pair_daily["walkforward_cumulative_pnl"] = pair_daily["pnl"].cumsum()
        pnl_ax.plot(
            pair_daily["day"],
            pair_daily["walkforward_cumulative_pnl"],
            color="#386641",
            linewidth=1.4,
        )
        pnl_ax.axhline(0.0, color="black", linewidth=0.7)
        pnl_ax.axvline(750, color="#e63946", linestyle=":", linewidth=1.1)
        pnl_ax.set_ylabel("Cumulative PnL ($)")
        pnl_ax.set_title(
            f"Singular walk-forward: {result['overall_status']}",
            color=colors[result["overall_status"]],
        )

    for ax in axes[-1]:
        ax.set_xlabel("Day")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices_frame = pd.read_csv(ROOT / "prices.txt", sep=r"\s+")
    prices = prices_frame.to_numpy(dtype=float).T
    if prices.shape != (strategy.N_INST, 1000):
        raise RuntimeError(f"Unexpected price matrix shape: {prices.shape}")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0.0):
        raise RuntimeError("Prices must be finite and positive")
    log_prices = np.log(prices)
    specs = target_specs()

    fold_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    rolling_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for name in TARGET_PAIRS:
        spec = specs[name]
        pair_fold_rows: list[dict[str, Any]] = []
        pair_daily_frames: list[pd.DataFrame] = []

        rolling = rolling_correlation(log_prices, spec)
        rolling.insert(0, "pair", name)
        rolling_frames.append(rolling)
        finite_rolling = rolling["rolling_return_corr"].dropna().to_numpy(dtype=float)
        yearly_return_corrs = [
            correlation_window(log_prices, spec, start, start + 250)[0]
            for start in (0, 250, 500, 750)
        ]

        for fold_name, start, end in FOLDS:
            trailing_start = start - 250
            trailing_return_corr, trailing_level_corr = correlation_window(
                log_prices,
                spec,
                trailing_start,
                start,
            )
            forward_return_corr, forward_level_corr = correlation_window(
                log_prices,
                spec,
                start,
                end,
                include_leading_return=True,
            )
            metrics, day_detail = simulate_fold(prices, spec, start, end)
            day_detail.insert(0, "fold", fold_name)
            day_detail.insert(0, "pair", name)
            pair_daily_frames.append(day_detail)
            row = {
                "pair": name,
                "fold": fold_name,
                "is_oos": fold_name.startswith("OOS"),
                "trailing_start_day": trailing_start + 1,
                "trailing_end_day": start,
                "forward_start_day": start + 1,
                "forward_end_day": end,
                "trailing_return_corr": trailing_return_corr,
                "trailing_level_corr": trailing_level_corr,
                "forward_return_corr": forward_return_corr,
                "forward_level_corr": forward_level_corr,
                "corr_strength_pass": abs(forward_return_corr)
                >= MIN_ABS_FORWARD_CORR,
                **metrics,
            }
            fold_rows.append(row)
            pair_fold_rows.append(row)

        pair_daily = pd.concat(pair_daily_frames, ignore_index=True)
        daily_frames.append(pair_daily)
        pair_folds = pd.DataFrame(pair_fold_rows)
        forward_correlations = pair_folds["forward_return_corr"].tolist()
        sign = dominant_sign(forward_correlations)
        sign_consistent_folds = int(
            (np.sign(pair_folds["forward_return_corr"]) == sign).sum()
        )
        corr_pass_folds = int(pair_folds["corr_strength_pass"].sum())
        rolling_abs_coverage = float(
            np.mean(np.abs(finite_rolling) >= MIN_ABS_FORWARD_CORR)
        )
        rolling_sign_consistency = float(
            np.mean(np.sign(finite_rolling) == sign)
        )
        correlation_pass = (
            corr_pass_folds >= MIN_CORR_PASS_FOLDS
            and sign_consistent_folds >= MIN_SIGN_CONSISTENT_FOLDS
            and rolling_abs_coverage >= MIN_ROLLING_CORR_COVERAGE
        )

        dev_daily = pair_daily.loc[
            pair_daily["fold"].str.startswith("DEV"), "pnl"
        ].to_numpy(dtype=float)
        oos_daily = pair_daily.loc[
            pair_daily["fold"].str.startswith("OOS"), "pnl"
        ].to_numpy(dtype=float)
        full_metrics = algothon_metrics(pair_daily["pnl"].to_numpy(dtype=float))
        dev_metrics = algothon_metrics(dev_daily)
        oos_metrics = algothon_metrics(oos_daily)
        positive_folds = int((pair_folds["score"] > 0.0).sum())
        oos_positive_folds = int(
            (
                pair_folds.loc[pair_folds["is_oos"], "score"]
                > 0.0
            ).sum()
        )
        trading_pass = (
            positive_folds >= 4
            and oos_positive_folds == 2
            and full_metrics["score"] > 0.0
            and oos_metrics["score"] > 0.0
        )
        oos_score_change_pct = (
            100.0
            * (oos_metrics["score"] - dev_metrics["score"])
            / abs(dev_metrics["score"])
            if abs(dev_metrics["score"]) > 1.0e-12
            else 0.0
        )
        summary_rows.append(
            {
                "pair": name,
                "correlation_status": "PASS" if correlation_pass else "FAIL",
                "trading_status": "PASS" if trading_pass else "FAIL",
                "overall_status": "PASS"
                if correlation_pass and trading_pass
                else "FAIL",
                "corr_pass_folds": corr_pass_folds,
                "sign_consistent_folds": sign_consistent_folds,
                "rolling_abs_corr_median": float(np.median(np.abs(finite_rolling))),
                "rolling_abs_corr_10pct": float(
                    np.quantile(np.abs(finite_rolling), 0.10)
                ),
                "rolling_abs_corr_coverage": rolling_abs_coverage,
                "rolling_sign_consistency": rolling_sign_consistency,
                "year1_return_corr": yearly_return_corrs[0],
                "year2_return_corr": yearly_return_corrs[1],
                "year3_return_corr": yearly_return_corrs[2],
                "year4_return_corr": yearly_return_corrs[3],
                "positive_folds": positive_folds,
                "oos_positive_folds": oos_positive_folds,
                "dev_score": dev_metrics["score"],
                "dev_total_pnl": dev_metrics["total_pnl"],
                "oos_score": oos_metrics["score"],
                "oos_total_pnl": oos_metrics["total_pnl"],
                "oos_score_change_pct": oos_score_change_pct,
                "full_score": full_metrics["score"],
                "full_total_pnl": full_metrics["total_pnl"],
                "full_sharpe": full_metrics["sharpe"],
                "total_entries": int(pair_folds["entries"].sum()),
            }
        )

    fold_detail = pd.DataFrame(fold_rows)
    daily = pd.concat(daily_frames, ignore_index=True)
    rolling = pd.concat(rolling_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    fold_detail.to_csv(OUTPUT_DIR / "walkforward_fold_detail.csv", index=False)
    daily.to_csv(OUTPUT_DIR / "walkforward_daily_pnl.csv", index=False)
    rolling.to_csv(OUTPUT_DIR / "rolling_return_correlation.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "walkforward_summary.csv", index=False)
    (OUTPUT_DIR / "validation_config.json").write_text(
        json.dumps(
            {
                "target_pairs": TARGET_PAIRS,
                "folds": FOLDS,
                "leg": LEG,
                "commission": COMMISSION,
                "rolling_corr_window": ROLLING_CORR_WINDOW,
                "min_abs_forward_corr": MIN_ABS_FORWARD_CORR,
                "min_corr_pass_folds": MIN_CORR_PASS_FOLDS,
                "min_sign_consistent_folds": MIN_SIGN_CONSISTENT_FOLDS,
                "min_rolling_corr_coverage": MIN_ROLLING_CORR_COVERAGE,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    build_report(summary, fold_detail, OUTPUT_DIR / "walkforward_report.md")
    make_plot(
        summary,
        rolling,
        daily,
        OUTPUT_DIR / "walkforward_correlation_and_pnl.png",
    )

    print(
        summary[
            [
                "pair",
                "correlation_status",
                "corr_pass_folds",
                "sign_consistent_folds",
                "rolling_abs_corr_coverage",
                "positive_folds",
                "oos_positive_folds",
                "dev_score",
                "oos_score",
                "oos_score_change_pct",
                "full_score",
                "overall_status",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

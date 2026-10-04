#!/usr/bin/env python
"""Explain IS-to-OOS decay for the Discord-approved top30 TestbedV2 book.

The approved set is treated as fixed. Days 751-1000 are evaluated only; this
script does not select pairs or alter parameters using OOS results.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.shortlist_pair_review import (
    account_window,
    load_prices,
    load_strategy,
    official_score,
)

STRATEGY_MODULE = "top30_testbedv2"
APPROVALS_PATH = ROOT / "pair_shortlist_review" / "discord_approved_pairs.json"
OUTPUT_DIR = ROOT / "analysis_outputs"
WINDOWS = (
    ("year1", 31, 250),
    ("year2", 251, 500),
    ("year3", 501, 750),
    ("oos", 751, 1000),
)
END_DAY = 1000


def pair_name(cfg: Any, symbols: dict[int, str]) -> str:
    return f"{symbols[int(cfg.y)]}-{symbols[int(cfg.x)]}"


def live_pair_replays(
    prices: np.ndarray,
    strategy: Any,
    symbols: dict[int, str],
) -> dict[str, dict[str, Any]]:
    """Replay each pair alone using the live getMyPosition wrapper semantics."""
    n_instruments = prices.shape[0]
    warmup = int(strategy.WARMUP) + 10
    eps = float(strategy._EPS_DENOM)
    outputs: dict[str, dict[str, Any]] = {}

    for cfg in strategy.PAIR_CFGS:
        name = pair_name(cfg, symbols)
        agent = strategy.PairAgent(cfg)
        targets = np.zeros((END_DAY + 1, n_instruments), dtype=int)
        z = np.zeros(END_DAY, dtype=float)
        kappa = np.zeros(END_DAY, dtype=float)
        sigma = np.zeros(END_DAY, dtype=float)
        entries: list[int] = []
        cuts: list[int] = []
        previous_side = 0
        previous_segment = None

        for day in range(1, END_DAY + 1):
            agent.catch_up(prices[:, :day])
            z[day - 1] = agent.current_z
            kappa[day - 1] = agent.current_kappa
            sigma[day - 1] = agent.current_sigma

            segment = agent.filt.seg_start
            if (
                day > warmup
                and previous_segment is not None
                and segment != previous_segment
            ):
                cuts.append(day)
            previous_segment = segment

            effective_side = agent.side if day >= warmup else 0
            if previous_side == 0 and effective_side != 0:
                entries.append(day)
            previous_side = effective_side

            if day < warmup:
                continue

            py, px, _ = agent.dollar_positions()
            dollars = np.zeros(n_instruments, dtype=float)
            dollars[int(cfg.y)] = py
            dollars[int(cfg.x)] = px
            if day >= 2:
                last_return = np.diff(
                    np.log(np.maximum(prices[:, day - 2 : day], eps)),
                    axis=1,
                )[:, 0]
                crash = np.abs(last_return) > 0.40
                crash[0] = False
                dollars[crash] = 0.0
            dollars[1:] = np.clip(dollars[1:], -strategy.LEG, strategy.LEG)
            # Match top30_testbedv2.getMyPosition exactly.
            dollars[0] = 0.0
            targets[day] = np.round(
                dollars / np.maximum(prices[:, day - 1], eps)
            ).astype(int)

        outputs[name] = {
            "cfg": cfg,
            "targets": targets,
            "z": z,
            "kappa": kappa,
            "sigma": sigma,
            "entries": entries,
            "cuts": cuts,
        }
    return outputs


def trace_book_selection(
    prices: np.ndarray,
    strategy: Any,
    symbols: dict[int, str],
) -> dict[str, dict[str, Any]]:
    """Trace candidate, selected, and blocked days under AtlasBook arbitration."""
    n_instruments = prices.shape[0]
    n_pairs = len(strategy.PAIR_CFGS)
    warmup = int(strategy.WARMUP) + 10
    eps = float(strategy._EPS_DENOM)
    agents = [strategy.PairAgent(cfg) for cfg in strategy.PAIR_CFGS]

    targets = [
        np.zeros((END_DAY + 1, n_instruments), dtype=int)
        for _ in range(n_pairs)
    ]
    candidates = np.zeros((n_pairs, END_DAY + 1), dtype=bool)
    selected = np.zeros((n_pairs, END_DAY + 1), dtype=bool)

    for day in range(warmup, END_DAY + 1):
        history = prices[:, :day]
        for agent in agents:
            agent.catch_up(history)

        ranked: list[tuple[float, int, Any, float, float]] = []
        for index, agent in enumerate(agents):
            py, px, abs_z = agent.dollar_positions()
            if abs(py) < 1.0e-9:
                continue
            candidates[index, day] = True
            ranked.append(
                (agent.current_kappa * abs_z, index, agent, py, px)
            )
        ranked.sort(key=lambda item: item[0], reverse=True)

        taken: set[int] = set()
        for _, index, agent, py, px in ranked:
            y_index = int(agent.spec.y)
            x_index = int(agent.spec.x)
            if y_index in taken or x_index in taken:
                continue
            taken.update((y_index, x_index))
            selected[index, day] = True

            dollars = np.zeros(n_instruments, dtype=float)
            dollars[y_index] = py
            dollars[x_index] = px
            if day >= 2:
                last_return = np.diff(
                    np.log(np.maximum(prices[:, day - 2 : day], eps)),
                    axis=1,
                )[:, 0]
                crash = np.abs(last_return) > 0.40
                crash[0] = False
                dollars[crash] = 0.0
            dollars[1:] = np.clip(dollars[1:], -strategy.LEG, strategy.LEG)
            dollars[0] = 0.0
            targets[index][day] = np.round(
                dollars / np.maximum(prices[:, day - 1], eps)
            ).astype(int)

    output: dict[str, dict[str, Any]] = {}
    for index, cfg in enumerate(strategy.PAIR_CFGS):
        output[pair_name(cfg, symbols)] = {
            "targets": targets[index],
            "candidate": candidates[index],
            "selected": selected[index],
        }
    return output


def price_correlation(
    prices: np.ndarray,
    y_index: int,
    x_index: int,
    start_day: int,
    end_day: int,
) -> float:
    block = np.log(prices[[y_index, x_index], start_day - 1 : end_day])
    returns = np.diff(block, axis=1)
    if min(float(returns[0].std()), float(returns[1].std())) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(returns)[0, 1])


def max_drawdown(daily_pnl: np.ndarray) -> float:
    cumulative = np.cumsum(daily_pnl)
    if cumulative.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(np.r_[0.0, cumulative])
    drawdowns = np.r_[0.0, cumulative] - running_peak
    return float(drawdowns.min())


def window_record(
    prices: np.ndarray,
    item: dict[str, Any],
    book_item: dict[str, Any],
    label: str,
    start_day: int,
    end_day: int,
) -> dict[str, float | int]:
    cfg = item["cfg"]
    metrics = account_window(
        prices, item["targets"], start_day, end_day
    )
    book_metrics = account_window(
        prices, book_item["targets"], start_day, end_day
    )
    day_slice = slice(start_day, end_day + 1)
    candidate_days = int(book_item["candidate"][day_slice].sum())
    selected_days = int(book_item["selected"][day_slice].sum())
    active_days = int(
        np.any(item["targets"][day_slice] != 0, axis=1).sum()
    )
    entries = sum(start_day <= day <= end_day for day in item["entries"])
    cuts = sum(start_day <= day <= end_day for day in item["cuts"])
    series_slice = slice(start_day - 1, end_day)

    return {
        f"{label}_score": float(metrics["score"]),
        f"{label}_mean_pnl": float(metrics["mean_pnl"]),
        f"{label}_sharpe": float(metrics["sharpe"]),
        f"{label}_total_pnl": float(metrics["total_pnl"]),
        f"{label}_max_drawdown": max_drawdown(metrics["daily_pnl"]),
        f"{label}_entries": entries,
        f"{label}_active_days": active_days,
        f"{label}_cuts": cuts,
        f"{label}_corr": price_correlation(
            prices, int(cfg.y), int(cfg.x), start_day, end_day
        ),
        f"{label}_mean_kappa": float(np.mean(item["kappa"][series_slice])),
        f"{label}_mean_sigma": float(np.mean(item["sigma"][series_slice])),
        f"{label}_z_std": float(np.std(item["z"][series_slice])),
        f"{label}_candidate_days": candidate_days,
        f"{label}_selected_days": selected_days,
        f"{label}_blocked_rate": (
            1.0 - selected_days / candidate_days
            if candidate_days
            else 0.0
        ),
        f"{label}_book_score": float(book_metrics["score"]),
        f"{label}_book_total_pnl": float(book_metrics["total_pnl"]),
    }


def diagnose(row: pd.Series) -> str:
    reasons: list[str] = []
    is_entries = float(row["is_median_entries"])
    is_cuts = float(row["is_median_cuts"])

    if bool(row["live_unhedged"]):
        reasons.append("ALGO leg disabled live")
    if row["oos_marginal_score"] < -2.0:
        reasons.append("removal improves live OOS book")
    elif row["oos_marginal_score"] > 2.0:
        reasons.append("adds material live-book score")
    if row["oos_total_pnl"] <= 0.0:
        reasons.append("standalone signal reversed")
    elif row["oos_retention"] < 0.40:
        reasons.append("standalone edge weakened")
    elif row["oos_retention"] > 1.10:
        reasons.append("standalone edge improved")
    else:
        reasons.append("standalone edge broadly retained")

    if row["oos_entries"] < 0.60 * max(is_entries, 1.0):
        reasons.append("trade drought")
    elif row["oos_entries"] > 1.50 * max(is_entries, 1.0):
        reasons.append("more signals")

    if row["oos_mean_kappa"] < 0.70 * max(row["is_mean_kappa"], 1.0e-9):
        reasons.append("slower reversion")
    if row["oos_mean_sigma"] > 1.30 * max(row["is_mean_sigma"], 1.0e-9):
        reasons.append("spread volatility rose")
    if row["oos_cuts"] > max(1.5 * is_cuts, is_cuts + 2.0):
        reasons.append("more CUSUM breaks")
    if max(abs(row["corr_shift"]), abs(row["corr_shift_vs_y3"])) >= 0.15:
        reasons.append("price correlation shifted")
    if row["blocked_rate_change_y3_to_oos"] >= 0.20:
        reasons.append("book conflicts worsened")
    if row["oos_blocked_rate"] >= 0.50:
        reasons.append("mostly blocked by symbol conflicts")
    elif row["oos_blocked_rate"] >= 0.25:
        reasons.append("material book conflicts")
    return "; ".join(reasons)


def counterfactual_book_targets(
    prices: np.ndarray,
    strategy: Any,
    symbols: dict[int, str],
    standalone: dict[str, dict[str, Any]],
    book: dict[str, dict[str, Any]],
    *,
    omit: str | None = None,
) -> np.ndarray:
    """Re-run daily arbitration from cached pair signals, optionally omitting one."""
    targets = np.zeros((END_DAY + 1, prices.shape[0]), dtype=int)
    names = [pair_name(cfg, symbols) for cfg in strategy.PAIR_CFGS]
    for day in range(int(strategy.WARMUP) + 10, END_DAY + 1):
        ranked: list[tuple[float, int, str]] = []
        for index, name in enumerate(names):
            if name == omit or not book[name]["candidate"][day]:
                continue
            item = standalone[name]
            priority = float(
                item["kappa"][day - 1] * abs(item["z"][day - 1])
            )
            ranked.append((priority, index, name))
        ranked.sort(key=lambda item: item[0], reverse=True)

        taken: set[int] = set()
        for _, _, name in ranked:
            cfg = standalone[name]["cfg"]
            y_index, x_index = int(cfg.y), int(cfg.x)
            if y_index in taken or x_index in taken:
                continue
            taken.update((y_index, x_index))
            targets[day] += standalone[name]["targets"][day]
    return targets


def portfolio_metrics(
    prices: np.ndarray,
    strategy: Any,
    start_day: int,
    end_day: int,
) -> dict[str, float]:
    strategy.reset()
    targets = np.zeros((END_DAY + 1, prices.shape[0]), dtype=int)
    for day in range(1, end_day + 1):
        targets[day] = strategy.getMyPosition(prices[:, :day])
    metrics = account_window(prices, targets, start_day, end_day)
    return {
        "score": float(metrics["score"]),
        "mean_pnl": float(metrics["mean_pnl"]),
        "std_pnl": float(metrics["std_pnl"]),
        "sharpe": float(metrics["sharpe"]),
        "total_pnl": float(metrics["total_pnl"]),
    }


def make_plot(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("oos_retention")
    colors = np.where(
        ordered["oos_score"] < 0.0,
        "#c44e52",
        np.where(ordered["oos_retention"] < 0.5, "#dd8452", "#55a868"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(17, 12))
    axes[0].barh(ordered["pair_name"], ordered["oos_retention"], color=colors)
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].axvline(0.5, color="grey", linestyle=":", linewidth=1)
    axes[0].set_title("Standalone OOS Score / Median IS-Year Score")
    axes[0].set_xlabel("Score retention")
    axes[0].grid(axis="x", alpha=0.25)

    blocked = frame.sort_values("oos_blocked_rate")
    axes[1].barh(
        blocked["pair_name"],
        blocked["oos_blocked_rate"],
        color="#4c72b0",
    )
    axes[1].axvline(0.5, color="grey", linestyle=":", linewidth=1)
    axes[1].set_title("OOS Candidate Days Blocked by Book Conflicts")
    axes[1].set_xlabel("Blocked fraction")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    portfolio: dict[str, dict[str, float]],
    symbol_counts: Counter[str],
    path: Path,
) -> None:
    oos = portfolio["oos"]
    train = portfolio["full_is"]
    drop = oos["score"] / train["score"] - 1.0
    negative = int((frame["oos_total_pnl"] <= 0.0).sum())
    weak = int((frame["oos_retention"] < 0.5).sum())
    blocked = int((frame["oos_blocked_rate"] >= 0.5).sum())
    harmful = int((frame["oos_marginal_score"] < 0.0).sum())
    year3_candidates = float(frame["year3_candidate_days"].sum())
    year3_selected = float(frame["year3_selected_days"].sum())
    oos_candidates = float(frame["oos_candidate_days"].sum())
    oos_selected = float(frame["oos_selected_days"].sum())
    lines = [
        "# Top30 TestbedV2 OOS decay analysis",
        "",
        "The approved pair set and parameters are fixed. Days 751-1000 are "
        "used only for evaluation.",
        "",
        "## Portfolio result",
        "",
        f"- Full IS (days 31-750) score: **{train['score']:.2f}**",
        f"- OOS (days 751-1000) score: **{oos['score']:.2f}**",
        f"- Score change: **{drop:.1%}**",
        f"- IS mean daily P&L: **${train['mean_pnl']:.2f}**",
        f"- OOS mean daily P&L: **${oos['mean_pnl']:.2f}**",
        f"- IS/OOS daily P&L volatility: **${train['std_pnl']:.2f} / "
        f"${oos['std_pnl']:.2f}**",
        "",
        "## Cross-sectional findings",
        "",
        f"- {negative}/{len(frame)} pairs have non-positive standalone OOS P&L.",
        f"- {weak}/{len(frame)} retain less than half their median IS-year score.",
        f"- {blocked}/{len(frame)} are blocked on at least half their OOS "
        "candidate days by higher-priority pairs sharing a symbol.",
        f"- {harmful}/{len(frame)} have negative leave-one-out OOS score "
        "contribution: removing them improves the actual book.",
        "- Aggregate conflict blocking was already high and changed only from "
        f"{1.0 - year3_selected / year3_candidates:.1%} in year 3 to "
        f"{1.0 - oos_selected / oos_candidates:.1%} OOS.",
        f"- Pair concentration: RTTH appears in {symbol_counts['RTTH']} pairs; "
        f"NAYO appears in {symbol_counts['NAYO']} pairs.",
        "- ALGO-NAYO and ALGO-RCRI were reviewed with two legs, but the live "
        "wrapper zeros instrument 0 (ALGO), leaving one-legged exposure.",
        "",
        "## Pair-by-pair results",
        "",
        "| Pair | IS median score | OOS score | Retention | OOS P&L | "
        "OOS blocked | OOS marginal score | Book P&L change vs Y3 | Diagnosis |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame.sort_values("oos_retention").itertuples(index=False):
        lines.append(
            f"| {row.pair_name} | {row.is_median_score:.2f} | "
            f"{row.oos_score:.2f} | {row.oos_retention:.1%} | "
            f"${row.oos_total_pnl:,.0f} | {row.oos_blocked_rate:.1%} | "
            f"{row.oos_marginal_score:+.2f} | "
            f"${row.book_pnl_change_y3_to_oos:,.0f} | "
            f"{row.diagnosis} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy = load_strategy(STRATEGY_MODULE)
    prices, price_symbols = load_prices(ROOT / "prices.txt")
    symbols = {index: symbol for index, symbol in enumerate(price_symbols)}
    approvals = json.loads(APPROVALS_PATH.read_text(encoding="utf-8"))
    approved_names = [str(row["pair_name"]) for row in approvals]
    configured_names = [
        pair_name(cfg, symbols) for cfg in strategy.PAIR_CFGS
    ]
    if configured_names != approved_names:
        raise ValueError(
            "top30_testbedv2 active configs do not exactly match Discord approvals"
        )

    standalone = live_pair_replays(prices, strategy, symbols)
    book = trace_book_selection(prices, strategy, symbols)
    records: list[dict[str, Any]] = []
    for name in approved_names:
        item = standalone[name]
        cfg = item["cfg"]
        record: dict[str, Any] = {
            "pair_name": name,
            "y_symbol": symbols[int(cfg.y)],
            "x_symbol": symbols[int(cfg.x)],
            "live_unhedged": int(cfg.y) == 0 or int(cfg.x) == 0,
        }
        for label, start_day, end_day in WINDOWS:
            record.update(
                window_record(
                    prices,
                    item,
                    book[name],
                    label,
                    start_day,
                    end_day,
                )
            )

        is_scores = [record[f"year{i}_score"] for i in range(1, 4)]
        is_entries = [record[f"year{i}_entries"] for i in range(1, 4)]
        is_cuts = [record[f"year{i}_cuts"] for i in range(1, 4)]
        record["is_median_score"] = float(median(is_scores))
        record["is_median_entries"] = float(median(is_entries))
        record["is_median_cuts"] = float(median(is_cuts))
        record["is_mean_corr"] = float(
            np.nanmean([record[f"year{i}_corr"] for i in range(1, 4)])
        )
        record["is_mean_kappa"] = float(
            np.mean([record[f"year{i}_mean_kappa"] for i in range(1, 4)])
        )
        record["is_mean_sigma"] = float(
            np.mean([record[f"year{i}_mean_sigma"] for i in range(1, 4)])
        )
        record["oos_retention"] = (
            record["oos_score"] / max(abs(record["is_median_score"]), 1.0e-12)
        )
        record["corr_shift"] = record["oos_corr"] - record["is_mean_corr"]
        record["score_drop"] = record["oos_score"] - record["is_median_score"]
        records.append(record)

    portfolio = {
        "year1": portfolio_metrics(prices, strategy, 31, 250),
        "year2": portfolio_metrics(prices, strategy, 251, 500),
        "year3": portfolio_metrics(prices, strategy, 501, 750),
        "full_is": portfolio_metrics(prices, strategy, 31, 750),
        "oos": portfolio_metrics(prices, strategy, 751, 1000),
    }
    frame = pd.DataFrame.from_records(records)
    base_targets = counterfactual_book_targets(
        prices, strategy, symbols, standalone, book
    )
    base_year3 = account_window(prices, base_targets, 501, 750)
    base_oos = account_window(prices, base_targets, 751, 1000)
    if not np.isclose(
        float(base_oos["score"]), portfolio["oos"]["score"], atol=1.0e-9
    ):
        raise AssertionError("Cached book arbitration does not match live strategy")

    marginal_records: dict[str, dict[str, float]] = {}
    for name in approved_names:
        without_targets = counterfactual_book_targets(
            prices,
            strategy,
            symbols,
            standalone,
            book,
            omit=name,
        )
        without_year3 = account_window(prices, without_targets, 501, 750)
        without_oos = account_window(prices, without_targets, 751, 1000)
        marginal_records[name] = {
            "year3_marginal_score": (
                float(base_year3["score"]) - float(without_year3["score"])
            ),
            "oos_marginal_score": (
                float(base_oos["score"]) - float(without_oos["score"])
            ),
            "year3_marginal_pnl": (
                float(base_year3["total_pnl"])
                - float(without_year3["total_pnl"])
            ),
            "oos_marginal_pnl": (
                float(base_oos["total_pnl"]) - float(without_oos["total_pnl"])
            ),
        }
    for column in (
        "year3_marginal_score",
        "oos_marginal_score",
        "year3_marginal_pnl",
        "oos_marginal_pnl",
    ):
        frame[column] = frame["pair_name"].map(
            lambda name: marginal_records[str(name)][column]
        )
    frame["book_pnl_change_y3_to_oos"] = (
        frame["oos_book_total_pnl"] - frame["year3_book_total_pnl"]
    )
    frame["blocked_rate_change_y3_to_oos"] = (
        frame["oos_blocked_rate"] - frame["year3_blocked_rate"]
    )
    frame["corr_shift_vs_y3"] = frame["oos_corr"] - frame["year3_corr"]
    frame["diagnosis"] = frame.apply(diagnose, axis=1)
    csv_path = OUTPUT_DIR / "top30_oos_pair_analysis.csv"
    frame.sort_values("score_drop").to_csv(csv_path, index=False)
    symbol_counts: Counter[str] = Counter()
    for name in approved_names:
        symbol_counts.update(name.split("-"))

    report_path = OUTPUT_DIR / "top30_oos_decay_report.md"
    write_report(frame, portfolio, symbol_counts, report_path)
    plot_path = OUTPUT_DIR / "top30_oos_pair_diagnostics.png"
    make_plot(frame, plot_path)
    summary_path = OUTPUT_DIR / "top30_oos_portfolio_metrics.json"
    summary_path.write_text(
        json.dumps(portfolio, indent=2), encoding="utf-8"
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")
    print(
        "Portfolio score: "
        f"{portfolio['full_is']['score']:.2f} IS -> "
        f"{portfolio['oos']['score']:.2f} OOS"
    )


if __name__ == "__main__":
    main()

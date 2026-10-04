#!/usr/bin/env python
"""Shortlist robust pair configurations and generate visual trade reviews.

The default universe comes from ``testbedv2_trainonly`` and the default
research period ends at day 750.  No stored database test/OOS metrics are read.

For every configuration the script:

1. replays the actual PairAgent logic;
2. evaluates days 31-250, 251-500, and 501-750 independently;
3. keeps pairs that are profitable and active across the requested regimes;
4. ranks survivors by a conservative worst/median/full-period score; and
5. writes one four-panel PNG per shortlisted pair plus CSV/JSON/HTML outputs.

Examples
--------
Headless shortlist and HTML gallery:

    py -B research/shortlist_pair_review.py

Visually review each shortlisted pair and approve/reject it:

    py -B research/shortlist_pair_review.py --review

Review the currently embedded testbedv2_tuned universe instead:

    py -B research/shortlist_pair_review.py --module testbedv2_tuned --review
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import types
import webbrowser
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "pair_shortlist_review"
DEFAULT_PRICES = ROOT / "prices.txt"
DEFAULT_WINDOWS = (
    ("year1", 31, 250),
    ("year2", 251, 500),
    ("year3", 501, 750),
)
FULL_WINDOW = ("full", 31, 750)


@dataclass
class Replay:
    target_shares: np.ndarray
    entries: list[tuple[int, str, float, int]]
    z: np.ndarray | None = None
    kappa: np.ndarray | None = None
    sigma: np.ndarray | None = None


def official_score(mean_pnl: float, std_pnl: float) -> float:
    if mean_pnl <= 0.0 or std_pnl < 1.0e-10:
        return mean_pnl
    sharpe = sqrt(250.0) * mean_pnl / std_pnl
    return mean_pnl * sharpe * sharpe / (sharpe * sharpe + 1.0)


def load_strategy(module_name: str) -> types.ModuleType:
    """Load source directly so stale ``.pyc`` files cannot mask local edits."""
    path = ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
    if not path.exists():
        raise FileNotFoundError(f"Strategy module does not exist: {path}")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = module_name.rpartition(".")[0]
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def load_prices(path: Path) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path, sep=r"\s+", header=0)
    values = frame.to_numpy(dtype=float).T
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("Prices must be finite and strictly positive")
    return values, [str(column) for column in frame.columns]


def replay_pair(
    prices: np.ndarray,
    cfg: Any,
    strategy: types.ModuleType,
    end_day: int,
    *,
    capture: bool,
) -> Replay:
    """Replay one PairAgent while retaining the official position convention."""
    n_instruments = prices.shape[0]
    target_shares = np.zeros((end_day + 1, n_instruments), dtype=int)
    z_hist = np.zeros(end_day, dtype=float) if capture else None
    kappa_hist = np.zeros(end_day, dtype=float) if capture else None
    sigma_hist = np.zeros(end_day, dtype=float) if capture else None
    entries: list[tuple[int, str, float, int]] = []

    agent = strategy.PairAgent(cfg)
    previous_effective_side = 0
    warmup = int(strategy.WARMUP) + 10
    leg = float(strategy.LEG)
    eps = float(getattr(strategy, "_EPS_DENOM", 1.0e-12))

    for day in range(1, end_day + 1):
        # Catching up every day gives the complete diagnostic path. Positions
        # remain zero until the same warmup used by getMyPosition.
        agent.catch_up(prices[:, :day])

        if capture:
            assert z_hist is not None
            assert kappa_hist is not None
            assert sigma_hist is not None
            z_hist[day - 1] = agent.current_z
            kappa_hist[day - 1] = agent.current_kappa
            sigma_hist[day - 1] = agent.current_sigma

        effective_side = agent.side if day >= warmup else 0
        if effective_side != previous_effective_side:
            event = "ENTRY" if previous_effective_side == 0 else "EXIT"
            entries.append((day, event, agent.current_z, effective_side))
        previous_effective_side = effective_side

        if day < warmup:
            continue

        py, px, _ = agent.dollar_positions()
        dollars = np.zeros(n_instruments, dtype=float)
        dollars[cfg.y] = py
        dollars[cfg.x] = px

        if day >= 2:
            last_return = np.diff(
                np.log(np.maximum(prices[:, day - 2 : day], eps)),
                axis=1,
            )[:, 0]
            crash = np.abs(last_return) > 0.40
            crash[0] = False
            dollars[crash] = 0.0

        dollars[1:] = np.clip(dollars[1:], -leg, leg)
        dollars[0] = np.clip(dollars[0], -10.0 * leg, 10.0 * leg)
        target_shares[day] = (
            dollars / np.maximum(prices[:, day - 1], eps)
        ).astype(int)

    return Replay(
        target_shares=target_shares,
        entries=entries,
        z=z_hist,
        kappa=kappa_hist,
        sigma=sigma_hist,
    )


def account_window(
    prices: np.ndarray,
    target_shares: np.ndarray,
    start_day: int,
    end_day: int,
) -> dict[str, Any]:
    """Apply the same position limits and delayed commission accounting as eval.py."""
    n_instruments = prices.shape[0]
    commission_rates = np.full(n_instruments, 1.0e-4)
    commission_rates[0] = 2.0e-5
    dollar_limits = np.full(n_instruments, 10_000.0)
    dollar_limits[0] = 100_000.0

    cash = 0.0
    current_position = np.zeros(n_instruments)
    total_volume = 0.0
    value = 0.0
    commission = 0.0
    daily_pnl: list[float] = []
    days: list[int] = []

    for day in range(start_day - 1, end_day + 1):
        current_prices = prices[:, day - 1]
        if day < end_day:
            limits = (dollar_limits / current_prices).astype(int)
            new_position = np.clip(
                target_shares[day], -limits, limits
            ).astype(float)
        else:
            new_position = current_position.copy()

        delta = new_position - current_position
        cash -= float(current_prices.dot(delta)) + commission
        daily_volume = current_prices * np.abs(delta)
        total_volume += float(daily_volume.sum())
        commission = float(np.sum(daily_volume * commission_rates))
        current_position = new_position

        new_value = cash + float(current_position.dot(current_prices))
        pnl = new_value - value
        value = new_value
        if day >= start_day:
            days.append(day)
            daily_pnl.append(pnl)

    pnl_values = np.asarray(daily_pnl, dtype=float)
    mean_pnl = float(pnl_values.mean())
    std_pnl = float(pnl_values.std())
    sharpe = sqrt(250.0) * mean_pnl / std_pnl if std_pnl > 0.0 else 0.0
    return {
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "sharpe": sharpe,
        "score": official_score(mean_pnl, std_pnl),
        "total_pnl": float(pnl_values.sum()),
        "return": value / total_volume if total_volume > 0.0 else 0.0,
        "dollar_volume": total_volume,
        "days": np.asarray(days, dtype=int),
        "daily_pnl": pnl_values,
    }


def pair_symbols(cfg: Any, inverse_symbols: dict[int, str]) -> tuple[str, str]:
    return inverse_symbols[int(cfg.y)], inverse_symbols[int(cfg.x)]


def evaluate_config(
    prices: np.ndarray,
    cfg: Any,
    strategy: types.ModuleType,
    inverse_symbols: dict[int, str],
    windows: tuple[tuple[str, int, int], ...],
    full_window: tuple[str, int, int],
) -> dict[str, Any]:
    end_day = max(end for _, _, end in (*windows, full_window))
    replay = replay_pair(prices, cfg, strategy, end_day, capture=False)
    y_symbol, x_symbol = pair_symbols(cfg, inverse_symbols)

    result: dict[str, Any] = {
        "pair_name": f"{y_symbol}-{x_symbol}",
        "y_symbol": y_symbol,
        "x_symbol": x_symbol,
        "q_beta": float(cfg.filt.q_beta),
        "h": float(cfg.filt.h),
        "cusum_k": float(cfg.filt.cusum_k),
        "max_cov_inflate": float(cfg.filt.max_cov_inflate),
        "cov_decay": float(cfg.filt.cov_decay),
        "z_entry": float(cfg.trade.z_entry),
        "z_exit": float(cfg.trade.z_exit),
        "z_stop": float(cfg.trade.z_stop),
    }

    year_scores: list[float] = []
    year_pnls: list[float] = []
    year_entries: list[int] = []
    for label, start_day, window_end in windows:
        metrics = account_window(
            prices, replay.target_shares, start_day, window_end
        )
        entry_count = sum(
            start_day <= day <= window_end and event == "ENTRY"
            for day, event, _, _ in replay.entries
        )
        result.update(
            {
                f"{label}_score": metrics["score"],
                f"{label}_mean_pnl": metrics["mean_pnl"],
                f"{label}_sharpe": metrics["sharpe"],
                f"{label}_total_pnl": metrics["total_pnl"],
                f"{label}_entries": entry_count,
            }
        )
        year_scores.append(float(metrics["score"]))
        year_pnls.append(float(metrics["total_pnl"]))
        year_entries.append(entry_count)

    full_label, full_start, full_end = full_window
    full_metrics = account_window(
        prices, replay.target_shares, full_start, full_end
    )
    result.update(
        {
            f"{full_label}_score": full_metrics["score"],
            f"{full_label}_mean_pnl": full_metrics["mean_pnl"],
            f"{full_label}_std_pnl": full_metrics["std_pnl"],
            f"{full_label}_sharpe": full_metrics["sharpe"],
            f"{full_label}_total_pnl": full_metrics["total_pnl"],
            f"{full_label}_return": full_metrics["return"],
            f"{full_label}_dollar_volume": full_metrics["dollar_volume"],
            "profitable_years": sum(value > 0.0 for value in year_pnls),
            "min_year_entries": min(year_entries),
            "worst_year_score": min(year_scores),
            "median_year_score": float(np.median(year_scores)),
            "best_year_score": max(year_scores),
        }
    )
    result["score_retention"] = result["worst_year_score"] / max(
        abs(result["best_year_score"]), 1.0e-12
    )
    result["robust_score"] = (
        0.50 * result["worst_year_score"]
        + 0.30 * result["median_year_score"]
        + 0.20 * result["full_score"]
    )
    return result


def select_shortlist(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    mask = (
        (frame["profitable_years"] >= args.min_profitable_years)
        & (frame["min_year_entries"] >= args.min_entries)
        & (frame["worst_year_score"] >= args.min_year_score)
        & (frame["full_score"] >= args.min_full_score)
        & (frame["score_retention"] >= args.min_score_retention)
    )
    passing = frame.loc[mask].sort_values(
        ["robust_score", "worst_year_score", "full_score"],
        ascending=False,
    )

    if not args.disjoint:
        return passing.head(args.max_pairs).copy()

    selected_indices: list[int] = []
    used_symbols: set[str] = set()
    for index, row in passing.iterrows():
        if row["y_symbol"] in used_symbols or row["x_symbol"] in used_symbols:
            continue
        selected_indices.append(index)
        used_symbols.update((row["y_symbol"], row["x_symbol"]))
        if len(selected_indices) >= args.max_pairs:
            break
    return passing.loc[selected_indices].copy()


def plot_diagnostic(
    prices: np.ndarray,
    replay: Replay,
    metrics: dict[str, Any],
    cfg: Any,
    pair_name: str,
    output_path: Path,
    *,
    show: bool,
) -> None:
    if replay.z is None or replay.kappa is None or replay.sigma is None:
        raise ValueError("Diagnostic replay requires capture=True")

    days = np.arange(1, len(replay.z) + 1)
    fig, (ax_price, ax1, ax2, ax3) = plt.subplots(
        4, 1, figsize=(12, 15), sharex=True
    )

    y_symbol, x_symbol = pair_name.split("-", maxsplit=1)
    y_prices = np.asarray(prices[int(cfg.y), : len(days)], dtype=float)
    x_prices = np.asarray(prices[int(cfg.x), : len(days)], dtype=float)
    y_normalized = 100.0 * y_prices / y_prices[0]
    x_normalized = 100.0 * x_prices / x_prices[0]
    return_correlation = float(
        np.corrcoef(np.diff(np.log(y_prices)), np.diff(np.log(x_prices)))[0, 1]
    )
    ax_price.plot(
        days,
        y_normalized,
        label=f"{y_symbol} normalized price",
        color="#4C72B0",
        linewidth=1.4,
    )
    ax_price.plot(
        days,
        x_normalized,
        label=f"{x_symbol} normalized price",
        color="#C44E52",
        linewidth=1.4,
    )
    ax_price.set_title(
        "Pair Price Comparison "
        f"(Day 1 = 100; daily log-return correlation: {return_correlation:.3f})"
    )
    ax_price.set_ylabel("Normalized Price")
    ax_price.legend(loc="upper left", fontsize=9)
    ax_price.grid(alpha=0.3)

    z_entry = float(cfg.trade.z_entry)
    ax1.plot(
        days,
        replay.z,
        label="Dynamic Z-Score",
        color="dodgerblue",
        alpha=0.7,
    )
    ax1.axhline(
        z_entry,
        color="red",
        linestyle="--",
        label=f"Entry K ({z_entry:.3g})",
    )
    ax1.axhline(-z_entry, color="red", linestyle="--")
    ax1.axhline(0.0, color="black", linestyle="-", alpha=0.5)

    entry_long = [
        event for event in replay.entries if event[1] == "ENTRY" and event[3] > 0
    ]
    entry_short = [
        event for event in replay.entries if event[1] == "ENTRY" and event[3] < 0
    ]
    exits = [event for event in replay.entries if event[1] == "EXIT"]

    if entry_long:
        ax1.scatter(
            [event[0] for event in entry_long],
            [event[2] for event in entry_long],
            color="lime",
            marker="^",
            s=120,
            zorder=10,
            label="Entry (Long)",
            edgecolors="black",
            linewidths=0.8,
        )
    if entry_short:
        ax1.scatter(
            [event[0] for event in entry_short],
            [event[2] for event in entry_short],
            color="red",
            marker="v",
            s=120,
            zorder=10,
            label="Entry (Short)",
            edgecolors="black",
            linewidths=0.8,
        )
    if exits:
        ax1.scatter(
            [event[0] for event in exits],
            [event[2] for event in exits],
            color="black",
            marker="x",
            s=100,
            zorder=10,
            label="Exit",
        )

    ax1.set_title(
        f"Shortlist Review: Dynamic OU Regime & Trade Execution ({pair_name})"
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(
        days,
        replay.kappa,
        label="Dynamic Kappa (Reversion Speed)",
        color="purple",
    )
    ax2.plot(
        days,
        replay.sigma,
        label="Dynamic Sigma (Volatility)",
        color="darkorange",
    )
    ax2.set_title("Rolling OU Parameters (Adapts to Market Regime)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)

    pnl_days = metrics["days"]
    cumulative_pnl = np.cumsum(metrics["daily_pnl"])
    ax3.plot(
        pnl_days,
        cumulative_pnl,
        label="Sim Cumulative PnL (Log Spread)",
        color="darkgreen",
        linewidth=2,
    )
    ax3.set_title(
        "Simulated Spread PnL (Official Eval Score: {:.4f})".format(
            metrics["score"]
        )
    )
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_xlabel("Day")

    for axis in (ax_price, ax1, ax2, ax3):
        for boundary in (250, 500):
            if boundary <= days[-1]:
                axis.axvline(
                    boundary,
                    color="grey",
                    linestyle=":",
                    linewidth=1,
                    alpha=0.45,
                )

    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show(block=True)
    plt.close(fig)


def write_gallery(
    shortlist: pd.DataFrame,
    image_paths: dict[str, Path],
    output_path: Path,
) -> None:
    cards: list[str] = []
    for rank, row in enumerate(shortlist.itertuples(index=False), start=1):
        pair_name = str(row.pair_name)
        if pair_name not in image_paths:
            continue
        image_path = image_paths[pair_name].relative_to(output_path.parent)
        cards.append(
            f"""
            <article class="card">
              <h2>#{rank} {html.escape(pair_name)}</h2>
              <p>
                Robust score: <strong>{row.robust_score:.3f}</strong> ·
                Full score: {row.full_score:.3f} ·
                Worst year: {row.worst_year_score:.3f} ·
                Retention: {row.score_retention:.1%}
              </p>
              <a href="{image_path.as_posix()}">
                <img src="{image_path.as_posix()}" alt="{html.escape(pair_name)}">
              </a>
            </article>
            """
        )
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pair shortlist review</title>
  <style>
    body {{ margin: 0; padding: 24px; background: #eef2f6;
            color: #16202a; font-family: Arial, sans-serif; }}
    main {{ max-width: 1320px; margin: auto; }}
    .card {{ background: white; border-radius: 10px; padding: 18px;
             margin: 0 0 24px; box-shadow: 0 2px 10px #0002; }}
    h1, h2 {{ margin-top: 0; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #ccd3da; }}
  </style>
</head>
<body>
<main>
  <h1>Pair shortlist visual review</h1>
  <p>{len(shortlist)} pairs, ranked without using days after 750.</p>
  {''.join(cards)}
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def serialisable_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        default="testbedv2_trainonly",
        help="Strategy module supplying PairAgent and PAIR_CFGS",
    )
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--min-profitable-years", type=int, default=3)
    parser.add_argument("--min-entries", type=int, default=2)
    parser.add_argument("--min-year-score", type=float, default=0.0)
    parser.add_argument("--min-full-score", type=float, default=2.0)
    parser.add_argument("--min-score-retention", type=float, default=0.10)
    parser.add_argument(
        "--disjoint",
        action="store_true",
        help="Greedily prevent shortlisted pairs from sharing instruments",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Show each image and prompt to approve or reject the pair",
    )
    parser.add_argument(
        "--open-gallery",
        action="store_true",
        help="Open the generated HTML gallery in the default browser",
    )
    parser.add_argument(
        "--max-universe",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for the complete module universe",
    )
    args = parser.parse_args()
    if args.max_pairs < 1:
        parser.error("--max-pairs must be positive")
    if args.min_profitable_years not in range(1, 4):
        parser.error("--min-profitable-years must be from 1 to 3")
    if args.min_entries < 0:
        parser.error("--min-entries cannot be negative")
    if args.max_universe is not None and args.max_universe < 1:
        parser.error("--max-universe must be positive")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output.resolve()
    image_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    strategy = load_strategy(args.module)
    prices, price_symbols = load_prices(args.prices.resolve())
    if prices.shape[1] < FULL_WINDOW[2]:
        raise ValueError(
            f"Need at least {FULL_WINDOW[2]} days, found {prices.shape[1]}"
        )

    inverse_symbols = {int(index): str(symbol) for symbol, index in strategy.I.items()}
    expected_symbols = [inverse_symbols[index] for index in range(strategy.N_INST)]
    if price_symbols != expected_symbols:
        raise ValueError("prices.txt columns do not match the strategy instrument map")

    configs = list(strategy.PAIR_CFGS)
    if args.max_universe is not None:
        configs = configs[: args.max_universe]
    print(
        f"Screening {len(configs)} configurations from {args.module} "
        "using days 1-750 only..."
    )

    records: list[dict[str, Any]] = []
    for number, cfg in enumerate(configs, start=1):
        records.append(
            evaluate_config(
                prices,
                cfg,
                strategy,
                inverse_symbols,
                DEFAULT_WINDOWS,
                FULL_WINDOW,
            )
        )
        if number % 50 == 0 or number == len(configs):
            print(f"  evaluated {number}/{len(configs)}")

    all_pairs = pd.DataFrame.from_records(records).sort_values(
        "robust_score", ascending=False
    )
    shortlist = select_shortlist(all_pairs, args)
    all_pairs.to_csv(output_dir / "all_pair_metrics.csv", index=False)
    shortlist.to_csv(output_dir / "shortlist.csv", index=False)
    (output_dir / "shortlist.json").write_text(
        json.dumps(serialisable_records(shortlist), indent=2),
        encoding="utf-8",
    )

    config_by_name = {
        "-".join(pair_symbols(cfg, inverse_symbols)): cfg for cfg in configs
    }
    image_paths: dict[str, Path] = {}
    approved: list[dict[str, Any]] = []

    print(f"Shortlisted {len(shortlist)} pairs. Generating diagnostic images...")
    for rank, row in enumerate(shortlist.itertuples(index=False), start=1):
        pair_name = str(row.pair_name)
        cfg = config_by_name[pair_name]
        replay = replay_pair(
            prices, cfg, strategy, FULL_WINDOW[2], capture=True
        )
        full_metrics = account_window(
            prices,
            replay.target_shares,
            FULL_WINDOW[1],
            FULL_WINDOW[2],
        )
        image_path = image_dir / f"{rank:02d}_{pair_name.replace('-', '_')}.png"
        plot_diagnostic(
            prices,
            replay,
            full_metrics,
            cfg,
            pair_name,
            image_path,
            show=args.review,
        )
        image_paths[pair_name] = image_path
        print(f"  [{rank}/{len(shortlist)}] {pair_name}: {image_path}")

        if args.review:
            decision = input(
                f"Keep {pair_name}? [Y]es / [n]o / [q]uit: "
            ).strip().lower()
            if decision in ("", "y", "yes"):
                approved.append(shortlist.iloc[rank - 1].to_dict())
            elif decision in ("q", "quit"):
                break

    gallery_path = output_dir / "index.html"
    write_gallery(shortlist, image_paths, gallery_path)
    if args.review:
        (output_dir / "approved_pairs.json").write_text(
            json.dumps(approved, indent=2),
            encoding="utf-8",
        )

    print(f"All metrics: {output_dir / 'all_pair_metrics.csv'}")
    print(f"Shortlist:   {output_dir / 'shortlist.csv'}")
    print(f"Gallery:     {gallery_path}")
    if args.open_gallery:
        webbrowser.open(gallery_path.as_uri())


if __name__ == "__main__":
    main()

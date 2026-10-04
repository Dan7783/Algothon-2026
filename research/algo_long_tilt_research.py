"""Leakage-safe ALGO long-index tilt research for ``four_sigmas``.

Selection physically reads only days 1-750. It compares the unchanged strategy
against deliberately simple long-only ALGO overlays:

* constant target-dollar exposure;
* long exposure only when trailing ALGO return is positive; and
* long exposure only when ALGO is above its trailing moving average.

A candidate must improve at least four of five 125-day development folds and
have positive median improvement. The selected rule is frozen to JSON before
the separate Year-4 phase can read days 751-1000. If no long tilt passes, the
baseline is frozen and the research rejects index tilting without mining OOS.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "algo_long_tilt"
DEVELOPMENT_PATH = OUTPUT_DIR / "development_results.json"
FROZEN_PATH = OUTPUT_DIR / "frozen_selection.json"
HELDOUT_PATH = OUTPUT_DIR / "heldout_results.json"
REPORT_PATH = OUTPUT_DIR / "algo_long_tilt_report.md"
PLOT_PATH = OUTPUT_DIR / "algo_long_tilt.png"

BASE_MODULE = "four_sigmas"
N_DEVELOPMENT_DAYS = 750
HELDOUT_START = 751
HELDOUT_END = 1000
ALGO_INDEX = 0

COMMISSION = np.full(51, 1.0e-4)
COMMISSION[ALGO_INDEX] = 2.0e-5
LIMIT = np.full(51, 10_000.0)
LIMIT[ALGO_INDEX] = 100_000.0

DEVELOPMENT_FOLDS = (
    ("DEV1", 126, 250),
    ("DEV2", 251, 375),
    ("DEV3", 376, 500),
    ("VAL1", 501, 625),
    ("VAL2", 626, 750),
)
ANNUAL_WINDOWS = (
    ("YEAR1", 2, 250),
    ("YEAR2", 251, 500),
    ("YEAR3", 501, 750),
)


@dataclass(frozen=True)
class TiltSpec:
    family: str
    notional: float = 0.0
    window: int = 0

    @property
    def name(self) -> str:
        if self.family == "baseline":
            return "baseline"
        return (
            f"{self.family}_w{self.window}_"
            f"{self.notional / 1000:g}k"
            if self.window
            else f"{self.family}_{self.notional / 1000:g}k"
        )


def load_prices(nrows: int | None) -> np.ndarray:
    frame = pd.read_csv(PRICE_PATH, sep=r"\s+", nrows=nrows)
    prices = frame.to_numpy(dtype=float).T
    if prices.shape[0] != 51:
        raise RuntimeError(f"Expected 51 assets, received {prices.shape[0]}")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0.0):
        raise RuntimeError("Prices must be finite and positive")
    return prices


def load_base():
    return importlib.import_module(BASE_MODULE)


def score(pnl: np.ndarray) -> float:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe**2 / (sharpe**2 + 1.0)


def metrics(
    pnl: np.ndarray,
    turnover: float,
    fees: float,
) -> dict[str, float]:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    return {
        "score": score(pnl),
        "mean": mean,
        "std": std,
        "sharpe": sqrt(250.0) * mean / std if std > 0.0 else 0.0,
        "total_pnl": float(np.sum(pnl)),
        "hit_rate": float(np.mean(pnl > 0.0)),
        "turnover": float(turnover),
        "fees": float(fees),
    }


def build_base_targets(prices: np.ndarray) -> np.ndarray:
    base = load_base()
    base.reset()
    n_assets, n_days = prices.shape
    targets = np.zeros((n_days + 1, n_assets), dtype=int)
    for history_size in range(1, n_days):
        requested = np.asarray(
            base.getMyPosition(prices[:, :history_size]),
            dtype=int,
        )
        current_prices = prices[:, history_size - 1]
        limits = (LIMIT / current_prices).astype(int)
        targets[history_size] = np.clip(
            requested,
            -limits,
            limits,
        ).astype(int)
    return targets


def tilt_candidates() -> tuple[TiltSpec, ...]:
    candidates: list[TiltSpec] = [TiltSpec("baseline")]
    for notional in (10_000.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0):
        candidates.append(TiltSpec("constant_long", notional=notional))
    for window in (20, 60, 125, 250):
        for notional in (25_000.0, 50_000.0, 100_000.0):
            candidates.append(
                TiltSpec(
                    "positive_return_long",
                    notional=notional,
                    window=window,
                )
            )
            candidates.append(
                TiltSpec(
                    "above_sma_long",
                    notional=notional,
                    window=window,
                )
            )
    return tuple(candidates)


def algo_notional(
    prices: np.ndarray,
    history_size: int,
    spec: TiltSpec,
) -> float:
    if spec.family == "baseline":
        return 0.0
    if spec.family == "constant_long":
        return spec.notional
    if history_size <= spec.window:
        return 0.0

    current = float(prices[ALGO_INDEX, history_size - 1])
    if spec.family == "positive_return_long":
        old = float(prices[ALGO_INDEX, history_size - 1 - spec.window])
        return spec.notional if current > old else 0.0
    if spec.family == "above_sma_long":
        trailing = prices[
            ALGO_INDEX,
            history_size - spec.window : history_size,
        ]
        return spec.notional if current > float(np.mean(trailing)) else 0.0
    raise ValueError(f"Unknown tilt family: {spec.family}")


def apply_tilt(
    prices: np.ndarray,
    base_targets: np.ndarray,
    spec: TiltSpec,
) -> np.ndarray:
    targets = base_targets.copy()
    for history_size in range(1, prices.shape[1]):
        notional = algo_notional(prices, history_size, spec)
        price = float(prices[ALGO_INDEX, history_size - 1])
        requested = int(np.round(notional / price))
        limit = int(LIMIT[ALGO_INDEX] / price)
        targets[history_size, ALGO_INDEX] = int(
            np.clip(requested, 0, limit)
        )
    return targets


def account(
    prices: np.ndarray,
    targets: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, float, float]:
    """Match eval.py's fee lag, position timing, and terminal mark."""
    cash = 0.0
    value = 0.0
    prior_fee = 0.0
    current = np.zeros(prices.shape[0])
    pnl: list[float] = []
    turnover = 0.0
    fees = 0.0

    for t in range(start - 1, end + 1):
        current_prices = prices[:, t - 1]
        new_position = (
            targets[t].astype(float)
            if t < end
            else current.copy()
        )
        delta = new_position - current
        cash -= float(current_prices @ delta) + prior_fee
        volumes = current_prices * np.abs(delta)
        prior_fee = float(volumes @ COMMISSION)
        fees += prior_fee
        turnover += float(np.sum(volumes))
        current = new_position

        new_value = cash + float(current @ current_prices)
        daily_pnl = new_value - value
        value = new_value
        if t >= start:
            pnl.append(daily_pnl)
    return np.asarray(pnl), turnover, fees


def evaluate_windows(
    prices: np.ndarray,
    targets: np.ndarray,
    windows: tuple[tuple[str, int, int], ...],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, start, end in windows:
        pnl, turnover, fees = account(prices, targets, start, end)
        result[name] = metrics(pnl, turnover, fees)
    return result


def index_diagnostics(
    prices: np.ndarray,
    base_targets: np.ndarray,
    windows: tuple[tuple[str, int, int], ...],
) -> dict[str, dict[str, float]]:
    diagnostics: dict[str, dict[str, float]] = {}
    for name, start, end in windows:
        pnl, _, _ = account(prices, base_targets, start, end)
        algo_returns = np.diff(
            np.log(prices[ALGO_INDEX, start - 2 : end])
        )
        variance = float(np.var(algo_returns))
        beta_dollars = (
            float(np.cov(pnl, algo_returns, ddof=0)[0, 1] / variance)
            if variance > 1.0e-18
            else 0.0
        )
        corr = (
            float(np.corrcoef(pnl, algo_returns)[0, 1])
            if np.std(pnl) > 1.0e-12
            and np.std(algo_returns) > 1.0e-12
            else 0.0
        )
        simple_return = float(
            prices[ALGO_INDEX, end - 1]
            / prices[ALGO_INDEX, start - 2]
            - 1.0
        )
        diagnostics[name] = {
            "algo_simple_return": simple_return,
            "algo_ann_mean_return": float(250.0 * np.mean(algo_returns)),
            "algo_ann_vol": float(sqrt(250.0) * np.std(algo_returns)),
            "algo_return_sharpe": float(
                sqrt(250.0) * np.mean(algo_returns) / np.std(algo_returns)
            )
            if np.std(algo_returns) > 0.0
            else 0.0,
            "portfolio_algo_pnl_corr": corr,
            "portfolio_algo_beta_dollars": beta_dollars,
        }
    return diagnostics


def select() -> None:
    prices = load_prices(N_DEVELOPMENT_DAYS)
    print(f"Selection loaded exactly {prices.shape[1]} rows")
    base_targets = build_base_targets(prices)
    candidates = tilt_candidates()

    detailed: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for spec in candidates:
        targets = apply_tilt(prices, base_targets, spec)
        folds = evaluate_windows(prices, targets, DEVELOPMENT_FOLDS)
        detailed[spec.name] = {
            "spec": asdict(spec),
            "folds": folds,
        }
        fold_scores = np.asarray(
            [folds[name]["score"] for name, _, _ in DEVELOPMENT_FOLDS]
        )
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "median_fold_score": float(np.median(fold_scores)),
                "mean_fold_score": float(np.mean(fold_scores)),
                "min_fold_score": float(np.min(fold_scores)),
            }
        )

    baseline_scores = np.asarray(
        [
            detailed["baseline"]["folds"][name]["score"]
            for name, _, _ in DEVELOPMENT_FOLDS
        ]
    )
    for row in rows:
        scores = np.asarray(
            [
                detailed[row["name"]]["folds"][name]["score"]
                for name, _, _ in DEVELOPMENT_FOLDS
            ]
        )
        delta = scores - baseline_scores
        row.update(
            {
                "positive_folds": int(np.sum(delta > 0.0)),
                "median_fold_delta": float(np.median(delta)),
                "mean_fold_delta": float(np.mean(delta)),
                "min_fold_delta": float(np.min(delta)),
            }
        )

    eligible = [
        row
        for row in rows
        if row["family"] != "baseline"
        and row["positive_folds"] >= 4
        and row["median_fold_delta"] > 0.0
    ]
    eligible.sort(
        key=lambda row: (
            row["positive_folds"],
            row["median_fold_delta"],
            row["mean_fold_delta"],
            row["min_fold_delta"],
        ),
        reverse=True,
    )
    baseline_row = next(row for row in rows if row["name"] == "baseline")
    selected_row = eligible[0] if eligible else baseline_row
    selected_spec = detailed[selected_row["name"]]["spec"]

    annual_diagnostics = index_diagnostics(
        prices,
        base_targets,
        ANNUAL_WINDOWS,
    )
    ranking = sorted(
        rows,
        key=lambda row: (
            row["positive_folds"],
            row["median_fold_delta"],
            row["mean_fold_delta"],
            row["min_fold_delta"],
        ),
        reverse=True,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    development = {
        "embargo": {
            "rows_loaded": int(prices.shape[1]),
            "selection_last_day": N_DEVELOPMENT_DAYS,
            "heldout": [HELDOUT_START, HELDOUT_END],
        },
        "selection_rule": (
            "Long tilt needs >=4/5 positive score deltas and positive median "
            "delta; rank by positive folds, median, mean, then worst delta"
        ),
        "index_diagnostics": annual_diagnostics,
        "ranking": ranking,
        "details": detailed,
    }
    DEVELOPMENT_PATH.write_text(
        json.dumps(development, indent=2),
        encoding="utf-8",
    )
    frozen = {
        "base_module": BASE_MODULE,
        "selected_name": selected_row["name"],
        "selected_spec": selected_spec,
        "selected_development_summary": selected_row,
        "selection_data_last_day": N_DEVELOPMENT_DAYS,
        "heldout_window": [HELDOUT_START, HELDOUT_END],
        "heldout_evaluated": False,
    }
    FROZEN_PATH.write_text(
        json.dumps(frozen, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(frozen, indent=2))
    print(f"Wrote {FROZEN_PATH}")


def write_report(
    frozen: dict[str, Any],
    development: dict[str, Any],
    heldout: dict[str, Any],
) -> None:
    top = development["ranking"][:12]
    lines = [
        "# ALGO long-index tilt investigation",
        "",
        "## Selection protocol",
        "",
        "- Current `four_sigmas` baseline.",
        "- Selection phase physically loaded only days 1-750.",
        "- Five sequential 125-day development/validation folds.",
        "- Candidate must improve at least 4/5 folds and have positive median "
        "score improvement.",
        "- Only the frozen selection was subsequently tested on days 751-1000.",
        "",
        "## Frozen result",
        "",
        f"- Selected: **{frozen['selected_name']}**",
        f"- Year-4 baseline score: **{heldout['baseline']['score']:.2f}**",
        f"- Year-4 selected score: **{heldout['selected']['score']:.2f}**",
        f"- Year-4 score delta: **{heldout['score_delta']:+.2f}**",
        f"- Year-4 ALGO return: "
        f"**{heldout['index_diagnostics']['algo_simple_return']:.1%}**",
        f"- Year-4 portfolio/ALGO PnL correlation: "
        f"**{heldout['index_diagnostics']['portfolio_algo_pnl_corr']:.3f}**",
        "",
        "## Development index diagnostics",
        "",
        "| Window | ALGO return | ALGO return Sharpe | Portfolio/ALGO PnL corr | "
        "Equivalent ALGO beta dollars |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in development["index_diagnostics"].items():
        lines.append(
            f"| {name} | {values['algo_simple_return']:.1%} | "
            f"{values['algo_return_sharpe']:.2f} | "
            f"{values['portfolio_algo_pnl_corr']:.3f} | "
            f"${values['portfolio_algo_beta_dollars']:,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Top development candidates",
            "",
            "| Candidate | Positive folds | Median delta | Mean delta | Worst delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in top:
        lines.append(
            f"| {row['name']} | {row['positive_folds']}/5 | "
            f"{row['median_fold_delta']:+.2f} | "
            f"{row['mean_fold_delta']:+.2f} | "
            f"{row['min_fold_delta']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Heldout details",
            "",
            "| Strategy | Score | Mean PnL | Total PnL | Sharpe | Turnover | Fees |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("baseline", "selected"):
        values = heldout[name]
        lines.append(
            f"| {name} | {values['score']:.2f} | {values['mean']:.2f} | "
            f"${values['total_pnl']:,.0f} | {values['sharpe']:.2f} | "
            f"${values['turnover']:,.0f} | ${values['fees']:,.0f} |"
        )
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def make_plot(
    development: dict[str, Any],
    heldout: dict[str, Any],
) -> None:
    top = development["ranking"][:10]
    names = [row["name"] for row in top]
    median_delta = np.asarray([row["median_fold_delta"] for row in top])
    worst_delta = np.asarray([row["min_fold_delta"] for row in top])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    positions = np.arange(len(names))
    axes[0].barh(
        positions - 0.18,
        median_delta,
        height=0.34,
        color="#2a9d8f",
        label="median fold delta",
    )
    axes[0].barh(
        positions + 0.18,
        worst_delta,
        height=0.34,
        color="#e76f51",
        label="worst fold delta",
    )
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Median development score delta")
    axes[0].set_title("Median gains are not robust to the worst fold")
    axes[0].legend()

    labels = ["baseline", "selected"]
    values = [heldout[name]["score"] for name in labels]
    axes[1].bar(labels, values, color=["#457b9d", "#2a9d8f"])
    axes[1].set_ylabel("Year-4 official score")
    axes[1].set_title("Frozen Year-4 comparison")
    for index, value in enumerate(values):
        axes[1].text(index, value, f"{value:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def test() -> None:
    if not FROZEN_PATH.exists() or not DEVELOPMENT_PATH.exists():
        raise RuntimeError("Run --phase select before heldout evaluation")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    development = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))
    if frozen.get("selection_data_last_day") != N_DEVELOPMENT_DAYS:
        raise RuntimeError("Invalid frozen development boundary")

    prices = load_prices(HELDOUT_END)
    base_targets = build_base_targets(prices)
    spec = TiltSpec(**frozen["selected_spec"])
    selected_targets = apply_tilt(prices, base_targets, spec)
    windows = (("YEAR4", HELDOUT_START, HELDOUT_END),)
    baseline = evaluate_windows(prices, base_targets, windows)["YEAR4"]
    selected = evaluate_windows(prices, selected_targets, windows)["YEAR4"]
    heldout_index = index_diagnostics(
        prices,
        base_targets,
        windows,
    )["YEAR4"]
    result = {
        "frozen_selection": frozen,
        "baseline": baseline,
        "selected": selected,
        "score_delta": selected["score"] - baseline["score"],
        "index_diagnostics": heldout_index,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HELDOUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    frozen["heldout_evaluated"] = True
    frozen["heldout_result_path"] = str(HELDOUT_PATH.relative_to(ROOT))
    FROZEN_PATH.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    write_report(frozen, development, result)
    make_plot(development, result)
    print(json.dumps(result, indent=2))
    print(f"Wrote {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "test"), required=True)
    args = parser.parse_args()
    if args.phase == "select":
        select()
    else:
        test()


if __name__ == "__main__":
    main()

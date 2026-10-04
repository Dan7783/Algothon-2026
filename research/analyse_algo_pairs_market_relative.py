"""Determine what the historical ALGO pair signals were actually trading.

``testbedv2_tuned`` contained twelve ALGO-X PairAgents, but its final wrapper
forced the ALGO leg to zero. This audit compares each profile in isolation:

* ``both_legs``: the intended adaptive spread trade;
* ``asset_only``: the historical live implementation after ALGO was zeroed;
* ``index_only``: the ALGO leg by itself.

It also estimates each asset's annual return beta to ALGO and tests the
asset-on-index log-price residual for fixed-beta stationarity. The current
``four_sigmas`` strategy is inspected but never modified.
"""

from __future__ import annotations

import sys
from math import log, sqrt
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas
import testbedv2_tuned as legacy


OUTPUT_DIR = ROOT / "analysis_outputs" / "algo_pair_market_relative"
ALGO_INDEX = legacy.I["ALGO"]
LEG = 10_000.0
COMMISSION = np.full(legacy.N_INST, 1.0e-4)
COMMISSION[ALGO_INDEX] = 2.0e-5
LIMIT = np.full(legacy.N_INST, LEG)
LIMIT[ALGO_INDEX] = 100_000.0

YEARS = (
    ("YEAR1", 0, 250),
    ("YEAR2", 250, 500),
    ("YEAR3", 500, 750),
    ("YEAR4", 750, 1000),
)
FORWARD_FOLDS = (
    ("YEAR2", 251, 500),
    ("YEAR3", 501, 750),
    ("YEAR4_OOS", 751, 1000),
)
MODES = ("both_legs", "asset_only", "index_only")


def inverse_symbols() -> dict[int, str]:
    return {index: symbol for symbol, index in legacy.I.items()}


def pair_name(spec: legacy.PairCfg) -> str:
    symbols = inverse_symbols()
    return f"{symbols[int(spec.y)]}-{symbols[int(spec.x)]}"


def algo_specs() -> tuple[legacy.PairCfg, ...]:
    return tuple(
        spec
        for spec in legacy.PAIR_CFGS
        if int(spec.y) == ALGO_INDEX or int(spec.x) == ALGO_INDEX
    )


def score(pnl: np.ndarray) -> float:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe**2 / (sharpe**2 + 1.0)


def metrics(pnl: np.ndarray) -> dict[str, float]:
    values = np.asarray(pnl, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "score": score(values),
        "mean_pnl": mean,
        "total_pnl": float(np.sum(values)),
        "sharpe": sqrt(250.0) * mean / std if std > 0.0 else 0.0,
    }


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if (
        len(left) < 3
        or float(np.std(left)) < 1.0e-12
        or float(np.std(right)) < 1.0e-12
    ):
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def safe_adf(series: np.ndarray) -> float:
    values = np.asarray(series, dtype=float)
    if len(values) < 25 or float(np.std(values)) < 1.0e-12:
        return 1.0
    try:
        return float(adfuller(values, autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        return 1.0


def half_life(residual: np.ndarray) -> float:
    lagged = residual[:-1]
    delta = np.diff(residual)
    design = np.column_stack([np.ones(len(lagged)), lagged])
    coefficient = float(np.linalg.lstsq(design, delta, rcond=None)[0][1])
    if coefficient >= -1.0e-12:
        return float("inf")
    value = -log(2.0) / coefficient
    return float(value) if value > 0.0 else float("inf")


def structural_rows(
    log_prices: np.ndarray,
    spec: legacy.PairCfg,
) -> list[dict[str, Any]]:
    name = pair_name(spec)
    asset_index = int(spec.x) if int(spec.y) == ALGO_INDEX else int(spec.y)
    rows: list[dict[str, Any]] = []
    for year, start, end in YEARS:
        market_levels = log_prices[ALGO_INDEX, start:end]
        asset_levels = log_prices[asset_index, start:end]
        market_returns = np.diff(market_levels)
        asset_returns = np.diff(asset_levels)

        market_centered = market_returns - market_returns.mean()
        asset_centered = asset_returns - asset_returns.mean()
        variance = float(market_centered @ market_centered)
        beta_return = (
            float(asset_centered @ market_centered / variance)
            if variance > 1.0e-14
            else 0.0
        )
        residual_returns = asset_returns - beta_return * market_returns

        design = np.column_stack([np.ones(len(market_levels)), market_levels])
        intercept, beta_level = np.linalg.lstsq(
            design,
            asset_levels,
            rcond=None,
        )[0]
        residual_level = asset_levels - (
            intercept + beta_level * market_levels
        )
        rows.append(
            {
                "pair": name,
                "asset": inverse_symbols()[asset_index],
                "year": year,
                "return_corr": safe_corr(asset_returns, market_returns),
                "return_beta": beta_return,
                "return_r2": safe_corr(asset_returns, market_returns) ** 2,
                "residual_return_lag1": safe_corr(
                    residual_returns[:-1],
                    residual_returns[1:],
                ),
                "level_beta": float(beta_level),
                "residual_adf_p": safe_adf(residual_level),
                "residual_half_life": half_life(residual_level),
            }
        )
    return rows


class IsolatedPair:
    def __init__(self, spec: legacy.PairCfg, mode: str) -> None:
        self.spec = spec
        self.mode = mode
        self.agent = legacy.PairAgent(spec)

    def get_position(self, prices: np.ndarray) -> np.ndarray:
        result = np.zeros(legacy.N_INST, dtype=int)
        if prices.shape[1] < legacy.WARMUP + 10:
            return result
        self.agent.catch_up(prices)
        pos_y, pos_x, _ = self.agent.dollar_positions()
        if self.mode == "asset_only":
            if int(self.spec.y) == ALGO_INDEX:
                pos_y = 0.0
            else:
                pos_x = 0.0
        elif self.mode == "index_only":
            if int(self.spec.y) == ALGO_INDEX:
                pos_x = 0.0
            else:
                pos_y = 0.0
        elif self.mode != "both_legs":
            raise ValueError(f"Unknown mode: {self.mode}")

        dollars = np.zeros(legacy.N_INST)
        dollars[int(self.spec.y)] = pos_y
        dollars[int(self.spec.x)] = pos_x
        last_return = np.diff(
            np.log(np.maximum(prices[:, -2:], legacy._EPS_DENOM)),
            axis=1,
        )[:, 0]
        crash = np.abs(last_return) > 0.40
        crash[ALGO_INDEX] = False
        dollars[crash] = 0.0
        dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
        safe_prices = np.maximum(prices[:, -1], legacy._EPS_DENOM)
        return np.round(dollars / safe_prices).astype(int)


def official_eval(
    prices: np.ndarray,
    runner: IsolatedPair,
    start_day: int,
    end_day: int,
) -> dict[str, float]:
    cash = 0.0
    value = 0.0
    previous_fee = 0.0
    current_position = np.zeros(legacy.N_INST)
    pnl: list[float] = []
    turnover = 0.0

    for t in range(start_day - 1, end_day + 1):
        history = prices[:, :t]
        current_prices = history[:, -1]
        if t < end_day:
            requested = runner.get_position(history)
            limits = (LIMIT / current_prices).astype(int)
            new_position = np.clip(
                requested,
                -limits,
                limits,
            ).astype(int)
        else:
            new_position = current_position.copy()

        delta = new_position - current_position
        cash -= float(current_prices @ delta) + previous_fee
        volume = current_prices * np.abs(delta)
        previous_fee = float(volume @ COMMISSION)
        turnover += float(np.sum(volume))
        current_position = new_position
        new_value = cash + float(current_position @ current_prices)
        daily_pnl = new_value - value
        value = new_value
        if t >= start_day:
            pnl.append(daily_pnl)

    result = metrics(np.asarray(pnl))
    result["turnover"] = turnover
    return result


def classify_pair(
    structure: pd.DataFrame,
    performance: pd.DataFrame,
) -> str:
    adf_passes = int((structure["residual_adf_p"] < 0.05).sum())
    reversal_years = int((structure["residual_return_lag1"] < 0.0).sum())
    both = performance[performance["mode"] == "both_legs"].set_index("fold")
    asset = performance[performance["mode"] == "asset_only"].set_index("fold")
    both_better = int((both["score"] > asset["score"]).sum())
    asset_positive = int((asset["score"] > 0.0).sum())

    if adf_passes >= 2 and both_better >= 2:
        return "MARKET_RELATIVE_REVERSION"
    if asset_positive >= 2 and both_better <= 1 and reversal_years >= 2:
        return "MOSTLY_NAKED_ASSET_REVERSION"
    return "UNSTABLE_OR_MIXED"


def write_report(
    summary: pd.DataFrame,
    structure: pd.DataFrame,
    performance: pd.DataFrame,
) -> None:
    active_current = [
        four_sigmas._pair_name(spec)
        for spec in four_sigmas.PAIR_CFGS
        if int(spec.y) == four_sigmas.I["ALGO"]
        or int(spec.x) == four_sigmas.I["ALGO"]
    ]
    lines = [
        "# What the historical ALGO pair signals were trading",
        "",
        "## Live configuration",
        "",
        f"- Current `four_sigmas` active ALGO pairs: **{len(active_current)}** "
        f"({', '.join(active_current) if active_current else 'none'}).",
        "- Historical `testbedv2_tuned` ALGO pair agents: **12**.",
        "- Historical wrapper set `dollars[0] = 0`, so those agents traded only "
        "the non-ALGO asset after pair selection.",
        "",
        "## Conclusion by relationship",
        "",
        "| Pair | Median return corr | Median market beta | ADF passes | "
        "Residual reversal years | Both-leg score Y4 | Asset-only score Y4 | "
        "Both-minus-asset Y4 | Interpretation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.pair} | {row.median_return_corr:.3f} | "
            f"{row.median_return_beta:.2f} | {row.adf_pass_years}/4 | "
            f"{row.residual_reversal_years}/4 | {row.year4_both_score:.2f} | "
            f"{row.year4_asset_score:.2f} | {row.year4_both_minus_asset:+.2f} | "
            f"{row.interpretation} |"
        )

    lines.extend(
        [
            "",
            "## Structural detail",
            "",
            "| Pair | Year | Return corr | Return beta | R² | Residual ADF p | "
            "Residual half-life | Residual-return lag-1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in structure.itertuples(index=False):
        half_life_text = (
            f"{row.residual_half_life:.1f}"
            if np.isfinite(row.residual_half_life)
            else "inf"
        )
        lines.append(
            f"| {row.pair} | {row.year} | {row.return_corr:.3f} | "
            f"{row.return_beta:.2f} | {row.return_r2:.3f} | "
            f"{row.residual_adf_p:.4f} | {half_life_text} | "
            f"{row.residual_return_lag1:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A genuine index-relative pair is the residual "
            "`asset return - beta * ALGO return`; its cumulative relative-price "
            "residual should repeatedly mean-revert.",
            "- Negative residual-return lag-1 correlation indicates short-horizon "
            "idiosyncratic reversal after removing the market move.",
            "- If the asset-only implementation performs as well as or better than "
            "both legs, the historical PnL was not dependent on neutralising the "
            "market. It was effectively a single-asset reversal signal.",
            "- Fixed-beta ADF is deliberately reported per year. The PairAgent uses "
            "adaptive beta, but adaptivity alone cannot prove economic cointegration.",
            "",
        ]
    )
    (OUTPUT_DIR / "algo_pair_interpretation.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def make_plot(summary: pd.DataFrame) -> None:
    positions = np.arange(len(summary))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].barh(
        positions - width / 2,
        summary["year4_both_score"],
        height=width,
        label="both legs",
        color="#457b9d",
    )
    axes[0].barh(
        positions + width / 2,
        summary["year4_asset_score"],
        height=width,
        label="asset only (historical live)",
        color="#e76f51",
    )
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(summary["pair"])
    axes[0].invert_yaxis()
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Year-4 singular score")
    axes[0].set_title("Did the ALGO hedge leg add the edge?")
    axes[0].legend()

    colors = {
        "MARKET_RELATIVE_REVERSION": "#2a9d8f",
        "MOSTLY_NAKED_ASSET_REVERSION": "#e9c46a",
        "UNSTABLE_OR_MIXED": "#e76f51",
    }
    axes[1].scatter(
        summary["median_return_corr"],
        summary["year4_both_minus_asset"],
        c=[colors[value] for value in summary["interpretation"]],
        s=90,
    )
    for row in summary.itertuples(index=False):
        axes[1].annotate(
            row.pair,
            (row.median_return_corr, row.year4_both_minus_asset),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Median annual ALGO/asset return correlation")
    axes[1].set_ylabel("Year-4 score: both legs minus asset only")
    axes[1].set_title("Correlation alone does not prove pair-trading value")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "algo_pair_interpretation.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "prices.txt", sep=r"\s+")
    prices = frame.to_numpy(dtype=float).T
    log_prices = np.log(prices)
    specs = algo_specs()

    structure_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    for spec in specs:
        name = pair_name(spec)
        structure_rows.extend(structural_rows(log_prices, spec))
        for fold, start, end in FORWARD_FOLDS:
            for mode in MODES:
                result = official_eval(
                    prices,
                    IsolatedPair(spec, mode),
                    start,
                    end,
                )
                performance_rows.append(
                    {
                        "pair": name,
                        "fold": fold,
                        "mode": mode,
                        **result,
                    }
                )

    structure = pd.DataFrame(structure_rows)
    performance = pd.DataFrame(performance_rows)
    summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        name = pair_name(spec)
        pair_structure = structure[structure["pair"] == name]
        pair_performance = performance[performance["pair"] == name]
        year4 = pair_performance[
            pair_performance["fold"] == "YEAR4_OOS"
        ].set_index("mode")
        summary_rows.append(
            {
                "pair": name,
                "median_return_corr": float(
                    pair_structure["return_corr"].median()
                ),
                "median_return_beta": float(
                    pair_structure["return_beta"].median()
                ),
                "adf_pass_years": int(
                    (pair_structure["residual_adf_p"] < 0.05).sum()
                ),
                "residual_reversal_years": int(
                    (pair_structure["residual_return_lag1"] < 0.0).sum()
                ),
                "year4_both_score": float(
                    year4.loc["both_legs", "score"]
                ),
                "year4_asset_score": float(
                    year4.loc["asset_only", "score"]
                ),
                "year4_index_score": float(
                    year4.loc["index_only", "score"]
                ),
                "year4_both_minus_asset": float(
                    year4.loc["both_legs", "score"]
                    - year4.loc["asset_only", "score"]
                ),
                "interpretation": classify_pair(
                    pair_structure,
                    pair_performance,
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["interpretation", "year4_both_minus_asset"],
        ascending=[True, False],
    )
    structure.to_csv(OUTPUT_DIR / "structural_detail.csv", index=False)
    performance.to_csv(OUTPUT_DIR / "singular_performance.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    write_report(summary, structure, performance)
    make_plot(summary)
    print(summary.to_string(index=False))
    print(f"\nWrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

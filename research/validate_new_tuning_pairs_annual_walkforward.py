#!/usr/bin/env python
"""Annual expanding walk-forward audit for the seven restored pair profiles.

The exact fixed profiles in ``four_sigmas.py`` are run one at a time, without
ridge, other pairs, or conflict voting. The accounting loop matches ``eval.py``.

Annual expanding folds:
    days 1-250 available  -> score days 251-500
    days 1-500 available  -> score days 501-750
    days 1-750 available  -> score days 751-1000 (true parameter holdout)

Because the current parameters were selected jointly on development regimes
through day 750, the first two forward years are chronological stability
checks, not unbiased parameter-selection tests. Year 4 is the genuine OOS
test. A separate causal correlation-gated replay is diagnostic only and does
not change ``four_sigmas.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

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
OUTPUT_DIR = ROOT / "analysis_outputs" / "new_tuning_pairs_annual_walkforward"

LEG = 10_000.0
COMMISSION = 0.0001
CORR_LOOKBACK = 60
CORR_GATE = 0.10

# One price year is 250 days. Start/end are the one-based values consumed by
# eval.py's accounting convention.
FOLDS = (
    ("YEAR2", 251, 500, False),
    ("YEAR3", 501, 750, False),
    ("YEAR4_OOS", 751, 1000, True),
)
YEARS = (
    ("YEAR1", 0, 250),
    ("YEAR2", 250, 500),
    ("YEAR3", 500, 750),
    ("YEAR4", 750, 1000),
)

# Structural tiers are declared before looking at results.
MIN_YEAR_CORR = 0.10
MIN_STRONG_SIGNIFICANT_YEARS = 3
MIN_STRONG_ACTIVE_COVERAGE = 0.60
MIN_MODERATE_MEDIAN_CORR = 0.10
MIN_MODERATE_SIGNIFICANT_YEARS = 2
MIN_MODERATE_ACTIVE_COVERAGE = 0.50


def pair_name(spec: strategy.PairCfg) -> str:
    symbols = {index: symbol for symbol, index in strategy.I.items()}
    return f"{symbols[int(spec.y)]}-{symbols[int(spec.x)]}"


def target_specs() -> dict[str, strategy.PairCfg]:
    configured = {pair_name(spec): spec for spec in strategy.PAIR_CFGS}
    missing = sorted(set(TARGET_PAIRS) - set(configured))
    if missing:
        raise RuntimeError(f"Inactive target profiles: {missing}")
    return {name: configured[name] for name in TARGET_PAIRS}


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if (
        left.size < 3
        or np.std(left) < 1.0e-12
        or np.std(right) < 1.0e-12
    ):
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def corr_with_pvalue(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    corr = safe_corr(left, right)
    if len(left) < 3 or np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
        return corr, 1.0
    result = pearsonr(left, right)
    return float(result.statistic), float(result.pvalue)


def trailing_return_corr(
    prices: np.ndarray,
    y_index: int,
    x_index: int,
    lookback: int = CORR_LOOKBACK,
) -> float:
    if prices.shape[1] < lookback + 1:
        return 0.0
    log_window = np.log(
        np.maximum(
            prices[[y_index, x_index], -(lookback + 1) :],
            strategy._EPS_DENOM,
        )
    )
    returns = np.diff(log_window, axis=1)
    return safe_corr(returns[0], returns[1])


def algothon_metrics(pnl: np.ndarray) -> dict[str, float]:
    values = np.asarray(pnl, dtype=float)
    mean = float(np.mean(values)) if values.size else 0.0
    std = float(np.std(values)) if values.size else 0.0
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


class SingularPairStrategy:
    """The current pair signal and bang-bang allocation, isolated."""

    def __init__(
        self,
        spec: strategy.PairCfg,
        *,
        correlation_gate: float | None,
    ) -> None:
        self.spec = spec
        self.agent = strategy.PairAgent(spec)
        self.correlation_gate = correlation_gate
        self.call_count = 0
        self.previous_side = 0
        self.boundary_side = 0
        self.entry_correlations: list[float] = []
        self.active_correlations: list[float] = []
        self.gated_days = 0
        self.records: list[dict[str, Any]] = []

    def get_position(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        result = np.zeros(strategy.N_INST, dtype=int)
        if prices.shape[1] < strategy.WARMUP + 10:
            return result

        self.agent.catch_up(prices)
        y_index, x_index = int(self.spec.y), int(self.spec.x)
        corr = trailing_return_corr(prices, y_index, x_index)
        raw_side = int(self.agent.side)

        gate_blocked = (
            self.correlation_gate is not None
            and corr < self.correlation_gate
        )
        if gate_blocked:
            self.agent.side = 0
            self.gated_days += 1
        effective_side = int(self.agent.side)

        if self.call_count == 0:
            self.boundary_side = effective_side
            if effective_side != 0:
                self.entry_correlations.append(corr)
        elif self.previous_side == 0 and effective_side != 0:
            self.entry_correlations.append(corr)

        if effective_side != 0:
            self.active_correlations.append(corr)

        pos_y, pos_x, _ = self.agent.dollar_positions()
        dollars = np.zeros(strategy.N_INST)
        dollars[y_index] = LEG * np.sign(pos_y)
        dollars[x_index] = LEG * np.sign(pos_x)

        last_returns = np.diff(
            np.log(np.maximum(prices[:, -2:], strategy._EPS_DENOM)),
            axis=1,
        )[:, 0]
        dollars[np.abs(last_returns) > 0.40] = 0.0
        safe_prices = np.maximum(prices[:, -1], strategy._EPS_DENOM)
        result = np.round(dollars / safe_prices).astype(int)

        self.records.append(
            {
                "signal_day": prices.shape[1],
                "raw_side": raw_side,
                "effective_side": effective_side,
                "trailing_corr": corr,
                "gate_blocked": gate_blocked,
                "z": float(self.agent.current_z),
                "beta": float(self.agent.filt.beta),
            }
        )
        self.previous_side = effective_side
        self.call_count += 1
        return result


def official_eval(
    prices: np.ndarray,
    pair_strategy: SingularPairStrategy,
    start_day: int,
    end_day: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Copy eval.py's position, fee, mark and score timing exactly."""
    n_instruments = prices.shape[0]
    cash = 0.0
    current_position = np.zeros(n_instruments)
    total_volume = 0.0
    value = 0.0
    commission = 0.0
    rows: list[dict[str, Any]] = []

    for t in range(start_day - 1, end_day + 1):
        history = prices[:, :t]
        current_prices = history[:, -1]
        if t < end_day:
            proposed = pair_strategy.get_position(history)
            limits = (LEG / current_prices).astype(int)
            new_position = np.clip(proposed, -limits, limits).astype(int)
        else:
            new_position = np.array(current_position)

        delta = new_position - current_position
        cash -= float(current_prices @ delta) + commission
        day_volume = float(current_prices @ np.abs(delta))
        total_volume += day_volume
        commission = COMMISSION * day_volume
        current_position = np.asarray(new_position)

        position_value = float(current_position @ current_prices)
        today_pnl = cash + position_value - value
        value = cash + position_value
        if t >= start_day:
            rows.append(
                {
                    "day": t,
                    "pnl": today_pnl,
                    "value": value,
                    "day_volume": day_volume,
                    "y_shares": int(current_position[int(pair_strategy.spec.y)]),
                    "x_shares": int(current_position[int(pair_strategy.spec.x)]),
                }
            )

    detail = pd.DataFrame(rows)
    metrics = algothon_metrics(detail["pnl"].to_numpy(dtype=float))
    metrics.update(
        {
            "turnover": total_volume,
            "boundary_side": float(pair_strategy.boundary_side),
            "entry_count": float(len(pair_strategy.entry_correlations)),
            "gated_days": float(pair_strategy.gated_days),
            "entry_corr_median": float(
                np.median(pair_strategy.entry_correlations)
            )
            if pair_strategy.entry_correlations
            else 0.0,
            "entry_corr_min": float(min(pair_strategy.entry_correlations))
            if pair_strategy.entry_correlations
            else 0.0,
            "active_corr_median": float(
                np.median(pair_strategy.active_correlations)
            )
            if pair_strategy.active_correlations
            else 0.0,
            "active_corr_coverage": float(
                np.mean(
                    np.asarray(pair_strategy.active_correlations)
                    >= CORR_GATE
                )
            )
            if pair_strategy.active_correlations
            else 0.0,
        }
    )
    return metrics, detail


def yearly_correlations(
    log_prices: np.ndarray,
    spec: strategy.PairCfg,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_index, x_index = int(spec.y), int(spec.x)
    for year, start, end in YEARS:
        segment = log_prices[[y_index, x_index], start:end]
        returns = np.diff(segment, axis=1)
        return_corr, return_p = corr_with_pvalue(returns[0], returns[1])
        level_corr, level_p = corr_with_pvalue(segment[0], segment[1])
        rows.append(
            {
                "year": year,
                "start_day": start + 1,
                "end_day": end,
                "return_corr": return_corr,
                "return_corr_pvalue": return_p,
                "level_corr": level_corr,
                "level_corr_pvalue": level_p,
            }
        )
    return rows


def classify_correlation(
    year_rows: list[dict[str, Any]],
    active_coverage: float,
) -> str:
    correlations = np.asarray(
        [row["return_corr"] for row in year_rows],
        dtype=float,
    )
    pvalues = np.asarray(
        [row["return_corr_pvalue"] for row in year_rows],
        dtype=float,
    )
    dominant = 1 if np.sum(correlations > 0.0) >= 2 else -1
    same_sign = bool(np.all(np.sign(correlations) == dominant))
    significant_years = int(np.sum(pvalues < 0.05))

    if (
        same_sign
        and float(np.min(np.abs(correlations))) >= MIN_YEAR_CORR
        and significant_years >= MIN_STRONG_SIGNIFICANT_YEARS
        and active_coverage >= MIN_STRONG_ACTIVE_COVERAGE
    ):
        return "STRONG"
    if (
        same_sign
        and float(np.median(np.abs(correlations)))
        >= MIN_MODERATE_MEDIAN_CORR
        and significant_years >= MIN_MODERATE_SIGNIFICANT_YEARS
        and active_coverage >= MIN_MODERATE_ACTIVE_COVERAGE
    ):
        return "MODERATE"
    return "FAIL"


def write_report(
    summary: pd.DataFrame,
    performance: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    lines = [
        "# Annual expanding walk-forward: seven restored pairs",
        "",
        "## What was tested",
        "",
        "- The exact profiles currently active in `four_sigmas.py`.",
        "- Each pair alone: no ridge, other pairs, conflict votes or netting.",
        "- Exact `eval.py` accounting, $10k bang-bang asset limits and 1bp fees.",
        "- YEAR2: train/state history 1-250, score 251-500.",
        "- YEAR3: train/state history 1-500, score 501-750.",
        "- YEAR4_OOS: train/state history 1-750, score 751-1000.",
        "- The current parameters were selected with data through day 750, so "
        "YEAR2/YEAR3 measure chronological stability; YEAR4 is the only true "
        "parameter-selection holdout.",
        "- The correlation-gated replay is a diagnostic candidate, not a change "
        "to the submitted strategy. It blocks/flatlines a pair whenever its "
        f"causal {CORR_LOOKBACK}-day return correlation is below {CORR_GATE:.2f}.",
        "",
        "## Result",
        "",
        "| Pair | Correlation | Year return corr Y1/Y2/Y3/Y4 | Baseline scores "
        "Y2/Y3/Y4 | Positive years | Y4 PnL | Gate Y4 score | Gate delta | "
        "Conclusion |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        pair_perf = performance[
            (performance["pair"] == row.pair)
            & (~performance["correlation_gated"])
        ]
        scores = "/".join(f"{score:.2f}" for score in pair_perf["score"])
        pair_corr = correlations[correlations["pair"] == row.pair]
        corr_values = "/".join(
            f"{corr:.3f}" for corr in pair_corr["return_corr"]
        )
        lines.append(
            f"| {row.pair} | {row.correlation_tier} | {corr_values} | "
            f"{scores} | {row.positive_years}/3 | ${row.year4_total_pnl:,.0f} | "
            f"{row.gated_year4_score:.2f} | {row.gated_year4_score_delta:+.2f} | "
            f"{row.conclusion} |"
        )

    lines.extend(
        [
            "",
            "## Exact annual performance",
            "",
            "| Pair | Fold | Mode | Score | Mean PnL | Total PnL | Sharpe | "
            "Turnover | Entries | Active corr median | Active corr >= 0.10 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in performance.itertuples(index=False):
        lines.append(
            f"| {row.pair} | {row.fold} | "
            f"{'gated' if row.correlation_gated else 'baseline'} | "
            f"{row.score:.2f} | {row.mean_pnl:.2f} | "
            f"${row.total_pnl:,.0f} | {row.sharpe:.2f} | "
            f"${row.turnover:,.0f} | {int(row.entry_count)} | "
            f"{row.active_corr_median:.3f} | {row.active_corr_coverage:.0%} |"
        )

    lines.extend(
        [
            "",
            "## Correlation evidence",
            "",
            "| Pair | Year | Return corr | IID p-value | Level corr (diagnostic) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in correlations.itertuples(index=False):
        lines.append(
            f"| {row.pair} | {row.year} | {row.return_corr:.3f} | "
            f"{row.return_corr_pvalue:.4f} | {row.level_corr:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Return correlation is used for structural classification. Price-level "
            "correlation is diagnostic only because common trends can make it spurious.",
            "- Pearson p-values use the IID approximation and are supporting evidence, "
            "not a standalone keep rule.",
            "- STRONG means same correlation sign in all four years, minimum yearly "
            f"|r| >= {MIN_YEAR_CORR:.2f}, at least "
            f"{MIN_STRONG_SIGNIFICANT_YEARS}/4 nominally significant years, and "
            f">= {MIN_STRONG_ACTIVE_COVERAGE:.0%} of active observations above "
            f"the {CORR_GATE:.2f} causal correlation floor.",
            "- MODERATE keeps the same sign, median yearly |r| >= "
            f"{MIN_MODERATE_MEDIAN_CORR:.2f}, at least "
            f"{MIN_MODERATE_SIGNIFICANT_YEARS}/4 significant years, and >= "
            f"{MIN_MODERATE_ACTIVE_COVERAGE:.0%} active correlation coverage.",
            "- No production pair is changed automatically by this report.",
            "",
        ]
    )
    (OUTPUT_DIR / "annual_walkforward_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def make_plot(
    performance: pd.DataFrame,
    correlations: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    x = np.arange(len(TARGET_PAIRS))
    width = 0.22
    baseline = performance[~performance["correlation_gated"]]
    for fold_index, (fold, _, _, _) in enumerate(FOLDS):
        values = [
            float(
                baseline[
                    (baseline["pair"] == pair)
                    & (baseline["fold"] == fold)
                ]["score"].iloc[0]
            )
            for pair in TARGET_PAIRS
        ]
        axes[0].bar(
            x + (fold_index - 1) * width,
            values,
            width,
            label=fold,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(TARGET_PAIRS, rotation=25, ha="right")
    axes[0].set_ylabel("Official score")
    axes[0].set_title("Exact singular-pair annual forward scores")
    axes[0].legend()

    for pair in TARGET_PAIRS:
        pair_corr = correlations[correlations["pair"] == pair]
        axes[1].plot(
            [1, 2, 3, 4],
            pair_corr["return_corr"],
            marker="o",
            label=pair,
        )
    axes[1].axhline(CORR_GATE, color="grey", linestyle="--", linewidth=0.9)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_xticklabels(["YEAR1", "YEAR2", "YEAR3", "YEAR4"])
    axes[1].set_ylabel("Annual log-return correlation")
    axes[1].set_title("Correlation persistence across all four years")
    axes[1].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "prices.txt", sep=r"\s+")
    prices = frame.to_numpy(dtype=float).T
    if prices.shape != (strategy.N_INST, 1000):
        raise RuntimeError(f"Unexpected price shape: {prices.shape}")
    if np.any(prices <= 0.0) or not np.all(np.isfinite(prices)):
        raise RuntimeError("Prices must be finite and positive")
    log_prices = np.log(prices)
    specs = target_specs()

    correlation_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []

    for name in TARGET_PAIRS:
        spec = specs[name]
        for row in yearly_correlations(log_prices, spec):
            correlation_rows.append({"pair": name, **row})

        for fold, start_day, end_day, is_oos in FOLDS:
            for gated in (False, True):
                runner = SingularPairStrategy(
                    spec,
                    correlation_gate=CORR_GATE if gated else None,
                )
                metrics, detail = official_eval(
                    prices,
                    runner,
                    start_day,
                    end_day,
                )
                row = {
                    "pair": name,
                    "fold": fold,
                    "is_oos": is_oos,
                    "correlation_gated": gated,
                    "start_day": start_day,
                    "end_day": end_day,
                    **metrics,
                }
                performance_rows.append(row)
                detail.insert(0, "correlation_gated", gated)
                detail.insert(0, "fold", fold)
                detail.insert(0, "pair", name)
                daily_rows.append(detail)

    correlations = pd.DataFrame(correlation_rows)
    performance = pd.DataFrame(performance_rows)
    daily = pd.concat(daily_rows, ignore_index=True)
    summary_rows: list[dict[str, Any]] = []

    for name in TARGET_PAIRS:
        baseline = performance[
            (performance["pair"] == name)
            & (~performance["correlation_gated"])
        ]
        gated = performance[
            (performance["pair"] == name)
            & (performance["correlation_gated"])
        ]
        year_rows = correlations[correlations["pair"] == name].to_dict("records")
        active_coverage = float(
            baseline["active_corr_coverage"].mean()
        )
        correlation_tier = classify_correlation(year_rows, active_coverage)
        year4 = baseline[baseline["fold"] == "YEAR4_OOS"].iloc[0]
        gated_year4 = gated[gated["fold"] == "YEAR4_OOS"].iloc[0]
        positive_years = int((baseline["score"] > 0.0).sum())

        if correlation_tier == "FAIL":
            conclusion = "REJECT_AS_PAIR"
        elif positive_years < 3 or year4["score"] <= 0.0:
            conclusion = "REVIEW_TRADING_INSTABILITY"
        elif correlation_tier == "MODERATE":
            conclusion = "KEEP_WITH_CORRELATION_MONITOR"
        else:
            conclusion = "KEEP"

        summary_rows.append(
            {
                "pair": name,
                "correlation_tier": correlation_tier,
                "positive_years": positive_years,
                "year4_score": float(year4["score"]),
                "year4_total_pnl": float(year4["total_pnl"]),
                "gated_year4_score": float(gated_year4["score"]),
                "gated_year4_total_pnl": float(gated_year4["total_pnl"]),
                "gated_year4_score_delta": float(
                    gated_year4["score"] - year4["score"]
                ),
                "active_corr_coverage": active_coverage,
                "conclusion": conclusion,
            }
        )

    summary = pd.DataFrame(summary_rows)
    correlations.to_csv(OUTPUT_DIR / "annual_correlations.csv", index=False)
    performance.to_csv(OUTPUT_DIR / "annual_performance.csv", index=False)
    daily.to_csv(OUTPUT_DIR / "annual_daily_pnl.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "annual_summary.csv", index=False)
    (OUTPUT_DIR / "validation_config.json").write_text(
        json.dumps(
            {
                "target_pairs": TARGET_PAIRS,
                "folds": FOLDS,
                "correlation_lookback": CORR_LOOKBACK,
                "correlation_gate": CORR_GATE,
                "leg": LEG,
                "commission": COMMISSION,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(summary, performance, correlations)
    make_plot(
        performance,
        correlations,
        OUTPUT_DIR / "annual_walkforward.png",
    )

    print(
        summary[
            [
                "pair",
                "correlation_tier",
                "positive_years",
                "year4_score",
                "year4_total_pnl",
                "gated_year4_score",
                "gated_year4_score_delta",
                "conclusion",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

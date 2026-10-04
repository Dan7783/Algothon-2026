"""Economic audit of pair, ridge, and ALGO-residual factor exposure.

Increasing all positions uniformly would mechanically scale PnL and drawdown
without changing edge. This audit instead changes one signal sleeve's
portfolio authority at a time while preserving the current signals, dollar
limits, confirmation model, crash gate, and integer-share execution.

Selection/conclusions use days 251-750 only. Days 751-1000 are descriptive.
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import algo12_equal_sleeve_research as evaluator


PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "factor_exposure"
ALGO = strategy.I["ALGO"]
EPS = 1.0e-12

WINDOWS = (
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("TRAIN", 251, 500),
    ("VALIDATION_A", 501, 625),
    ("VALIDATION_B", 626, 750),
    ("VALIDATION", 501, 750),
    ("EXPOSED_A", 751, 875),
    ("EXPOSED_B", 876, 1000),
    ("EXPOSED", 751, 1000),
)

CURRENT_PAIR_SCALE = 1.0
CURRENT_RIDGE_WEIGHT = strategy.RIDGE_CONFLICT_WEIGHT
CURRENT_ALGO_SCALE = 1.0

VARIANT_SPECS: dict[str, tuple[str, float, float, float]] = {
    "current_live": (
        "baseline",
        CURRENT_PAIR_SCALE,
        CURRENT_RIDGE_WEIGHT,
        CURRENT_ALGO_SCALE,
    ),
}
for value in (0.50, 0.75, 1.25, 1.50, 2.00):
    VARIANT_SPECS[f"pair_x{value:.2f}"] = (
        "pair",
        value,
        CURRENT_RIDGE_WEIGHT,
        CURRENT_ALGO_SCALE,
    )
for value in (0.25, 0.50, 1.00, 1.25, 1.50, 2.00):
    VARIANT_SPECS[f"ridge_x{value:.2f}"] = (
        "ridge",
        CURRENT_PAIR_SCALE,
        value,
        CURRENT_ALGO_SCALE,
    )
for value in (0.00, 0.50, 1.50, 2.00):
    VARIANT_SPECS[f"algo_x{value:.2f}"] = (
        "algo",
        CURRENT_PAIR_SCALE,
        CURRENT_RIDGE_WEIGHT,
        value,
    )


def dollars_to_shares(
    dollars: np.ndarray,
    prices: np.ndarray,
) -> np.ndarray:
    safe = np.maximum(prices, strategy._EPS_DENOM)
    limits = np.floor(strategy.BANG_BANG_LIMITS / safe).astype(int)
    raw = np.nan_to_num(
        dollars / safe,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return np.clip(np.round(raw), -limits, limits).astype(int)


def resolve_direction(
    sleeves: dict[str, Any],
    pair_scale: float,
    ridge_weight: float,
) -> np.ndarray:
    """Reproduce current confirmed arbitration with alternate authority."""
    pair_vote = (
        pair_scale
        * np.asarray(sleeves["pair_vote"], dtype=float)
    )
    pair_active = np.asarray(
        sleeves["pair_active_mask"],
        dtype=bool,
    )
    ridge_sign = np.sign(np.asarray(sleeves["ridge_raw"], dtype=float))
    ridge_strength = np.asarray(sleeves["ridge_strength"], dtype=float)
    ridge_active = np.asarray(
        sleeves["ridge_active_mask"],
        dtype=bool,
    )
    confirm_sign = np.asarray(
        sleeves["ridge_confirmation_sign"],
        dtype=float,
    )
    confirm_active = np.asarray(
        sleeves["ridge_confirmation_active_mask"],
        dtype=bool,
    )

    combined = pair_vote + ridge_weight * ridge_sign * ridge_strength
    preliminary = np.sign(combined)
    direction = preliminary.copy()
    active = pair_active | ridge_active
    tied = active & (direction == 0.0) & ridge_active
    direction[tied] = ridge_sign[tied]

    pair_direction = np.sign(pair_vote)
    conflict = (
        pair_active
        & ridge_active
        & (pair_direction != 0.0)
        & (pair_direction != ridge_sign)
    )
    confirmed = (
        conflict
        & confirm_active
        & (confirm_sign == ridge_sign)
        & (preliminary == ridge_sign)
    )
    direction[conflict] = pair_direction[conflict]
    direction[confirmed] = ridge_sign[confirmed]
    direction[np.asarray(sleeves["crash_mask"], dtype=bool)] = 0.0
    return direction


def variant_dollars(
    sleeves: dict[str, Any],
    pair_scale: float,
    ridge_weight: float,
    algo_scale: float,
) -> np.ndarray:
    direction = resolve_direction(sleeves, pair_scale, ridge_weight)
    dollars = strategy.BANG_BANG_LIMITS * direction
    deadlock = np.asarray(
        sleeves["deadlock_ridge_overlay_mask"],
        dtype=bool,
    )
    deadlock_dollars = np.asarray(
        sleeves["deadlock_ridge_overlay_dollars"],
        dtype=float,
    )
    dollars[deadlock] = deadlock_dollars[deadlock]
    crash = np.asarray(sleeves["crash_mask"], dtype=bool)
    if bool(sleeves["algo_beta_ready"]) and algo_scale != 0.0:
        dollars, _, _, _ = strategy._algo_beta_neutral_overlay(
            dollars,
            algo_scale
            * np.asarray(
                sleeves["algo_residual_requested"],
                dtype=float,
            ),
            np.asarray(sleeves["algo_return_betas"], dtype=float),
            crash,
        )
    dollars[ALGO] = 0.0
    return dollars


def fixed_pc1(prices: np.ndarray) -> np.ndarray:
    """First standardized-return PC estimated only from days 1-250."""
    returns = np.diff(np.log(prices[:, :250]), axis=1)
    scale = returns.std(axis=1, keepdims=True) + EPS
    standardized = (
        returns - returns.mean(axis=1, keepdims=True)
    ) / scale
    correlation = standardized @ standardized.T / (
        standardized.shape[1] - 1
    )
    values, vectors = np.linalg.eigh(correlation)
    loading = vectors[:, int(np.argmax(values))]
    if loading[ALGO] < 0.0:
        loading *= -1.0
    return loading / max(float(np.linalg.norm(loading)), EPS)


def build_targets(
    prices: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    shape = (prices.shape[1] + 1, strategy.N_INST)
    targets = {
        variant: np.zeros(shape, dtype=int)
        for variant in VARIANT_SPECS
    }
    algo_betas = np.zeros(shape)
    algo_ready = np.zeros(prices.shape[1] + 1, dtype=bool)
    requested_dollars = {
        variant: np.zeros(shape)
        for variant in VARIANT_SPECS
    }
    reconstruction_mismatches = 0

    strategy.reset()
    for history_size in range(1, prices.shape[1]):
        known = prices[:, :history_size]
        sleeves = strategy.sleeve_breakdown(known)
        if sleeves is None:
            continue
        algo_betas[history_size] = np.asarray(
            sleeves["algo_return_betas"],
            dtype=float,
        )
        algo_ready[history_size] = bool(sleeves["algo_beta_ready"])
        current_shares = dollars_to_shares(
            np.asarray(sleeves["dollars"], dtype=float),
            known[:, -1],
        )
        targets["current_live"][history_size] = current_shares
        requested_dollars["current_live"][history_size] = np.asarray(
            sleeves["dollars"],
            dtype=float,
        )

        reconstructed = variant_dollars(
            sleeves,
            CURRENT_PAIR_SCALE,
            CURRENT_RIDGE_WEIGHT,
            CURRENT_ALGO_SCALE,
        )
        if not np.array_equal(
            dollars_to_shares(reconstructed, known[:, -1]),
            current_shares,
        ):
            reconstruction_mismatches += 1

        for variant, (
            family,
            pair_scale,
            ridge_weight,
            algo_scale,
        ) in VARIANT_SPECS.items():
            if family == "baseline":
                continue
            dollars = variant_dollars(
                sleeves,
                pair_scale,
                ridge_weight,
                algo_scale,
            )
            requested_dollars[variant][history_size] = dollars
            targets[variant][history_size] = dollars_to_shares(
                dollars,
                known[:, -1],
            )
    strategy.reset()
    if reconstruction_mismatches:
        raise RuntimeError(
            "Current arbitration reconstruction differed on "
            f"{reconstruction_mismatches} days"
        )
    return targets, algo_betas, algo_ready, requested_dollars


def risk_metrics(pnl: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    equity = np.cumsum(pnl)
    running_max = np.maximum.accumulate(np.insert(equity, 0, 0.0))[1:]
    drawdown = running_max - equity
    downside = np.minimum(pnl, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside * downside)))
    positive = float(np.sum(pnl[pnl > 0.0]))
    negative = float(-np.sum(pnl[pnl < 0.0]))
    cutoff = max(1, int(np.ceil(0.05 * len(pnl))))
    tail = np.sort(pnl)[:cutoff]
    return {
        "mean_pnl": mean,
        "std_pnl": std,
        "annualized_sharpe": (
            sqrt(250.0) * mean / std if std > EPS else 0.0
        ),
        "max_drawdown": float(np.max(drawdown)) if len(drawdown) else 0.0,
        "downside_deviation": downside_deviation,
        "sortino": (
            sqrt(250.0) * mean / downside_deviation
            if downside_deviation > EPS
            else 0.0
        ),
        "daily_hit_rate": float(np.mean(pnl > 0.0)),
        "profit_factor": positive / negative if negative > EPS else 0.0,
        "cvar_5": float(np.mean(tail)),
    }


def exposure_metrics(
    prices: np.ndarray,
    targets: np.ndarray,
    algo_betas: np.ndarray,
    pc1: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    gross_values: list[float] = []
    net_ratios: list[float] = []
    algo_ratios: list[float] = []
    pc_ratios: list[float] = []
    concentration: list[float] = []
    for day in range(start - 1, end):
        dollars = (
            targets[day].astype(float)
            * prices[:, day - 1]
        )
        gross = float(np.abs(dollars).sum())
        if gross <= EPS:
            continue
        gross_values.append(gross)
        net_ratios.append(abs(float(dollars.sum())) / gross)
        algo_ratios.append(
            abs(float(algo_betas[day] @ dollars)) / gross
        )
        pc_ratios.append(abs(float(pc1 @ dollars)) / gross)
        concentration.append(float(np.max(np.abs(dollars))) / gross)
    return {
        "mean_gross_dollars": float(np.mean(gross_values)),
        "mean_abs_net_ratio": float(np.mean(net_ratios)),
        "mean_abs_algo_beta_ratio": float(np.mean(algo_ratios)),
        "mean_abs_pc1_ratio": float(np.mean(pc_ratios)),
        "mean_max_asset_concentration": float(np.mean(concentration)),
    }


def evaluate(
    prices: np.ndarray,
    targets: dict[str, np.ndarray],
    algo_betas: np.ndarray,
) -> pd.DataFrame:
    pc1 = fixed_pc1(prices)
    rows: list[dict[str, Any]] = []
    for variant, target in targets.items():
        family, pair_scale, ridge_weight, algo_scale = (
            VARIANT_SPECS[variant]
        )
        for window, start, end in WINDOWS:
            pnl, turnover, fees = evaluator.account(
                prices,
                target,
                start,
                end,
            )
            risk = risk_metrics(pnl)
            exposure = exposure_metrics(
                prices,
                target,
                algo_betas,
                pc1,
                start,
                end,
            )
            total_pnl = float(np.sum(pnl))
            gross_days = (
                exposure["mean_gross_dollars"] * len(pnl)
            )
            rows.append(
                {
                    "variant": variant,
                    "family": family,
                    "pair_scale": pair_scale,
                    "ridge_weight": ridge_weight,
                    "algo_scale": algo_scale,
                    "window": window,
                    "score": evaluator.score(pnl),
                    "total_pnl": total_pnl,
                    "turnover": float(turnover),
                    "fees": float(fees),
                    "net_bps_per_turnover": (
                        10_000.0 * total_pnl / turnover
                        if turnover > EPS
                        else 0.0
                    ),
                    "net_bps_per_gross_day": (
                        10_000.0 * total_pnl / gross_days
                        if gross_days > EPS
                        else 0.0
                    ),
                    **risk,
                    **exposure,
                }
            )
    return pd.DataFrame(rows)


def compare_development(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        metrics[metrics["variant"] == "current_live"]
        .set_index("window")
    )
    rows: list[dict[str, Any]] = []
    for variant in VARIANT_SPECS:
        if variant == "current_live":
            continue
        sample = metrics[metrics["variant"] == variant].set_index("window")
        family, pair_scale, ridge_weight, algo_scale = (
            VARIANT_SPECS[variant]
        )
        row: dict[str, Any] = {
            "variant": variant,
            "family": family,
            "pair_scale": pair_scale,
            "ridge_weight": ridge_weight,
            "algo_scale": algo_scale,
        }
        for window in ("TRAIN", "VALIDATION"):
            for field in (
                "total_pnl",
                "annualized_sharpe",
                "max_drawdown",
                "net_bps_per_turnover",
                "net_bps_per_gross_day",
                "mean_abs_algo_beta_ratio",
                "mean_abs_pc1_ratio",
            ):
                row[f"{window.lower()}_{field}"] = float(
                    sample.loc[window, field]
                )
                row[f"{window.lower()}_{field}_delta"] = float(
                    sample.loc[window, field]
                    - baseline.loc[window, field]
                )
        row["non_score_dominates"] = bool(
            row["train_total_pnl_delta"] > 0.0
            and row["validation_total_pnl_delta"] > 0.0
            and row["train_annualized_sharpe_delta"] >= 0.0
            and row["validation_annualized_sharpe_delta"] >= 0.0
            and row["train_net_bps_per_turnover_delta"] >= 0.0
            and row["validation_net_bps_per_turnover_delta"] >= 0.0
            and row["train_max_drawdown"]
            <= 1.10 * baseline.loc["TRAIN", "max_drawdown"]
            and row["validation_max_drawdown"]
            <= 1.10 * baseline.loc["VALIDATION", "max_drawdown"]
        )
        row["worst_pnl_delta"] = min(
            row["train_total_pnl_delta"],
            row["validation_total_pnl_delta"],
        )
        row["worst_sharpe_delta"] = min(
            row["train_annualized_sharpe_delta"],
            row["validation_annualized_sharpe_delta"],
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["non_score_dominates", "worst_pnl_delta", "worst_sharpe_delta"],
        ascending=False,
    )


def is_increased(row: pd.Series) -> bool:
    if row["family"] == "pair":
        return float(row["pair_scale"]) > CURRENT_PAIR_SCALE
    if row["family"] == "ridge":
        return float(row["ridge_weight"]) > CURRENT_RIDGE_WEIGHT
    if row["family"] == "algo":
        return float(row["algo_scale"]) > CURRENT_ALGO_SCALE
    return False


def select_family_diagnostics(
    comparison: pd.DataFrame,
) -> dict[str, Any]:
    selections: dict[str, Any] = {}
    for family in ("pair", "ridge", "algo"):
        candidates = comparison[
            (comparison["family"] == family)
            & comparison.apply(is_increased, axis=1)
        ].copy()
        candidates = candidates.sort_values(
            ["non_score_dominates", "worst_pnl_delta", "worst_sharpe_delta"],
            ascending=False,
        )
        winner = candidates.iloc[0]
        selections[family] = {
            "selected_variant": str(winner["variant"]),
            "increased_exposure_improves_non_score_metrics": bool(
                winner["non_score_dominates"]
            ),
            "selection_last_day": 750,
            "train_pnl_delta": float(winner["train_total_pnl_delta"]),
            "validation_pnl_delta": float(
                winner["validation_total_pnl_delta"]
            ),
            "train_sharpe_delta": float(
                winner["train_annualized_sharpe_delta"]
            ),
            "validation_sharpe_delta": float(
                winner["validation_annualized_sharpe_delta"]
            ),
            "train_drawdown_delta": float(
                winner["train_max_drawdown_delta"]
            ),
            "validation_drawdown_delta": float(
                winner["validation_max_drawdown_delta"]
            ),
            "train_efficiency_delta_bps": float(
                winner["train_net_bps_per_turnover_delta"]
            ),
            "validation_efficiency_delta_bps": float(
                winner["validation_net_bps_per_turnover_delta"]
            ),
        }
    return selections


def write_report(
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    selections: dict[str, Any],
) -> None:
    baseline = metrics[
        (metrics["variant"] == "current_live")
        & metrics["window"].isin(("TRAIN", "VALIDATION", "EXPOSED"))
    ].set_index("window")
    lines = [
        "# Factor-exposure economic audit",
        "",
        "Signals and execution limits remain fixed. Pair and ridge exposure "
        "change their authority in conflicts; ALGO exposure scales the "
        "requested residual basket before the existing beta-neutral projection.",
        "",
        "## Current economic profile",
        "",
        "| Window | PnL | Sharpe | Max drawdown | Hit rate | Net bps/turnover "
        "| Gross | ALGO beta ratio | PC1 ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window, row in baseline.iterrows():
        lines.append(
            f"| {window} | ${row.total_pnl:,.0f} | "
            f"{row.annualized_sharpe:.2f} | ${row.max_drawdown:,.0f} | "
            f"{row.daily_hit_rate:.1%} | "
            f"{row.net_bps_per_turnover:.2f} | "
            f"${row.mean_gross_dollars:,.0f} | "
            f"{row.mean_abs_algo_beta_ratio:.2%} | "
            f"{row.mean_abs_pc1_ratio:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Development-only increased-exposure verdicts",
            "",
            "| Family | Best increased variant | Dominates without score? | "
            "Train PnL change | Validation PnL change | "
            "Train Sharpe change | Validation Sharpe change | "
            "Train drawdown change | Validation drawdown change |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family, result in selections.items():
        lines.append(
            f"| {family} | {result['selected_variant']} | "
            f"{result['increased_exposure_improves_non_score_metrics']} | "
            f"${result['train_pnl_delta']:+,.0f} | "
            f"${result['validation_pnl_delta']:+,.0f} | "
            f"{result['train_sharpe_delta']:+.2f} | "
            f"{result['validation_sharpe_delta']:+.2f} | "
            f"${result['train_drawdown_delta']:+,.0f} | "
            f"${result['validation_drawdown_delta']:+,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Frozen descriptive results on days 751-1000",
            "",
            "| Family | Variant | PnL | Sharpe | Max drawdown | "
            "Net bps/turnover | ALGO beta ratio | PC1 ratio |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family, result in selections.items():
        variant = result["selected_variant"]
        row = metrics[
            (metrics["variant"] == variant)
            & (metrics["window"] == "EXPOSED")
        ].iloc[0]
        lines.append(
            f"| {family} | {variant} | ${row.total_pnl:,.0f} | "
            f"{row.annualized_sharpe:.2f} | ${row.max_drawdown:,.0f} | "
            f"{row.net_bps_per_turnover:.2f} | "
            f"{row.mean_abs_algo_beta_ratio:.2%} | "
            f"{row.mean_abs_pc1_ratio:.2%} |"
        )
    lines.extend(
        [
            "",
            "A variant is marked as dominating only when train and validation "
            "both improve total PnL, Sharpe, and PnL per turnover, while "
            "neither maximum drawdown rises by more than 10%. Score is not "
            "part of this decision.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+").to_numpy().T
    targets, algo_betas, _, _ = build_targets(prices)
    metrics = evaluate(prices, targets, algo_betas)
    comparison = compare_development(metrics)
    selections = select_family_diagnostics(comparison)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    comparison.to_csv(
        OUTPUT_DIR / "development_comparison.csv",
        index=False,
    )
    (OUTPUT_DIR / "selection.json").write_text(
        json.dumps(selections, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(metrics, comparison, selections)
    print(json.dumps(selections, indent=2))
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Test twelve independent ALGO pair sub-books sharing the $100k ALGO cap.

Each historical ALGO-X relationship receives an equal, non-transferable gross
ALGO slot of ``100_000 / 12`` dollars. Every PairAgent updates independently;
no priority/taken-set logic can suppress another ALGO signal. Only after all
signals are formed are their ALGO legs netted at the broker level.

The pair profiles are read from ``tuning_data.db`` and use the current
``four_sigmas`` PairAgent implementation. This is research only: it does not
modify the production strategy.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from math import sqrt
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as base


OUTPUT_DIR = ROOT / "analysis_outputs" / "algo12_equal_sleeves"
PRICE_PATH = ROOT / "prices.txt"
DATABASE_PATH = ROOT / "tuning_data.db"

PAIR_NAMES = (
    "ALGO-ACIX",
    "ALGO-AETS",
    "ALGO-ANSO",
    "ALGO-CUBO",
    "ALGO-FCSG",
    "ALGO-HRET",
    "ALGO-LSST",
    "ALGO-MHRM",
    "ALGO-NGTE",
    "ALGO-NPCK",
    "ALGO-OTCS",
    "ALGO-SPLZ",
)
ALGO_INDEX = base.I["ALGO"]
TOTAL_ALGO_CAP = 100_000.0
SLOT = TOTAL_ALGO_CAP / len(PAIR_NAMES)
COMMISSION = np.full(base.N_INST, 1.0e-4)
COMMISSION[ALGO_INDEX] = 2.0e-5
LIMIT = np.full(base.N_INST, base.LEG)
LIMIT[ALGO_INDEX] = TOTAL_ALGO_CAP

DEVELOPMENT_FOLDS = (
    ("DEV1", 126, 250),
    ("DEV2", 251, 375),
    ("DEV3", 376, 500),
    ("VAL1", 501, 625),
    ("VAL2", 626, 750),
)
ANNUAL_FOLDS = (
    ("YEAR2", 251, 500),
    ("YEAR3", 501, 750),
    ("CONTAMINATED_751_1000", 751, 1000),
)
ALL_WINDOWS = DEVELOPMENT_FOLDS + ANNUAL_FOLDS


def load_specs() -> tuple[base.PairCfg, ...]:
    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(
            "SELECT pair_name, y_symbol, x_symbol, auto_params "
            f"FROM pair_results WHERE pair_name IN "
            f"({','.join('?' for _ in PAIR_NAMES)})",
            PAIR_NAMES,
        ).fetchall()
    by_name = {row[0]: row for row in rows}
    missing = sorted(set(PAIR_NAMES) - set(by_name))
    if missing:
        raise RuntimeError(f"Missing tuning profiles: {missing}")

    specs: list[base.PairCfg] = []
    for index, name in enumerate(PAIR_NAMES):
        _, y_symbol, x_symbol, encoded = by_name[name]
        params = json.loads(encoded)
        specs.append(
            base.PairCfg(
                idx=index,
                y=base.I[y_symbol],
                x=base.I[x_symbol],
                filt=base.FiltCfg(
                    q_beta=float(params["q_beta"]),
                    cusum_k=float(params["cusum_k"]),
                    h=float(params["h"]),
                    max_cov_inflate=float(params["max_cov_inflate"]),
                    cov_decay=float(params["cov_decay"]),
                ),
                trade=base.TradeCfg(
                    z_entry=float(params["z_entry"]),
                    z_exit=float(params["z_exit"]),
                    z_stop=10.0,
                ),
            )
        )
    return tuple(specs)


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
        "mean_pnl": mean,
        "std_pnl": std,
        "sharpe": sqrt(250.0) * mean / std if std > 0.0 else 0.0,
        "total_pnl": float(np.sum(pnl)),
        "turnover": float(turnover),
        "fees": float(fees),
    }


class EqualSleeveBook:
    """Twelve independent signals with fixed, non-transferable ALGO slots."""

    def __init__(self, specs: tuple[base.PairCfg, ...]) -> None:
        self.agents = [base.PairAgent(spec) for spec in specs]
        self.records: list[dict[str, float]] = []
        self.sleeve_dollars: dict[str, list[np.ndarray]] = {
            name: [] for name in PAIR_NAMES
        }

    def dollars(self, prices: np.ndarray) -> np.ndarray:
        total = np.zeros(base.N_INST)
        gross_algo = 0.0
        net_algo = 0.0
        active = 0

        for name, agent in zip(PAIR_NAMES, self.agents, strict=True):
            agent.catch_up(prices)
            sleeve = np.zeros(base.N_INST)
            if agent.side != 0:
                beta = float(agent.filt.beta)
                scale = SLOT / max(1.0, abs(beta))
                pos_y = scale * float(agent.side)
                pos_x = -beta * pos_y
                sleeve[int(agent.spec.y)] = pos_y
                sleeve[int(agent.spec.x)] = pos_x
                active += 1
                gross_algo += abs(float(sleeve[ALGO_INDEX]))
                net_algo += float(sleeve[ALGO_INDEX])
            self.sleeve_dollars[name].append(sleeve.copy())
            total += sleeve

        last_return = np.diff(
            np.log(np.maximum(prices[:, -2:], base._EPS_DENOM)),
            axis=1,
        )[:, 0]
        crash = np.abs(last_return) > 0.40
        crash[ALGO_INDEX] = False
        total[crash] = 0.0
        total[ALGO_INDEX] = np.clip(
            total[ALGO_INDEX],
            -TOTAL_ALGO_CAP,
            TOTAL_ALGO_CAP,
        )
        total[1:] = np.clip(total[1:], -base.LEG, base.LEG)
        cancellation = (
            1.0 - abs(net_algo) / gross_algo
            if gross_algo > 1.0e-12
            else 0.0
        )
        self.records.append(
            {
                "history_size": float(prices.shape[1]),
                "active_sleeves": float(active),
                "gross_algo": gross_algo,
                "net_algo": net_algo,
                "algo_cancellation": cancellation,
            }
        )
        return total


def dollars_to_shares(
    dollars: np.ndarray,
    prices: np.ndarray,
) -> np.ndarray:
    limits = (LIMIT / prices).astype(int)
    requested = np.round(
        dollars / np.maximum(prices, base._EPS_DENOM)
    ).astype(int)
    return np.clip(requested, -limits, limits).astype(int)


def build_base_targets(prices: np.ndarray) -> np.ndarray:
    base.reset()
    targets = np.zeros((prices.shape[1] + 1, base.N_INST), dtype=int)
    for history_size in range(1, prices.shape[1]):
        current_prices = prices[:, history_size - 1]
        requested = base.getMyPosition(prices[:, :history_size])
        limits = (LIMIT / current_prices).astype(int)
        targets[history_size] = np.clip(
            requested,
            -limits,
            limits,
        ).astype(int)
    return targets


def build_sleeve_targets(
    prices: np.ndarray,
    specs: tuple[base.PairCfg, ...],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    book = EqualSleeveBook(specs)
    targets = np.zeros((prices.shape[1] + 1, base.N_INST), dtype=int)
    for history_size in range(1, prices.shape[1]):
        if history_size < base.WARMUP + 10:
            continue
        current_prices = prices[:, history_size - 1]
        targets[history_size] = dollars_to_shares(
            book.dollars(prices[:, :history_size]),
            current_prices,
        )

    diagnostics = pd.DataFrame(book.records)
    individual: dict[str, np.ndarray] = {}
    for name in PAIR_NAMES:
        pair_targets = np.zeros_like(targets)
        # Records begin at history size WARMUP+10. Align them explicitly.
        for offset, sleeve in enumerate(book.sleeve_dollars[name]):
            history_size = base.WARMUP + 10 + offset
            if history_size >= prices.shape[1]:
                break
            pair_targets[history_size] = dollars_to_shares(
                sleeve,
                prices[:, history_size - 1],
            )
        individual[name] = pair_targets
    return targets, diagnostics, individual


def combine_targets(
    prices: np.ndarray,
    base_targets: np.ndarray,
    sleeve_targets: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    combined = np.zeros_like(base_targets)
    records: list[dict[str, float]] = []
    for history_size in range(1, prices.shape[1]):
        current_prices = prices[:, history_size - 1]
        base_dollars = base_targets[history_size] * current_prices
        sleeve_dollars = sleeve_targets[history_size] * current_prices
        raw = base_dollars + sleeve_dollars
        combined[history_size] = dollars_to_shares(raw, current_prices)
        effective_dollars = combined[history_size] * current_prices
        requested_overlay = float(np.sum(np.abs(sleeve_dollars[1:])))
        delivered_overlay = float(
            np.sum(
                np.abs(effective_dollars[1:] - base_dollars[1:])
            )
        )
        records.append(
            {
                "history_size": float(history_size),
                "requested_non_algo_overlay": requested_overlay,
                "delivered_non_algo_change": delivered_overlay,
                "delivery_ratio": (
                    delivered_overlay / requested_overlay
                    if requested_overlay > 1.0e-12
                    else 0.0
                ),
            }
        )
    return combined, pd.DataFrame(records)


def account(
    prices: np.ndarray,
    targets: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, float, float]:
    cash = 0.0
    value = 0.0
    prior_fee = 0.0
    current = np.zeros(base.N_INST)
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
        volume = current_prices * np.abs(delta)
        prior_fee = float(volume @ COMMISSION)
        turnover += float(np.sum(volume))
        fees += prior_fee
        current = new_position
        new_value = cash + float(current @ current_prices)
        daily_pnl = new_value - value
        value = new_value
        if t >= start:
            pnl.append(daily_pnl)
    return np.asarray(pnl), turnover, fees


def evaluate(
    prices: np.ndarray,
    targets: np.ndarray,
    windows: tuple[tuple[str, int, int], ...],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, start, end in windows:
        pnl, turnover, fees = account(prices, targets, start, end)
        result[name] = metrics(pnl, turnover, fees)
    return result


def randomize_active_episodes(
    targets: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomize each held episode's direction while preserving activity."""
    randomized = np.zeros_like(targets)
    for instrument in range(1, targets.shape[1]):
        prior_source_sign = 0
        random_sign = 0
        for day in range(1, targets.shape[0]):
            source = int(targets[day, instrument])
            source_sign = int(np.sign(source))
            if source_sign == 0:
                prior_source_sign = 0
                random_sign = 0
                continue
            if prior_source_sign == 0 or source_sign != prior_source_sign:
                random_sign = -1 if rng.random() < 0.5 else 1
            randomized[day, instrument] = (
                random_sign * abs(source)
            )
            prior_source_sign = source_sign
    return randomized


def asset_overlay_random_control(
    prices: np.ndarray,
    base_targets: np.ndarray,
    asset_leg_targets: np.ndarray,
    observed_combined: np.ndarray,
    seeds: int = 500,
) -> pd.DataFrame:
    """Compare residual timing with random directions at identical activity."""
    baselines = {}
    observed = {}
    for window, start, end in ANNUAL_FOLDS:
        baseline_pnl, _, _ = account(
            prices,
            base_targets,
            start,
            end,
        )
        observed_pnl, _, _ = account(
            prices,
            observed_combined,
            start,
            end,
        )
        baselines[window] = score(baseline_pnl)
        observed[window] = score(observed_pnl) - baselines[window]

    random_by_window = {
        window: [] for window, _, _ in ANNUAL_FOLDS
    }
    for seed in range(seeds):
        random_targets = randomize_active_episodes(
            asset_leg_targets,
            np.random.default_rng(30_000 + seed),
        )
        random_combined, _ = combine_targets(
            prices,
            base_targets,
            random_targets,
        )
        for window, start, end in ANNUAL_FOLDS:
            random_pnl, _, _ = account(
                prices,
                random_combined,
                start,
                end,
            )
            random_by_window[window].append(
                score(random_pnl) - baselines[window]
            )

    rows = []
    for window, _, _ in ANNUAL_FOLDS:
        values = np.asarray(random_by_window[window])
        observed_delta = observed[window]
        rows.append(
            {
                "window": window,
                "observed_delta": observed_delta,
                "random_mean_delta": float(values.mean()),
                "random_std_delta": float(values.std()),
                "random_95pct": float(np.quantile(values, 0.95)),
                "randomization_p": float(
                    (1 + np.sum(values >= observed_delta))
                    / (seeds + 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    results: dict[str, dict[str, dict[str, float]]],
    pair_results: pd.DataFrame,
    sleeve_diagnostics: pd.DataFrame,
    combine_diagnostics: pd.DataFrame,
    random_control: pd.DataFrame,
    selected_asset_pairs: tuple[str, ...],
) -> None:
    lines = [
        "# Twelve independent ALGO equal sleeves",
        "",
        "## Construction",
        "",
        f"- ALGO capacity: **${TOTAL_ALGO_CAP:,.0f}**.",
        f"- Independent pair slots: **{len(PAIR_NAMES)} × ${SLOT:,.2f}**.",
        "- Every pair updates and signals independently; no priority competition.",
        "- Each sleeve sizes both ALGO and asset legs so neither exceeds its slot.",
        "- ALGO legs are netted only after all twelve decisions are formed.",
        "- Non-ALGO overlay legs are combined with the current strategy and then "
        "subject to the existing $10k per-asset limit.",
        "",
        "## Portfolio results",
        "",
        "| Window | Baseline | Full overlay delta | ALGO-leg-only delta | "
        "Asset-leg-only delta | Base+full PnL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for window, _, _ in ALL_WINDOWS:
        baseline = results["baseline"][window]
        combined = results["base_plus_sleeves"][window]
        algo_only = results["base_plus_algo_legs"][window]
        asset_only = results["base_plus_asset_legs"][window]
        lines.append(
            f"| {window} | {baseline['score']:.2f} | "
            f"{combined['score'] - baseline['score']:+.2f} | "
            f"{algo_only['score'] - baseline['score']:+.2f} | "
            f"{asset_only['score'] - baseline['score']:+.2f} | "
            f"${combined['total_pnl']:,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Sleeve mechanics",
            "",
            f"- Mean active sleeves: "
            f"**{sleeve_diagnostics['active_sleeves'].mean():.2f}/12**.",
            f"- Mean gross ALGO allocation: "
            f"**${sleeve_diagnostics['gross_algo'].mean():,.0f}**.",
            f"- Mean absolute net ALGO exposure: "
            f"**${sleeve_diagnostics['net_algo'].abs().mean():,.0f}**.",
            f"- Mean internal ALGO cancellation: "
            f"**{sleeve_diagnostics['algo_cancellation'].mean():.1%}**.",
            f"- Mean delivery of requested non-ALGO overlay after combining "
            f"with the base book: **"
            f"{combine_diagnostics.loc[combine_diagnostics['requested_non_algo_overlay'] > 0, 'delivery_ratio'].mean():.1%}**.",
            f"- Development/validation-stable asset-only shortlist: **"
            f"{', '.join(selected_asset_pairs) if selected_asset_pairs else 'none'}**.",
            "",
            "## Individual sleeves",
            "",
        "| Pair | Full Y2 | Full Y3 | Full Y4 | Asset-only Y2 | "
        "Asset-only Y3 | Asset-only Y4 | ALGO-only Y4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pair_results.itertuples(index=False):
        lines.append(
            f"| {row.pair} | {row.year2_score:.2f} | "
            f"{row.year3_score:.2f} | {row.year4_score:.2f} | "
            f"{row.asset_year2_score:.2f} | "
            f"{row.asset_year3_score:.2f} | "
            f"{row.asset_year4_score:.2f} | "
            f"{row.algo_year4_score:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Asset-leg timing versus random directions",
            "",
            "Random controls preserve each asset's active episodes and "
            "position magnitudes but replace the residual signal direction.",
            "",
            "| Window | Observed delta | Random mean | Random SD | "
            "Random 95% | p(random >= observed) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in random_control.itertuples(index=False):
        lines.append(
            f"| {row.window} | {row.observed_delta:+.2f} | "
            f"{row.random_mean_delta:+.2f} | "
            f"{row.random_std_delta:.2f} | "
            f"{row.random_95pct:+.2f} | "
            f"{row.randomization_p:.4f} |"
        )
    if selected_asset_pairs:
        lines.extend(
            [
                "",
                "## Stable asset-only shortlist",
                "",
                "A pair qualifies only when its asset-only overlay improves "
                "the current book in at least 4/5 development folds and in "
                "both validation halves. Original $8,333 slots are retained.",
                "",
                "| Window | Baseline | Shortlist | Delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for window, _, _ in ALL_WINDOWS:
            baseline = results["baseline"][window]["score"]
            shortlist = results["stable_asset_shortlist"][window]["score"]
            lines.append(
                f"| {window} | {baseline:.2f} | {shortlist:.2f} | "
                f"{shortlist - baseline:+.2f} |"
            )
    lines.append("")
    (OUTPUT_DIR / "equal_sleeve_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def make_plot(
    results: dict[str, dict[str, dict[str, float]]],
    pair_results: pd.DataFrame,
) -> None:
    windows = [name for name, _, _ in ALL_WINDOWS]
    baseline = [results["baseline"][name]["score"] for name in windows]
    combined = [
        results["base_plus_sleeves"][name]["score"] for name in windows
    ]
    deltas = np.asarray(combined) - np.asarray(baseline)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = np.where(deltas >= 0.0, "#2a9d8f", "#e76f51")
    axes[0].bar(windows, deltas, color=colors)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Score delta vs current four_sigmas")
    axes[0].set_title("Does the equal-sleeve overlay improve the book?")

    axes[1].barh(
        pair_results["pair"],
        pair_results["year4_score"],
        color=np.where(
            pair_results["year4_score"] >= 0.0,
            "#457b9d",
            "#e76f51",
        ),
    )
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Year-4 standalone sleeve score")
    axes[1].set_title(f"Each pair gets a ${SLOT:,.0f} slot")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "equal_sleeve_results.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+").to_numpy(dtype=float).T
    specs = load_specs()
    base_targets = build_base_targets(prices)
    sleeve_targets, sleeve_diagnostics, individual = build_sleeve_targets(
        prices,
        specs,
    )
    combined_targets, combine_diagnostics = combine_targets(
        prices,
        base_targets,
        sleeve_targets,
    )
    algo_leg_targets = sleeve_targets.copy()
    algo_leg_targets[:, 1:] = 0
    asset_leg_targets = sleeve_targets.copy()
    asset_leg_targets[:, ALGO_INDEX] = 0
    combined_algo_legs, _ = combine_targets(
        prices,
        base_targets,
        algo_leg_targets,
    )
    combined_asset_legs, _ = combine_targets(
        prices,
        base_targets,
        asset_leg_targets,
    )
    random_control = asset_overlay_random_control(
        prices,
        base_targets,
        asset_leg_targets,
        combined_asset_legs,
    )

    results = {
        "baseline": evaluate(prices, base_targets, ALL_WINDOWS),
        "sleeves_only": evaluate(prices, sleeve_targets, ALL_WINDOWS),
        "base_plus_sleeves": evaluate(prices, combined_targets, ALL_WINDOWS),
        "base_plus_algo_legs": evaluate(
            prices,
            combined_algo_legs,
            ALL_WINDOWS,
        ),
        "base_plus_asset_legs": evaluate(
            prices,
            combined_asset_legs,
            ALL_WINDOWS,
        ),
    }
    pair_rows: list[dict[str, Any]] = []
    asset_targets_by_pair: dict[str, np.ndarray] = {}
    for name in PAIR_NAMES:
        pair_metrics = evaluate(prices, individual[name], ANNUAL_FOLDS)
        pair_asset_targets = individual[name].copy()
        pair_asset_targets[:, ALGO_INDEX] = 0
        asset_targets_by_pair[name] = pair_asset_targets
        pair_algo_targets = individual[name].copy()
        pair_algo_targets[:, 1:] = 0
        pair_asset_metrics = evaluate(
            prices,
            pair_asset_targets,
            ANNUAL_FOLDS,
        )
        pair_algo_metrics = evaluate(
            prices,
            pair_algo_targets,
            ANNUAL_FOLDS,
        )
        pair_asset_combined, _ = combine_targets(
            prices,
            base_targets,
            pair_asset_targets,
        )
        pair_asset_overlay_metrics = evaluate(
            prices,
            pair_asset_combined,
            ALL_WINDOWS,
        )
        scores = [
            pair_metrics["YEAR2"]["score"],
            pair_metrics["YEAR3"]["score"],
            pair_metrics["CONTAMINATED_751_1000"]["score"],
        ]
        pair_rows.append(
            {
                "pair": name,
                "year2_score": scores[0],
                "year3_score": scores[1],
                "year4_score": scores[2],
                "positive_years": int(np.sum(np.asarray(scores) > 0.0)),
                "year4_total_pnl": pair_metrics[
                    "CONTAMINATED_751_1000"
                ]["total_pnl"],
                "asset_year2_score": pair_asset_metrics["YEAR2"]["score"],
                "asset_year3_score": pair_asset_metrics["YEAR3"]["score"],
                "asset_year4_score": pair_asset_metrics[
                    "CONTAMINATED_751_1000"
                ]["score"],
                "algo_year2_score": pair_algo_metrics["YEAR2"]["score"],
                "algo_year3_score": pair_algo_metrics["YEAR3"]["score"],
                "algo_year4_score": pair_algo_metrics[
                    "CONTAMINATED_751_1000"
                ]["score"],
                **{
                    f"overlay_delta_{window}": (
                        pair_asset_overlay_metrics[window]["score"]
                        - results["baseline"][window]["score"]
                    )
                    for window, _, _ in DEVELOPMENT_FOLDS
                },
            }
        )
    pair_results = pd.DataFrame(pair_rows).sort_values(
        "year4_score",
        ascending=False,
    )
    development_delta_columns = tuple(
        f"overlay_delta_{window}"
        for window, _, _ in DEVELOPMENT_FOLDS
    )
    selected_asset_pairs = tuple(
        row.pair
        for row in pair_results.itertuples(index=False)
        if sum(
            getattr(row, column) > 0.0
            for column in development_delta_columns
        )
        >= 4
        and row.overlay_delta_VAL1 > 0.0
        and row.overlay_delta_VAL2 > 0.0
    )
    if selected_asset_pairs:
        stable_targets = np.zeros_like(sleeve_targets)
        for name in selected_asset_pairs:
            stable_targets += asset_targets_by_pair[name]
        stable_combined, _ = combine_targets(
            prices,
            base_targets,
            stable_targets,
        )
        results["stable_asset_shortlist"] = evaluate(
            prices,
            stable_combined,
            ALL_WINDOWS,
        )

    payload = {
        "slot_dollars": SLOT,
        "pair_names": PAIR_NAMES,
        "selected_asset_pairs": selected_asset_pairs,
        "results": results,
        "mechanics": {
            "mean_active_sleeves": float(
                sleeve_diagnostics["active_sleeves"].mean()
            ),
            "mean_gross_algo": float(
                sleeve_diagnostics["gross_algo"].mean()
            ),
            "mean_abs_net_algo": float(
                sleeve_diagnostics["net_algo"].abs().mean()
            ),
            "mean_algo_cancellation": float(
                sleeve_diagnostics["algo_cancellation"].mean()
            ),
            "mean_non_algo_delivery": float(
                combine_diagnostics.loc[
                    combine_diagnostics["requested_non_algo_overlay"] > 0,
                    "delivery_ratio",
                ].mean()
            ),
        },
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    pair_results.to_csv(OUTPUT_DIR / "pair_results.csv", index=False)
    random_control.to_csv(
        OUTPUT_DIR / "asset_overlay_random_control.csv",
        index=False,
    )
    sleeve_diagnostics.to_csv(
        OUTPUT_DIR / "sleeve_diagnostics.csv",
        index=False,
    )
    combine_diagnostics.to_csv(
        OUTPUT_DIR / "combine_diagnostics.csv",
        index=False,
    )
    write_report(
        results,
        pair_results,
        sleeve_diagnostics,
        combine_diagnostics,
        random_control,
        selected_asset_pairs,
    )
    make_plot(results, pair_results)

    rows = []
    for window, _, _ in ALL_WINDOWS:
        baseline_score = results["baseline"][window]["score"]
        combined_score = results["base_plus_sleeves"][window]["score"]
        rows.append(
            {
                "window": window,
                "baseline": baseline_score,
                "sleeves_only": results["sleeves_only"][window]["score"],
                "base_plus_sleeves": combined_score,
                "delta": combined_score - baseline_score,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nIndividual sleeves:")
    print(pair_results.to_string(index=False))
    print(f"\nWrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Focused train-led robustness screen and beta diagnostics for CTGI/EELT.

The broad explorer grid is useful for discovery, but this screen answers the
pair-specific question directly: does making the extremely sticky live beta
faster improve a stable threshold neighborhood?  Candidate ranking uses only
days 251--500.  Test metrics are attached afterward for validation and never
enter ``train_rank`` or ``plateau_rank``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for module_dir in (REPO_ROOT, REPO_ROOT / "research", REPO_ROOT / "archive"):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import archive.old_tuners.daniel_filter as dfil
from pair_book_lab import load_raw_prices
from tune_daniel_pairs import (
    TEST_DECIDE_START,
    TEST_END,
    TEST_START,
    TRAIN_DECIDE_START,
    TRAIN_END,
    TRAIN_START,
    gate_and_z_paths,
    pnl_path,
    record_paths,
    trade_sim,
)
from tune_pair_seed_explorer import window_metrics


PAIR_NAME = "CTGI/EELT"
OUT_DIR = Path("cusum_output/pair_seed_explorer/ctgi_eelt")
Q_MULTIPLIERS = (
    0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 20.0, 32.0, 64.0,
    128.0, 256.0,
)
ENTRIES = (0.25, 0.30, 0.325, 0.35, 0.375, 0.40, 0.45, 0.50)
EXITS = (0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20)
STOPS = (3.0, 3.25, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)


def percentile(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=higher_is_better)


def rolling_origin_beta(log_y: np.ndarray, log_x: np.ndarray, window: int = 100) -> np.ndarray:
    out = np.full(len(log_y), np.nan)
    for index in range(window, len(log_y)):
        xs = log_x[index - window : index]
        ys = log_y[index - window : index]
        denominator = float(xs @ xs)
        if denominator > 1.0e-12:
            out[index] = float(xs @ ys) / denominator
    return out


def beta_window_diagnostics(
    beta: np.ndarray,
    rolling: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    scope = slice(start - 1, end)
    scoped_beta = beta[scope]
    scoped_rolling = rolling[scope]
    valid = np.isfinite(scoped_rolling)
    changes = np.diff(scoped_beta)
    return {
        "mae_to_rolling100": float(np.mean(np.abs(scoped_beta[valid] - scoped_rolling[valid]))),
        "beta_total_variation": float(np.sum(np.abs(changes))),
        "beta_change_std": float(np.std(changes)),
        "beta_range": float(np.ptp(scoped_beta)),
    }


def main() -> None:
    prices = load_raw_prices()
    seed = next(spec for spec in dfil.PAIR_CFGS if spec.name == PAIR_NAME)
    log_y, log_x = np.log(prices[seed.y]), np.log(prices[seed.x])
    price_y, price_x = prices[seed.y], prices[seed.x]
    rolling = rolling_origin_beta(log_y, log_x)

    rows: list[dict[str, float | int]] = []
    beta_summary: list[dict[str, float | int]] = []
    for q_multiplier in Q_MULTIPLIERS:
        filt = seed.filt._replace(q_beta=float(seed.filt.q_beta * q_multiplier))
        path = record_paths(log_y, log_x, filt)
        gates = gate_and_z_paths(log_y, log_x, path["beta"])
        beta_summary.append(
            {
                "q_multiplier": q_multiplier,
                "q_beta": filt.q_beta,
                "cuts": int(np.sum(path["cut"])),
                "train_adf_block_rate": float(np.mean(gates["adf_block"][TRAIN_START - 2 : TRAIN_END - 1])),
                "train_kappa_block_rate": float(np.mean(gates["kappa_block"][TRAIN_START - 2 : TRAIN_END - 1])),
                "test_adf_block_rate": float(np.mean(gates["adf_block"][TEST_START - 2 : TEST_END - 1])),
                "test_kappa_block_rate": float(np.mean(gates["kappa_block"][TEST_START - 2 : TEST_END - 1])),
                **{f"train_{key}": value for key, value in beta_window_diagnostics(path["beta"], rolling, TRAIN_START, TRAIN_END).items()},
                **{f"test_{key}": value for key, value in beta_window_diagnostics(path["beta"], rolling, TEST_START, TEST_END).items()},
            }
        )

        for entry in ENTRIES:
            for exit_ in EXITS:
                if exit_ >= entry:
                    continue
                for stop in STOPS:
                    trade = seed.trade._replace(
                        z_entry=float(entry),
                        z_exit=float(exit_),
                        z_stop=float(stop),
                    )
                    train_y, train_x = trade_sim(
                        gates, price_y, price_x, trade, TRAIN_DECIDE_START
                    )
                    test_y, test_x = trade_sim(
                        gates, price_y, price_x, trade, TEST_DECIDE_START
                    )
                    train = window_metrics(
                        pnl_path(train_y, train_x, price_y, price_x),
                        train_y,
                        train_x,
                        price_y,
                        price_x,
                        TRAIN_START,
                        TRAIN_END,
                    )
                    test = window_metrics(
                        pnl_path(test_y, test_x, price_y, price_x),
                        test_y,
                        test_x,
                        price_y,
                        price_x,
                        TEST_START,
                        TEST_END,
                    )
                    row: dict[str, float | int] = {
                        "q_multiplier": q_multiplier,
                        "q_beta": filt.q_beta,
                        "z_entry": entry,
                        "z_exit": exit_,
                        "z_stop": stop,
                    }
                    for prefix, metrics in (("train", train), ("test", test)):
                        for key in (
                            "score", "mean", "sharpe", "sortino", "max_drawdown",
                            "cvar_5", "turnover", "entries", "positive_blocks",
                            "worst_block_mean", "median_block_score", "exposure",
                        ):
                            row[f"{prefix}_{key}"] = metrics[key]
                        for block_index, block in enumerate(metrics["blocks"], start=1):
                            row[f"{prefix}_block_{block_index}_mean"] = block["mean"]
                            row[f"{prefix}_block_{block_index}_score"] = block["score"]
                    rows.append(row)

    frame = pd.DataFrame(rows)
    frame["train_rank"] = (
        0.18 * percentile(frame["train_score"])
        + 0.12 * percentile(frame["train_mean"])
        + 0.12 * percentile(frame["train_sharpe"])
        + 0.08 * percentile(frame["train_sortino"])
        + 0.18 * percentile(frame["train_worst_block_mean"])
        + 0.12 * percentile(frame["train_median_block_score"])
        + 0.10 * percentile(frame["train_max_drawdown"], higher_is_better=False)
        + 0.05 * percentile(frame["train_cvar_5"])
        + 0.05 * percentile(frame["train_turnover"], higher_is_better=False)
    )
    eligible = frame[
        (frame["train_mean"] > 0.0)
        & (frame["train_entries"] >= 3)
        & (frame["train_positive_blocks"] >= 4)
    ].copy()
    eligible["neighbor_median_rank"] = np.nan
    eligible["neighbor_p10_rank"] = np.nan
    for index, candidate in eligible.iterrows():
        neighbors = eligible[
            (eligible["q_multiplier"] == candidate["q_multiplier"])
            & (np.abs(eligible["z_entry"] - candidate["z_entry"]) <= 0.05)
            & (np.abs(eligible["z_exit"] - candidate["z_exit"]) <= 0.05)
            & (np.abs(eligible["z_stop"] - candidate["z_stop"]) <= 0.75)
        ]
        eligible.loc[index, "neighbor_median_rank"] = float(neighbors["train_rank"].median())
        eligible.loc[index, "neighbor_p10_rank"] = float(neighbors["train_rank"].quantile(0.10))
    eligible["plateau_rank"] = (
        0.55 * eligible["train_rank"]
        + 0.30 * eligible["neighbor_median_rank"]
        + 0.15 * eligible["neighbor_p10_rank"]
    )

    baseline_mask = (
        np.isclose(frame["q_multiplier"], 1.0)
        & np.isclose(frame["z_entry"], seed.trade.z_entry)
        & np.isclose(frame["z_exit"], seed.trade.z_exit)
        & np.isclose(frame["z_stop"], seed.trade.z_stop)
    )
    baseline = frame[baseline_mask].iloc[0]
    leaderboard = eligible.nlargest(60, "plateau_rank")
    by_q = (
        eligible.sort_values("plateau_rank", ascending=False)
        .groupby("q_multiplier", as_index=False)
        .first()
        .sort_values("q_multiplier")
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / "focused_beta_threshold_surface.csv", index=False)
    leaderboard.to_csv(OUT_DIR / "focused_beta_threshold_leaderboard.csv", index=False)
    by_q.to_csv(OUT_DIR / "best_train_candidate_by_q.csv", index=False)
    pd.DataFrame(beta_summary).to_csv(OUT_DIR / "beta_speed_diagnostics.csv", index=False)
    summary = {
        "pair": PAIR_NAME,
        "selection_uses_test": False,
        "candidate_count": int(len(frame)),
        "eligible_count": int(len(eligible)),
        "baseline": baseline.to_dict(),
        "top_train_plateau": leaderboard.iloc[0].to_dict(),
        "beta_diagnostics": beta_summary,
    }
    (OUT_DIR / "focused_screen.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    columns = [
        "q_multiplier", "z_entry", "z_exit", "z_stop", "plateau_rank",
        "train_score", "train_mean", "train_sharpe", "train_max_drawdown",
        "test_score", "test_mean", "test_sharpe", "test_max_drawdown",
    ]
    print("baseline")
    print(baseline[columns[0:4] + columns[5:]].to_string())
    print("\nbest train-led plateau candidate at each beta speed")
    print(by_q[columns].to_string(index=False))
    print("\nbeta speed diagnostics")
    print(pd.DataFrame(beta_summary).to_string(index=False))


if __name__ == "__main__":
    main()

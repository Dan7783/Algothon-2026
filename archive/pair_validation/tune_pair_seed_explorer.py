"""Robust multi-metric tuner for a pair selected in pair_seed_explorer.

Candidate ranking uses only train days 251-500 and balances after-cost
mean/score, Sharpe, five chronological 50-day blocks, drawdown, left-tail PnL,
turnover, and trade count.  Test days 501-750 are computed only after the train
ranking and are reported as validation, not included in the ranking formula.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for module_dir in (REPO_ROOT, REPO_ROOT / "research", REPO_ROOT / "archive"):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import archive.old_tuners.daniel_filter as dfil
from archive.old_tuners.daniel_filter import FiltCfg, PairCfg, TradeCfg
from pair_book_lab import eval_score, load_raw_prices
from tune_daniel_pairs import (
    TEST_END,
    TEST_START,
    TRAIN_DECIDE_START,
    TRAIN_END,
    TRAIN_START,
    TEST_DECIDE_START,
    gate_and_z_paths,
    pnl_path,
    record_paths,
    trade_sim,
)


DEFAULT_PAIR_NAME = "CCNS/SRTX"
OUT_ROOT = Path("cusum_output/pair_seed_explorer")

ENTRY_GRID = (0.25, 0.35, 0.40, 0.50, 0.60, 0.625, 0.75, 0.90, 1.10, 1.30)
EXIT_GRID = (
    0.02, 0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225,
    0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40,
)
STOP_GRID = (3.00, 4.00, 5.00, 6.00)
Q_MULTIPLIERS = (0.50, 1.00, 2.00, 3.00)
GATE_MODES = ((True, True), (False, True), (False, False))


def max_drawdown(daily: np.ndarray) -> float:
    equity = np.r_[0.0, np.cumsum(daily)]
    return float(np.max(np.maximum.accumulate(equity) - equity))


def window_metrics(
    pnl: np.ndarray,
    shares_y: np.ndarray,
    shares_x: np.ndarray,
    price_y: np.ndarray,
    price_x: np.ndarray,
    start: int,
    end: int,
) -> dict[str, object]:
    daily = np.asarray(pnl[start : end + 1], dtype=float)
    mean = float(daily.mean())
    std = float(daily.std())
    downside = daily[daily < 0.0]
    downside_std = float(np.sqrt(np.mean(downside * downside))) if len(downside) else 0.0
    cvar_count = max(1, int(np.ceil(0.05 * len(daily))))
    cvar_5 = float(np.sort(daily)[:cvar_count].mean())

    block_metrics = []
    for block_start in range(start, end + 1, 50):
        block = daily[block_start - start : block_start - start + 50]
        block_mean = float(block.mean())
        block_std = float(block.std())
        block_metrics.append(
            {
                "start": block_start,
                "end": block_start + len(block) - 1,
                "mean": block_mean,
                "score": float(eval_score(block_mean, block_std)),
                "net": float(block.sum()),
            }
        )

    lo = max(start - 2, 1)
    hi = min(end - 1, len(price_y) - 1)
    turnover = float(
        np.sum(np.abs(np.diff(shares_y[lo - 1 : hi + 1])) * price_y[lo:hi + 1])
        + np.sum(np.abs(np.diff(shares_x[lo - 1 : hi + 1])) * price_x[lo:hi + 1])
    )
    side = np.sign(shares_y).astype(int)
    scope = side[lo : hi + 1]
    previous = side[lo - 1 : hi]
    entries = int(np.sum((scope != 0) & (previous == 0)))
    flips = int(np.sum((scope != 0) & (previous != 0) & (scope != previous)))

    positive = daily[daily > 0.0].sum()
    negative = -daily[daily < 0.0].sum()
    return {
        "net": float(daily.sum()),
        "mean": mean,
        "std": std,
        "score": float(eval_score(mean, std)),
        "sharpe": float(sqrt(250.0) * mean / std) if std > 0.0 else 0.0,
        "sortino": float(sqrt(250.0) * mean / downside_std) if downside_std > 0.0 else 0.0,
        "max_drawdown": max_drawdown(daily),
        "cvar_5": cvar_5,
        "daily_hit_rate": float(np.mean(daily > 0.0)),
        "profit_factor": float(positive / negative) if negative > 0.0 else float("inf"),
        "turnover": turnover,
        "entries": entries,
        "flips": flips,
        "exposure": float(np.mean(scope != 0)),
        "positive_blocks": int(sum(block["mean"] > 0.0 for block in block_metrics)),
        "worst_block_mean": float(min(block["mean"] for block in block_metrics)),
        "worst_block_score": float(min(block["score"] for block in block_metrics)),
        "median_block_score": float(np.median([block["score"] for block in block_metrics])),
        "blocks": block_metrics,
    }


def path_metrics(path: dict[str, np.ndarray]) -> dict[str, float | int]:
    beta = np.asarray(path["beta"])
    cut_idx = np.flatnonzero(path["cut"])
    jumps = (
        np.abs(beta[cut_idx] - beta[np.maximum(cut_idx - 1, 0)])
        if len(cut_idx)
        else np.zeros(1)
    )
    return {
        "cuts": int(len(cut_idx)),
        "beta_total_variation": float(np.abs(np.diff(beta)).sum()),
        "max_cut_jump": float(jumps.max()),
    }


def gate_metrics(gates: dict[str, np.ndarray]) -> dict[str, float]:
    def rate(key: str, start: int, end: int) -> float:
        return float(np.mean(gates[key][start - 2 : end - 1]))

    def median(key: str, start: int, end: int) -> float:
        return float(np.nanmedian(gates[key][start - 2 : end - 1]))

    return {
        "train_adf_block_rate": rate("adf_block", TRAIN_START, TRAIN_END),
        "train_kappa_block_rate": rate("kappa_block", TRAIN_START, TRAIN_END),
        "test_adf_block_rate": rate("adf_block", TEST_START, TEST_END),
        "test_kappa_block_rate": rate("kappa_block", TEST_START, TEST_END),
        "train_median_adf_t": median("adf_t", TRAIN_START, TRAIN_END),
        "test_median_adf_t": median("adf_t", TEST_START, TEST_END),
        "train_median_kappa": median("kappa", TRAIN_START, TRAIN_END),
        "test_median_kappa": median("kappa", TEST_START, TEST_END),
    }


def filter_grid(seed: PairCfg) -> list[FiltCfg]:
    base = seed.filt
    h_grid = sorted({max(2.0, base.h - 2.0), base.h, base.h + 2.0})
    k_grid = sorted({max(0.20, base.cusum_k - 0.10), base.cusum_k, min(0.90, base.cusum_k + 0.10)})
    configs = {
        base._replace(
            q_beta=float(base.q_beta * q_mult),
            h=float(h),
            cusum_k=float(k),
        )
        for q_mult in Q_MULTIPLIERS
        for h in h_grid
        for k in k_grid
    }
    configs.add(base)
    return sorted(configs, key=repr)


def trade_grid(seed: PairCfg) -> list[TradeCfg]:
    base = seed.trade
    entry_grid = sorted({*ENTRY_GRID, float(base.z_entry)})
    exit_grid = sorted({*EXIT_GRID, float(base.z_exit)})
    stop_grid = sorted({*STOP_GRID, float(base.z_stop)})
    configs = {
        base._replace(
            z_entry=float(entry),
            z_exit=float(min(exit_, 0.85 * entry)),
            z_stop=float(max(stop, 1.25 * entry)),
            require_adf=bool(require_adf),
            require_kappa=bool(require_kappa),
        )
        for entry in entry_grid
        for exit_ in exit_grid
        if exit_ < entry
        for stop in stop_grid
        for require_adf, require_kappa in GATE_MODES
    }
    configs.add(base)
    return sorted(configs, key=repr)


def percentile(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=higher_is_better)


def main(
    pair_name: str = DEFAULT_PAIR_NAME,
    reference_exit: float | None = None,
    out_dir: Path | None = None,
) -> None:
    if out_dir is None:
        out_dir = OUT_ROOT / pair_name.lower().replace("/", "_")
    prices = load_raw_prices()
    seed = next(spec for spec in dfil.PAIR_CFGS if spec.name == pair_name)
    ly, lx = np.log(prices[seed.y]), np.log(prices[seed.x])
    py, px = prices[seed.y], prices[seed.x]

    filters = filter_grid(seed)
    trades = trade_grid(seed)
    rows: list[dict[str, object]] = []
    for filter_index, filt in enumerate(filters):
        path = record_paths(ly, lx, filt)
        gates = gate_and_z_paths(ly, lx, path["beta"])
        path_info = {**path_metrics(path), **gate_metrics(gates)}
        for trade in trades:
            sy_train, sx_train = trade_sim(gates, py, px, trade, TRAIN_DECIDE_START)
            sy_test, sx_test = trade_sim(gates, py, px, trade, TEST_DECIDE_START)
            pnl_train = pnl_path(sy_train, sx_train, py, px)
            pnl_test = pnl_path(sy_test, sx_test, py, px)
            train = window_metrics(
                pnl_train, sy_train, sx_train, py, px, TRAIN_START, TRAIN_END
            )
            test = window_metrics(
                pnl_test, sy_test, sx_test, py, px, TEST_START, TEST_END
            )
            rows.append(
                {
                    "filter_index": filter_index,
                    "filt": filt,
                    "trade": trade,
                    "path": path_info,
                    "train": train,
                    "test": test,
                }
            )

    frame = pd.DataFrame(
        {
            "row": np.arange(len(rows)),
            "score": [row["train"]["score"] for row in rows],
            "mean": [row["train"]["mean"] for row in rows],
            "sharpe": [row["train"]["sharpe"] for row in rows],
            "sortino": [row["train"]["sortino"] for row in rows],
            "worst_block_mean": [row["train"]["worst_block_mean"] for row in rows],
            "median_block_score": [row["train"]["median_block_score"] for row in rows],
            "max_drawdown": [row["train"]["max_drawdown"] for row in rows],
            "cvar_5": [row["train"]["cvar_5"] for row in rows],
            "turnover": [row["train"]["turnover"] for row in rows],
            "entries": [row["train"]["entries"] for row in rows],
            "positive_blocks": [row["train"]["positive_blocks"] for row in rows],
        }
    )
    frame["test_score"] = [row["test"]["score"] for row in rows]
    frame["test_mean"] = [row["test"]["mean"] for row in rows]
    frame["test_sharpe"] = [row["test"]["sharpe"] for row in rows]
    frame["test_max_drawdown"] = [row["test"]["max_drawdown"] for row in rows]
    frame["q_beta"] = [row["filt"].q_beta for row in rows]
    frame["h"] = [row["filt"].h for row in rows]
    frame["cusum_k"] = [row["filt"].cusum_k for row in rows]
    frame["snap"] = [row["filt"].snap_overshoot for row in rows]
    frame["z_entry"] = [row["trade"].z_entry for row in rows]
    frame["z_exit"] = [row["trade"].z_exit for row in rows]
    frame["z_stop"] = [row["trade"].z_stop for row in rows]
    frame["require_adf"] = [row["trade"].require_adf for row in rows]
    frame["require_kappa"] = [row["trade"].require_kappa for row in rows]
    frame["balanced_rank"] = (
        0.18 * percentile(frame["score"])
        + 0.12 * percentile(frame["mean"])
        + 0.12 * percentile(frame["sharpe"])
        + 0.08 * percentile(frame["sortino"])
        + 0.18 * percentile(frame["worst_block_mean"])
        + 0.12 * percentile(frame["median_block_score"])
        + 0.10 * percentile(frame["max_drawdown"], higher_is_better=False)
        + 0.05 * percentile(frame["cvar_5"])
        + 0.05 * percentile(frame["turnover"], higher_is_better=False)
    )
    eligible = frame[
        (frame["mean"] > 0.0)
        & (frame["entries"] >= 3)
        & (frame["positive_blocks"] >= 4)
    ].copy()
    if eligible.empty:
        raise RuntimeError("no candidate passed minimum robustness constraints")

    # The winning neighborhood should be a plateau, not one lucky grid point.
    top = eligible.nlargest(40, "balanced_rank").copy()
    for index, item in top.iterrows():
        row = rows[int(item["row"])]
        filt, trade = row["filt"], row["trade"]
        neighbors = frame[
            frame["row"].map(
                lambda r: (
                    rows[int(r)]["filt"].h == filt.h
                    and rows[int(r)]["filt"].cusum_k == filt.cusum_k
                    and rows[int(r)]["filt"].q_beta == filt.q_beta
                    and rows[int(r)]["filt"].snap_overshoot == filt.snap_overshoot
                    and abs(rows[int(r)]["trade"].z_entry - trade.z_entry) <= 0.20
                    and abs(rows[int(r)]["trade"].z_exit - trade.z_exit) <= 0.10
                    and abs(rows[int(r)]["trade"].z_stop - trade.z_stop) <= 0.75
                    and rows[int(r)]["trade"].require_adf == trade.require_adf
                    and rows[int(r)]["trade"].require_kappa == trade.require_kappa
                )
            )
        ]
        top.loc[index, "neighbor_median_rank"] = float(neighbors["balanced_rank"].median())
        top.loc[index, "neighbor_p10_rank"] = float(neighbors["balanced_rank"].quantile(0.10))
        top.loc[index, "neighbor_count"] = int(len(neighbors))

    top["selection_value"] = (
        0.55 * top["balanced_rank"]
        + 0.30 * top["neighbor_median_rank"]
        + 0.15 * top["neighbor_p10_rank"]
    )
    chosen_index = int(top.nlargest(1, "selection_value").iloc[0]["row"])
    chosen = rows[chosen_index]

    # Prefer the smallest defensible production change.  The filter and stop
    # remain frozen at the live seed; only entry/exit hysteresis may move.  A
    # local threshold plateau is again required, so the result is not a lucky
    # single grid point.
    simple = eligible[
        (eligible["q_beta"] == seed.filt.q_beta)
        & (eligible["h"] == seed.filt.h)
        & (eligible["cusum_k"] == seed.filt.cusum_k)
        & (eligible["snap"] == seed.filt.snap_overshoot)
        & (eligible["z_stop"] == seed.trade.z_stop)
    ].copy()
    for index, item in simple.iterrows():
        neighbors = simple[
            (np.abs(simple["z_entry"] - item["z_entry"]) <= 0.15)
            & (np.abs(simple["z_exit"] - item["z_exit"]) <= 0.05)
            & (simple["require_adf"] == item["require_adf"])
            & (simple["require_kappa"] == item["require_kappa"])
        ]
        simple.loc[index, "neighbor_median_rank"] = float(
            neighbors["balanced_rank"].median()
        )
        simple.loc[index, "neighbor_p10_rank"] = float(
            neighbors["balanced_rank"].quantile(0.10)
        )
    simple["selection_value"] = (
        0.60 * simple["balanced_rank"]
        + 0.25 * simple["neighbor_median_rank"]
        + 0.15 * simple["neighbor_p10_rank"]
    )
    simple_index = int(simple.nlargest(1, "selection_value").iloc[0]["row"])
    simple_chosen = rows[simple_index]
    legacy_trade = (
        seed.trade
        if reference_exit is None
        else seed.trade._replace(z_exit=float(reference_exit))
    )
    baseline = next(
        row for row in rows if row["filt"] == seed.filt and row["trade"] == legacy_trade
    )
    live_seed = next(
        row for row in rows if row["filt"] == seed.filt and row["trade"] == seed.trade
    )
    max_score = rows[int(frame.nlargest(1, "score").iloc[0]["row"])]

    def serialize(row: dict[str, object]) -> dict[str, object]:
        return {
            "filt": row["filt"]._asdict(),
            "trade": row["trade"]._asdict(),
            "path": row["path"],
            "train": row["train"],
            "test": row["test"],
        }

    report = {
        "pair": pair_name,
        "selection_uses_test": False,
        "candidate_count": len(rows),
        "eligible_count": int(len(eligible)),
        "baseline": serialize(baseline),
        "live_seed": serialize(live_seed),
        "balanced_chosen": serialize(chosen),
        "minimal_change_chosen": serialize(simple_chosen),
        "max_train_score": serialize(max_score),
        "chosen_train_rank": float(frame.loc[chosen_index, "balanced_rank"]),
        "chosen_neighborhood": {
            key: float(top[top["row"] == chosen_index].iloc[0][key])
            for key in ("neighbor_median_rank", "neighbor_p10_rank", "neighbor_count")
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "robust_tuning.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    leaderboard = top.nlargest(40, "selection_value").copy()
    leaderboard.to_csv(out_dir / "robust_leaderboard.csv", index=False)
    frame.to_csv(out_dir / "all_candidates.csv", index=False)

    for label, row in (
        ("baseline", baseline),
        ("live", live_seed),
        ("balanced", chosen),
        ("minimal", simple_chosen),
        ("max-score", max_score),
    ):
        print(
            f"{label:9s} train score/mean/sharpe/dd "
            f"{row['train']['score']:7.2f}/{row['train']['mean']:7.2f}/"
            f"{row['train']['sharpe']:5.2f}/{row['train']['max_drawdown']:7.0f}  "
            f"test {row['test']['score']:7.2f}/{row['test']['mean']:7.2f}/"
            f"{row['test']['sharpe']:5.2f}/{row['test']['max_drawdown']:7.0f}"
        )
        print("  filt ", row["filt"])
        print("  trade", row["trade"])
        print("  train blocks", [round(block["mean"], 1) for block in row["train"]["blocks"]])
        print("  test  blocks", [round(block["mean"], 1) for block in row["test"]["blocks"]])
    print(f"wrote {out_dir / 'robust_tuning.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default=DEFAULT_PAIR_NAME)
    parser.add_argument("--reference-exit", type=float)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    main(args.pair, args.reference_exit, args.out_dir)

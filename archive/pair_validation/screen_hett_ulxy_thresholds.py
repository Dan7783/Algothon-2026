"""Focused threshold plateau for the live HETT/ULXY filter and gates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for module_dir in (ROOT, ROOT / "research", ROOT / "archive"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

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
from tune_pair_seed_explorer import percentile, window_metrics


PAIR_NAME = "HETT/ULXY"
ENTRIES = tuple(sorted({
    0.27, 0.50, 0.55, 0.60, 0.625, 0.65, 0.675,
    *(round(0.70 + 0.005 * index, 3) for index in range(21)),
    0.825, 0.85, 0.875, 0.90, 0.95,
}))
EXITS = tuple(sorted({
    *(round(0.02 + 0.005 * index, 3) for index in range(17)),
    0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30,
    0.325, 0.35, 0.375, 0.40,
}))
OUT = Path("cusum_output/pair_seed_explorer/hett_ulxy/threshold_surface.csv")


def main() -> None:
    prices = load_raw_prices()
    seed = next(spec for spec in dfil.PAIR_CFGS if spec.name == PAIR_NAME)
    py, px = prices[seed.y], prices[seed.x]
    path = record_paths(np.log(py), np.log(px), seed.filt)
    gates = gate_and_z_paths(np.log(py), np.log(px), path["beta"])
    rows = []
    for entry in ENTRIES:
        for exit_z in EXITS:
            if exit_z >= entry:
                continue
            trade = seed.trade._replace(z_entry=entry, z_exit=exit_z)
            sy, sx = trade_sim(gates, py, px, trade, TRAIN_DECIDE_START)
            train = window_metrics(
                pnl_path(sy, sx, py, px), sy, sx, py, px, TRAIN_START, TRAIN_END
            )
            sy, sx = trade_sim(gates, py, px, trade, TEST_DECIDE_START)
            test = window_metrics(
                pnl_path(sy, sx, py, px), sy, sx, py, px, TEST_START, TEST_END
            )
            rows.append({
                "z_entry": entry,
                "z_exit": exit_z,
                **{f"train_{key}": train[key] for key in (
                    "score", "mean", "sharpe", "sortino", "worst_block_mean",
                    "max_drawdown", "cvar_5", "turnover", "entries", "positive_blocks",
                )},
                "train_block_means": "|".join(
                    f"{block['mean']:.12g}" for block in train["blocks"]
                ),
                **{f"test_{key}": test[key] for key in (
                    "score", "mean", "sharpe", "sortino", "worst_block_mean",
                    "max_drawdown", "cvar_5", "turnover", "entries", "positive_blocks",
                )},
                "test_block_means": "|".join(
                    f"{block['mean']:.12g}" for block in test["blocks"]
                ),
            })

    frame = pd.DataFrame(rows)
    frame["train_rank"] = (
        0.20 * percentile(frame["train_score"])
        + 0.15 * percentile(frame["train_mean"])
        + 0.15 * percentile(frame["train_sharpe"])
        + 0.10 * percentile(frame["train_sortino"])
        + 0.20 * percentile(frame["train_worst_block_mean"])
        + 0.10 * percentile(frame["train_cvar_5"])
        + 0.10 * percentile(frame["train_turnover"], higher_is_better=False)
    )
    frame.to_csv(OUT, index=False)
    columns = [
        "z_entry", "z_exit", "train_score", "train_mean", "train_sharpe",
        "train_worst_block_mean", "train_cvar_5", "train_turnover",
        "test_score", "test_mean", "test_sharpe", "test_worst_block_mean",
        "train_rank",
    ]
    print(frame.nlargest(30, "train_rank")[columns].to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

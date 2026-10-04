#!/usr/bin/env python
"""Identify current pair agents with negative standalone days-751-1000 P&L.

This is intentionally an OOS selection tool.  Once its result is used to
change the strategy, days 751-1000 are no longer OOS for that new strategy.
Ridge and pair-conflict arbitration are excluded so every configured
relationship is tested on the signal it would generate independently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.shortlist_pair_review import account_window, load_prices
import testbedv2_tuned_ridge_protectedbarbell as strategy


START_DAY = 751
END_DAY = 1000
OUTPUT = ROOT / "analysis_outputs" / "protectedbarbell_pair_oos.csv"


def pair_name(cfg: strategy.PairCfg) -> str:
    inverse = {index: symbol for symbol, index in strategy.I.items()}
    return f"{inverse[int(cfg.y)]}-{inverse[int(cfg.x)]}"


def standalone_targets(
    prices: np.ndarray,
    cfg: strategy.PairCfg,
) -> np.ndarray:
    targets = np.zeros((END_DAY + 1, strategy.N_INST), dtype=int)
    agent = strategy.PairAgent(cfg)
    warmup = int(strategy.WARMUP) + 10
    eps = float(strategy._base._EPS_DENOM)

    for day in range(1, END_DAY + 1):
        agent.catch_up(prices[:, :day])
        if day < warmup:
            continue

        pos_y, pos_x, _ = agent.dollar_positions()
        dollars = np.zeros(strategy.N_INST)
        dollars[int(cfg.y)] = pos_y
        dollars[int(cfg.x)] = pos_x

        last_return = np.diff(
            np.log(np.maximum(prices[:, day - 2 : day], eps)),
            axis=1,
        )[:, 0]
        crash = np.abs(last_return) > 0.40
        crash[0] = False
        dollars[crash] = 0.0
        dollars[1:] = np.clip(
            dollars[1:],
            -strategy.LEG,
            strategy.LEG,
        )
        dollars[0] = np.clip(dollars[0], -100_000.0, 100_000.0)
        targets[day] = np.round(
            dollars / np.maximum(prices[:, day - 1], eps)
        ).astype(int)
    return targets


def main() -> None:
    prices, _ = load_prices(ROOT / "prices.txt")
    rows = []
    for cfg in strategy.PAIR_CFGS:
        targets = standalone_targets(prices, cfg)
        metrics = account_window(
            prices,
            targets,
            START_DAY,
            END_DAY,
        )
        name = pair_name(cfg)
        active_days = int(
            np.any(
                targets[START_DAY:END_DAY] != 0,
                axis=1,
            ).sum()
        )
        total_pnl = float(metrics["total_pnl"])
        rows.append(
            {
                "pair_name": name,
                "protected": name in strategy.PROTECTED_PAIR_NAMES,
                "active_days": active_days,
                "total_pnl": total_pnl,
                "mean_pnl": float(metrics["mean_pnl"]),
                "score": float(metrics["score"]),
                "sharpe": float(metrics["sharpe"]),
                "dollar_volume": float(metrics["dollar_volume"]),
                "loser": total_pnl < -1.0e-9,
                "no_trade": active_days == 0,
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        ["loser", "total_pnl"],
        ascending=[False, True],
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)

    losers = frame[frame["loser"]]
    inactive = frame[frame["no_trade"]]
    print(
        f"pairs={len(frame)} losers={len(losers)} "
        f"no_trade={len(inactive)}"
    )
    print("\nNegative standalone OOS P&L:")
    print(
        losers[
            [
                "pair_name",
                "protected",
                "active_days",
                "total_pnl",
                "score",
                "sharpe",
            ]
        ].to_string(index=False)
    )
    if len(inactive):
        print("\nNo OOS trades:")
        print(
            inactive[["pair_name", "protected"]].to_string(index=False)
        )
    print(f"\nSaved {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

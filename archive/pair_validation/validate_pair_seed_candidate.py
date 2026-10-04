"""Portfolio-level validation for the RTTH/NAYO minimal-change candidates.

This is deliberately a one-dimensional check: the live filter, entry and stop
remain fixed while z_exit moves across the stable pair-level plateau.  Each
candidate is replayed independently on train and test with the selected
robust_hybrid_v2 production policy and eval-style delayed commissions.
"""

from __future__ import annotations

import sys
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
import robust_hybrid_v2 as strategy


PAIR_NAME = "RTTH/NAYO"
EXIT_CANDIDATES = (0.02, 0.175, 0.225, 0.25, 0.275, 0.30, 0.35, 0.40)
COMM = np.full(51, 0.0001)
COMM[0] = 0.00002
LIMIT = np.full(51, 10_000.0)
LIMIT[0] = 100_000.0


def score(mean: float, std: float) -> float:
    if mean <= 0.0 or std < 1e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe * sharpe / (sharpe * sharpe + 1.0)


def stats(daily: np.ndarray, turnover: float) -> dict[str, object]:
    mean, std = float(daily.mean()), float(daily.std())
    negative = daily[daily < 0.0]
    downside = float(np.sqrt(np.mean(negative * negative))) if len(negative) else 0.0
    equity = np.r_[0.0, np.cumsum(daily)]
    drawdown = np.maximum.accumulate(equity) - equity
    tail_n = max(1, int(np.ceil(0.05 * len(daily))))
    halves = np.array_split(daily, 2)
    return {
        "net": float(daily.sum()),
        "mean": mean,
        "std": std,
        "score": float(score(mean, std)),
        "sharpe": float(sqrt(250.0) * mean / std) if std > 0.0 else 0.0,
        "sortino": float(sqrt(250.0) * mean / downside) if downside > 0.0 else 0.0,
        "max_drawdown": float(drawdown.max()),
        "cvar_5": float(np.sort(daily)[:tail_n].mean()),
        "turnover": float(turnover),
        "positive_days": float(np.mean(daily > 0.0)),
        "half_scores": [
            float(score(float(block.mean()), float(block.std()))) for block in halves
        ],
        "half_means": [float(block.mean()) for block in halves],
    }


def replay(prices: np.ndarray, start: int, end: int) -> dict[str, object]:
    strategy.apply_policy(dict(strategy.SELECTED_POLICY))
    strategy.USE_EXTRA_LL = True
    strategy.USE_RIDGE = True
    strategy.USE_DISCOVERY = True
    strategy.RIDGE_ENTRY = 1.0
    strategy.DISCOVERY_SCALE = 1.0
    strategy.reset()

    cash = value = previous_commission = turnover = 0.0
    current_position = np.zeros(prices.shape[0])
    daily: list[float] = []
    for t in range(start - 1, end + 1):
        history = prices[:, :t]
        current_prices = history[:, -1]
        if t < end:
            requested = strategy.getMyPosition(history)
            limits = (LIMIT / current_prices).astype(int)
            new_position = np.clip(requested, -limits, limits).astype(int)
        else:
            new_position = current_position.copy()

        delta = new_position - current_position
        cash -= float(current_prices @ delta) + previous_commission
        dollar_volume = current_prices * np.abs(delta)
        turnover += float(dollar_volume.sum())
        previous_commission = float(dollar_volume @ COMM)
        current_position = new_position
        new_value = cash + float(current_position @ current_prices)
        today = new_value - value
        value = new_value
        if t >= start:
            daily.append(today)
    return stats(np.asarray(daily), turnover)


def main() -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    original = tuple(dfil.PAIR_CFGS)
    try:
        for exit_z in EXIT_CANDIDATES:
            candidate = tuple(
                spec._replace(trade=spec.trade._replace(z_exit=exit_z))
                if spec.name == PAIR_NAME
                else spec
                for spec in original
            )
            dfil.apply_pair_cfgs(candidate)
            train = replay(prices, 251, 500)
            test = replay(prices, 501, 750)
            print(
                f"exit={exit_z:>5g}  "
                f"train score/mean/sharpe/dd={train['score']:8.3f}/"
                f"{train['mean']:8.3f}/{train['sharpe']:6.3f}/"
                f"{train['sortino']:6.3f}/{train['max_drawdown']:8.1f}/"
                f"{train['cvar_5']:8.1f} turnover={train['turnover']:10.0f} "
                f"halves={train['half_scores']}  "
                f"test={test['score']:8.3f}/{test['mean']:8.3f}/"
                f"{test['sharpe']:6.3f}/{test['sortino']:6.3f}/"
                f"{test['max_drawdown']:8.1f}/{test['cvar_5']:8.1f} "
                f"turnover={test['turnover']:10.0f} "
                f"halves={test['half_scores']}"
            )
    finally:
        dfil.apply_pair_cfgs(original)
        strategy.apply_policy(dict(strategy.SELECTED_POLICY))


if __name__ == "__main__":
    main()

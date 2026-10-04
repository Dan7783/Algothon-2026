"""Reproducible train/test ablation audit for robust_hybrid.py."""

from __future__ import annotations

# --- repo path bootstrap: run as `python research/<script>.py` from root ---
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "research"), str(_REPO_ROOT / "archive")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

import robust_hybrid_v2 as strategy


OUT = Path("cusum_output/robust_hybrid/audit.json")
COMM = np.full(51, 0.0001)
COMM[0] = 0.00002
LIMIT = np.full(51, 10_000.0)
LIMIT[0] = 100_000.0


def score(mean: float, std: float) -> float:
    if mean <= 0.0 or std < 1e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe * sharpe / (sharpe * sharpe + 1.0)


def stats(pnl: np.ndarray, turnover: float) -> dict[str, float | list[float]]:
    mean = float(pnl.mean())
    std = float(pnl.std())
    blocks = np.array_split(pnl, 2)
    return {
        "net": float(pnl.sum()),
        "mean": mean,
        "std": std,
        "sharpe": float(sqrt(250.0) * mean / std) if std > 0.0 else 0.0,
        "score": float(score(mean, std)),
        "turnover": float(turnover),
        "half_scores": [float(score(float(x.mean()), float(x.std()))) for x in blocks],
    }


def replay(prices: np.ndarray, start: int, end: int) -> dict[str, object]:
    strategy.reset()
    cash = value = previous_commission = turnover = 0.0
    current_position = np.zeros(prices.shape[0])
    pnl: list[float] = []
    max_dollars = 0.0

    for t in range(start - 1, end + 1):
        history = prices[:, :t]
        current_prices = history[:, -1]
        if t < end:
            requested = strategy.getMyPosition(history)
            limits = (LIMIT / current_prices).astype(int)
            new_position = np.clip(requested, -limits, limits).astype(int)
            max_dollars = max(
                max_dollars,
                float(np.max(np.abs(new_position * current_prices))),
            )
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
            pnl.append(today)

    result = stats(np.asarray(pnl), turnover)
    result["max_abs_dollars"] = max_dollars
    return result


def configure(
    *,
    discovery: bool,
    extra: bool,
    ridge: float,
    scale: float = 1.0,
    ridge_on: bool = True,
) -> None:
    strategy.USE_DISCOVERY = discovery
    strategy.USE_EXTRA_LL = extra
    strategy.USE_RIDGE = ridge_on
    strategy.RIDGE_ENTRY = ridge
    strategy.DISCOVERY_SCALE = scale


def main() -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    variants = {
        "daniel_core": dict(discovery=False, extra=True, ridge=1.0),
        "hybrid_scale_025": dict(discovery=True, extra=True, ridge=1.0, scale=0.25),
        "hybrid_scale_050": dict(discovery=True, extra=True, ridge=1.0, scale=0.50),
        "hybrid_scale_075": dict(discovery=True, extra=True, ridge=1.0, scale=0.75),
        "hybrid_default": dict(discovery=True, extra=True, ridge=1.0),
        "hybrid_no_extra": dict(discovery=True, extra=False, ridge=1.0),
        "hybrid_ridge_075": dict(discovery=True, extra=True, ridge=0.75),
        "hybrid_no_extra_ridge_075": dict(discovery=True, extra=False, ridge=0.75),
    }
    report: dict[str, object] = {}
    for name, config in variants.items():
        configure(**config)
        train = replay(prices, 251, 500)
        test = replay(prices, 501, 750)
        report[name] = {"config": config, "train": train, "test": test}
        print(
            f"{name:28s} train {train['score']:8.2f} / {train['mean']:8.2f}  "
            f"test {test['score']:8.2f} / {test['mean']:8.2f}"
        )

    # Restore production constants even when this module is run interactively.
    configure(discovery=True, extra=True, ridge=1.0, scale=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

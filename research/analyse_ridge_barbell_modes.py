#!/usr/bin/env python
"""Train-only attribution of ridge/pair priority rules.

Only days 1-750 are loaded into the replay.  The script compares standalone
ridge, standalone pairs, classic pair-priority, full ridge-priority, and
high-confidence ridge override thresholds.  It does not modify a strategy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.shortlist_pair_review import load_prices
import testbedv2_tuned_ridge_smallbarbell as strategy


COMMISSIONS = np.full(strategy.N_INST, 1.0e-4)
COMMISSIONS[0] = 2.0e-5
LIMITS = np.full(strategy.N_INST, strategy.LEG)
LIMITS[0] = 100_000.0
THRESHOLDS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
WINDOWS = (("year2", 251, 500), ("year3", 501, 750))


def score(mean: float, std: float) -> float:
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = np.sqrt(250.0) * mean / std
    return mean * sharpe * sharpe / (sharpe * sharpe + 1.0)


def pair_conflict_mask(
    pair_dollars: np.ndarray,
    strong_ridge: np.ndarray,
) -> np.ndarray:
    suppressed = np.zeros(strategy.N_INST, dtype=bool)
    for spec in strategy.PAIR_CFGS:
        if (
            abs(pair_dollars[spec.y]) > 1.0
            and abs(pair_dollars[spec.x]) > 1.0
            and (strong_ridge[spec.y] or strong_ridge[spec.x])
        ):
            suppressed[spec.y] = True
            suppressed[spec.x] = True
    return suppressed


def subordinate_ridge(
    ridge_raw: np.ndarray,
    pair_dollars: np.ndarray,
) -> np.ndarray:
    allocation = np.zeros(strategy.N_INST)
    held = pair_dollars[1:]
    raw = ridge_raw[1:]
    idle = np.abs(held) <= 1.0
    allocation[1:][idle] = raw[idle]
    agrees = (
        (~idle)
        & (np.abs(raw) > 1.0)
        & (np.sign(raw) == np.sign(held))
    )
    room = np.maximum(
        0.0,
        strategy.LEG - np.minimum(np.abs(held), strategy.LEG),
    )
    allocation[1:][agrees] = (
        np.sign(raw[agrees])
        * np.minimum(np.abs(raw[agrees]), room[agrees])
    )
    return allocation


def barbell(
    pair_raw: np.ndarray,
    ridge_raw: np.ndarray,
    ridge_strength: np.ndarray,
    override_threshold: float,
) -> np.ndarray:
    strong = (
        (np.abs(ridge_raw) > 1.0)
        & (ridge_strength > override_threshold)
    )
    suppressed = pair_conflict_mask(pair_raw, strong)
    pairs = pair_raw.copy()
    pairs[suppressed] = 0.0
    ridge = subordinate_ridge(ridge_raw, pairs)
    ridge[strong] = ridge_raw[strong]
    return pairs + ridge


def to_shares(
    dollars: np.ndarray,
    prices: np.ndarray,
) -> np.ndarray:
    target = dollars.copy()
    last_return = np.diff(np.log(prices[:, -2:]), axis=1)[:, 0]
    crash = np.abs(last_return) > 0.40
    crash[0] = False
    target[crash] = 0.0
    target[1:] = np.clip(target[1:], -strategy.LEG, strategy.LEG)
    target[0] = 0.0
    return np.round(target / prices[:, -1]).astype(int)


def account(
    prices: np.ndarray,
    share_map: dict[int, np.ndarray],
    start: int,
    end: int,
) -> tuple[float, float, float]:
    cash = value = commission = 0.0
    positions = np.zeros(strategy.N_INST)
    pnl: list[float] = []
    for day in range(start - 1, end + 1):
        current = prices[:, day - 1]
        if day < end and day in share_map:
            position_limit = (LIMITS / current).astype(int)
            new_positions = np.clip(
                share_map[day],
                -position_limit,
                position_limit,
            ).astype(int)
        else:
            new_positions = positions.astype(int)
        delta = new_positions - positions
        cash -= float(current @ delta) + commission
        commission = float(
            np.sum(current * np.abs(delta) * COMMISSIONS)
        )
        positions = new_positions.astype(float)
        new_value = cash + float(positions @ current)
        if day >= start:
            pnl.append(new_value - value)
        value = new_value
    values = np.asarray(pnl, dtype=float)
    mean = float(values.mean())
    std = float(values.std())
    return mean, std, score(mean, std)


def main() -> None:
    all_prices, _ = load_prices(ROOT / "prices.txt")
    prices = all_prices[:, :750].copy()
    book = strategy.AtlasBook()

    modes = {
        "ridge_only": {},
        "pairs_only": {},
        "pair_priority": {},
        **{f"override_{threshold:g}": {} for threshold in THRESHOLDS},
    }
    for length in range(strategy.WARMUP + 10, 751):
        history = prices[:, :length]
        returns = np.diff(np.log(history), axis=1)
        ridge_raw, ridge_strength = strategy._ridge._ridge_forecast(
            returns
        )
        pair_raw, _ = book.dollars(history, ridge_active=None)

        targets = {
            "ridge_only": ridge_raw,
            "pairs_only": pair_raw,
            "pair_priority": (
                pair_raw + subordinate_ridge(ridge_raw, pair_raw)
            ),
            **{
                f"override_{threshold:g}": barbell(
                    pair_raw,
                    ridge_raw,
                    ridge_strength,
                    threshold,
                )
                for threshold in THRESHOLDS
            },
        }
        for name, dollars in targets.items():
            modes[name][length] = to_shares(dollars, history)

    print(
        f"{'mode':>16} | "
        + " | ".join(
            f"{name + ' score':>13} {'mean':>8}"
            for name, _, _ in WINDOWS
        )
        + " | floor"
    )
    print("-" * 82)
    for name, share_map in modes.items():
        results = [
            account(prices, share_map, start, end)
            for _, start, end in WINDOWS
        ]
        floor = min(result[2] for result in results)
        fields = " | ".join(
            f"{result[2]:13.2f} {result[0]:8.2f}"
            for result in results
        )
        print(f"{name:>16} | {fields} | {floor:5.2f}")


if __name__ == "__main__":
    main()

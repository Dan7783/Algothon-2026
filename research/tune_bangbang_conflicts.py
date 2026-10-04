#!/usr/bin/env python
"""Train/validation/test tuning of bang-bang pair/ridge conflicts.

Conflict and sizing configurations are fitted on days 31-500.  The top 20
training configurations advance to days 501-750 validation, which selects one
frozen configuration.  Days 751-1000 are evaluated only for that winner.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import testbedv2_tuned_ridge_protectedbarbell as strategy


COMMISSIONS = np.full(strategy.N_INST, 1.0e-4)
COMMISSIONS[0] = 2.0e-5
LIMITS = np.full(strategy.N_INST, strategy.LEG)
LIMITS[0] = 100_000.0
TRAIN_FOLDS = (
    ("y1a", 31, 140),
    ("y1b", 141, 250),
    ("y2a", 251, 375),
    ("y2b", 376, 500),
)
TRAIN = ("train", 31, 500)
VALIDATION = ("validation", 501, 750)
TEST = ("test", 751, 1000)
TRAIN_SHORTLIST_SIZE = 20


@dataclass(frozen=True)
class ConflictCfg:
    pair_mode: str
    protected_boost: float
    ridge_weight: float
    ridge_mode: str
    algo_limit: float


@dataclass
class DaySignals:
    ordinary_equal_num: np.ndarray
    ordinary_equal_den: np.ndarray
    protected_equal_num: np.ndarray
    protected_equal_den: np.ndarray
    ordinary_priority_num: np.ndarray
    ordinary_priority_den: np.ndarray
    protected_priority_num: np.ndarray
    protected_priority_den: np.ndarray
    ridge_sign: np.ndarray
    ridge_strength: np.ndarray
    crash: np.ndarray


def score(mean: float, std: float) -> float:
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = np.sqrt(250.0) * mean / std
    return mean * sharpe * sharpe / (sharpe * sharpe + 1.0)


def account(
    prices: np.ndarray,
    shares: dict[int, np.ndarray],
    start: int,
    end: int,
) -> tuple[float, float, float]:
    cash = value = previous_commission = 0.0
    positions = np.zeros(strategy.N_INST)
    pnl = []
    for day in range(start - 1, end + 1):
        current_prices = prices[:, day - 1]
        if day < end:
            requested = shares.get(day, np.zeros(strategy.N_INST))
            limits = (LIMITS / current_prices).astype(int)
            new_positions = np.clip(
                requested,
                -limits,
                limits,
            ).astype(int)
        else:
            new_positions = positions.astype(int)
        delta = new_positions - positions
        cash -= float(current_prices @ delta) + previous_commission
        volume = current_prices * np.abs(delta)
        previous_commission = float(volume @ COMMISSIONS)
        positions = new_positions.astype(float)
        new_value = cash + float(positions @ current_prices)
        if day >= start:
            pnl.append(new_value - value)
        value = new_value
    values = np.asarray(pnl)
    mean = float(values.mean())
    std = float(values.std())
    return mean, std, score(mean, std)


def add_vote(
    numerator: np.ndarray,
    denominator: np.ndarray,
    index: int,
    position: float,
    weight: float,
) -> None:
    if abs(position) <= 1.0e-9:
        return
    numerator[index] += weight * np.sign(position)
    denominator[index] += weight


def precompute(prices: np.ndarray) -> dict[int, DaySignals]:
    book = strategy.AtlasBook()
    output = {}
    for length in range(strategy.WARMUP + 10, prices.shape[1] + 1):
        history = prices[:, :length]
        candidates = book.candidates(history)
        arrays = [np.zeros(strategy.N_INST) for _ in range(8)]
        (
            ordinary_equal_num,
            ordinary_equal_den,
            protected_equal_num,
            protected_equal_den,
            ordinary_priority_num,
            ordinary_priority_den,
            protected_priority_num,
            protected_priority_den,
        ) = arrays

        for priority, agent, pos_y, pos_x in candidates:
            protected = (
                strategy._pair_name(agent.spec)
                in strategy.PROTECTED_PAIR_NAMES
            )
            if protected:
                equal_num, equal_den = (
                    protected_equal_num,
                    protected_equal_den,
                )
                priority_num, priority_den = (
                    protected_priority_num,
                    protected_priority_den,
                )
            else:
                equal_num, equal_den = (
                    ordinary_equal_num,
                    ordinary_equal_den,
                )
                priority_num, priority_den = (
                    ordinary_priority_num,
                    ordinary_priority_den,
                )
            for index, position in (
                (int(agent.spec.y), pos_y),
                (int(agent.spec.x), pos_x),
            ):
                add_vote(equal_num, equal_den, index, position, 1.0)
                add_vote(
                    priority_num,
                    priority_den,
                    index,
                    position,
                    max(float(priority), 1.0e-12),
                )

        returns = np.diff(np.log(history), axis=1)
        ridge_raw, ridge_strength = strategy._ridge_forecast(returns)
        ridge_sign = np.sign(ridge_raw)
        last_return = np.diff(np.log(history[:, -2:]), axis=1)[:, 0]
        crash = np.abs(last_return) > 0.40
        crash[0] = False
        output[length] = DaySignals(
            ordinary_equal_num,
            ordinary_equal_den,
            protected_equal_num,
            protected_equal_den,
            ordinary_priority_num,
            ordinary_priority_den,
            protected_priority_num,
            protected_priority_den,
            ridge_sign,
            ridge_strength,
            crash,
        )
    return output


def pair_consensus(
    signals: DaySignals,
    cfg: ConflictCfg,
) -> tuple[np.ndarray, np.ndarray]:
    if cfg.pair_mode == "equal":
        ordinary_num = signals.ordinary_equal_num
        ordinary_den = signals.ordinary_equal_den
        protected_num = signals.protected_equal_num
        protected_den = signals.protected_equal_den
    else:
        ordinary_num = signals.ordinary_priority_num
        ordinary_den = signals.ordinary_priority_den
        protected_num = signals.protected_priority_num
        protected_den = signals.protected_priority_den
    numerator = (
        ordinary_num + cfg.protected_boost * protected_num
    )
    denominator = (
        ordinary_den + cfg.protected_boost * protected_den
    )
    consensus = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return consensus, denominator > 0.0


def targets_for_cfg(
    prices: np.ndarray,
    all_signals: dict[int, DaySignals],
    cfg: ConflictCfg,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    shares = {}
    ridge_active_count = ridge_wins = active_count = 0
    ridge_conflicts = ridge_conflict_wins = 0
    for length, signals in all_signals.items():
        pair_vote, pair_active = pair_consensus(signals, cfg)
        ridge_active = signals.ridge_sign != 0.0
        if cfg.ridge_mode == "confidence":
            ridge_vote = (
                cfg.ridge_weight
                * signals.ridge_sign
                * signals.ridge_strength
            )
        else:
            ridge_vote = cfg.ridge_weight * signals.ridge_sign

        combined = pair_vote + ridge_vote
        active = pair_active | ridge_active
        direction = np.sign(combined)
        tied = active & (direction == 0.0)
        direction[tied & ridge_active] = signals.ridge_sign[
            tied & ridge_active
        ]
        # A pair-only exact tie is left flat; no directional consensus exists.
        direction[signals.crash] = 0.0

        dollar_targets = LIMITS.copy()
        dollar_targets[0] = cfg.algo_limit
        dollars = dollar_targets * direction
        target = np.round(
            dollars / prices[:, length - 1]
        ).astype(int)
        shares[length] = target

        ridge_agrees = (
            ridge_active
            & (direction == signals.ridge_sign)
            & ~signals.crash
        )
        conflict = (
            ridge_active
            & pair_active
            & (np.sign(pair_vote) != signals.ridge_sign)
            & (pair_vote != 0.0)
        )
        ridge_active_count += int(ridge_active.sum())
        ridge_wins += int(ridge_agrees.sum())
        active_count += int((direction != 0.0).sum())
        ridge_conflicts += int(conflict.sum())
        ridge_conflict_wins += int(
            (conflict & ridge_agrees).sum()
        )
    attribution = {
        "ridge_direction_share": (
            ridge_wins / ridge_active_count
            if ridge_active_count
            else 0.0
        ),
        "ridge_conflict_win_rate": (
            ridge_conflict_wins / ridge_conflicts
            if ridge_conflicts
            else 0.0
        ),
        "ridge_active_share": (
            ridge_active_count / active_count
            if active_count
            else 0.0
        ),
    }
    return shares, attribution


def main() -> None:
    all_prices = pd.read_csv(
        ROOT / "prices.txt",
        sep=r"\s+",
    ).to_numpy(float).T
    development_prices = all_prices[:, :750].copy()
    development_signals = precompute(development_prices)
    configs = (
        ConflictCfg(
            pair_mode,
            protected_boost,
            ridge_weight,
            ridge_mode,
            algo_limit,
        )
        for pair_mode in ("equal", "priority")
        for protected_boost in (1.0, 2.0, 4.0)
        for ridge_mode in ("binary", "confidence")
        for ridge_weight in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
        for algo_limit in (0.0, 10_000.0, 100_000.0)
    )

    training_rows = []
    for cfg in configs:
        shares, attribution = targets_for_cfg(
            development_prices,
            development_signals,
            cfg,
        )
        train_fold_scores = tuple(
            account(development_prices, shares, start, end)[2]
            for _, start, end in TRAIN_FOLDS
        )
        train_result = account(
            development_prices,
            shares,
            TRAIN[1],
            TRAIN[2],
        )
        train_objective = (
            min(train_fold_scores) + 0.03 * train_result[2]
        )
        training_rows.append(
            (
                train_objective,
                min(train_fold_scores),
                train_result,
                cfg,
                train_fold_scores,
                attribution,
                shares,
            )
        )

    training_rows.sort(
        key=lambda row: (row[0], row[2][2]),
        reverse=True,
    )
    shortlisted = training_rows[:TRAIN_SHORTLIST_SIZE]
    validation_rows = []
    for row in shortlisted:
        validation_result = account(
            development_prices,
            row[6],
            VALIDATION[1],
            VALIDATION[2],
        )
        validation_rows.append((*row[:6], validation_result, row[6]))
    validation_rows.sort(
        key=lambda row: (row[6][2], row[0]),
        reverse=True,
    )

    print(
        "rank mode boost ridge_mode weight algo | "
        "train_floor train validation | ridge_dir ridge_conflict"
    )
    for rank, row in enumerate(validation_rows, start=1):
        (
            train_objective,
            train_floor,
            train_result,
            cfg,
            train_fold_scores,
            attribution,
            validation_result,
            shares,
        ) = row
        del train_objective, train_fold_scores, shares
        print(
            f"{rank:>4} {cfg.pair_mode:>8} "
            f"{cfg.protected_boost:5.1f} {cfg.ridge_mode:>10} "
            f"{cfg.ridge_weight:5.2f} {cfg.algo_limit / 1000:4.0f}k | "
            f"{train_floor:11.2f} {train_result[2]:5.1f} "
            f"{validation_result[2]:10.2f} | "
            f"{attribution['ridge_direction_share']:8.1%} "
            f"{attribution['ridge_conflict_win_rate']:13.1%}"
        )

    winner = validation_rows[0]
    winner_cfg = winner[3]
    full_signals = precompute(all_prices)
    full_shares, full_attribution = targets_for_cfg(
        all_prices,
        full_signals,
        winner_cfg,
    )
    test_result = account(
        all_prices,
        full_shares,
        TEST[1],
        TEST[2],
    )
    print("\nSelected on validation (test was not used for ranking):")
    print(winner_cfg)
    print(
        f"train_score={winner[2][2]:.2f} "
        f"validation_score={winner[6][2]:.2f} "
        f"test_score={test_result[2]:.2f} "
        f"test_mean={test_result[0]:.2f} "
        f"ridge_direction_share={full_attribution['ridge_direction_share']:.1%}"
    )


if __name__ == "__main__":
    main()

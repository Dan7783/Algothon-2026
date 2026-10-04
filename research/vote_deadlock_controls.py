"""Walk-forward controls for vote abstention, hysteresis, joint resolution,
and a persistence/cost-aware continuous aim portfolio.

The current catalog has no clean holdout.  Days 31-500 estimate structural
quantities, days 501-750 compare families, and 751-1000 is reported only as a
contaminated final in-sample window.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import whole_vector_ev_conflicts as study


OUTPUT_DIR = ROOT / "analysis_outputs" / "vote_deadlock_deepdive"
TRAIN = (31, 500)
VALIDATION = (501, 750)
FINAL = (751, 1000)
REPORT_WINDOWS = (
    ("YEAR1", 31, 250),
    ("YEAR2", 251, 500),
    ("VALIDATION", 501, 750),
    ("FINAL_IN_SAMPLE", 751, 1000),
)
BLOCKS = (
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("VAL_A", 501, 625),
    ("VAL_B", 626, 750),
    ("FINAL_A", 751, 875),
    ("FINAL_B", 876, 1000),
)
ABSTENTION_SWEEP = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
HYSTERESIS_EXIT_RATIOS = (0.0, 0.50, 0.75)
GP_SCALES = (0.25, 0.50, 1.0, 2.0, 4.0, 8.0, 16.0)
ROUND_TRIP_COMMISSION = 2.0 * strategy.LEG * 1.0e-4


@dataclass
class SignalCache:
    prices: np.ndarray
    baseline: np.ndarray
    pair_vote: np.ndarray
    pair_active: np.ndarray
    combined_vote: np.ndarray
    desired_direction: np.ndarray
    ridge_sign: np.ndarray
    ridge_strength: np.ndarray
    crash: np.ndarray
    pair_alpha: np.ndarray
    pair_decay: np.ndarray
    edges: list[list[tuple[int, int, int, int, float, str]]]
    train_beta_relations: dict[str, tuple[int, int, int]]


def _edge_weight(
    priority: float,
    agent: strategy.PairAgent,
) -> float:
    boost = (
        strategy.PROTECTED_VOTE_BOOST
        if strategy._pair_name(agent.spec) in strategy.PROTECTED_PAIR_NAMES
        else 1.0
    )
    if strategy.PAIR_CONFLICT_MODE == "priority":
        return boost * max(float(priority), 1.0e-12)
    return boost


def collect_cache(prices: np.ndarray) -> SignalCache:
    n_history = prices.shape[1]
    shape = (n_history + 1, strategy.N_INST)
    baseline = np.zeros(shape)
    pair_vote_all = np.zeros(shape)
    pair_active_all = np.zeros(shape, dtype=bool)
    combined_all = np.zeros(shape)
    desired_all = np.zeros(shape)
    ridge_sign_all = np.zeros(shape)
    ridge_strength_all = np.zeros(shape)
    crash_all = np.zeros(shape, dtype=bool)
    pair_alpha = np.zeros(shape)
    pair_decay = np.zeros(shape)
    edges: list[list[tuple[int, int, int, int, float, str]]] = [
        [] for _ in range(n_history + 1)
    ]
    atlas = strategy.AtlasBook()
    train_beta_relations: dict[str, tuple[int, int, int]] = {}

    for history_size in range(1, n_history):
        history = prices[:, :history_size]
        candidates = atlas.candidates(history)
        if history_size < strategy.WARMUP + 10:
            continue
        pair_vote, pair_active, _ = strategy._pair_conflict_consensus(
            candidates
        )
        returns = np.diff(
            np.log(np.maximum(history, strategy._EPS_DENOM)),
            axis=1,
        )
        ridge_raw, ridge_strength = strategy._ridge_forecast(returns)
        ridge_sign = np.sign(ridge_raw)
        combined = (
            pair_vote
            + strategy.RIDGE_CONFLICT_WEIGHT * ridge_sign * ridge_strength
        )
        active = pair_active | (ridge_sign != 0.0)
        desired = np.sign(combined)
        tied = active & (desired == 0.0) & (ridge_sign != 0.0)
        desired[tied] = ridge_sign[tied]
        last_return = np.diff(
            np.log(np.maximum(history[:, -2:], strategy._EPS_DENOM)),
            axis=1,
        )[:, 0]
        crash = np.abs(last_return) > 0.40
        crash[0] = False
        desired[crash] = 0.0

        pair_vote_all[history_size] = pair_vote
        pair_active_all[history_size] = pair_active
        combined_all[history_size] = combined
        desired_all[history_size] = desired
        ridge_sign_all[history_size] = ridge_sign
        ridge_strength_all[history_size] = ridge_strength
        crash_all[history_size] = crash
        baseline[history_size] = strategy.BANG_BANG_LIMITS * desired

        alpha_numerator = np.zeros(strategy.N_INST)
        decay_numerator = np.zeros(strategy.N_INST)
        alpha_denominator = np.zeros(strategy.N_INST)
        day_edges: list[tuple[int, int, int, int, float, str]] = []
        for priority, agent, pos_y, pos_x in candidates:
            kappa = max(float(agent.current_kappa), 0.0)
            sigma = max(float(agent.current_sigma), 0.0)
            abs_z = abs(float(agent.current_z))
            expected_move = (
                sigma * abs_z * (1.0 - np.exp(-kappa))
            )
            weight = _edge_weight(priority, agent)
            y = int(agent.spec.y)
            x = int(agent.spec.x)
            qy = int(np.sign(pos_y))
            qx = int(np.sign(pos_x))
            day_edges.append(
                (y, x, qy, qx, float(weight), strategy._pair_name(agent.spec))
            )
            for instrument, position in ((y, pos_y), (x, pos_x)):
                leg_scale = abs(position) / max(strategy.LEG, 1.0e-12)
                signed_alpha = (
                    expected_move * leg_scale * np.sign(position)
                )
                alpha_numerator[instrument] += signed_alpha
                decay_numerator[instrument] += (
                    abs(signed_alpha) * (1.0 - np.exp(-kappa))
                )
                alpha_denominator[instrument] += abs(signed_alpha)
        edges[history_size] = day_edges
        pair_alpha[history_size] = alpha_numerator
        pair_decay[history_size] = np.divide(
            decay_numerator,
            alpha_denominator,
            out=np.zeros(strategy.N_INST),
            where=alpha_denominator > 0.0,
        )

        if history_size == TRAIN[1]:
            for agent in atlas.agents:
                beta = float(agent.filt.beta)
                relation = -int(np.sign(beta)) if beta != 0.0 else -1
                train_beta_relations[strategy._pair_name(agent.spec)] = (
                    int(agent.spec.y),
                    int(agent.spec.x),
                    relation,
                )

    return SignalCache(
        prices=prices,
        baseline=baseline,
        pair_vote=pair_vote_all,
        pair_active=pair_active_all,
        combined_vote=combined_all,
        desired_direction=desired_all,
        ridge_sign=ridge_sign_all,
        ridge_strength=ridge_strength_all,
        crash=crash_all,
        pair_alpha=pair_alpha,
        pair_decay=pair_decay,
        edges=edges,
        train_beta_relations=train_beta_relations,
    )


def score_rows(cache: SignalCache, start: int, end: int) -> np.ndarray:
    score_days = np.arange(len(cache.baseline))
    return (score_days >= start) & (score_days <= end)


def fit_chow_threshold(cache: SignalCache) -> dict[str, Any]:
    train_rows = score_rows(cache, *TRAIN)
    active = (
        train_rows[:, None]
        & (cache.desired_direction != 0.0)
        & ~cache.crash
    )
    history, instrument = np.nonzero(active)
    forward_return = (
        cache.prices[instrument, history]
        / np.maximum(
            cache.prices[instrument, history - 1],
            strategy._EPS_DENOM,
        )
        - 1.0
    )
    correct = (
        cache.desired_direction[history, instrument] * forward_return > 0.0
    ).astype(float)
    confidence = np.abs(cache.combined_vote[history, instrument])
    movement = strategy.LEG * np.abs(forward_return)
    mean_movement = float(np.mean(movement))
    required_probability = float(
        min(
            1.0,
            0.5
            + ROUND_TRIP_COMMISSION
            / max(2.0 * mean_movement, 1.0e-12),
        )
    )
    calibrator = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    )
    calibrator.fit(confidence, correct, sample_weight=np.maximum(movement, 1.0))
    grid = np.linspace(0.0, max(float(np.max(confidence)), 1.0), 2001)
    probability = calibrator.predict(grid)
    eligible = np.flatnonzero(probability >= required_probability)
    tau = float(grid[eligible[0]]) if len(eligible) else float(grid[-1] + 1.0e-6)
    return {
        "tau": tau,
        "required_probability": required_probability,
        "mean_abs_move_dollars": mean_movement,
        "round_trip_commission": ROUND_TRIP_COMMISSION,
        "calibration_rows": int(len(confidence)),
        "train_weighted_accuracy": float(
            np.average(correct, weights=np.maximum(movement, 1.0))
        ),
        "grid_probability_at_tau": float(
            calibrator.predict([min(tau, grid[-1])])[0]
        ),
    }


def abstention_dollars(cache: SignalCache, tau: float) -> np.ndarray:
    direction = cache.desired_direction.copy()
    reject = (
        (np.abs(cache.combined_vote) < tau)
        & (
            cache.pair_active
            | (cache.ridge_sign != 0.0)
        )
    )
    direction[reject] = 0.0
    direction[cache.crash] = 0.0
    return strategy.BANG_BANG_LIMITS[None, :] * direction


def hysteresis_dollars(
    cache: SignalCache,
    enter: float,
    exit_threshold: float,
) -> np.ndarray:
    dollars = np.zeros_like(cache.baseline)
    state = np.zeros(strategy.N_INST)
    for history_size in range(1, len(dollars)):
        desired = cache.desired_direction[history_size]
        confidence = np.abs(cache.combined_vote[history_size])
        for instrument in range(1, strategy.N_INST):
            if cache.crash[history_size, instrument]:
                state[instrument] = 0.0
                continue
            proposed = desired[instrument]
            if state[instrument] == 0.0:
                if proposed != 0.0 and confidence[instrument] >= enter:
                    state[instrument] = proposed
                continue
            if proposed == state[instrument]:
                if confidence[instrument] <= exit_threshold:
                    state[instrument] = 0.0
            elif proposed == 0.0:
                if confidence[instrument] <= exit_threshold:
                    state[instrument] = 0.0
            elif confidence[instrument] >= enter:
                state[instrument] = proposed
            elif confidence[instrument] <= exit_threshold:
                state[instrument] = 0.0
        dollars[history_size] = strategy.BANG_BANG_LIMITS * state
    return dollars


def _joint_objective(
    signs: np.ndarray,
    edges: list[tuple[int, int, int, int, float, str]],
    ridge_sign: np.ndarray,
    ridge_strength: np.ndarray,
) -> float:
    value = 0.0
    for y, x, qy, qx, weight, _ in edges:
        if signs[y] == qy and signs[x] == qx:
            value += weight
    ridge_active = ridge_sign != 0.0
    value += float(
        np.sum(
            strategy.RIDGE_CONFLICT_WEIGHT
            * ridge_strength[ridge_active]
            * (signs[ridge_active] == ridge_sign[ridge_active])
        )
    )
    return value


def _local_search_assignment(
    initial: np.ndarray,
    edges: list[tuple[int, int, int, int, float, str]],
    ridge_sign: np.ndarray,
    ridge_strength: np.ndarray,
) -> np.ndarray:
    active = np.zeros(strategy.N_INST, dtype=bool)
    unary = np.zeros(strategy.N_INST)
    for y, x, qy, qx, weight, _ in edges:
        active[y] = True
        active[x] = True
        unary[y] += weight * qy
        unary[x] += weight * qx
    active |= ridge_sign != 0.0
    unary += (
        strategy.RIDGE_CONFLICT_WEIGHT * ridge_sign * ridge_strength
    )
    base = initial.copy()
    unresolved = active & (base == 0.0)
    base[unresolved] = np.where(unary[unresolved] >= 0.0, 1.0, -1.0)
    base[~active] = 0.0

    best_signs = base
    best_value = _joint_objective(
        best_signs, edges, ridge_sign, ridge_strength
    )
    for seed in (base, np.where(active, -base, 0.0)):
        signs = seed.copy()
        current = _joint_objective(signs, edges, ridge_sign, ridge_strength)
        for _ in range(10):
            improved = False
            for instrument in np.flatnonzero(active):
                candidate = signs.copy()
                candidate[instrument] *= -1.0
                value = _joint_objective(
                    candidate, edges, ridge_sign, ridge_strength
                )
                if value > current + 1.0e-12:
                    signs = candidate
                    current = value
                    improved = True
            if not improved:
                break
        if current > best_value:
            best_signs = signs
            best_value = current
    return best_signs


def joint_dollars(cache: SignalCache) -> tuple[np.ndarray, dict[str, float]]:
    start_time = time.perf_counter()
    dollars = np.zeros_like(cache.baseline)
    baseline_complete = 0.0
    joint_complete = 0.0
    edge_count = 0
    changed_asset_days = 0
    for history_size in range(1, len(dollars)):
        edges = cache.edges[history_size]
        if not edges and not np.any(cache.ridge_sign[history_size]):
            continue
        signs = _local_search_assignment(
            cache.desired_direction[history_size],
            edges,
            cache.ridge_sign[history_size],
            cache.ridge_strength[history_size],
        )
        signs[cache.crash[history_size]] = 0.0
        dollars[history_size] = strategy.BANG_BANG_LIMITS * signs
        changed_asset_days += int(
            np.sum(signs != cache.desired_direction[history_size])
        )
        for y, x, qy, qx, _, _ in edges:
            baseline_complete += float(
                cache.desired_direction[history_size, y] == qy
                and cache.desired_direction[history_size, x] == qx
            )
            joint_complete += float(signs[y] == qy and signs[x] == qx)
            edge_count += 1
    return dollars, {
        "runtime_seconds": float(time.perf_counter() - start_time),
        "edge_observations": edge_count,
        "baseline_complete_agreement": baseline_complete / max(edge_count, 1),
        "joint_complete_agreement": joint_complete / max(edge_count, 1),
        "changed_asset_days": changed_asset_days,
    }


def fit_gp_scales(cache: SignalCache) -> dict[str, np.ndarray]:
    train_rows = score_rows(cache, *TRAIN)
    pair_abs = np.abs(cache.pair_alpha[train_rows])
    ridge_signed = cache.ridge_sign * cache.ridge_strength
    ridge_abs = np.abs(ridge_signed[train_rows])
    pair_scale = np.nanmedian(
        np.where(pair_abs > 0.0, pair_abs, np.nan),
        axis=0,
    )
    ridge_scale = np.nanmedian(
        np.where(ridge_abs > 0.0, ridge_abs, np.nan),
        axis=0,
    )
    pair_fallback = float(np.nanmedian(pair_scale))
    ridge_fallback = float(np.nanmedian(ridge_scale))
    pair_scale = np.nan_to_num(
        pair_scale,
        nan=max(pair_fallback, 1.0e-8),
        posinf=max(pair_fallback, 1.0e-8),
    )
    ridge_scale = np.nan_to_num(
        ridge_scale,
        nan=max(ridge_fallback, 1.0e-8),
        posinf=max(ridge_fallback, 1.0e-8),
    )
    return {
        "pair": np.maximum(pair_scale, 1.0e-8),
        "ridge": np.maximum(ridge_scale, 1.0e-8),
    }


def gp_aim_dollars(
    cache: SignalCache,
    scale: float,
    adjustment_mode: str,
    fitted_scales: dict[str, np.ndarray],
) -> np.ndarray:
    dollars = np.zeros_like(cache.baseline)
    current = np.zeros(strategy.N_INST)
    ridge_signed = cache.ridge_sign * cache.ridge_strength
    normalized_alpha = (
        cache.pair_alpha / fitted_scales["pair"][None, :]
        + strategy.RIDGE_CONFLICT_WEIGHT
        * ridge_signed
        / fitted_scales["ridge"][None, :]
    )
    for history_size in range(1, len(dollars)):
        aim = (
            strategy.BANG_BANG_LIMITS
            * np.clip(scale * normalized_alpha[history_size], -1.0, 1.0)
        )
        aim[cache.crash[history_size]] = 0.0
        if adjustment_mode == "immediate":
            rate = np.ones(strategy.N_INST)
        elif adjustment_mode == "persistence":
            ridge_rate = np.where(
                cache.ridge_sign[history_size] != 0.0,
                0.25,
                0.0,
            )
            rate = np.maximum(
                cache.pair_decay[history_size],
                ridge_rate,
            )
            rate = np.clip(rate, 0.05, 1.0)
        else:
            raise ValueError(adjustment_mode)
        current += rate * (aim - current)
        current = np.clip(
            current,
            -strategy.BANG_BANG_LIMITS,
            strategy.BANG_BANG_LIMITS,
        )
        current[cache.crash[history_size]] = 0.0
        dollars[history_size] = current
    return dollars


def metrics(
    cache: SignalCache,
    dollars: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        name: study.region_metrics(cache.prices, dollars, start, end)
        for name, start, end in REPORT_WINDOWS
    }


def traded_volume(
    cache: SignalCache,
    dollars: np.ndarray,
    start: int,
    end: int,
) -> float:
    prior = np.zeros(strategy.N_INST)
    total = 0.0
    for history_size in range(max(start - 1, 1), end):
        prices = cache.prices[:, history_size - 1]
        shares = np.round(
            dollars[history_size]
            / np.maximum(prices, strategy._EPS_DENOM)
        )
        total += float(prices @ np.abs(shares - prior))
        prior = shares
    return total


def critical_value(trials: int, alpha: float = 0.05) -> float:
    if trials <= 1:
        return float(norm.ppf(1.0 - alpha / 2.0))
    return float(norm.ppf((1.0 - alpha / 2.0) ** (1.0 / trials)))


def block_metrics(
    cache: SignalCache,
    dollars: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        name: study.region_metrics(cache.prices, dollars, start, end)
        for name, start, end in BLOCKS
    }


def position_diagnostics(
    cache: SignalCache,
    dollars: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    direction = np.sign(dollars)
    current = direction[start - 1 : end]
    previous = direction[start - 2 : end - 1]
    opposite_flip = (
        (current != 0.0)
        & (previous != 0.0)
        & (current == -previous)
    )
    any_change = current != previous
    source_active = (
        cache.pair_active[start - 1 : end]
        | (cache.ridge_sign[start - 1 : end] != 0.0)
    )
    flat_with_source = (
        (current == 0.0)
        & source_active
        & ~cache.crash[start - 1 : end]
    )
    return {
        "opposite_sign_flips": int(np.sum(opposite_flip)),
        "position_changes": int(np.sum(any_change)),
        "flat_active_asset_days": int(np.sum(flat_with_source)),
        "mean_active_assets": float(np.mean(np.sum(current != 0.0, axis=1))),
        "traded_volume": traded_volume(cache, dollars, start, end),
    }


def cycle_diagnostics(
    relations: dict[str, tuple[int, int, int]],
    max_length: int = 6,
) -> dict[str, Any]:
    adjacency: dict[int, set[int]] = {
        instrument: set() for instrument in range(strategy.N_INST)
    }
    edge_relation: dict[tuple[int, int], int] = {}
    edge_name: dict[tuple[int, int], str] = {}
    for name, (y, x, relation) in relations.items():
        edge = tuple(sorted((y, x)))
        adjacency[y].add(x)
        adjacency[x].add(y)
        edge_relation[edge] = relation
        edge_name[edge] = name

    cycles: set[tuple[int, ...]] = set()
    nodes = sorted(node for node, peers in adjacency.items() if peers)
    for start in nodes:
        stack: list[tuple[int, tuple[int, ...]]] = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            if len(path) > max_length:
                continue
            for neighbour in adjacency[node]:
                if neighbour == start and len(path) >= 3:
                    forward = path
                    reverse = (path[0],) + tuple(reversed(path[1:]))
                    cycles.add(min(forward, reverse))
                elif (
                    neighbour > start
                    and neighbour not in path
                    and len(path) < max_length
                ):
                    stack.append((neighbour, path + (neighbour,)))

    frustrated = []
    participation: dict[str, int] = defaultdict(int)
    for cycle in sorted(cycles, key=lambda item: (len(item), item)):
        product = 1
        names = []
        for left, right in zip(cycle, cycle[1:] + cycle[:1]):
            edge = tuple(sorted((left, right)))
            product *= edge_relation[edge]
            names.append(edge_name[edge])
        if product == -1:
            frustrated.append(
                {
                    "assets": [
                        strategy._SYMBOL_BY_INDEX[index] for index in cycle
                    ],
                    "pairs": names,
                    "length": len(cycle),
                }
            )
            for name in names:
                participation[name] += 1
    return {
        "enumerated_cycles_up_to_length": max_length,
        "cycle_count": len(cycles),
        "frustrated_cycle_count": len(frustrated),
        "balanced": len(frustrated) == 0,
        "frustrated_cycles": frustrated,
        "pair_frustration_participation": sorted(
            participation.items(),
            key=lambda item: (-item[1], item[0]),
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = study.load_prices(1000)
    cache = collect_cache(prices)
    baseline_metrics = metrics(cache, cache.baseline)
    chow = fit_chow_threshold(cache)

    abstention_rows = []
    abstention_candidates = sorted(set(ABSTENTION_SWEEP + (chow["tau"],)))
    for tau in abstention_candidates:
        dollars = abstention_dollars(cache, tau)
        result = metrics(cache, dollars)
        abstention_rows.append(
            {
                "tau": tau,
                "derived": bool(abs(tau - chow["tau"]) <= 1.0e-12),
                "validation_score": result["VALIDATION"]["score"],
                "validation_delta": (
                    result["VALIDATION"]["score"]
                    - baseline_metrics["VALIDATION"]["score"]
                ),
                "final_score": result["FINAL_IN_SAMPLE"]["score"],
                "final_delta": (
                    result["FINAL_IN_SAMPLE"]["score"]
                    - baseline_metrics["FINAL_IN_SAMPLE"]["score"]
                ),
                "validation_volume": traded_volume(
                    cache, dollars, *VALIDATION
                ),
            }
        )
    abstention_grid = pd.DataFrame(abstention_rows)
    abstention_grid.to_csv(OUTPUT_DIR / "abstention_grid.csv", index=False)

    hysteresis_rows = []
    enter_candidates = sorted(
        set(ABSTENTION_SWEEP + (chow["tau"],))
    )
    for enter in enter_candidates:
        for exit_ratio in HYSTERESIS_EXIT_RATIOS:
            exit_threshold = enter * exit_ratio
            dollars = hysteresis_dollars(cache, enter, exit_threshold)
            result = metrics(cache, dollars)
            hysteresis_rows.append(
                {
                    "enter": enter,
                    "exit": exit_threshold,
                    "exit_ratio": exit_ratio,
                    "derived": bool(
                        abs(enter - chow["tau"]) <= 1.0e-12
                        and exit_ratio == 0.50
                    ),
                    "validation_score": result["VALIDATION"]["score"],
                    "validation_delta": (
                        result["VALIDATION"]["score"]
                        - baseline_metrics["VALIDATION"]["score"]
                    ),
                    "final_score": result["FINAL_IN_SAMPLE"]["score"],
                    "final_delta": (
                        result["FINAL_IN_SAMPLE"]["score"]
                        - baseline_metrics["FINAL_IN_SAMPLE"]["score"]
                    ),
                    "validation_volume": traded_volume(
                        cache, dollars, *VALIDATION
                    ),
                }
            )
    hysteresis_grid = pd.DataFrame(hysteresis_rows)
    hysteresis_grid.to_csv(OUTPUT_DIR / "hysteresis_grid.csv", index=False)

    joint, joint_diagnostics = joint_dollars(cache)
    joint_metrics = metrics(cache, joint)
    cycle = cycle_diagnostics(cache.train_beta_relations)
    (OUTPUT_DIR / "frustrated_cycles.json").write_text(
        json.dumps(cycle, indent=2),
        encoding="utf-8",
    )

    fitted_scales = fit_gp_scales(cache)
    gp_rows = []
    for adjustment_mode in ("immediate", "persistence"):
        for scale in GP_SCALES:
            dollars = gp_aim_dollars(
                cache,
                scale,
                adjustment_mode,
                fitted_scales,
            )
            result = metrics(cache, dollars)
            gp_rows.append(
                {
                    "adjustment_mode": adjustment_mode,
                    "scale": scale,
                    "validation_score": result["VALIDATION"]["score"],
                    "validation_delta": (
                        result["VALIDATION"]["score"]
                        - baseline_metrics["VALIDATION"]["score"]
                    ),
                    "final_score": result["FINAL_IN_SAMPLE"]["score"],
                    "final_delta": (
                        result["FINAL_IN_SAMPLE"]["score"]
                        - baseline_metrics["FINAL_IN_SAMPLE"]["score"]
                    ),
                    "validation_volume": traded_volume(
                        cache, dollars, *VALIDATION
                    ),
                }
            )
    gp_grid = pd.DataFrame(gp_rows)
    gp_grid.to_csv(OUTPUT_DIR / "gp_aim_grid.csv", index=False)

    derived_abstention = abstention_grid[abstention_grid["derived"]].iloc[0]
    derived_hysteresis = hysteresis_grid[hysteresis_grid["derived"]].iloc[0]
    best_abstention = abstention_grid.sort_values(
        "validation_score", ascending=False
    ).iloc[0]
    best_hysteresis = hysteresis_grid.sort_values(
        "validation_score", ascending=False
    ).iloc[0]
    best_gp = gp_grid.sort_values(
        "validation_score", ascending=False
    ).iloc[0]
    candidate_dollars = {
        "baseline": cache.baseline,
        "chow_derived": abstention_dollars(cache, chow["tau"]),
        "chow_validation_best": abstention_dollars(
            cache, float(best_abstention["tau"])
        ),
        "hysteresis_derived": hysteresis_dollars(
            cache, chow["tau"], chow["tau"] * 0.50
        ),
        "hysteresis_validation_best": hysteresis_dollars(
            cache,
            float(best_hysteresis["enter"]),
            float(best_hysteresis["exit"]),
        ),
        "joint_local_search": joint,
        "gp_validation_best": gp_aim_dollars(
            cache,
            float(best_gp["scale"]),
            str(best_gp["adjustment_mode"]),
            fitted_scales,
        ),
    }
    candidate_blocks = {
        name: block_metrics(cache, dollars)
        for name, dollars in candidate_dollars.items()
    }
    candidate_position_diagnostics = {
        name: position_diagnostics(cache, dollars, *VALIDATION)
        for name, dollars in candidate_dollars.items()
    }
    family_trials = {
        "abstention": len(abstention_candidates),
        "hysteresis": len(hysteresis_rows),
        "joint": 1,
        "gp_aim": len(gp_rows),
    }
    total_trials = int(sum(family_trials.values()))
    summary = {
        "protocol": {
            "train": list(TRAIN),
            "validation": list(VALIDATION),
            "final_in_sample": list(FINAL),
            "clean_holdout_available": False,
            "pair_vote_mode": (
                "priority"
                if strategy.PAIR_CONFLICT_MODE == "priority"
                else "equal"
            ),
        },
        "baseline": baseline_metrics,
        "chow": {
            **chow,
            "derived_validation_score": float(
                derived_abstention["validation_score"]
            ),
            "derived_validation_delta": float(
                derived_abstention["validation_delta"]
            ),
            "derived_final_delta": float(
                derived_abstention["final_delta"]
            ),
            "best_validation_grid_row": abstention_grid.sort_values(
                "validation_score", ascending=False
            ).iloc[0].to_dict(),
        },
        "hysteresis": {
            "derived_enter": chow["tau"],
            "derived_exit": chow["tau"] * 0.50,
            "derived_validation_delta": float(
                derived_hysteresis["validation_delta"]
            ),
            "derived_final_delta": float(
                derived_hysteresis["final_delta"]
            ),
            "best_validation_grid_row": hysteresis_grid.sort_values(
                "validation_score", ascending=False
            ).iloc[0].to_dict(),
        },
        "joint": {
            **joint_diagnostics,
            "metrics": joint_metrics,
            "validation_delta": (
                joint_metrics["VALIDATION"]["score"]
                - baseline_metrics["VALIDATION"]["score"]
            ),
            "final_delta": (
                joint_metrics["FINAL_IN_SAMPLE"]["score"]
                - baseline_metrics["FINAL_IN_SAMPLE"]["score"]
            ),
        },
        "structural_balance": {
            key: value
            for key, value in cycle.items()
            if key != "frustrated_cycles"
        },
        "gp_aim": {
            "best_validation_grid_row": best_gp.to_dict(),
        },
        "candidate_blocks": candidate_blocks,
        "validation_position_diagnostics": candidate_position_diagnostics,
        "multiple_testing": {
            "family_trials": family_trials,
            "family_critical_z_5pct": {
                name: critical_value(count)
                for name, count in family_trials.items()
            },
            "total_trials": total_trials,
            "overall_critical_z_5pct": critical_value(total_trials),
            "warning": (
                "Critical z is a lower-bound Gaussian best-of-N threshold; "
                "score deltas are not t-statistics and no clean holdout exists."
            ),
        },
    }
    (OUTPUT_DIR / "control_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_control_report(summary)
    print(json.dumps(summary, indent=2))


def write_control_report(summary: dict[str, Any]) -> None:
    baseline_blocks = summary["candidate_blocks"]["baseline"]
    lines = [
        "# Vote deadlock control families",
        "",
        "Days 751-1000 are contaminated in-sample data, not a holdout.",
        "",
        "## Family results",
        "",
        "| Family | Validation delta | Final-window delta | Trials | Critical z |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Chow derived | {summary['chow']['derived_validation_delta']:.4f} | "
            f"{summary['chow']['derived_final_delta']:.4f} | "
            f"{summary['multiple_testing']['family_trials']['abstention']} | "
            f"{summary['multiple_testing']['family_critical_z_5pct']['abstention']:.3f} |"
        ),
        (
            f"| Hysteresis derived | "
            f"{summary['hysteresis']['derived_validation_delta']:.4f} | "
            f"{summary['hysteresis']['derived_final_delta']:.4f} | "
            f"{summary['multiple_testing']['family_trials']['hysteresis']} | "
            f"{summary['multiple_testing']['family_critical_z_5pct']['hysteresis']:.3f} |"
        ),
        (
            f"| Joint local search | {summary['joint']['validation_delta']:.4f} | "
            f"{summary['joint']['final_delta']:.4f} | 1 | 1.960 |"
        ),
        (
            f"| Continuous aim, validation-best | "
            f"{summary['gp_aim']['best_validation_grid_row']['validation_delta']:.4f} | "
            f"{summary['gp_aim']['best_validation_grid_row']['final_delta']:.4f} | "
            f"{summary['multiple_testing']['family_trials']['gp_aim']} | "
            f"{summary['multiple_testing']['family_critical_z_5pct']['gp_aim']:.3f} |"
        ),
        "",
        "## 125-day block score deltas",
        "",
        "| Candidate | TRAIN_A | TRAIN_B | VAL_A | VAL_B | FINAL_A | FINAL_B |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, blocks in summary["candidate_blocks"].items():
        if candidate == "baseline":
            continue
        deltas = {
            block: blocks[block]["score"] - baseline_blocks[block]["score"]
            for block, _, _ in BLOCKS
        }
        lines.append(
            f"| {candidate} | "
            + " | ".join(f"{deltas[block]:.2f}" for block, _, _ in BLOCKS)
            + " |"
        )
    lines.extend(
        [
            "",
            f"The overall best-of-{summary['multiple_testing']['total_trials']} "
            f"Gaussian lower-bound critical z is "
            f"**{summary['multiple_testing']['overall_critical_z_5pct']:.3f}**.",
            "Score deltas are not t-statistics, so this is a warning threshold,",
            "not a claim of corrected significance.",
        ]
    )
    (OUTPUT_DIR / "control_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
